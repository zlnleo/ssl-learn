"""窗口多头自注意力（完整版：相对位置偏置 + SW-MSA 掩码广播）。

对应学习模块：01（最简版注意力）、03（相对位置偏置）、05（mask 广播）。
"""
import torch
import torch.nn as nn

from .window import build_relative_position_index

__all__ = ["WindowAttention"]


class WindowAttention(nn.Module):
    """窗口内多头自注意力。

    输入：x (B_, N, C)，其中 B_ = B * nW（所有窗口按 batch 拼接），N = M^2。
    输出：(B_, N, C)。

    计算流：
      QKV 投影 -> 拆多头 -> 缩放点积 -> + 相对位置偏置 -> + 掩码(-100) -> softmax -> AV -> 输出投影
    掩码广播：mask (nW, N, N) 在 batch 间共享（窗口几何结构对所有样本一致），
    通过 view(B, nW, h, N, N) + mask[None, :, None] 广播。
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) 必须能被 num_heads({num_heads}) 整除"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5          # 缩放点积：防止点积方差随 d 增长导致 softmax 饱和
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 可学习相对位置偏置表：窗口内所有相对位移共 (2M-1)^2 种，每种每个头一个参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer("relative_position_index",
                             build_relative_position_index(window_size), persistent=False)
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        # (B_,N,3C) -> (B_,N,3,h,d) -> (3,B_,h,N,d)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # 各 (B_, h, N, d)
        attn = (q * self.scale) @ k.transpose(-2, -1)           # (B_, h, N, N)

        # 相对位置偏置：table[(M^2,M^2)] -> (M^2,M^2,h) -> permute -> (1,h,N,N) 广播
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]                                   # 单样本窗口数
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)           # 屏蔽位 -100 -> softmax 后 ≈ 0

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)       # (B_, N, C)
        return self.proj_drop(self.proj(out))
