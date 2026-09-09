# -*- coding: utf-8 -*-
"""
模块 07：Swin Block（W-MSA / SW-MSA 全部机制的总装）
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

本模块实现 Swin Transformer 的基本块，包含：
  - window_partition / window_reverse      ：把特征图切成 7x7 窗口再还原
  - build_relative_position_index          ：相对位置偏置的查表索引
  - build_attn_mask                        ：SW-MSA 的循环移位注意力掩码
  - Mlp / DropPath                         ：前馈网络与随机深度
  - WindowAttention                        ：窗口内多头自注意力（含相对位置偏置与 mask）
  - SwinBlock                              ：pre-norm + (W/SW-MSA) + 残差 + pre-norm + MLP + 残差

设计说明：本实现对任意尺寸特征图都先 pad 到 window_size 的整数倍（官方实现假设整除），
移位、mask 均在 pad 后的坐标系进行，最后 crop 回原尺寸；整除情形下与官方行为完全一致。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe swin_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 依赖的小部件（自包含，供后续模块复用）
# ----------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window_size, window_size, C)。

    把特征图按 window_size 切成不重叠的窗口，每个窗口展平成一个 batch 元素。
    """
    B, H, W, C = x.shape
    # 先按网格切： (B, H/M, M, W/M, M, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute 把窗口网格的 (行网格, 列网格) 提到前面： (B, nH, nW, M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 展平 batch 与窗口网格 -> (B*nW, M, M, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """window_partition 的逆运算：(B*nW, M, M, C) -> (B, H, W, C)。"""
    # 由窗口总数反推 batch：nW = (H/M)*(W/M)
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def build_relative_position_index(window_size: int) -> torch.Tensor:
    """构造相对位置偏置查表索引，返回 (M^2, M^2) 的整型索引。

    对窗口内 M^2 个 token 两两配对，计算其相对坐标 (dx, dy)，量化为
    [-(M-1), M-1] 范围后映射到一维偏置表的下标。
    """
    # coords: (2, M, M)，通道 0 是行坐标、通道 1 是列坐标
    coords = torch.stack(torch.meshgrid(torch.arange(window_size),
                                        torch.arange(window_size), indexing="ij"))
    coords = coords.reshape(2, -1)                       # (2, M^2)
    # rel: (2, M^2, M^2)，rel[0] = dx, rel[1] = dy
    rel = coords[:, :, None] - coords[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()              # (M^2, M^2, 2)
    # 把 dx, dy 平移到 [0, 2M-2]
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    # 行方向乘以 (2M-1)，做成 (2M-1)x(2M-1) 二维表的行优先一维下标
    rel[:, :, 0] *= 2 * window_size - 1
    return rel.sum(-1)                                   # (M^2, M^2)


def build_attn_mask(H: int, W: int, window_size: int, shift_size: int,
                    device: str = "cpu") -> torch.Tensor:
    """构造 SW-MSA 的注意力掩码 (nW, M^2, M^2)。

    思想：循环移位后，部分窗口内的 token 来自原图不相邻区域，它们之间不该互相
    注意。用 9 宫格编号标记每个 token 的原区域，窗口内编号不同的两 token 掩码为 -100。
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)
    # 把 (H, W) 按移位切分线分成 3x3 共 9 块
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt                      # 给 9 块分别标 0..8
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # (nW, M, M, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)  # (nW, M^2)
    # 同一窗口内，编号不同 -> 掩码 -100；编号相同 -> 0
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, M^2, M^2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask                                        # (nW, M^2, M^2)


class Mlp(nn.Module):
    """两层 MLP：fc1 升维 -> GELU -> fc2 降维，中间用 Dropout。"""
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """drop_path：训练时以 drop_prob 概率把整条残差支路置零，推理时恒等。
    除以 keep_prob 保持期望不变（stochastic depth）。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 掩码 shape (B, 1, 1, ...)，按 batch 元素整条支路随机置零
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = noise.floor()          # 以 keep_prob 概率为 1，其余为 0
        return x / keep_prob * noise   # 除以 keep_prob 保持期望不变


class WindowAttention(nn.Module):
    """完整版窗口注意力：含相对位置偏置与 mask 广播。
    输入 x (B_, N, C)，B_ = B*nW，N = M^2；mask (nW, N, N) 或 None。"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5              # 缩放因子 1/sqrt(d)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)    # 一次投影出 Q,K,V
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)                      # 多头拼接后的输出投影
        self.proj_drop = nn.Dropout(proj_drop)
        # 相对位置偏置表：(2M-1)^2 种相对位置 x h 头
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        # 索引表是常量，注册为 buffer（不参与训练、不参与 state_dict）
        self.register_buffer("relative_position_index",
                             build_relative_position_index(window_size), persistent=False)
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # (B_, N, 3C) -> (B_, N, 3, h, d) -> (3, B_, h, N, d)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # 各 (B_, h, N, d)
        attn = (q * self.scale) @ k.transpose(-2, -1)       # (B_, h, N, N)
        # 相对位置偏置：按索引查表 -> (N, N, h) -> (h, N, N) -> (1, h, N, N)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        if mask is not None:
            # mask (nW, N, N) 广播到 (B, nW, h, N, N) 再压回 (B_, h, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        return self.proj_drop(self.proj(out))


class SwinBlock(nn.Module):
    """Swin Transformer 基本块：pre-norm + (W-MSA 或 SW-MSA) + 残差 + pre-norm + MLP + 残差。
    shift_size=0 -> W-MSA；shift_size=window_size//2 -> SW-MSA。"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self._mask_cache = None
        self._mask_key = None

    def _get_mask(self, H, W, device):
        """惰性缓存：mask 只与 (H, W, window_size, shift_size) 有关，与输入内容无关，可复用。"""
        key = (H, W, str(device))
        if key != self._mask_key:
            self._mask_cache = build_attn_mask(H, W, self.window_size, self.shift_size, device=device) \
                if self.shift_size > 0 else None
            self._mask_key = key
        return self._mask_cache

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, L, C)，L = H*W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)                 # pre-norm 后恢复 2D

        # 1) pad 到 window_size 整数倍（整除时 pad_r/pad_b 均为 0）
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))           # 只补右下 (B, Hp, Wp, C)
        Hp, Wp = H + pad_b, W + pad_r

        # 2) 循环移位（仅 SW-MSA），在 pad 后的坐标系进行
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # 3) 分窗 + 注意力
        x = window_partition(x, self.window_size)          # (B*nW, M, M, C)
        x = x.view(-1, self.window_size ** 2, C)           # (B*nW, M^2, C)
        x = self.attn(x, mask=self._get_mask(Hp, Wp, x.device))  # (B*nW, M^2, C)
        x = x.view(-1, self.window_size, self.window_size, C)    # (B*nW, M, M, C)

        # 4) 还原：reverse -> unshift -> crop
        x = window_reverse(x, self.window_size, Hp, Wp)    # (B, Hp, Wp, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :H, :W, :].contiguous().view(B, L, C)     # crop 回原尺寸

        x = shortcut + self.drop_path(x)                    # 残差 1
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # 残差 2
        return x


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)

    print("=" * 70)
    print("SwinBlock 演示")
    print("=" * 70)

    B, H, W, C = 2, 8, 8, 96
    window_size, num_heads = 4, 3
    x = torch.randn(B, H * W, C)

    print(f"输入 x 形状: {tuple(x.shape)}  (H={H}, W={W}, C={C})")

    # W-MSA（shift=0）
    blk_w = SwinBlock(dim=C, num_heads=num_heads, window_size=window_size, shift_size=0)
    y_w = blk_w(x, H, W)
    print(f"W-MSA block 输出: {tuple(y_w.shape)}（应与输入一致）")

    # SW-MSA（shift=window_size//2）
    blk_sw = SwinBlock(dim=C, num_heads=num_heads, window_size=window_size,
                       shift_size=window_size // 2)
    y_sw = blk_sw(x, H, W)
    print(f"SW-MSA block 输出: {tuple(y_sw.shape)}（应与输入一致）")

    # 参数量
    total = sum(p.numel() for p in blk_w.parameters())
    print(f"\nW-MSA block 参数量: {total:,}")

    # 相对位置索引（M=4 时 (2M-1)^2 = 49 种相对位置）
    idx = build_relative_position_index(window_size)
    print(f"相对位置索引形状: {tuple(idx.shape)}，取值范围 [0, {(2 * window_size - 1) ** 2 - 1}]")

    # mask 形状演示：SwinBlock 内部会先把 10x10 pad 到 12x12（4 的整数倍）再建 mask
    mask = build_attn_mask(12, 12, 4, 2)
    print(f"\nSW-MSA mask (H=W=12, M=4, shift=2) 形状: {tuple(mask.shape)}")
    print(f"mask 中 -100 元素数: {(mask == -100).sum().item()}，0 元素数: {(mask == 0).sum().item()}")
