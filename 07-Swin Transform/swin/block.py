"""Swin Transformer 基本块：pre-norm + (W-MSA / SW-MSA) + 残差 + pre-norm + MLP + 残差。

对应学习模块：07（Swin Block）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import WindowAttention
from .utils import Mlp, DropPath
from .window import window_partition, window_reverse, build_attn_mask

__all__ = ["SwinBlock"]


class SwinBlock(nn.Module):
    """Swin 基本块。

    shift_size = 0          -> W-MSA（常规窗口）
    shift_size = window//2  -> SW-MSA（循环移位窗口 + 注意力掩码）

    结构（pre-norm 双残差）：
        x ──┬─ LN ── (W/SW-MSA) ── DropPath ──(+)── LN ── MLP ── DropPath ──(+)── 输出
            └─────────────────────────────────┘      └───────────────────────────┘

    设计说明：本实现对任意尺寸特征图先 pad 到 window_size 的整数倍，移位与掩码
    均在 pad 后的坐标系进行，最后 crop 回原尺寸。标准配置（56/28/14/7 与 window 7）
    下整除成立、不发生 pad，与官方实现行为完全一致；pad 分支让 window_size 消融
    实验（如 window=14 作用于 7x7 特征图）也能正常运行。
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size 必须满足 0 <= shift_size < window_size"
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        # mask 惰性缓存：mask 只依赖 (H, W, shift, window_size) 与设备，
        # 与输入内容、batch 无关，同一几何配置只需构造一次
        self._mask_cache = None
        self._mask_key = None

    def _get_mask(self, H: int, W: int, device) -> torch.Tensor:
        key = (H, W, str(device))
        if key != self._mask_key:
            self._mask_cache = build_attn_mask(H, W, self.window_size, self.shift_size,
                                               device=device) if self.shift_size > 0 else None
            self._mask_key = key
        return self._mask_cache

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)          # pre-norm 后回到 2D 布局

        # 1) pad 到 window_size 整数倍（pad 在 roll 之前，保证 roll 与分窗在统一坐标系）
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # 2) 循环移位（仅 SW-MSA）
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # 3) 分窗 + 窗口注意力
        x = window_partition(x, self.window_size)                 # (B*nW, M, M, C)
        x = x.view(-1, self.window_size ** 2, C)                  # (B*nW, M^2, C)
        x = self.attn(x, mask=self._get_mask(Hp, Wp, x.device))   # (B*nW, M^2, C)
        x = x.view(-1, self.window_size, self.window_size, C)

        # 4) 还原：reverse -> unshift -> crop 回原尺寸
        x = window_reverse(x, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :H, :W, :].contiguous().view(B, L, C)

        x = shortcut + self.drop_path(x)                          # 残差 1（注意力支路）
        x = x + self.drop_path(self.mlp(self.norm2(x)))           # 残差 2（MLP 支路）
        return x
