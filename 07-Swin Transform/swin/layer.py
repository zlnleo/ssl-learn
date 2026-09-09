"""BasicLayer：一个 stage = depth 个 SwinBlock（W/SW 交替）+ 可选 PatchMerging。

对应学习模块：08（BasicLayer）。
"""
import torch
import torch.nn as nn

from .block import SwinBlock
from .patch import PatchMerging

__all__ = ["BasicLayer"]


class BasicLayer(nn.Module):
    """一个层级（stage）。

    - depth 个 SwinBlock：偶数位 shift_size=0（W-MSA）、奇数位 shift_size=window_size//2（SW-MSA），
      保证每个 stage 内跨窗口信息能够流动。
    - 末尾可选 PatchMerging：分辨率减半、通道翻倍（最后一个 stage 不接）。
    - drop_path 可为标量或逐块列表（stochastic depth 沿深度线性增长时传列表）。
    """

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0.,
                 attn_drop: float = 0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample: nn.Module = None):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)                     # (B, L, C) 形状不变
        if self.downsample is not None:
            x = self.downsample(x, H, W)         # (B, L/4, 2C)
            H, W = (H + 1) // 2, (W + 1) // 2    # 分辨率减半（奇数向上取整）
        return x, H, W
