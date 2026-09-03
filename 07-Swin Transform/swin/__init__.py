"""模块化工程版 Swin Transformer 包。

组装顺序（也是学习顺序）：
  window.py    -> 窗口划分/还原、相对位置索引、注意力掩码   （学习模块 02/03/05）
  attention.py -> 窗口多头自注意力（相对偏置 + 掩码广播）   （学习模块 01/03/05）
  block.py     -> SwinBlock（W-MSA/SW-MSA + 双残差 MLP）     （学习模块 07）
  layer.py     -> BasicLayer（blocks + PatchMerging）        （学习模块 08）
  patch.py     -> PatchEmbed / PatchMerging                  （学习模块 06/09）
  model.py     -> 完整 SwinTransformer + Swin-Tiny/S/B/L     （学习模块 09）
  utils.py     -> Mlp / DropPath

用法：
  from swin import swin_tiny
  model = swin_tiny(num_classes=10)
"""
from .model import (SwinTransformer, swin_tiny, swin_small, swin_base, swin_large,
                    build_swin)
from .layer import BasicLayer
from .block import SwinBlock
from .attention import WindowAttention
from .patch import PatchEmbed, PatchMerging
from .window import window_partition, window_reverse, build_relative_position_index, build_attn_mask
from .config import SWIN_CONFIGS

__all__ = [
    "SwinTransformer", "swin_tiny", "swin_small", "swin_base", "swin_large", "build_swin",
    "BasicLayer", "SwinBlock", "WindowAttention", "PatchEmbed", "PatchMerging",
    "window_partition", "window_reverse", "build_relative_position_index", "build_attn_mask",
    "SWIN_CONFIGS",
]
__version__ = "1.0.0"
