"""完整 Swin Transformer（组装 01-08 全部机制）+ Swin-Tiny/S/B/L 工厂函数。

对应学习模块：09（完整 Swin）。
"""
import torch
import torch.nn as nn

from .block import SwinBlock
from .layer import BasicLayer
from .patch import PatchEmbed, PatchMerging

__all__ = ["SwinTransformer", "swin_tiny", "swin_small", "swin_base", "swin_large", "build_swin"]


class SwinTransformer(nn.Module):
    """完整 Swin Transformer。

    数据流（以 Swin-Tiny, img 224 为例）：
      PatchEmbed (224x224 -> 56x56, C=96)
      -> Stage1: 2 blocks (W,SW) + Merging -> 28x28, C=192
      -> Stage2: 2 blocks (W,SW) + Merging -> 14x14, C=384
      -> Stage3: 6 blocks (W,SW)x3 + Merging -> 7x7, C=768
      -> Stage4: 2 blocks (W,SW)
      -> LN -> 全局平均池化 -> 分类头

    参数：
      patch_merging=False 时禁用全部 PatchMerging（消融实验 3 专用）：
      所有 stage 保持 embed_dim 与 num_heads[0]，分辨率不变，形成"无层级"对照模型。
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 1000, embed_dim: int = 96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size: int = 7, mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm,
                 patch_norm: bool = True, patch_merging: bool = True):
        super().__init__()
        assert len(depths) == len(num_heads), "depths 与 num_heads 长度必须一致"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_merging = patch_merging

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim,
                                      norm_layer if patch_norm else None)

        # stochastic depth：drop_path 沿块索引从 0 线性增长到 drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if patch_merging:
                dim, n_head = int(embed_dim * 2 ** i), num_heads[i]
                downsample = PatchMerging(dim=dim, norm_layer=norm_layer) \
                    if i < self.num_layers - 1 else None
            else:
                dim, n_head = embed_dim, num_heads[0]
                downsample = None
            layer = BasicLayer(
                dim=dim, depth=depths[i], num_heads=n_head, window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer, downsample=downsample)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features if patch_merging else embed_dim)
        self.head = nn.Linear(self.num_features if patch_merging else embed_dim, num_classes) \
            if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.patch_embed(x)                  # (B, L, C)
        for layer in self.layers:
            x, H, W = layer(x, H, W)             # H, W 随 PatchMerging 减半
        x = self.norm(x)                         # (B, L, C)
        return x.mean(dim=1)                     # (B, C) 全局平均池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def swin_tiny(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-Tiny：embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window=7。"""
    kwargs.setdefault("window_size", 7)
    return SwinTransformer(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                           num_classes=num_classes, **kwargs)


def swin_small(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-S：96 / (2,2,18,2) / (3,6,12,24)。"""
    kwargs.setdefault("window_size", 7)
    return SwinTransformer(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
                           num_classes=num_classes, **kwargs)


def swin_base(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-B：128 / (2,2,18,2) / (4,8,16,32)。"""
    kwargs.setdefault("window_size", 7)
    return SwinTransformer(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                           num_classes=num_classes, **kwargs)


def swin_large(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-L：192 / (2,2,18,2) / (6,12,24,48)。"""
    kwargs.setdefault("window_size", 7)
    return SwinTransformer(embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
                           num_classes=num_classes, **kwargs)


def build_swin(name: str, num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """按名称构建：name ∈ {tiny, small, base, large}。"""
    factories = {"tiny": swin_tiny, "small": swin_small, "base": swin_base, "large": swin_large}
    if name not in factories:
        raise ValueError(f"未知模型 {name!r}，可选: {list(factories)}")
    return factories[name](num_classes=num_classes, **kwargs)
