"""Patch 化与层级降采样：PatchEmbed（切 patch）与 PatchMerging（2x2 合并）。

对应学习模块：06（Patch Merging）。PatchEmbed 属于 09 完整组装的一部分。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PatchEmbed", "PatchMerging"]


class PatchEmbed(nn.Module):
    """用 patch_size x patch_size、stride=patch_size 的卷积把图像切成 patch。

    (B, 3, H, W) -> (B, (H/p)*(W/p), embed_dim)
    """

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96,
                 norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, (H/p)*(W/p), embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """把相邻 2x2 patch 合并成一个，实现层级降采样：

    (B, H*W, C) -> (B, (H/2)*(W/2), 2C)

    步骤：reshape 回 2D -> 按 (行奇偶, 列奇偶) 分成 4 路，每路 (B, H/2, W/2, C)
    -> 通道拼接为 4C -> LayerNorm -> Linear(4C -> 2C)。
    效果 = 分辨率减半 + 通道翻倍（与 CNN 的 pooling + channel doubling 同构）。
    """

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:              # 奇数尺寸保护（标准配置下均为偶数）
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]                  # (B, H/2, W/2, C) 左上
        x1 = x[:, 1::2, 0::2, :]                  # 左下
        x2 = x[:, 0::2, 1::2, :]                  # 右上
        x3 = x[:, 1::2, 1::2, :]                  # 右下
        x = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)                  # (B, (H/2)*(W/2), 4C)
        x = self.reduction(self.norm(x))          # (B, (H/2)*(W/2), 2C)
        return x
