# -*- coding: utf-8 -*-
"""
模块 06：Patch Merging（相邻 patch 拼接降采样）
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer
本模块对应 Swin Transformer 中每个 stage 结束后的降采样操作。

作用：把 (B, H*W, C) 的特征序列下采样为 (B, (H/2)*(W/2), 2C)。
机制：先把序列 reshape 回 2D，按 (行奇偶, 列奇偶) 把相邻 2x2 patch 拆成 4 路，
每路分辨率减半，再把 4 路沿通道维拼接成 4C，最后 LayerNorm + Linear(4C -> 2C)。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe patch_merging.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMerging(nn.Module):
    """把相邻 2x2 patch 拼接降采样: (B, H*W, C) -> (B, (H/2)*(W/2), 2C)。

    步骤：reshape 回 2D -> 按 (行奇偶, 列奇偶) 分成 4 路，每路 (B, H/2, W/2, C)
    -> 通道拼接成 4C -> LayerNorm -> Linear(4C -> 2C)。
    """

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        # LayerNorm 作用在拼接后的 4C 维上（先归一化再线性降维，稳定训练）
        self.norm = norm_layer(4 * dim)
        # 4C -> 2C 的线性变换，无偏置
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, L, C) = (B, H*W, C)
        B, L, C = x.shape
        x = x.view(B, H, W, C)  # (B, H, W, C)

        # 奇数尺寸保护：本项目里各 stage 设计尺寸均为偶数，此分支不会触发
        if H % 2 == 1 or W % 2 == 1:
            # F.pad 顺序：最后两维是 (左, 右, 上, 下)；这里只补右下
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 按 (行奇偶, 列奇偶) 拆成 4 路，每路 (B, H/2, W/2, C)
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C) 左上
        x1 = x[:, 1::2, 0::2, :]  # 左下
        x2 = x[:, 0::2, 1::2, :]  # 右上
        x3 = x[:, 1::2, 1::2, :]  # 右下

        # 4 路沿通道维拼接 -> (B, H/2, W/2, 4C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        # 展平空间维 -> (B, (H/2)*(W/2), 4C)
        x = x.view(B, -1, 4 * C)
        # LayerNorm(4C) -> Linear(4C -> 2C) -> (B, (H/2)*(W/2), 2C)
        x = self.reduction(self.norm(x))
        return x


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)
    print("=" * 70)
    print("PatchMerging 演示")
    print("=" * 70)

    B, H, W, C = 2, 8, 8, 96
    x = torch.randn(B, H * W, C)
    print(f"输入 x 形状: {tuple(x.shape)}   (B={B}, H={H}, W={W}, C={C})")

    model = PatchMerging(dim=C)
    y = model(x, H, W)
    print(f"输出 y 形状: {tuple(y.shape)}")
    print(f"预期输出:    (B={B}, (H/2)*(W/2)={(H // 2) * (W // 2)}, 2C={2 * C})")
    assert y.shape == (B, (H // 2) * (W // 2), 2 * C), "输出形状不符合预期"

    # 参数量统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量：LayerNorm(4C) 的权重+偏置 = 2*4C，Linear(4C->2C) = 4C*2C")
    print(f"        C={C} 时总计 = {2 * 4 * C + 4 * C * 2 * C} = {n_params}")

    # 奇数尺寸保护演示
    print("\n--- 奇数尺寸保护演示 (H=5, W=5) ---")
    x_odd = torch.randn(1, 5 * 5, 4)
    y_odd = PatchMerging(4)(x_odd, 5, 5)
    print(f"输入 (1, 25, 4), H=W=5 -> 输出 {tuple(y_odd.shape)}（预期 (1, 9, 8)，(5+1)//2=3，3*3=9）")
