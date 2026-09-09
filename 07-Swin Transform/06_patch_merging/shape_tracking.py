# -*- coding: utf-8 -*-
"""
模块 06：Patch Merging 的 Tensor Shape 逐步跟踪
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

逐行打印 PatchMerging 内部每一步的 shape，并带断言校验，方便建立对
"reshape 回 2D -> 4 路切分 -> 拼接 -> LN -> Linear" 全流程的直观印象。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe shape_tracking.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def trace_patch_merging(x: torch.Tensor, H: int, W: int, dim: int,
                        norm_layer=nn.LayerNorm, verbose: bool = True):
    """逐步跟踪 PatchMerging 的每个形状变化，返回最终输出。"""
    def log(name, t):
        if verbose:
            print(f"  {name:<28} shape = {tuple(t.shape)}")

    B, L, C = x.shape
    assert L == H * W, f"序列长度 {L} 与 H*W={H*W} 不一致"
    log("输入 x", x)

    x2d = x.view(B, H, W, C)
    log("reshape 回 2D", x2d)
    assert x2d.shape == (B, H, W, C)

    padded = False
    if H % 2 == 1 or W % 2 == 1:
        padded = True
        x2d = F.pad(x2d, (0, 0, 0, W % 2, 0, H % 2))
        log("奇数尺寸 pad 之后", x2d)
    Hp, Wp = x2d.shape[1], x2d.shape[2]

    x0 = x2d[:, 0::2, 0::2, :]
    x1 = x2d[:, 1::2, 0::2, :]
    x2 = x2d[:, 0::2, 1::2, :]
    x3 = x2d[:, 1::2, 1::2, :]
    log("x0 (左上)", x0)
    log("x1 (左下)", x1)
    log("x2 (右上)", x2)
    log("x3 (右下)", x3)
    for t in (x0, x1, x2, x3):
        assert t.shape == (B, Hp // 2, Wp // 2, C)

    x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
    log("4 路拼接 (4C)", x_cat)
    assert x_cat.shape == (B, Hp // 2, Wp // 2, 4 * C)

    x_flat = x_cat.view(B, -1, 4 * C)
    log("展平空间维", x_flat)
    assert x_flat.shape == (B, (Hp // 2) * (Wp // 2), 4 * C)

    norm = norm_layer(4 * C)
    reduction = nn.Linear(4 * C, 2 * C, bias=False)
    x_norm = norm(x_flat)
    log("LayerNorm(4C)", x_norm)
    assert x_norm.shape == (B, (Hp // 2) * (Wp // 2), 4 * C)

    out = reduction(x_norm)
    log("Linear(4C -> 2C)", out)
    assert out.shape == (B, (Hp // 2) * (Wp // 2), 2 * C)

    if verbose:
        print(f"  {'(pad 分支命中)' if padded else '(偶数尺寸，无 pad)'}")
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)
    print("=" * 70)
    print("Shape Tracking 1：标准偶数尺寸 (B=2, H=8, W=8, C=96)")
    print("=" * 70)
    x = torch.randn(2, 8 * 8, 96)
    trace_patch_merging(x, 8, 8, 96)

    print("\n" + "=" * 70)
    print("Shape Tracking 2：奇数尺寸 pad 分支 (B=1, H=5, W=5, C=4)")
    print("=" * 70)
    x_odd = torch.randn(1, 5 * 5, 4)
    out_odd = trace_patch_merging(x_odd, 5, 5, 4)
    print(f"奇数输入输出形状: {tuple(out_odd.shape)}，期望 (1, (5+1)//2 * (5+1)//2, 8) = (1, 9, 8)")

    print("\n全部 shape 断言通过。")
