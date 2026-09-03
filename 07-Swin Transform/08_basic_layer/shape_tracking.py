# -*- coding: utf-8 -*-
"""
模块 08：BasicLayer 的 Tensor Shape 逐步跟踪
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

逐 stage 打印形状演化（含每个 block 内部形状不变、PatchMerging 后的分辨率/通道变化），
并断言最终形状正确。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe shape_tracking.py
"""

import torch

from basic_layer import BasicLayer, PatchMerging


def trace_layer(layer: BasicLayer, x: torch.Tensor, H: int, W: int, name: str, verbose=True):
    """跟踪一个 stage：打印进入时的形状，逐 block 确认形状不变，打印降采样后形状。"""
    if verbose:
        print(f"\n[stage: {name}]")
        print(f"  输入  (B, L, C) = {tuple(x.shape)}  (H={H}, W={W})")
    B, L, C = x.shape
    assert L == H * W

    for i, blk in enumerate(layer.blocks):
        x = blk(x, H, W)
        assert x.shape == (B, L, C), f"block {i} 后形状应不变"
        if verbose:
            print(f"  block {i} (shift={blk.shift_size}) 后形状不变: {tuple(x.shape)}  "
                  f"(H={H}, W={W}, C={C})")

    if layer.downsample is not None:
        x = layer.downsample(x, H, W)
        H, W = (H + 1) // 2, (W + 1) // 2
        if verbose:
            print(f"  PatchMerging 降采样后: {tuple(x.shape)}  (H={H}, W={W}, C={x.shape[-1]})")
    else:
        if verbose:
            print(f"  无降采样，输出形状不变: {tuple(x.shape)}  (H={H}, W={W})")

    return x, H, W


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)

    print("=" * 70)
    print("Shape Tracking：两 stage（56 -> 28 -> 14，96 -> 192 -> 384）")
    print("=" * 70)

    B, H, W = 1, 56, 56
    x = torch.randn(B, H * W, 96)

    stage1 = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7,
                        downsample=PatchMerging(96))
    stage2 = BasicLayer(dim=192, depth=2, num_heads=6, window_size=7,
                        downsample=PatchMerging(192))

    x, H, W = trace_layer(stage1, x, H, W, "stage1 (dim 96, depth 2, +downsample)")
    assert (H, W, x.shape[-1]) == (28, 28, 192)
    x, H, W = trace_layer(stage2, x, H, W, "stage2 (dim 192, depth 2, +downsample)")
    assert (H, W, x.shape[-1]) == (14, 14, 384)

    print("\n" + "=" * 70)
    print("Shape Tracking：downsample=None 时形状不变")
    print("=" * 70)
    x2 = torch.randn(1, 16 * 16, 96)
    stage_none = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7, downsample=None)
    x2, H2, W2 = trace_layer(stage_none, x2, 16, 16, "stage (downsample=None)")
    assert (H2, W2, x2.shape[-1]) == (16, 16, 96)
    assert x2.shape == (1, 16 * 16, 96)

    print("\n全部 shape 断言通过。")
