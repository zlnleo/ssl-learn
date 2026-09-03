# -*- coding: utf-8 -*-
"""
模块 08：BasicLayer 实验
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

实验目标：
1. 构造两 stage 小模型（56x56 起步，window=7，dim 96->192），打印每 stage 前后 (H,W,C)。
2. 参数量统计：每 stage 占比 + ASCII 条形图。
3. 粗算 MACs 分布（每 stage 占比 + ASCII 条形图）。
4. 打印 drop_path 列表随深度线性增长的值。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe experiment.py
"""

import torch

from basic_layer import (BasicLayer, PatchMerging, SwinBlock,
                         linear_drop_path_schedule)


def stage_params(layer: BasicLayer) -> int:
    """统计一个 stage 的参数量（含 blocks 与 downsample）。"""
    return sum(p.numel() for p in layer.parameters())


def estimate_stage_macs(dim: int, depth: int, H: int, W: int, window_size: int,
                        has_downsample: bool) -> int:
    """粗算一个 stage 的 MACs（乘加数）。

    每 block：投影+MLP ≈ 12 * hw * C^2；窗口注意力 ≈ 2 * hw * M^2 * C。
    PatchMerging（若有）：≈ hw_out * 8C^2 = 2 * hw * C^2。
    """
    hw = H * W
    C = dim
    per_block = 12 * hw * C * C + 2 * hw * window_size * window_size * C
    macs = depth * per_block
    if has_downsample:
        macs += 2 * hw * C * C
    return macs


def ascii_bar(label: str, value: int, total: int, width: int = 40):
    """打印一行带 ASCII 条形图的占比。"""
    frac = value / total if total > 0 else 0.0
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"  {label:<18} {value:>12,}  {frac*100:>5.1f}%  |{bar}|"


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("实验：两 stage 小模型的形状、参数量与计算量分布")
    print("=" * 70)

    # ---- 构建两 stage 模型 ----
    B, H, W = 1, 56, 56
    x = torch.randn(B, H * W, 96)
    stage1 = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7,
                        downsample=PatchMerging(96))
    stage2 = BasicLayer(dim=192, depth=2, num_heads=6, window_size=7,
                        downsample=PatchMerging(192))

    print(f"\n输入: {tuple(x.shape)}  (H={H}, W={W}, C=96)")
    print("\n每 stage 前后的 (H, W, C)：")
    print(f"  stage1 前: (H={H}, W={W}, C=96)")
    x, H, W = stage1(x, H, W)
    print(f"  stage1 后: (H={H}, W={W}, C={x.shape[-1]})")
    print(f"  stage2 前: (H={H}, W={W}, C=192)")
    x, H, W = stage2(x, H, W)
    print(f"  stage2 后: (H={H}, W={W}, C={x.shape[-1]})")

    # ---- 参数量统计 ----
    p1 = stage_params(stage1)
    p2 = stage_params(stage2)
    p_total = p1 + p2
    print("\n" + "=" * 70)
    print("参数量统计（每 stage 占比，ASCII 条形图）")
    print("=" * 70)
    print(ascii_bar("stage1 (96->192)", p1, p_total))
    print(ascii_bar("stage2 (192->384)", p2, p_total))
    print(f"  合计: {p_total:,}")

    # ---- 粗算 MACs 分布 ----
    m1 = estimate_stage_macs(96, 2, 56, 56, 7, has_downsample=True)
    m2 = estimate_stage_macs(192, 2, 28, 28, 7, has_downsample=True)
    m_total = m1 + m2
    print("\n" + "=" * 70)
    print("粗算 MACs 分布（每 stage 占比，ASCII 条形图）")
    print("=" * 70)
    print(ascii_bar("stage1", m1, m_total))
    print(ascii_bar("stage2", m2, m_total))
    print(f"  合计 MACs: {m_total:,}")

    # ---- drop_path 线性增长 ----
    print("\n" + "=" * 70)
    print("drop_path（stochastic depth）按深度线性增长")
    print("=" * 70)
    for depth, rate in ((4, 0.2), (6, 0.3), (8, 0.5)):
        sched = linear_drop_path_schedule(depth, rate)
        print(f"  depth={depth}, rate={rate}: " + ", ".join(f"{v:.3f}" for v in sched))
    print("  含义：浅层 block 几乎不被跳过，越深 block 被整条跳过的概率越大。")

    print("\n实验完成。")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
