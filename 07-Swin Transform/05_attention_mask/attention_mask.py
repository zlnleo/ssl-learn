# -*- coding: utf-8 -*-
"""
模块编号: 05
学习顺序: 04 移位窗口 -> 05 注意力掩码 (Swin 核心机制的最后一块拼图)

注意力掩码 (Attention Mask for SW-MSA)
======================================

【问题】模块 04 的循环移位会把空间上不相邻的区域卷进同一个新窗口("伪邻居")。
如果直接 softmax 做注意力, 这些相距最远的 token 会被当成真邻居加权求和,
产生跨整个特征图的错误依赖。

【解法】在 SW-MSA 里, 给注意力分数矩阵上"伪邻居"的位置加 -100。
softmax 后 exp(-100) ≈ 0, 权重严格归零。这个掩码只依赖窗口几何结构
(与输入内容、batch 无关), 因此可以构造一次、缓存复用。

核心对象:
- window_partition: 与模块 04 一致的分窗函数
- build_attn_mask: 用"9 宫格编号"构造 (nW, M^2, M^2) 的掩码, 0=允许, -100=屏蔽
"""

import sys

import torch

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def build_attn_mask(H: int, W: int, window_size: int, shift_size: int,
                    device: str = "cpu") -> torch.Tensor:
    """构造 SW-MSA 的注意力掩码, 形状 (nW, M^2, M^2), 0=允许注意力, -100=屏蔽。

    H, W 是移位窗口注意力实际分窗的特征图尺寸(本项目即 pad 到 window_size 整数倍后的尺寸)。

    原理: 把移位后的图按 9 宫格切成 9 块(3 行 x 3 列 slice), 每块一个编号 0..8;
    每个窗口内两个 token 若来自不同编号块, 则它们空间上不相邻(是 roll 带来的伪邻居), 必须屏蔽。
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)              # (nW, M, M, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)     # (nW, M^2)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # (nW, M^2, M^2) 区域号之差
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


if __name__ == "__main__":
    print("=" * 70)
    print("模块 05 演示: SW-MSA 注意力掩码")
    print("=" * 70)

    H = W = 8
    window_size = 4
    shift_size = 2
    mask = build_attn_mask(H, W, window_size, shift_size)
    nW = (H // window_size) * (W // window_size)
    N = window_size * window_size

    print(f"\nH=W={H}, window={window_size}, shift={shift_size}")
    print(f"mask 形状 = {tuple(mask.shape)}, 期望 (nW, M^2, M^2) = ({nW}, {N}, {N})")
    print(f"mask 取值集合 = {sorted(set(mask.flatten().tolist()))} (只含 0 与 -100)")
    print(f"被屏蔽元素数(-100) = {(mask == -100).sum().item()} / {mask.numel()}")
    print(f"对角线全为 0(自己永远可见): {bool((mask.diagonal(dim1=1, dim2=2) == 0).all())}")
    print(f"对称性 mask[i,j]==mask[j,i]: {bool((mask == mask.transpose(1, 2)).all())}")

    # shift_size=0 时等价无 mask(全 0)
    mask0 = build_attn_mask(H, W, window_size, 0)
    print(f"\nshift_size=0 时 mask 全 0(等价无 mask): {bool((mask0 == 0).all())}")

    print("\n演示完成 ✔")
