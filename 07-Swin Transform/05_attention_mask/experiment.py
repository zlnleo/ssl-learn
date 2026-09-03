# -*- coding: utf-8 -*-
"""
模块编号: 05
学习顺序: 05 注意力掩码 (本文件是"debug 实验"脚本)

experiment.py —— 注意力掩码的四个数值实验 (H=8, W=8, window=4, shift=2):

1) 打印 9 宫格编号图 (ASCII, 0..8 共 9 个区域)。
2) 打印每个窗口的 mask (非零位置画 X 的字符画)。
3) 统计被屏蔽元素数量 (每窗口 + 总计)。
4) 验证 mask 值域只含 {0, -100}。
"""

import sys

import torch

from attention_mask import build_attn_mask

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_region_map(H, W, window_size, shift_size):
    """构造 9 宫格编号图 (与 build_attn_mask 内部逻辑一致), 供可视化。"""
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    return img_mask


def experiment_1_region_map(H, W, M, S):
    print("=" * 72)
    print("实验 1: 9 宫格编号图")
    print("=" * 72)
    img_mask = build_region_map(H, W, M, S)
    grid = img_mask[0, :, :, 0].long()
    print(f"\nH=W={H}, window={M}, shift={S}: 3 行 slice × 3 列 slice 切成 9 块:")
    print("切片: 行 [0:4], [4:6], [6:8]; 列 [0:4], [4:6], [6:8]")
    print("\n区域编号 (每个格子的值 = 它所属的 9 宫格块编号):")
    print("   " + " ".join(f"{c:>3}" for c in range(W)))
    print("   " + "---" * W)
    for i in range(H):
        print(f"{i:2d} " + " ".join(f"{v:3d}" for v in grid[i].tolist()))
    return img_mask


def experiment_2_mask_art(H, W, M, S, mask):
    print("\n" + "=" * 72)
    print("实验 2: 每个窗口的 mask (X = 屏蔽, . = 允许)")
    print("=" * 72)
    nW = mask.shape[0]
    N = M * M
    for w in range(nW):
        print(f"\n[窗口 {w}] mask ({N}x{N}), 屏蔽数 = {(mask[w] == -100).sum().item()}:")
        m = mask[w]
        # 打印字符画 (X = -100 屏蔽, . = 0 允许)
        for i in range(N):
            line = "".join("X" if m[i, j].item() != 0 else "." for j in range(N))
            print("  " + line)
    return mask


def experiment_3_count_masked(mask):
    print("\n" + "=" * 72)
    print("实验 3: 统计被屏蔽元素数量")
    print("=" * 72)
    nW = mask.shape[0]
    total = 0
    for w in range(nW):
        c = int((mask[w] == -100).sum().item())
        total += c
        print(f"  窗口 {w}: 屏蔽 {c} 个 / 共 {mask[w].numel()} 个")
    print(f"  总计: 屏蔽 {total} 个 / 共 {mask.numel()} 个")
    assert total == 448, total   # 0 + 128 + 128 + 192
    print("  与理论值 448 一致 ✔")
    return total


def experiment_4_value_range(mask):
    print("\n" + "=" * 72)
    print("实验 4: 验证 mask 值域只含 {0, -100}")
    print("=" * 72)
    vals = set(mask.flatten().tolist())
    print(f"  mask 取值集合 = {vals}")
    assert vals == {0.0, -100.0}, vals
    diag_ok = bool((mask.diagonal(dim1=1, dim2=2) == 0).all())
    sym_ok = bool((mask == mask.transpose(1, 2)).all())
    print(f"  对角线全 0 (自己永远可见): {diag_ok}")
    print(f"  对称性 mask[i,j]==mask[j,i]: {sym_ok}")
    assert diag_ok and sym_ok
    print("  通过 ✔")


if __name__ == "__main__":
    H = W = 8
    M = 4
    S = 2
    img_mask = experiment_1_region_map(H, W, M, S)
    mask = build_attn_mask(H, W, M, S)
    experiment_2_mask_art(H, W, M, S, mask)
    experiment_3_count_masked(mask)
    experiment_4_value_range(mask)
    print("\n" + "=" * 72)
    print("全部实验完成, 无报错 ✔")
    print("=" * 72)
