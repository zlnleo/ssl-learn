# -*- coding: utf-8 -*-
"""
模块编号: 04
学习顺序: 04 移位窗口 (本文件是"debug 实验"脚本)

experiment.py —— 移位窗口的四个数值实验:

1) 构造位置 id 张量 (1, 8, 8, 1), id = i*8 + j, 打印 roll 前后的 id 矩阵 (ASCII)。
2) 统计移位后某个窗口内 id 的"行跨度/列跨度", 证明新窗口跨越了旧窗口边界。
3) 验证 unroll(roll(x)) == x。
4) 演示无 mask 时, 同一新窗口内混入了哪些空间上不相邻的区域 (用 id 集合展示)。
"""

import sys

import torch

from shifted_window import window_partition, cyclic_shift, cyclic_unshift

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def print_id_matrix(title: str, mat: torch.Tensor):
    """打印 8x8 的 id 矩阵 (ASCII), 每个元素是 0..63 的整数。"""
    print(title)
    h, w = mat.shape
    print("   " + " ".join(f"{c:>3}" for c in range(w)))
    print("   " + "---" * w)
    for i in range(h):
        print(f"{i:2d} " + " ".join(f"{v:3d}" for v in mat[i].tolist()))


def experiment_1_roll_before_after():
    print("=" * 72)
    print("实验 1: 位置 id 张量 (1,8,8,1) 在 roll 前后的对比")
    print("=" * 72)
    H = W = 8
    ids = torch.arange(H * W).view(1, H, W, 1)   # (1, 8, 8, 1), id = i*8 + j
    print("\n[roll 前] 原始位置 id (id = i*8 + j):")
    print_id_matrix("", ids[0, :, :, 0])

    shifted = cyclic_shift(ids, shift_size=2)
    print("\n[roll 后] 向左上循环移位 2 像素后:")
    print_id_matrix("", shifted[0, :, :, 0])
    return ids, shifted


def experiment_2_cross_boundary(shifted: torch.Tensor):
    print("\n" + "=" * 72)
    print("实验 2: 移位后新窗口跨越旧边界 (行/列跨度证明)")
    print("=" * 72)
    H = W = 8
    M = 4
    windows = window_partition(shifted, M)       # (4, 4, 4, 1), 2x2 个窗口
    nH = nW = H // M                              # 2

    print(f"移位后按 {M}x{M} 分窗, 得到 {nH}x{nW} = {nH*nW} 个窗口 (行优先编号):")
    for r in range(nH):
        for c in range(nW):
            w = windows[r * nW + c].squeeze(-1)       # (4, 4)
            ids = w.flatten().tolist()
            orig_rows = sorted({i // W for i in ids})  # id//8 = 原始行号
            orig_cols = sorted({i % W for i in ids})   # id%8  = 原始列号
            # 判断是否跨越边界: 连续 4 行(列)不跨; 有间隙即跨
            row_cross = orig_rows != list(range(min(orig_rows), min(orig_rows) + M))
            col_cross = orig_cols != list(range(min(orig_cols), min(orig_cols) + M))
            print(f"  窗口({r},{c}) [id={r*nW+c}]: 原始行={orig_rows}, 原始列={orig_cols} "
                  f"| 跨行边界={row_cross}, 跨列边界={col_cross}")

    # 重点: 窗口(1,1) 跨越了行与列两个边界
    w33 = windows[3].squeeze(-1)
    ids33 = w33.flatten().tolist()
    orig_rows33 = sorted({i // W for i in ids33})
    orig_cols33 = sorted({i % W for i in ids33})
    print(f"\n窗口(1,1) 的原始行集合 = {orig_rows33} (应为 {{0,1,6,7}}, 跨了行边界)")
    print(f"窗口(1,1) 的原始列集合 = {orig_cols33} (应为 {{0,1,6,7}}, 跨了列边界)")
    assert orig_rows33 == [0, 1, 6, 7], orig_rows33
    assert orig_cols33 == [0, 1, 6, 7], orig_cols33
    print("结论: 新窗口跨越旧窗口边界 ✔ (跨窗信息得以流动)")
    return windows


def experiment_3_unroll_roll_identity(ids: torch.Tensor):
    print("\n" + "=" * 72)
    print("实验 3: 验证 unroll(roll(x)) == x")
    print("=" * 72)
    rolled = cyclic_shift(ids, 2)
    restored = cyclic_unshift(rolled, 2)
    ok = bool(torch.equal(restored, ids))
    print(f"cyclic_unshift(cyclic_shift(x, 2), 2) == x : {ok}")
    assert ok
    print("可逆性成立 ✔ (循环移位一个元素不丢、可精确还原)")


def experiment_4_no_mask_mixing(shifted: torch.Tensor, windows: torch.Tensor):
    print("\n" + "=" * 72)
    print("实验 4: 无 mask 时, 同一新窗口内混入了哪些不相邻区域")
    print("=" * 72)
    W = 8
    M = 4
    w33 = windows[3].squeeze(-1)             # 窗口(1,1), (4,4)
    ids = w33.flatten().tolist()
    print(f"窗口(1,1) 内 16 个 token 的 id 集合 (按原始图坐标 (i,j) 分组):")
    # 按原始行/列把 16 个 id 归入 4 个角块
    blocks = {
        "左上角 (行0-1,列0-1)": [i for i in ids if i // W in (0, 1) and i % W in (0, 1)],
        "右上角 (行0-1,列6-7)": [i for i in ids if i // W in (0, 1) and i % W in (6, 7)],
        "左下角 (行6-7,列0-1)": [i for i in ids if i // W in (6, 7) and i % W in (0, 1)],
        "右下角 (行6-7,列6-7)": [i for i in ids if i // W in (6, 7) and i % W in (6, 7)],
    }
    for name, blk in blocks.items():
        print(f"  {name}: {sorted(blk)}")

    n_disjoint = sum(1 for b in blocks.values() if b)
    print(f"\n这 16 个 token 来自 {n_disjoint} 个空间上互不相邻的角块。")
    print("若无 mask, 这些相距最远的 token 会被当作邻居加权求和 —— 这就是模块 05 要屏蔽的'伪邻居'。")
    assert n_disjoint == 4
    assert sum(len(b) for b in blocks.values()) == 16
    print("演示完成 ✔")


if __name__ == "__main__":
    ids, shifted = experiment_1_roll_before_after()
    windows = experiment_2_cross_boundary(shifted)
    experiment_3_unroll_roll_identity(ids)
    experiment_4_no_mask_mixing(shifted, windows)
    print("\n" + "=" * 72)
    print("全部实验完成, 无报错 ✔")
    print("=" * 72)
