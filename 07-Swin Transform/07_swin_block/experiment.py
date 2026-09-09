# -*- coding: utf-8 -*-
"""
模块 07：Swin Block 实验（项目“实验2：Window vs Shifted Window”的机制部分）
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

实验目标：
1. 感受野/连接性：用位置 id 网格推算出每一层注意力的“连接矩阵”，比较
   “W-MSA + W-MSA” 与 “W-MSA + SW-MSA” 两层之后每个 token 能看到的 token 集合。
   打印对比表与 ASCII 覆盖图，证明 shift 让信息跨窗口流动、W-W 则永远困在本窗口。
2. 打印 SW-MSA 注意力 mask 的惰性缓存命中情况。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe experiment.py
"""

import torch

import swin_block as sb
from swin_block import SwinBlock, window_partition, build_attn_mask


def connectivity(H: int, W: int, M: int, shift: int) -> torch.Tensor:
    """返回 (H*W, H*W) 的 0/1 布尔连接矩阵 C，C[i, j]=True 表示 token i 会与 token j
    在窗口注意力中交互（同一窗口内；SW 时还要满足 mask 允许）。

    计算方式与 SwinBlock 完全一致：pad -> roll -> window_partition -> 逐窗口配对，
    SW 情形用 build_attn_mask 去掉跨区域对。
    """
    pad_r = (M - W % M) % M
    pad_b = (M - H % M) % M
    Hp, Wp = H + pad_b, W + pad_r

    # 位置 id 网格 (1, Hp, Wp, 1)：原图 token 标 0..HW-1，pad 区标 -1
    ids = torch.full((1, Hp, Wp, 1), -1, dtype=torch.long)
    grid = torch.arange(H * W, dtype=torch.long).view(H, W)
    ids[0, :H, :W, 0] = grid
    if shift > 0:
        ids = torch.roll(ids, shifts=(-shift, -shift), dims=(1, 2))

    win = window_partition(ids, M).view(-1, M * M)         # (nW, M^2) 每窗口内的 id
    mask = build_attn_mask(Hp, Wp, M, shift) if shift > 0 else None  # (nW, M^2, M^2)

    C = torch.zeros(H * W, H * W, dtype=torch.bool)
    for k in range(win.shape[0]):
        ids_k = win[k]
        for a in range(M * M):
            ia = ids_k[a].item()
            if ia < 0:
                continue
            for b in range(M * M):
                ib = ids_k[b].item()
                if ib < 0:
                    continue
                # mask 值 0 表示允许交互；-100 表示屏蔽（跨区域）
                if mask is None or mask[k, a, b] == 0:
                    C[ia, ib] = True
    return C


def receptive_field(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """两层堆叠后的感受野：R[j, i] = 存在 k 使 C2[j,k] 且 C1[k,i]。
    即 R = (C2 @ C1) > 0（布尔矩阵乘）。"""
    return (C2.float() @ C1.float()) > 0


def coverage_map(R: torch.Tensor, H: int, W: int, j: int):
    """打印某个输出 token j 的感受野 ASCII 覆盖图。"""
    r_j, c_j = j // W, j % W
    lines = []
    lines.append("    " + "".join(f"{c:>2}" for c in range(W)))
    for r in range(H):
        row = f" r{r:<2} "
        for c in range(W):
            i = r * W + c
            if i == j:
                row += " O"   # 该 token 自身
            elif R[j, i]:
                row += " #"   # 可看到
            else:
                row += " ."   # 看不到
        lines.append(row)
    return "\n".join(lines)


def bbox(R_row: torch.Tensor, H: int, W: int):
    """返回某 token 感受野的空间范围 (min_r, max_r, min_c, max_c)。"""
    idx = torch.nonzero(R_row).flatten().tolist()
    rows = [i // W for i in idx]
    cols = [i % W for i in idx]
    return (min(rows), max(rows), min(cols), max(cols))


def main():
    torch.manual_seed(0)
    H = W = 8
    M = 4
    shift = M // 2
    print("=" * 70)
    print("实验：W-MSA+W-MSA vs W-MSA+SW-MSA 的感受野 / 连接性")
    print("=" * 70)
    print(f"网格 {H}x{H}，window_size={M}，shift_size={shift}，共 {H*W} 个 token\n")

    # 第一层恒为 W-MSA（shift=0）
    C1 = connectivity(H, W, M, 0)
    # 第二层两种选择
    C2_ww = connectivity(H, W, M, 0)       # W-MSA（不移动窗口）
    C2_sw = connectivity(H, W, M, shift)   # SW-MSA（循环移位）

    R_ww = receptive_field(C1, C2_ww)
    R_sw = receptive_field(C1, C2_sw)

    # 汇总统计
    sizes_ww = R_ww.sum(dim=1).tolist()
    sizes_sw = R_sw.sum(dim=1).tolist()
    print(f"{'配置':<28} {'最小看到':>8} {'最大看到':>8} {'平均看到':>8}")
    print(f"{'W-MSA + W-MSA':<28} {min(sizes_ww):>8} {max(sizes_ww):>8} {sum(sizes_ww)/len(sizes_ww):>8.1f}")
    print(f"{'W-MSA + SW-MSA':<28} {min(sizes_sw):>8} {max(sizes_sw):>8} {sum(sizes_sw)/len(sizes_sw):>8.1f}")
    print("\n结论：W-W 两层后每个 token 最多看到自己窗口内的 16 个 token（M^2），")
    print("      W-SW 两层后感受野成倍扩大（跨窗口流动），且每层窗口数量恒定。\n")

    # 取中心 token (3,3) 和 (4,4) 作为代表
    for j in (3 * W + 3, 4 * W + 4):
        print("-" * 70)
        print(f"代表 token ({j // W},{j % W}) 的 ASCII 覆盖图（# 能看到，O 自身，. 看不到）")
        print("-" * 70)
        print(f"\n[W-MSA + W-MSA]  感受野大小 = {R_ww[j].sum().item()}")
        print(coverage_map(R_ww, H, W, j))
        rmin, rmax, cmin, cmax = bbox(R_ww[j], H, W)
        print(f"空间范围: 行[{rmin},{rmax}] 列[{cmin},{cmax}]（困在本窗口内）\n")

        print(f"[W-MSA + SW-MSA] 感受野大小 = {R_sw[j].sum().item()}")
        print(coverage_map(R_sw, H, W, j))
        rmin, rmax, cmin, cmax = bbox(R_sw[j], H, W)
        print(f"空间范围: 行[{rmin},{rmax}] 列[{cmin},{cmax}]（跨窗口扩大）\n")

    # ---- mask 惰性缓存命中情况 ----
    print("=" * 70)
    print("mask 惰性缓存命中情况")
    print("=" * 70)
    calls = {"n": 0}
    orig_build = sb.build_attn_mask

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig_build(*args, **kwargs)

    sb.build_attn_mask = counting
    try:
        blk = SwinBlock(dim=96, num_heads=3, window_size=M, shift_size=shift)
        x = torch.randn(2, H * W, 96)
        for _ in range(3):
            blk(x, H, W)     # 同一 (H, W, device) 反复前向
        print(f"同一 (H,W) 前向 3 次后，build_attn_mask 实际调用次数 = {calls['n']}")
        print(f"mask 对象已缓存（第 2、3 次前向命中缓存，不重复构造）。")
        blk2 = SwinBlock(dim=96, num_heads=3, window_size=M, shift_size=shift)
        x12 = torch.randn(2, 12 * 12, 96)
        blk2(x12, 12, 12)     # 不同尺寸 -> 需要新 mask
        print(f"换用 H=W=12 后，累计调用次数 = {calls['n']}（新尺寸触发一次新构造）")
    finally:
        sb.build_attn_mask = orig_build

    print("\n实验完成。")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
