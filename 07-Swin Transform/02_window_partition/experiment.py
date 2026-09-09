# -*- coding: utf-8 -*-
"""
模块 02 / 学习顺序第 2 步：实验 "Window Partition / Reverse"
================================================================
用「位置 id 张量」直观验证窗口划分的正确性：

    每个位置的数值 = 它的全局行优先序号 id = h*W + w

这样 partition 之后，每个窗口内的数字**直接就是它原来在特征图中的全局坐标序号**，
一眼就能看出「第 (i,j) 个窗口覆盖原图哪些位置」。同时：

    1. 打印每个窗口的 id 集合，验证窗口编号顺序 = 行优先；
    2. 随机张量 partition → reverse 的逐元素误差（应为精确 0）；
    3. 打印一张 id 映射的 ASCII 图（4×4 图、M=2，标注窗口边界）。

用法：
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe experiment.py
"""

import sys

import torch

from window_partition import window_partition, window_reverse

# Windows 控制台默认 GBK 编码；强制 UTF-8 保证中文输出不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def part1_position_id(B: int, H: int, W: int, C: int, M: int) -> None:
    print("=" * 78)
    print("实验 A：位置 id 张量 → 每个窗口的 id 集合")
    print("=" * 78)
    print(f"设定: B={B}, H={H}, W={W}, C={C}, M={M}  （每个位置的数值 = 全局序号 h*W+w）")
    print()

    # 位置 id：单通道先造 (H,W)，再复制到 C 个通道，保证每个通道都看到同一套 id。
    id_map = torch.arange(H * W).reshape(1, H, W, 1).expand(B, H, W, C).clone()
    win = window_partition(id_map, M)

    nW = win.shape[0] // B
    print(f"partition 输出形状: {tuple(win.shape)}  →  B*nW = {B}*{nW} = {B*nW} 个窗口")
    print()

    # 打印每个窗口的 id（取第 0 通道即可，各通道相同）
    for b in range(B):
        for k in range(nW):
            # 窗口编号 k 对应的 (i,j)：i = k // (W//M), j = k % (W//M)
            i, j = k // (W // M), k % (W // M)
            ids = win[b * nW + k, :, :, 0].long().flatten().tolist()
            print(f"  batch={b}  窗口 k={k:2d}  (i={i}, j={j})  id = {ids}")

    print()
    print("  [验证] 手算第 (0,0) 窗口应覆盖 行[0,2)×列[0,2)：全局 id = {0,1,4,5}")
    print("         第 (0,1) 窗口应覆盖 行[0,2)×列[2,4)：全局 id = {2,3,6,7}，依此类推。")
    print("         若打印结果与之一致 → 划分正确，窗口编号为行优先。")


def part2_reverse_exactness() -> None:
    print()
    print("=" * 78)
    print("实验 B：随机张量 partition → reverse 的逐元素误差")
    print("=" * 78)
    torch.manual_seed(42)
    for (B, H, W, C, M) in [(2, 8, 8, 3, 2), (4, 16, 8, 6, 4), (1, 12, 12, 1, 3)]:
        x = torch.randn(B, H, W, C)
        back = window_reverse(window_partition(x, M), M, H, W)
        err = (back - x).abs().max().item()
        exact = torch.equal(back, x)
        print(f"  (B={B}, H={H}, W={W}, C={C}, M={M})  "
              f"max|err| = {err:.3e}   精确相等 = {exact}")
    print()
    print("  [结论] 误差恒为 0 —— view+permute 只是重排内存解释，不改变任何数值。")


def part3_ascii_id_map() -> None:
    print()
    print("=" * 78)
    print("实验 C：id 映射的 ASCII 图（4×4 图，M=2）")
    print("=" * 78)
    H = W = 4
    M = 2
    ids = torch.arange(H * W).reshape(1, H, W, 1).expand(1, H, W, 1)
    win = window_partition(ids, M)  # (4, 2, 2, 1)

    # 打印原图 id
    print("原图（数字 = 全局行优先 id）:")
    print("    " + " ".join(f"c{j}" for j in range(W)))
    for h in range(H):
        print(f"  r{h} " + " ".join(f"{h*W + w:2d}" for w in range(W)))

    # 打印窗口边界图：每 2x2 一个窗口
    print()
    print("窗口边界示意（4 个 2×2 窗口）:")
    for h in range(H):
        row = ""
        for w in range(W):
            row += f"{h*W + w:2d}"
            row += " |" if (w + 1) % M == 0 and w + 1 < W else " "
        print(f"  {row}")
        if (h + 1) % M == 0 and h + 1 < H:
            print("  " + "----" * W)

    print()
    print("partition 后每个窗口的 id（行优先编号）:")
    for k in range(win.shape[0]):
        print(f"  window {k}: {win[k, :, :, 0].long().flatten().tolist()}")
    print()
    print("  [读法] window 0 = {0,1,4,5}（左上）、window 1 = {2,3,6,7}（右上）、")
    print("         window 2 = {8,9,12,13}（左下）、window 3 = {10,11,14,15}（右下）。")
    print("         对照「窗口边界示意」图，可直观确认每个窗口覆盖的位置。")


if __name__ == "__main__":
    part1_position_id(B=2, H=4, W=4, C=2, M=2)
    part2_reverse_exactness()
    part3_ascii_id_map()
