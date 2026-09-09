# -*- coding: utf-8 -*-
"""
模块 02 / 学习顺序第 2 步：Window Partition / Reverse（窗口划分与还原）
================================================================================
本模块解决「特征图 (B,H,W,C) 与窗口序列 (B*nW, M, M, C) 之间如何来回转换」的问题。
它是模块 01（WindowAttention）的**上游枢纽**：

    (B, H, W, C)  ──window_partition──►  (B*nW, M, M, C)  ──view──►  (B_, N, C) 序列
                                                                          │
                                                                    [模块 01] 注意力
                                                                          │
    (B, H, W, C)  ◄──window_reverse───  (B*nW, M, M, C)  ◄──view──  (B_, N, C) 序列

核心思想：这一切转换**不搬运任何数据**，只是改变「如何解释同一块内存」——
即 `view`（重解释形状）+ `permute`（换索引轴序）。因此必须精确理解索引映射公式，
以及为什么 `permute` 之后要先 `contiguous()` 才能再 `view(-1)`。

窗口按**行优先**编号：第 (i, j) 个窗口 = 原图行 [i*M:(i+1)*M]、列 [j*M:(j+1)*M]。
例如 4×4 图、M=2 时，窗口编号如下（每个数字代表一个 2×2 窗口）：

        ┌─────────┬─────────┐
        │  (0,0)  │  (0,1)  │    窗口 (0,0) 覆盖 行0-1、列0-1
        ├─────────┼─────────┤    窗口 (0,1) 覆盖 行0-1、列2-3
        │  (1,0)  │  (1,1)  │    窗口 (1,0) 覆盖 行2-3、列0-1
        └─────────┴─────────┘    窗口 (1,1) 覆盖 行2-3、列2-3

本文件只提供两个纯函数（无参数、无状态），与官方实现接口一致。
"""

import sys

import torch

# Windows 控制台默认 GBK 编码；强制 UTF-8 保证中文输出不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window_size, window_size, C)，nW = (H//M)*(W//M)
    窗口按行优先编号：第 (i,j) 个窗口 = 原图行 [i*M:(i+1)*M]、列 [j*M:(j+1)*M]。
    """
    B, H, W, C = x.shape
    # (B,H,W,C) -> (B, H/M, M, W/M, M, C)：把高、宽各拆成 (窗口个数, 窗口边长)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 把窗口编号维度 (H/M, W/M) 提到前面，得到 (B, H/M, W/M, M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """window_partition 的逆运算：(B*nW, window_size, window_size, C) -> (B, H, W, C)"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


if __name__ == "__main__":
    print("=" * 70)
    print("window_partition / window_reverse 演示入口")
    print("=" * 70)
    B, H, W, C, M = 2, 4, 4, 3, 2

    # 用"位置 id"张量：每个位置的值 = 其全局行优先序号 h*W + w
    id_map = torch.arange(H * W).reshape(1, H, W, 1).expand(B, H, W, C).clone().float()

    win = window_partition(id_map, M)
    print(f"输入 id_map 形状 : {tuple(id_map.shape)}  (每个位置值 = 全局序号 h*W+w)")
    print(f"partition 输出形状: {tuple(win.shape)}  (B*nW={B*4}, M={M}, M={M}, C={C})")
    print()

    print("每个窗口的 id 集合（取每窗口第 1 个通道）:")
    for b in range(B):
        for k in range(win.shape[0] // B):
            wid = win[b * 4 + k, :, :, 0].long()
            print(f"  batch={b}, window={k}: {wid.tolist()}")
    print()

    back = window_reverse(win, M, H, W)
    print(f"reverse 输出形状  : {tuple(back.shape)}")
    print(f"reverse(partition(x)) == x : {torch.equal(back, id_map)}")
