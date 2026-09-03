# -*- coding: utf-8 -*-
"""
模块 02 / 学习顺序第 2 步：window_partition / window_reverse 的 Tensor Shape 跟踪
================================================================================
用 (B=2, H=8, W=8, C=3)、window_size=2 的例子，逐步打印 partition 与 reverse
每一步的形状，并断言 reverse(partition(x)) == x（精确可逆，误差 = 0）。

partition 的形状流转（B=2, H=W=8, C=3, M=2）：

    x        (B, H, W, C)              = (2, 8, 8, 3)
     | view(B, H/M, M, W/M, M, C)
     v
    x6       (B, H/M, M, W/M, M, C)    = (2, 4, 2, 4, 2, 3)
     | permute(0,1,3,2,4,5)             —— 把窗口编号 (H/M, W/M) 提到前
     v
    xp       (B, H/M, W/M, M, M, C)    = (2, 4, 4, 2, 2, 3)
     | contiguous().view(-1, M, M, C)
     v
    windows  (B*nW, M, M, C)           = (32, 2, 2, 3)

reverse 是 partition 的镜像：

    windows  (B*nW, M, M, C)           = (32, 2, 2, 3)
     | view(B, H/M, W/M, M, M, C)
     v
    x6       (B, H/M, W/M, M, M, C)    = (2, 4, 4, 2, 2, 3)
     | permute(0,1,3,2,4,5)
     v
    xp       (B, H/M, M, W/M, M, C)    = (2, 4, 2, 4, 2, 3)
     | contiguous().view(B, H, W, C)
     v
    back     (B, H, W, C)              = (2, 8, 8, 3)
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


def step(label: str, t: torch.Tensor) -> None:
    print(f"{label:<34} shape = {str(tuple(t.shape)):<18} strides = {t.stride()}")


def trace_partition_and_reverse() -> None:
    B, H, W, C, M = 2, 8, 8, 3, 2
    x = torch.arange(B * H * W * C, dtype=torch.float32).reshape(B, H, W, C)

    print("=" * 78)
    print("PARTITION：(B,H,W,C) -> (B*nW, M, M, C)")
    print("=" * 78)
    step("x (B,H,W,C)", x)

    x6 = x.view(B, H // M, M, W // M, M, C)
    step("view (B,H/M,M,W/M,M,C)", x6)

    xp = x6.permute(0, 1, 3, 2, 4, 5)
    step("permute (B,H/M,W/M,M,M,C)", xp)
    print(f"      [观察] permute 后 is_contiguous = {xp.is_contiguous()}  ← 换轴后内存不连续")

    xc = xp.contiguous()
    print(f"      [观察] contiguous() 后 is_contiguous = {xc.is_contiguous()}")
    step("contiguous()", xc)

    windows = xc.view(-1, M, M, C)
    step("view (-1, M, M, C) = windows", windows)

    print()
    print("=" * 78)
    print("REVERSE：(B*nW, M, M, C) -> (B,H,W,C)")
    print("=" * 78)
    step("windows (B*nW,M,M,C)", windows)

    y6 = windows.view(B, H // M, W // M, M, M, C)
    step("view (B,H/M,W/M,M,M,C)", y6)

    yp = y6.permute(0, 1, 3, 2, 4, 5)
    step("permute (B,H/M,M,W/M,M,C)", yp)

    yc = yp.contiguous()
    print(f"      [观察] contiguous() 后 is_contiguous = {yc.is_contiguous()}")

    back = yc.view(B, H, W, C)
    step("view (B,H,W,C) = back", back)

    print()
    # 断言：精确可逆（逐元素相等，误差必须为 0）
    equal = torch.equal(back, x)
    max_err = (back - x).abs().max().item()
    print(f"reverse(partition(x)) == x : {equal}")
    print(f"逐元素最大误差            : {max_err}  （应为 0.0）")
    assert equal, "reverse(partition(x)) != x，索引映射有误！"
    assert max_err == 0.0
    print("\n全部形状跟踪与断言通过 [OK]")


if __name__ == "__main__":
    print("模块 02 · Window Partition / Reverse Shape Tracking")
    trace_partition_and_reverse()
