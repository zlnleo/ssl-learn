# -*- coding: utf-8 -*-
"""
模块编号: 04
学习顺序: 04 移位窗口 (本文件是"形状跟踪"脚本)

shape_tracking.py —— 逐步打印并断言 SW-MSA 管线的张量形状演变:
roll -> partition -> (窗内) -> reverse -> unroll。

目的: window_partition 的 view/permute 组合最容易在"哪个维度是窗口行/窗口列"上
犯迷糊; 本脚本用 (1, 8, 8, 4) 的输入把每一步形状与语义钉死。
"""

import sys

import torch

from shifted_window import (
    window_partition,
    window_reverse,
    cyclic_shift,
    cyclic_unshift,
)

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def track():
    B, H, W, C = 1, 8, 8, 4
    M = 4          # window_size
    S = 2          # shift_size
    x = torch.randn(B, H, W, C)

    print("=" * 72)
    print(f"形状跟踪: SW-MSA 管线 (B={B}, H={H}, W={W}, C={C}, window={M}, shift={S})")
    print("=" * 72)

    # 步骤 1: 循环移位
    shifted = cyclic_shift(x, S)
    print(f"[step 1] cyclic_shift(x, {S})            -> {tuple(shifted.shape)}  # (B, H, W, C) 尺寸不变")
    assert shifted.shape == (B, H, W, C), shifted.shape

    # 步骤 2: 分窗
    windows = window_partition(shifted, M)
    nW = (H // M) * (W // M)
    print(f"[step 2] window_partition(shifted, {M})  -> {tuple(windows.shape)}  # (B*nW, M, M, C) = ({B*nW}, {M}, {M}, {C})")
    assert windows.shape == (B * nW, M, M, C), windows.shape

    # 步骤 3: 观察单个窗口
    print(f"[step 3] 单个窗口形状 = {tuple(windows[0].shape)}  # (M, M, C) = ({M}, {M}, {C})")
    print(f"         窗口个数 nW = (H//M)*(W//M) = ({H//M})*({W//M}) = {nW}")

    # 步骤 4: 窗口内 reshape 成 token 序列 (供注意力使用)
    N = M * M
    tokens = windows.view(B * nW, N, C)
    print(f"[step 4] windows.view(B*nW, N, C)        -> {tuple(tokens.shape)}  # (B*nW, M^2, C) = ({B*nW}, {N}, {C})")
    assert tokens.shape == (B * nW, N, C), tokens.shape

    # 步骤 5: 拼回 (reverse)
    shifted_out = window_reverse(windows, M, H, W)
    print(f"[step 5] window_reverse(windows, ...)    -> {tuple(shifted_out.shape)}  # (B, H, W, C)")
    assert shifted_out.shape == (B, H, W, C), shifted_out.shape

    # 步骤 6: 滚回 (unroll)
    out = cyclic_unshift(shifted_out, S)
    print(f"[step 6] cyclic_unshift(shifted_out, {S}) -> {tuple(out.shape)}  # (B, H, W, C)")
    assert out.shape == (B, H, W, C), out.shape

    # ===== 关键断言 =====
    print("\n===== 关键断言 =====")
    # 1) roll/unroll 可逆
    print(f"[断言] unroll(roll(x)) == x : {bool(torch.equal(cyclic_unshift(cyclic_shift(x, S), S), x))}")
    assert torch.equal(cyclic_unshift(cyclic_shift(x, S), S), x)

    # 2) partition/reverse 可逆 (对移位前与移位后的图都成立)
    print(f"[断言] reverse(partition(x)) == x : {bool(torch.equal(window_reverse(window_partition(x, M), M, H, W), x))}")
    assert torch.equal(window_reverse(window_partition(x, M), M, H, W), x)

    # 3) roll 不改变内容多重集 (只是重排)
    before = torch.sort(x.flatten())[0]
    after = torch.sort(cyclic_shift(x, S).flatten())[0]
    print(f"[断言] roll 前后内容多重集一致 : {bool(torch.equal(before, after))}")
    assert torch.equal(before, after)

    # 4) 分窗后所有窗口拼起来的总 token 数守恒
    print(f"[断言] 分窗后总元素数 = B*H*W*C : {windows.numel() == B * H * W * C}")
    assert windows.numel() == B * H * W * C

    print("\n全部形状与断言通过 ✔")
    return windows


if __name__ == "__main__":
    track()
