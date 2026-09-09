# -*- coding: utf-8 -*-
"""
模块编号: 05
学习顺序: 05 注意力掩码 (本文件是"形状跟踪"脚本)

shape_tracking.py —— 逐步打印并断言 build_attn_mask 的张量形状演变:
img_mask -> partition -> view -> unsqueeze 相减 -> masked_fill。

目的: 9 宫格编号 -> 每个窗口展平成区域号向量 -> 两两相减得到"区域号之差" ->
非零位置填 -100。本脚本用 (H=8, W=8, window=4, shift=2) 把每一步形状与语义钉死。
"""

import sys

import torch

from attention_mask import window_partition, build_attn_mask

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def track():
    H = W = 8
    M = 4          # window_size
    S = 2          # shift_size
    nW = (H // M) * (W // M)   # 4

    print("=" * 72)
    print(f"形状跟踪: build_attn_mask (H={H}, W={W}, window={M}, shift={S})")
    print("=" * 72)

    # 步骤 0: 构造 9 宫格编号图
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (slice(0, -M), slice(-M, -S), slice(-S, None))
    w_slices = (slice(0, -M), slice(-M, -S), slice(-S, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    print(f"[step 0] img_mask (9 宫格编号)            -> {tuple(img_mask.shape)}  # (1, H, W, 1) = (1, {H}, {W}, 1)")
    assert img_mask.shape == (1, H, W, 1), img_mask.shape

    # 步骤 1: 分窗
    mask_windows = window_partition(img_mask, M)
    print(f"[step 1] window_partition(img_mask, {M})  -> {tuple(mask_windows.shape)}  # (nW, M, M, 1) = ({nW}, {M}, {M}, 1)")
    assert mask_windows.shape == (nW, M, M, 1), mask_windows.shape

    # 步骤 2: 展平成区域号向量
    mask_windows = mask_windows.view(-1, M * M)
    print(f"[step 2] mask_windows.view(-1, M^2)       -> {tuple(mask_windows.shape)}  # (nW, M^2) = ({nW}, {M*M})")
    assert mask_windows.shape == (nW, M * M), mask_windows.shape

    # 步骤 3: unsqueeze 相减得到区域号之差
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    print(f"[step 3] unsqueeze(1) - unsqueeze(2)      -> {tuple(attn_mask.shape)}  # (nW, M^2, M^2) = ({nW}, {M*M}, {M*M})")
    assert attn_mask.shape == (nW, M * M, M * M), attn_mask.shape

    # 步骤 4: 非零填 -100, 零填 0
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    print(f"[step 4] masked_fill(!=0, -100)           -> {tuple(attn_mask.shape)}  # (nW, M^2, M^2)")
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    print(f"[step 5] masked_fill(==0, 0)             -> {tuple(attn_mask.shape)}  # (nW, M^2, M^2)")

    # ===== 关键断言 =====
    print("\n===== 关键断言 =====")
    vals = set(attn_mask.flatten().tolist())
    print(f"[断言] mask 值域只含 {{0, -100}} : {vals == {0.0, -100.0}} (实际 {vals})")
    assert vals == {0.0, -100.0}, vals

    diag = attn_mask.diagonal(dim1=1, dim2=2)
    print(f"[断言] 对角线全 0(自己永远可见) : {bool((diag == 0).all())}")
    assert bool((diag == 0).all())

    sym = bool((attn_mask == attn_mask.transpose(1, 2)).all())
    print(f"[断言] 对称 mask[i,j]==mask[j,i] : {sym}")
    assert sym

    # 与权威实现一致
    ref = build_attn_mask(H, W, M, S)
    print(f"[断言] 手工逐步计算 == 权威实现 : {bool((attn_mask == ref).all())}")
    assert torch.equal(attn_mask, ref)

    masked = int((attn_mask == -100).sum().item())
    print(f"[统计] 被屏蔽元素总数 = {masked} (共 {attn_mask.numel()} 个元素)")
    assert masked == 448, masked   # 0 + 128 + 128 + 192

    print("\n全部形状与语义断言通过 ✔")
    return attn_mask


if __name__ == "__main__":
    track()
