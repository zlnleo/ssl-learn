# -*- coding: utf-8 -*-
"""
模块 07：Swin Block 的 Tensor Shape 逐步跟踪
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

对 W-MSA（shift=0）与 SW-MSA（shift=window_size//2）各走一遍，逐步打印形状
（含 pad 前后、mask 形状），并断言输出形状与输入一致。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe shape_tracking.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_block import (SwinBlock, window_partition, window_reverse,
                        build_attn_mask)


def trace_swin_block(blk: SwinBlock, x: torch.Tensor, H: int, W: int, verbose: bool = True):
    """逐步跟踪 SwinBlock.forward 内部每一步形状，返回输出。"""
    def log(name, t):
        if verbose:
            print(f"  {name:<30} shape = {tuple(t.shape)}")

    M = blk.window_size
    B, L, C = x.shape
    assert L == H * W
    log("输入 x (序列)", x)

    shortcut = x
    x = blk.norm1(x).view(B, H, W, C)
    log("norm1 + reshape 2D", x)
    assert x.shape == (B, H, W, C)

    # 1) pad 到 window_size 整数倍
    pad_r = (M - W % M) % M
    pad_b = (M - H % M) % M
    if verbose:
        print(f"  {'pad 尺寸 (右,下)':<30} = ({pad_r}, {pad_b})")
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    Hp, Wp = H + pad_b, W + pad_r
    log("pad 之后", x)
    assert x.shape == (B, Hp, Wp, C)

    # 2) 循环移位（仅 SW-MSA）
    if blk.shift_size > 0:
        x = torch.roll(x, shifts=(-blk.shift_size, -blk.shift_size), dims=(1, 2))
        log("roll 循环移位 (-s,-s)", x)

    # 3) 分窗 + 注意力
    x = window_partition(x, M)
    log("window_partition", x)
    nW = (Hp // M) * (Wp // M)
    assert x.shape == (B * nW, M, M, C)

    x = x.view(-1, M ** 2, C)
    log("展平窗口内 token", x)
    assert x.shape == (B * nW, M * M, C)

    mask = blk._get_mask(Hp, Wp, x.device)
    if mask is not None:
        log("attention mask", mask)
    else:
        if verbose:
            print(f"  {'attention mask':<30} = None（W-MSA 不需要）")
    x = blk.attn(x, mask=mask)
    log("window attention 输出", x)
    assert x.shape == (B * nW, M * M, C)

    x = x.view(-1, M, M, C)
    log("reshape 回窗口网格", x)

    # 4) 还原：reverse -> unshift -> crop
    x = window_reverse(x, M, Hp, Wp)
    log("window_reverse", x)
    assert x.shape == (B, Hp, Wp, C)

    if blk.shift_size > 0:
        x = torch.roll(x, shifts=(blk.shift_size, blk.shift_size), dims=(1, 2))
        log("roll 反向移位 (+s,+s)", x)

    x = x[:, :H, :W, :].contiguous().view(B, L, C)
    log("crop 回原尺寸", x)
    assert x.shape == (B, L, C)

    out = shortcut + blk.drop_path(x)
    log("残差 1 之后", out)
    out = out + blk.drop_path(blk.mlp(blk.norm2(out)))
    log("残差 2 (MLP) 之后", out)
    assert out.shape == (B, L, C)
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)

    B, H, W, C = 2, 8, 8, 96
    M, num_heads = 4, 3
    x = torch.randn(B, H * W, C)

    print("=" * 70)
    print("Shape Tracking 1：W-MSA（shift=0），H=W=8，window=4（整除，无 pad）")
    print("=" * 70)
    blk_w = SwinBlock(dim=C, num_heads=num_heads, window_size=M, shift_size=0)
    trace_swin_block(blk_w, x, H, W)

    print("\n" + "=" * 70)
    print("Shape Tracking 2：SW-MSA（shift=2），H=W=8，window=4")
    print("=" * 70)
    blk_sw = SwinBlock(dim=C, num_heads=num_heads, window_size=M, shift_size=M // 2)
    trace_swin_block(blk_sw, x, H, W)

    print("\n" + "=" * 70)
    print("Shape Tracking 3：非整除 pad 分支（H=W=10，window=4 -> pad 到 12）")
    print("=" * 70)
    x2 = torch.randn(2, 10 * 10, C)
    blk_pad = SwinBlock(dim=C, num_heads=num_heads, window_size=M, shift_size=M // 2)
    trace_swin_block(blk_pad, x2, 10, 10)

    print("\n全部 shape 断言通过：两个 block 输出形状均与输入一致 (B, L, C)。")
