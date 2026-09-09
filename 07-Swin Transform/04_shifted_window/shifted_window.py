# -*- coding: utf-8 -*-
"""
模块编号: 04
学习顺序: 03 相对位置偏置 -> 04 移位窗口 -> 05 注意力掩码

移位窗口 (Shifted Window) 与 torch.roll 循环移位
=================================================

【问题】W-MSA 的窗口是固定划分的: 每个窗口只在自已的 M×M 区域内做自注意力,
窗口与窗口之间"零交流"。逐层堆叠时, 每个 token 的感受野永远局限在它所在的窗口内,
深层网络退化成"一堆彼此独立的小块各自演化", 跨窗口的信息永远无法流动。

【解法】相邻两层交替使用两种窗口划分:
- 偶数层: 标准窗口划分 W-MSA
- 奇数层: 先把整张特征图"循环移位" shift 个像素, 再做同样大小的窗口划分 SW-MSA
移位后, 新窗口跨越旧窗口的边界, 注意力就能跨窗交流。

【为什么用 torch.roll(循环移位) 而不是普通平移?】
普通平移会丢失越界内容、改变特征图尺寸; torch.roll 是"wrap-around"循环滚动,
越界部分从另一侧绕回, 内容一个不丢、尺寸不变。移位后分窗计算、再滚回来,
中间不损失任何信息。代价是: 新窗口内可能混入空间上不相邻的区域(伪邻居),
这引出模块 05 的注意力掩码。
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window_size, window_size, C)

    nW = (H//window_size) * (W//window_size), 窗口按行优先编号。
    本质: view 把每个窗口的二维区域暴露成独立维度, permute 调整维度顺序
    使"窗口行/窗口列"两个维度相邻, contiguous 后即可一次性 view 成 (nW, M, M, C)。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # x: (B, nH, M, nW, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # windows: (B*nW, M, M, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """window_partition 的逆运算: (B*nW, M, M, C) -> (B, H, W, C)"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x: (B, nH, nW, M, M, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # x: (B, H, W, C)
    return x


def cyclic_shift(x: torch.Tensor, shift_size: int) -> torch.Tensor:
    """x: (B, H, W, C)。向左上循环移位 shift_size: 内容整体滚动, 越界部分从另一侧绕回 (wrap-around)。

    语义: out[i, j] = x[(i + shift_size) % H, (j + shift_size) % W]
    """
    return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))


def cyclic_unshift(x: torch.Tensor, shift_size: int) -> torch.Tensor:
    """cyclic_shift 的逆: 向右下滚回。"""
    return torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))


def simple_window_attention(windows: torch.Tensor) -> torch.Tensor:
    """简化版窗口自注意力(无 mask): 以 x 自身作为 q/k/v, 仅用于演示管线。

    windows: (nW, M, M, C) -> (nW, M, M, C)

    ⚠ 注意: 此处暂无 mask, 会把不相邻区域当作邻居 —— 这正是模块 05 要解决的问题。
    """
    nW, M, _, C = windows.shape
    N = M * M
    x = windows.view(nW, N, C)                     # (nW, N, C)
    q = k = v = x                                   # 简化: q=k=v=x
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # (nW, N, N) 缩放点积
    attn = F.softmax(attn, dim=-1)                  # (nW, N, N) 每行归一
    out = attn @ v                                  # (nW, N, C)
    return out.view(nW, M, M, C)                    # (nW, M, M, C)


def sw_msa_demo(x: torch.Tensor, window_size: int, shift_size: int) -> torch.Tensor:
    """完整 SW-MSA 演示管线: roll -> partition -> 无 mask 注意力 -> reverse -> unroll。

    x: (B, H, W, C) -> (B, H, W, C)
    注意: 中间的 simple_window_attention 没有 mask, 仅用于演示形状流转。
    """
    B, H, W, C = x.shape
    shifted = cyclic_shift(x, shift_size)                    # (B, H, W, C) 向左上循环移位
    windows = window_partition(shifted, window_size)         # (B*nW, M, M, C) 分窗
    windows = simple_window_attention(windows)               # (B*nW, M, M, C) 窗内注意力(无 mask)
    shifted_out = window_reverse(windows, window_size, H, W) # (B, H, W, C) 拼回
    out = cyclic_unshift(shifted_out, shift_size)            # (B, H, W, C) 向右下滚回
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("模块 04 演示: 移位窗口 SW-MSA 管线 (roll -> partition -> attn -> reverse -> unroll)")
    print("=" * 70)

    B, H, W, C = 1, 8, 8, 4
    x = torch.randn(B, H, W, C)
    print(f"\n输入 x 形状 = {tuple(x.shape)}")

    shifted = cyclic_shift(x, shift_size=2)
    print(f"cyclic_shift 后形状 = {tuple(shifted.shape)} (尺寸不变)")

    windows = window_partition(shifted, window_size=4)
    print(f"window_partition 后形状 = {tuple(windows.shape)} (nW = 2x2 = 4)")

    # 验证 roll/unroll 可逆
    restored = cyclic_unshift(cyclic_shift(x, 2), 2)
    print(f"\nunroll(roll(x)) == x: {bool(torch.equal(restored, x))}")

    # 验证 partition/reverse 可逆
    rev = window_reverse(window_partition(x, 4), 4, H, W)
    print(f"reverse(partition(x)) == x: {bool(torch.equal(rev, x))}")

    # 完整管线
    out = sw_msa_demo(x, window_size=4, shift_size=2)
    print(f"\nSW-MSA 演示管线输出形状 = {tuple(out.shape)}, 与输入一致: {out.shape == x.shape}")
    print("\n演示完成 ✔")
