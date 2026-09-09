# -*- coding: utf-8 -*-
"""
模块 01 / 学习顺序第 1 步：Window Attention（窗口多头自注意力，最简教学版）
================================================================================
本项目是 "从零学习 Swin Transformer" 的第 1 个模块。它回答一个最核心的问题：

    "为什么要把全局注意力限制在一个小窗口内？"

    全局多头自注意力（MSA）的复杂度是 O((hw)^2)（hw = 特征图的 token 总数），
    当特征图较大时（例如 56x56 = 3136 个 token）代价爆炸。
    窗口多头自注意力（W-MSA）把图切成一个个 MxM 的小窗口，在每个窗口内做注意力，
    复杂度降为 O(hw * M^2)，当 M 固定为常数时对 token 数是线性的。

本文件只包含"注意力算子本身"：它输入一批已经划分好的窗口序列 (B_, N, C)，
输出同样形状的序列。它**不负责**把 (B,H,W,C) 切成窗口——那是模块 02 的事。
这种"算子与划分解耦"的设计，正是 Swin 简洁优雅的关键。

    学习顺序：
        模块 01（本模块）: 窗口内的注意力算子
        模块 02          : window_partition / window_reverse（划分与还原）
        后续模块         : 相对位置偏置、shifted window、patch merging 等

    接口约定（与原论文实现保持一致）：
        输入  x : (B_, N, C)，其中 B_ = B * nW（所有窗口按 batch 拼在一起），
                  N = window_size ** 2（每个窗口内的 token 数）。
        输出  y : (B_, N, C)

    全局 MSA 是本模块的特例：令 window_size = H = W（整个特征图就是唯一一个窗口）。

代码中关键张量形状已逐行标注，便于对照 shape_tracking.py 的输出。
"""

import sys

import torch
import torch.nn as nn

# Windows 控制台默认 GBK 编码；强制 UTF-8 保证中文输出不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def msa_macs(hw: int, C: int, win_tokens: int) -> float:
    """一次注意力（QKV 投影 + 注意力 + 输出投影）的乘加数 MACs（1 MAC = 1 次乘加 ≈ 2 FLOPs）。

    参数
    ----
    hw         : 总 token 数（整张特征图，所有窗口的 token 之和）。
    C          : 通道数（embedding 维度）。
    win_tokens : 每个"注意力单元"内的 token 数。
                 全局注意力时 = hw（所有 token 互相看）；
                 窗口注意力时 = M^2（只在窗口内互相看）。

    推导
    ----
    QKV 投影 : 3 * (hw * C) * C = 3*hw*C^2
    输出投影 : (hw * C) * C     =   hw*C^2
    QK^T    : hw * win_tokens * C   （每个 token 只和 win_tokens 个 token 算点积）
    AV      : hw * win_tokens * C
    合计     : 4*hw*C^2 + 2*hw*win_tokens*C
    """
    return 4 * hw * C * C + 2 * hw * win_tokens * C


class WindowAttention(nn.Module):
    """窗口多头自注意力（最简教学版：无相对位置偏置、无 attention mask，后续模块再加）。

    输入: x (B_, N, C)，B_ = B * nW（所有窗口按 batch 拼接），N = window_size**2。
    输出: (B_, N, C)
    全局 MSA 就是它的特例：window_size = H = W（整个特征图是一个窗口）。
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5   # 为什么除以 sqrt(d)：防止点积方差随 d 线性增长
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        # (B_,N,3C) -> (B_,N,3,h,d) -> (3,B_,h,N,d)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # 各 (B_, h, N, d)
        attn = (q * self.scale) @ k.transpose(-2, -1)    # (B_, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        return self.proj_drop(self.proj(out))


if __name__ == "__main__":
    # 最小演示：一个 2 个窗口、每窗口 4 个 token、8 维通道的输入。
    print("=" * 70)
    print("WindowAttention 演示入口")
    print("=" * 70)
    torch.manual_seed(0)
    B_, N, C = 2, 4, 8          # 2 个窗口，窗口大小 2x2（N=4），通道 8
    win, heads = 2, 2
    x = torch.randn(B_, N, C)   # (B_=2, N=4, C=8)

    attn = WindowAttention(dim=C, window_size=win, num_heads=heads)
    y = attn(x)
    print(f"输入形状  : {tuple(x.shape)}")
    print(f"输出形状  : {tuple(y.shape)}")
    print(f"window_size = {win}, num_heads = {heads}, head_dim = {attn.head_dim}")
    print(f"scale = 1/sqrt(head_dim) = {attn.scale:.4f}")

    # 用公式算一下这个小例子的 MACs
    hw, M2 = B_ * N, win * win
    macs = msa_macs(hw, C, M2)
    print(f"本小例 MACs（hw={hw}, C={C}, win_tokens={M2}）: {macs:,.0f}")

    # 全局 vs 窗口：教科书数字（hw = 56*56 = 3136, C = 96, M = 7）
    print("\n教科书数字（hw=56^2=3136, C=96, M=7）:")
    g = msa_macs(3136, 96, 3136)
    w = msa_macs(3136, 96, 49)
    print(f"  全局 MSA   : {g/1e6:8.1f} M MACs")
    print(f"  窗口 W-MSA : {w/1e6:8.1f} M MACs")
    print(f"  总比值     : {g/w:8.2f} x")
    print(f"  注意力部分比值 = hw/M^2 = {3136/49:.1f} x")
