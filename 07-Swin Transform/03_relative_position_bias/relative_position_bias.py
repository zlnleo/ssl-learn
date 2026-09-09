# -*- coding: utf-8 -*-
"""
模块编号: 03
学习顺序: (前置 01/02 基础) -> 03 相对位置偏置 -> 04 移位窗口 -> 05 注意力掩码

相对位置偏置 (Relative Position Bias)
======================================

【问题】窗口内的自注意力 (Window Self-Attention) 本质是"集合到集合"的映射:
两个 token 只要 query/key 相同, 注意力分数就完全相同, 与它们在窗口内的相对位置无关。
换句话说, 把窗口内所有 token 的顺序任意打乱, 注意力输出(在无位置信息时)不变 ——
这就是"排列等变"。但图像是有空间结构的, "我左边是谁/右边是谁"必须能被模型区分。

【解法】在 softmax 之前, 给注意力分数矩阵加上一个只依赖"相对位移 (dh, dw)"的偏置:
    attn = softmax(Q K^T / sqrt(d) + RelativeBias)
偏置表只有 (2M-1)^2 行(所有可能的相对位移), 每行是 num_heads 维向量。
因为窗口内 token 的相对位移只有 (2M-1)^2 种, 一张小表就够覆盖全部情形。

本文件是"权威实现"的核心语义, 注释做了逐行润色, 接口保持一致:
- build_relative_position_index(window_size)  -> Tensor (M^2, M^2)
- RelativePositionBias(window_size, num_heads) -> nn.Module
"""

import sys

import torch
import torch.nn as nn

# Windows 控制台默认 GBK 编码，无法输出 ✔ 等符号；强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_relative_position_index(window_size: int) -> torch.Tensor:
    """构造相对位置索引表, 形状 (M^2, M^2), 元素是偏置表的行号 0..(2M-1)^2-1。

    原理: token a=(i1,j1) 与 b=(i2,j2) 的相对坐标 (i1-i2, j1-j2) 只有 (2M-1)^2 种取值,
    平移 M-1 归一到 [0, 2M-2]^2, 再按 (dh*(2M-1)+dw) 展平成一行。

    返回: long 类型张量, 形状 (M^2, M^2)。
    index[a, b] 表示"以 a 为 query、b 为 key 时, 应查偏置表第几行"。
    """
    # 生成 (2, M, M) 的坐标网格: coords[0] 是行坐标 i, coords[1] 是列坐标 j
    coords = torch.stack(
        torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
    )
    # coords: (2, M, M)
    coords = coords.reshape(2, -1)                    # (2, M^2): 每列是一个 token 的 (i, j)

    # 广播相减得到相对坐标: rel[:, a, b] = coords[:, a] - coords[:, b] = (i1-i2, j1-j2)
    rel = coords[:, :, None] - coords[:, None, :]     # (2, M^2, M^2)

    # 把坐标维挪到最后, 得到 (M^2, M^2, 2), 最后一个维度是 (dh, dw)
    rel = rel.permute(1, 2, 0).contiguous()           # (M^2, M^2, 2)

    rel[:, :, 0] += window_size - 1                   # 行偏移平移: [-(M-1), M-1] -> [0, 2M-2]
    rel[:, :, 1] += window_size - 1                   # 列偏移平移
    rel[:, :, 0] *= 2 * window_size - 1               # 行坐标 x 表宽(2M-1), 做行主序展平

    return rel.sum(-1)                                # (M^2, M^2) long 类型行号


def decode_relative_position_row(row: torch.Tensor, window_size: int) -> torch.Tensor:
    """把行号解码回相对位移 (dh, dw)。与 build_relative_position_index 的展平规则互为逆运算。

    row: 任意形状的整数张量。返回形状 row.shape + (2,), 最后一个维度是 (dh, dw)。
    学习辅助工具, 用于验证"行号 <-> 相对位移"的双射与转置对称性。
    """
    span = 2 * window_size - 1
    row = row.long()
    dh = row // span - (window_size - 1)
    dw = row % span - (window_size - 1)
    return torch.stack([dh, dw], dim=-1)


class RelativePositionBias(nn.Module):
    """可学习相对位置偏置: Parameter((2M-1)^2, num_heads)。

    forward() 输出 (M^2, M^2, num_heads), 在 WindowAttention 中 permute 成
    (h, M^2, M^2) 广播加到 attn 上。
    """

    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        # 偏置表: 每一行对应一种相对位移 (dh, dw), 共 (2M-1)^2 行, 每行 num_heads 维
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))

        # 索引表是"常量", 不参与反向传播, 也不写入 state_dict(persistent=False)
        self.register_buffer(
            "relative_position_index",
            build_relative_position_index(window_size),
            persistent=False,
        )

        # 用截断正态初始化, 与官方一致
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        # 高级索引 gather: table[(M^2, M^2)] -> (M^2, M^2, num_heads)
        return self.relative_position_bias_table[self.relative_position_index]


if __name__ == "__main__":
    print("=" * 70)
    print("模块 03 演示: 相对位置偏置 (Relative Position Bias)")
    print("=" * 70)

    M = 3
    H = 3  # num_heads
    idx = build_relative_position_index(M)
    print(f"\n[M={M}] 相对位置索引表 shape = {tuple(idx.shape)}, dtype = {idx.dtype}")
    print("索引表内容 (行号):")
    print(idx)

    # 解码验证: 把行号还原成 (dh, dw)
    dh, dw = decode_relative_position_row(idx, M)[..., 0], decode_relative_position_row(idx, M)[..., 1]
    print("\n行号解码回相对位移 dh (每行代表 query, 每列代表 key):")
    print(dh)
    print("\n行号解码回相对位移 dw:")
    print(dw)

    rpb = RelativePositionBias(window_size=M, num_heads=H)
    bias = rpb.forward()
    print(f"\n偏置表参数形状 = {tuple(rpb.relative_position_bias_table.shape)}")
    print(f"参数总量 = {rpb.relative_position_bias_table.numel()} (=(2*{M}-1)^2 * {H})")
    print(f"forward 输出形状 = {tuple(bias.shape)}, 期望 (M^2, M^2, num_heads) = {(M*M, M*M, H)}")
    print(f"索引表是否保存进 state_dict: {'relative_position_index' in rpb.state_dict()} (应为 False)")
    print(f"偏置表 requires_grad = {rpb.relative_position_bias_table.requires_grad} (应为 True)")
