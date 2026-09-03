# -*- coding: utf-8 -*-
"""
模块编号: 03
学习顺序: 03 相对位置偏置 (本文件是"形状跟踪"脚本)

shape_tracking.py —— 逐步打印并断言 build_relative_position_index 的张量形状演变。

目的: 广播相减 + permute + 平移 + 乘 + sum 这套操作最容易在"哪个维度是什么"上
犯迷糊。本脚本把每一步的形状、语义都打印出来, 并用 assert 把关键结论钉死,
让你亲眼看到 (2, M, M) 如何一步步变成 (M^2, M^2) 的行号表。
"""

import sys

import torch

from relative_position_bias import build_relative_position_index

# Windows 控制台默认 GBK 编码，无法输出 ✔ 等符号；强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def track(window_size: int = 3):
    M = window_size
    print("=" * 72)
    print(f"形状跟踪: build_relative_position_index(window_size={M})")
    print("=" * 72)

    # 步骤 0: 构造坐标网格
    arange = torch.arange(M)
    mesh = torch.meshgrid(arange, arange, indexing="ij")
    print(f"[step 0] torch.meshgrid(arange({M}), arange({M}), indexing='ij') -> 两个 (M, M)")
    print(f"         mesh[0] 是行坐标 i:\n{mesh[0]}")
    print(f"         mesh[1] 是列坐标 j:\n{mesh[1]}")

    # 步骤 1: stack
    coords = torch.stack(mesh)                       # (2, M, M)
    print(f"\n[step 1] torch.stack(mesh)                 -> shape {tuple(coords.shape)}  # (2, M, M)")
    assert coords.shape == (2, M, M), coords.shape

    # 步骤 2: reshape 展平
    coords = coords.reshape(2, -1)                   # (2, M^2)
    print(f"[step 2] coords.reshape(2, -1)             -> shape {tuple(coords.shape)}  # (2, M^2)")
    assert coords.shape == (2, M * M), coords.shape
    print(f"         coords[0] = 每个 token 的行坐标: {coords[0].tolist()}")
    print(f"         coords[1] = 每个 token 的列坐标: {coords[1].tolist()}")

    # 步骤 3: 广播相减
    rel = coords[:, :, None] - coords[:, None, :]    # (2, M^2, M^2)
    print(f"\n[step 3] coords[:, :, None] - coords[:, None, :] -> shape {tuple(rel.shape)}  # (2, M^2, M^2)")
    assert rel.shape == (2, M * M, M * M), rel.shape
    print(f"         rel[0, a, b] = i_a - i_b  (行相对位移 dh)")
    print("         rel[0] (dh 通道):")
    print(rel[0])
    print("         rel[1] (dw 通道):")
    print(rel[1])

    # 步骤 4: permute 把坐标维挪到最后
    rel = rel.permute(1, 2, 0).contiguous()          # (M^2, M^2, 2)
    print(f"\n[step 4] rel.permute(1,2,0).contiguous() -> shape {tuple(rel.shape)}  # (M^2, M^2, 2)")
    assert rel.shape == (M * M, M * M, 2), rel.shape

    # 步骤 5: 平移归一到非负区间
    rel[:, :, 0] += M - 1                            # dh: [-(M-1), M-1] -> [0, 2M-2]
    rel[:, :, 1] += M - 1                            # dw: 同理
    print(f"[step 5] rel[:, :, 0] += M-1; rel[:, :, 1] += M-1  (平移归一到 [0, {2*M-2}])")
    assert rel.min() >= 0 and rel.max() <= 2 * M - 2, (rel.min(), rel.max())
    print(f"         平移后 rel[..., 0] (dh) 范围 [{rel[..., 0].min()}, {rel[..., 0].max()}]")
    print(f"         平移后 rel[..., 1] (dw) 范围 [{rel[..., 1].min()}, {rel[..., 1].max()}]")

    # 步骤 6: 行坐标乘以表宽
    rel[:, :, 0] *= 2 * M - 1                        # 行主序展平
    print(f"[step 6] rel[:, :, 0] *= (2M-1)={2*M-1}   (行主序: dh * 表宽)")

    # 步骤 7: sum 得到一维行号
    index = rel.sum(-1)                              # (M^2, M^2)
    print(f"[step 7] rel.sum(-1)                       -> shape {tuple(index.shape)}  # (M^2, M^2)")
    assert index.shape == (M * M, M * M), index.shape
    assert index.dtype == torch.long, index.dtype

    # 关键断言: 行号落在 [0, (2M-1)^2)
    span = 2 * M - 1
    print(f"\n[断言] 行号取值范围: [{index.min()}, {index.max()}]")
    print(f"        合法区间应为 [0, (2M-1)^2) = [0, {span*span})")
    assert index.min() >= 0 and index.max() < span * span

    # 关键断言: 每种相对位移都出现过(双射的"满射"部分)
    uniq = torch.unique(index)
    print(f"[断言] 索引表中不同行号的个数 = {uniq.numel()}, 应等于 (2M-1)^2 = {span*span}")
    assert uniq.numel() == span * span, uniq.numel()

    # 关键断言: 对角线(自己对自己)统一映射到中心位移 (0, 0)
    center = (M - 1) * span + (M - 1)
    diag = index.diagonal()
    print(f"[断言] 对角线元素全部等于中心行号 {center} (位移 (0,0)): {bool((diag == center).all())}")
    assert bool((diag == center).all())

    # 与权威实现的一致性
    ref = build_relative_position_index(M)
    print(f"[断言] 手工逐步计算 == 权威实现: {bool((index == ref).all())}")

    print("\n全部形状与语义断言通过 ✔")
    return index


if __name__ == "__main__":
    for m in (2, 3, 4):
        track(m)
        print()
