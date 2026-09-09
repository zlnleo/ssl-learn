# -*- coding: utf-8 -*-
"""
模块编号: 03
学习顺序: 03 相对位置偏置 (本文件是"debug 实验"脚本)

experiment.py —— 相对位置偏置的四个数值实验:

1) 打印 M=3 的索引表(行号矩阵)。
2) 验证所有元素落在 [0, (2M-1)^2) 内。
3) 验证 (dh, dw) <-> 行号 的双射(每种位移恰好唯一对应一行, 且每个位移都出现)。
4) 数值实验: 构造固定 bias, 比较"加/不加 bias"两种情况下 softmax 后的注意力权重,
   展示相对位置偏置如何"在 softmax 前"重新分配注意力。
"""

import sys

import torch
import torch.nn.functional as F

# Windows 控制台默认 GBK 编码，无法输出 ✔ 等符号；强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from relative_position_bias import (
    build_relative_position_index,
    decode_relative_position_row,
    RelativePositionBias,
)


def print_matrix(title: str, mat: torch.Tensor):
    """把整型/浮点矩阵打印成对齐表格。"""
    print(title)
    for row in mat.tolist():
        print("  " + " ".join(f"{v:4d}" if isinstance(v, int) else f"{v:6.3f}" for v in row))


def experiment_1_index_table(M: int = 3):
    print("=" * 72)
    print("实验 1: 打印 M=3 的相对位置索引表")
    print("=" * 72)
    idx = build_relative_position_index(M)
    print(f"index shape = {tuple(idx.shape)}, dtype = {idx.dtype}\n")
    print_matrix(f"index (行=query, 列=key) 元素为偏置表行号:", idx)

    # 同时打印解码后的 (dh, dw) 帮助理解
    dec = decode_relative_position_row(idx, M)
    print("\n解码出 dh (行相对位移 i_query - i_key):")
    print_matrix("", dec[..., 0])
    print("\n解码出 dw (列相对位移 j_query - j_key):")
    print_matrix("", dec[..., 1])
    return idx


def experiment_2_range_check(idx: torch.Tensor, M: int = 3):
    print("\n" + "=" * 72)
    print("实验 2: 验证索引值全部落在合法区间 [0, (2M-1)^2)")
    print("=" * 72)
    span = 2 * M - 1
    lo, hi = idx.min().item(), idx.max().item()
    ok = (lo >= 0) and (hi < span * span)
    print(f"min = {lo}, max = {hi}")
    print(f"合法区间 = [0, {span*span})  =>  {'通过 ✔' if ok else '失败 ✘'}")
    assert ok
    return ok


def experiment_3_bijection(idx: torch.Tensor, M: int = 3):
    print("\n" + "=" * 72)
    print("实验 3: 验证 (dh, dw) <-> 行号 的双射")
    print("=" * 72)
    span = 2 * M - 1
    dec = decode_relative_position_row(idx, M)          # (M^2, M^2, 2)

    # 反向映射: 行号 -> (dh, dw), 由行号公式逆推
    rows = torch.arange(span * span)
    dh = rows // span - (M - 1)
    dw = rows % span - (M - 1)
    print("行号 0..(2M-1)^2-1 与相对位移的对应关系:")
    print(f"{'行号':>4} | {'dh':>4} | {'dw':>4}")
    print("-" * 22)
    for r, h, w in zip(rows.tolist(), dh.tolist(), dw.tolist()):
        print(f"{r:4d} | {h:4d} | {w:4d}")

    # 满射: 每个位移都在索引表中出现过
    present = set(idx.flatten().tolist())
    surjective = present == set(range(span * span))
    print(f"\n[满射] 索引表包含的行号集合大小 = {len(present)}, 应等于 {span*span}")
    print(f"        是否覆盖全部 {span*span} 种位移: {'是 ✔' if surjective else '否 ✘'}")

    # 单射: 同一位移(行号)在表中出现的次数 = (M-|dh|)(M-|dw|), 一致即可验证映射唯一
    print("\n[一致性] 每种位移出现次数 == (M-|dh|)(M-|dw|):")
    injective_all = True
    for r in range(span * span):
        h, w = dh[r].item(), dw[r].item()
        expect = (M - abs(h)) * (M - abs(w))
        actual = int((idx == r).sum().item())
        mark = "✔" if expect == actual else "✘"
        if expect != actual:
            injective_all = False
        print(f"  行号 {r:2d} (dh={h:2d}, dw={w:2d}): 出现 {actual:2d} 次, 期望 {expect:2d} 次  {mark}")

    assert surjective and injective_all
    return surjective and injective_all


def experiment_4_attention_redistribution(M: int = 3):
    print("\n" + "=" * 72)
    print("实验 4: bias 在 softmax 前如何改变注意力分布")
    print("=" * 72)
    torch.manual_seed(0)
    N = M * M          # 窗口内 token 数
    D = 8              # 每个 head 的特征维

    q = torch.randn(N, D)
    k = torch.randn(N, D)
    scores = q @ k.T   # (N, N), 纯内容决定的注意力分数

    # 构造一个"固定"的相对位置偏置: 距离越近, bias 越大(越不惩罚)
    # bias[a, b] = -0.5 * (|dh| + |dw|), 鼓励注意力集中在空间邻近的 token 上
    idx = build_relative_position_index(M)
    dec = decode_relative_position_row(idx, M)
    manhattan = dec[..., 0].abs() + dec[..., 1].abs()   # (N, N)
    fixed_bias = -0.5 * manhattan.float()

    attn_no_bias = F.softmax(scores, dim=-1)                    # (N, N)
    attn_bias = F.softmax(scores + fixed_bias, dim=-1)          # (N, N)

    center = (M // 2) * M + (M // 2)   # 中心 token 的索引
    print(f"考察中心 token (index={center}, 空间中心) 作为 query 时的注意力分布:")
    print(f"{'key位置':>8} {'manhattan':>10} {'无bias':>8} {'有bias':>8} {'变化':>8}")
    for j in range(N):
        d = manhattan[center, j].item()
        nb = attn_no_bias[center, j].item()
        yb = attn_bias[center, j].item()
        print(f"{j:8d} {d:10d} {nb:8.4f} {yb:8.4f} {yb-nb:+8.4f}")

    # 数值结论: 加 bias 后, 距离越近的 key 权重应整体上升
    # 注意: M=3 时中心 token 的最大曼哈顿距离是 2(四个角), 故"远邻"取距离 >= 2
    near_mask = manhattan[center] <= 1
    far_mask = manhattan[center] >= 2
    d_near = attn_bias[center, near_mask].sum() - attn_no_bias[center, near_mask].sum()
    d_far = attn_bias[center, far_mask].sum() - attn_no_bias[center, far_mask].sum()
    print(f"\n加 bias 后, 近邻(距离<=1)权重总变化: {d_near:+.4f}")
    print(f"加 bias 后, 远邻(距离>=2)权重总变化: {d_far:+.4f}")
    print(f"结论: bias 把注意力从远邻转移到近邻 => {'成立 ✔' if d_near > 0 and d_far < 0 else '需检查 ✘'}")
    assert d_near > 0 and d_far < 0
    return attn_no_bias, attn_bias


def experiment_5_parameter_count():
    print("\n" + "=" * 72)
    print("实验 5: 参数数量核对 (2M-1)^2 * num_heads")
    print("=" * 72)
    M, H = 7, 3
    rpb = RelativePositionBias(window_size=M, num_heads=H)
    n = rpb.relative_position_bias_table.numel()
    print(f"M=7, num_heads=3: 参数量 = {n}, 期望 (2*7-1)^2 * 3 = {13*13*3} = 507")
    assert n == 507
    print("通过 ✔")
    return n


if __name__ == "__main__":
    idx = experiment_1_index_table(3)
    experiment_2_range_check(idx, 3)
    experiment_3_bijection(idx, 3)
    experiment_4_attention_redistribution(3)
    experiment_5_parameter_count()
    print("\n" + "=" * 72)
    print("全部实验完成, 无报错 ✔")
    print("=" * 72)
