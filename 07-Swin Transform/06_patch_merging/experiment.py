# -*- coding: utf-8 -*-
"""
模块 06：Patch Merging 实验
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

实验目标：
1. 用 (1,4,4,2) 的位置 id 张量（两通道分别存行号、列号）展示 2x2 分组后，
   每个新位置由哪些旧位置组成，并画出 ASCII 分组图。
2. 验证输出形状。
3. 打印参数量统计与公式核对。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe experiment.py
"""

import torch
from patch_merging import PatchMerging


def build_position_id_tensor(H: int, W: int) -> torch.Tensor:
    """构造 (1, H, W, 2) 张量：通道 0 存行号 r，通道 1 存列号 c，用于追踪空间来源。"""
    r = torch.arange(H).view(H, 1).expand(H, W)      # (H, W) 行号
    c = torch.arange(W).view(1, W).expand(H, W)      # (H, W) 列号
    # 转成 float32：后续 LayerNorm 要求浮点输入
    return torch.stack([r, c], dim=-1).unsqueeze(0).float()  # (1, H, W, 2)


def print_grid(x2d):
    """打印 2D 网格里每个位置存的内容（取前两个通道作为 (r,c) id）。"""
    H, W = x2d.shape[1], x2d.shape[2]
    print("    " + " ".join(f"  c{c:<2}" for c in range(W)))
    for r in range(H):
        row = f" r{r:<2}"
        for c in range(W):
            ids = x2d[0, r, c].long().tolist()
            row += f" ({ids[0]},{ids[1]})"
        print(row)


def print_grouping(H, W):
    """打印 2x2 分组 ASCII 图：每个新位置由哪些旧位置（含通道段归属）组成。"""
    print("\n2x2 分组的空间语义：每个新位置 (i,j) 由旧位置 2x2 邻域拼成，")
    print("拼接后通道顺序为 [x0 左上 | x1 左下 | x2 右上 | x3 右下]，每段 C 维。\n")
    for i in range(H // 2):
        for j in range(W // 2):
            r0, c0 = 2 * i, 2 * j
            print(f"新位置 ({i},{j})  <-  旧 2x2 邻域:")
            print(f"   ┌─────────────────┬─────────────────┐")
            print(f"   │ 旧({r0},{c0})   │ 旧({r0},{c0+1})   │")
            print(f"   │ -> x0 段(左上)  │ -> x2 段(右上)  │")
            print(f"   ├─────────────────┼─────────────────┤")
            print(f"   │ 旧({r0+1},{c0}) │ 旧({r0+1},{c0+1}) │")
            print(f"   │ -> x1 段(左下)  │ -> x3 段(右下)  │")
            print(f"   └─────────────────┴─────────────────┘")
            if not (i == H // 2 - 1 and j == W // 2 - 1):
                print()


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("实验：Patch Merging 的 2x2 分组语义 + 形状 + 参数量")
    print("=" * 70)

    H, W, C = 4, 4, 2
    ids = build_position_id_tensor(H, W)            # (1, 4, 4, 2)
    print(f"\n位置 id 张量形状: {tuple(ids.shape)}（两通道分别存 (行号, 列号)）")
    print("\n原始 4x4 网格（每个格子是该位置的 (r,c) id）：")
    print_grid(ids)

    print_grouping(H, W)

    # ---- 手工做 4 路切分 + 拼接，验证通道段归属 ----
    x = ids.view(1, H, W, C)
    x0 = x[:, 0::2, 0::2, :]   # 左上
    x1 = x[:, 1::2, 0::2, :]   # 左下
    x2 = x[:, 0::2, 1::2, :]   # 右上
    x3 = x[:, 1::2, 1::2, :]   # 右下
    x_cat = torch.cat([x0, x1, x2, x3], dim=-1)     # (1, 2, 2, 8)

    print("-" * 70)
    print("手工拼接结果（每个新位置的 8 个通道值 = 4 段 x 2 通道 id）：")
    print("通道含义：[x0 左上(r,c) | x1 左下(r,c) | x2 右上(r,c) | x3 右下(r,c)]")
    for i in range(H // 2):
        for j in range(W // 2):
            vals = x_cat[0, i, j].long().tolist()
            segs = [f"({vals[k]},{vals[k+1]})" for k in range(0, 8, 2)]
            print(f"  新位置 ({i},{j}) = " + " | ".join(segs))
    print("  -> 每个新位置都恰好收集了 2x2 邻域 4 个旧位置，互不重叠、无遗漏。")

    # ---- 走真正的 PatchMerging，验证输出形状 ----
    print("\n" + "-" * 70)
    model = PatchMerging(dim=C)
    x_seq = ids.view(1, H * W, C)                   # (1, 16, 2)
    y = model(x_seq, H, W)
    print(f"PatchMerging 输入  : {tuple(x_seq.shape)}")
    print(f"PatchMerging 输出  : {tuple(y.shape)}")
    print(f"预期输出           : (1, (H/2)*(W/2), 2C) = (1, {H//2 * W//2}, {2*C})")
    assert y.shape == (1, (H // 2) * (W // 2), 2 * C)

    # ---- 参数量统计 ----
    print("\n" + "=" * 70)
    print("参数量统计（C=96 为 Swin-Tiny 第一个 PatchMerging 的实际通道数）")
    print("=" * 70)
    for dim in (C, 96):
        m = PatchMerging(dim=dim)
        ln_params = sum(p.numel() for p in m.norm.parameters())     # 2 * 4C
        lin_params = sum(p.numel() for p in m.reduction.parameters())  # 4C * 2C
        total = sum(p.numel() for p in m.parameters())
        print(f"dim(C)={dim:>3}: LayerNorm(4C) 参数 = {ln_params:>6}  "
              f"Linear(4C->2C) 参数 = {lin_params:>7}  合计 = {total}")
        # 公式核对：LN = 2*4C，Linear = 4C*2C = 8C^2
        assert ln_params == 2 * 4 * dim
        assert lin_params == 4 * dim * 2 * dim
        print(f"           公式核对: 2*4C={2*4*dim} OK,  8C^2={8*dim*dim} OK")

    print("\n实验完成。")

    # ---- 计算量（FLOPs）公式示意 ----
    print("\n" + "=" * 70)
    print("Linear(4C -> 2C) 的 MACs（乘加数）推导")
    print("=" * 70)
    print("每个输出 token 一次矩阵乘：输入 4C 维 -> 输出 2C 维，需 4C * 2C 次乘法 + 加法")
    print("对 h'*w' = (H/2)*(W/2) 个 token、batch B，总 MACs ~= B * h'w' * 4C * 2C = 2 * B * h'w' * C^2")
    hw = (H // 2) * (W // 2)
    print(f"本例 H=W=4, C=2: B=1, h'w'={hw}, MACs ~= 1 * {hw} * (4*2) * (2*2) = {hw * 8 * 4}")
    print(f"C=96, 输入 56x56 -> 28x28: h'w'=784, MACs ~= 784 * 384 * 192 = {784 * 384 * 192:,}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
