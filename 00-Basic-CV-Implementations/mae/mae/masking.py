"""Stage 2 模块: random_masking —— MAE 唯一的核心新逻辑。

一个 forward 里发生三件事:
    1. 生成随机排列 ids_shuffle, 每张图独立;
    2. 取前 V 个为可见 token (x_vis), 其余 M = N - V 个被遮;
    3. 给出逆排列 ids_restore, 供 decoder 把 [x_vis + mask_token] 还原成原始 patch 顺序。

N=4, V=2 的最小数值例子 (运行仓库根目录 masking_example.py 查看完整演示):
    ids_shuffle           = [3, 1, 0, 2]        # 随机排列
    ids_keep              = [3, 1]              # 前 V 个 -> 可见 patch 是 3 号和 1 号
    ids_restore           = argsort([3,1,0,2])
                          = [2, 1, 3, 0]        # 逆排列
    concat                = [x3, x1, m, m]      # x_vis 拼上 mask token
    gather(concat, ids_restore) = [m, x1, m, x3]  # 恢复原始顺序
"""
import torch


def random_masking(x, mask_ratio, generator=None):
    """
    Args:
        x: (B, N, D) 完整 token 序列 (已加位置编码)
        mask_ratio: 掩码比例, 例如 0.75
        generator: 可选 torch.Generator, 用于显式控制随机性

    Returns: (按顺序)
        x_vis       (B, V, D)  float  可见 token, 按 ids_keep 的顺序排列
        mask        (B, N)     bool   True = 该位置被遮 (原始 patch 顺序)
        ids_restore (B, N)     int64  逆排列索引, decoder 用它还原顺序
        ids_keep    (B, V)     int64  可见 patch 在原始序列中的序号
        ids_shuffle (B, N)     int64  随机排列 (调试/教学用)
    """
    B, N, D = x.shape
    len_keep = int(N * (1.0 - mask_ratio))  # V, 例如 N=16, r=0.75 -> V=4

    noise = torch.rand(B, N, device=x.device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)  # (B, N) 每行一个随机排列
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N) 逆排列

    ids_keep = ids_shuffle[:, :len_keep]  # (B, V) 前 V 个 -> 可见
    x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # 原始顺序的 mask: 位置 i 被遮 <=> 还原索引 ids_restore[i] >= V
    # (等价于官方写法: 先造 [False]*V + [True]*M 再按 ids_restore gather 回填)
    mask = ids_restore >= len_keep  # (B, N) bool

    return x_vis, mask, ids_restore, ids_keep, ids_shuffle
