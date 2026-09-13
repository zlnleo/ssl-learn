"""Stage 3 模块: per-patch 归一化 target。

为什么不用原始像素做 target:
    直接回归 RGB, 模型最先学会的是"输出平均颜色";
    对每个 patch 做 mean/std 归一化后, target 只剩"结构信息",
    且归一化后每个 patch 方差 = 1 -> 随机初始化模型的初始 loss 理论值 ~ 1.0。
"""
import torch


def patch_normalize(target):
    """(B, N, T) -> norm (B, N, T), mean (B, N, 1), var (B, N, 1)

    对每个 patch 的全部 T 个像素一起统计 (不分通道)。
    """
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True, unbiased=False)
    norm = (target - mean) / (var + 1e-6).sqrt()
    return norm, mean, var


def patch_denormalize(norm, mean, var):
    """可视化时把归一化 patch 还原成真实像素。"""
    return norm * (var + 1e-6).sqrt() + mean
