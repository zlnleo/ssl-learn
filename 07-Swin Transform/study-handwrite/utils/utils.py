# 已修复：DropPath 里 noise.floor() 返回值需重新赋值才生效（见《修改记录.md》）
import torch
from torch import nn


class DropPath(nn.Module):
    """stochastic depth：训练时以 drop_path_rate 概率把整条残差支路置零，推理时恒等。

    除以 keep_prob 保持训练阶段期望值不变（缩放补偿）。
    与官方实现一致：只有 batch 维随机，其余维度广播。
    """
    def __init__(self, drop_path_rate=0.):
        super().__init__()
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = noise.floor()          # 注意：floor() 返回新张量，必须重新赋值（不能只写 noise.floor()）
        return x / keep_prob * noise
