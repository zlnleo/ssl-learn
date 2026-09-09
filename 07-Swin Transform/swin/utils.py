"""基础网络部件：MLP 与 DropPath（stochastic depth）。"""
import torch
import torch.nn as nn

__all__ = ["Mlp", "DropPath"]


class Mlp(nn.Module):
    """两层全连接 + GELU，隐藏层宽度 mlp_ratio × 输入宽度（Swin 默认 4x）。

    FC1 (C -> 4C) -> GELU -> Dropout -> FC2 (4C -> C) -> Dropout
    参数量 2 * C * 4C = 8C^2，是每个 block 参数的主体（约 2/3）。
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """stochastic depth：训练时以 drop_prob 概率把整条残差支路置零，推理时恒等。

    除以 keep_prob 保持训练期望不变（缩放补偿）。
    在 Swin 中 drop_prob 沿网络深度线性增长（stochastic depth rate），
    浅层接近 0、深层接近 drop_path_rate。
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)   # 只在 batch 维随机
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = noise.floor()
        return x / keep_prob * noise
