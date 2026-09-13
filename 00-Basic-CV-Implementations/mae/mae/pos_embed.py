"""Stage 4 模块: 位置编码。encoder 与 decoder 各用一份 (维度不同)。

MAE 预训练默认用固定 2D sincos (论文做法, 不参与梯度);
也可选 learnable=True 换成可学习位置编码 (和你的 ViT 一样)。
"""
import torch
import torch.nn as nn


def get_1d_sincos_from_grid(embed_dim, pos):
    """pos: (N,) -> (N, embed_dim), 标准 ViT 1D sincos。"""
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2)
    omega = 1.0 / (10000 ** omega)
    out = pos[:, None].float() * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """(1, grid_h*grid_w, embed_dim), 行主序, 与 patchify 的顺序约定一致。"""
    grid_h_pos = torch.arange(grid_h)
    grid_w_pos = torch.arange(grid_w)
    grid = torch.stack(torch.meshgrid(grid_h_pos, grid_w_pos, indexing="ij"), dim=-1)  # (gh, gw, 2)
    grid = grid.reshape(-1, 2)  # (N, 2) 行主序
    emb_h = get_1d_sincos_from_grid(embed_dim // 2, grid[:, 0])
    emb_w = get_1d_sincos_from_grid(embed_dim // 2, grid[:, 1])
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)


def build_pos_embed(num_patches, dim, learnable=False):
    """返回 nn.Parameter (1, num_patches, dim)。

    learnable=False: 论文默认的固定 2D sincos (不参与梯度)
    learnable=True : 可学习, trunc_normal(std=0.02) 初始化
    """
    if learnable:
        pe = torch.zeros(1, num_patches, dim)
        nn.init.trunc_normal_(pe, std=0.02)
    else:
        gh = int(num_patches ** 0.5)  # 方形图像, 网格 = sqrt(N) x sqrt(N)
        pe = get_2d_sincos_pos_embed(dim, gh, gh)
    return nn.Parameter(pe, requires_grad=learnable)
