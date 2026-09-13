"""Stage 5 模块: MAEEncoder —— ViT encoder 去掉 [CLS], 只编码可见 token。

数据流: images -> PatchEmbed -> +pos_embed(全部 N 个) -> random_masking -> LxBlock -> LayerNorm

phase-1 学习版: forward 返回 dict, 把每个中间张量暴露出来, 便于 test_mae_flow.py 逐步打印;
后续阶段会收紧成接口契约里的精简签名 (x_vis, mask, ids_restore)。
"""
import torch.nn as nn

from .masking import random_masking
from .modules import Block, PatchEmbed
from .pos_embed import build_pos_embed


class MAEEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = build_pos_embed(self.num_patches, embed_dim, learnable=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, imgs, mask_ratio=None):
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio

        x = self.patch_embed(imgs)  # (B, N, D) patch embedding
        x_full = x + self.pos_embed  # 先给全部 N 个 token 加位置编码, 再 shuffle (官方顺序)

        x_vis_in, mask, ids_restore, ids_keep, ids_shuffle = random_masking(x_full, mask_ratio)

        x_vis = x_vis_in
        for blk in self.blocks:  # 只对 V 个可见 token 做注意力
            x_vis = blk(x_vis)
        latent = self.norm(x_vis)  # (B, V, D)

        return {
            "patch_embed": x,  # (B, N, D) PatchEmbed 输出
            "tokens_full": x_full,  # (B, N, D) +pos_embed 后、shuffle 前的完整序列
            "x_vis": x_vis_in,  # (B, V, D) gather 出的可见 token (进 blocks 之前)
            "mask": mask,  # (B, N) bool, True = 被遮
            "ids_keep": ids_keep,  # (B, V) int64
            "ids_restore": ids_restore,  # (B, N) int64
            "ids_shuffle": ids_shuffle,  # (B, N) int64
            "latent": latent,  # (B, V, D) encoder 最终输出
        }
