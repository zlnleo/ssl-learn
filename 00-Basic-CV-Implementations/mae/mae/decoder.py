"""Stage 6 模块: MAEDecoder —— 填 mask token, 还原顺序, 重建像素。

数据流: x_vis -> Linear(D->Dd) -> cat mask_token -> gather(ids_restore) -> +pos_embed
        -> Ld x Block -> LayerNorm -> Linear(Dd->T) -> pred

decoder 的可学习"新面孔"只有三处:
    mask_token        (1, 1, Dd)  被遮位置共用的占位 token
    decoder_pos_embed (1, N, Dd)  自己的位置编码
    pred              Linear(Dd -> T)  把特征映射回像素空间
"""
import torch
import torch.nn as nn

from .modules import Block
from .pos_embed import build_pos_embed


class MAEDecoder(nn.Module):
    def __init__(self, num_patches, encoder_embed_dim, decoder_embed_dim, depth, num_heads, patch_size, in_chans):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_size * patch_size * in_chans  # T = P*P*C

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = build_pos_embed(num_patches, decoder_embed_dim, learnable=False)
        self.blocks = nn.ModuleList([Block(decoder_embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.pred = nn.Linear(decoder_embed_dim, self.patch_dim)

    def forward(self, x_vis, ids_restore):
        B, V, _ = x_vis.shape
        N = ids_restore.shape[1]
        Dd = self.decoder_embed.out_features

        x_emb = self.decoder_embed(x_vis)  # (B, V, Dd)

        # [x_visible + mask_token]: 共享 mask_token 广播到每个被遮位置
        mask_tokens = self.mask_token.repeat(B, N - V, 1)  # (B, M, Dd)
        x = torch.cat([x_emb, mask_tokens], dim=1)  # (B, V+M, Dd)

        # gather(ids_restore): 恢复原始 patch 顺序 —— decoder 的核心一步
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, Dd))  # (B, N, Dd)
        restored = x + self.decoder_pos_embed  # (B, N, Dd)

        x = restored
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = self.pred(x)  # (B, N, T)

        return {
            "decoder_embed": x_emb,  # (B, V, Dd) Linear(D->Dd) 之后的可见 token
            "restored": restored,  # (B, N, Dd) gather 还原 + pos_embed 后的 token
            "pred": pred,  # (B, N, T) 每个 patch 重建出的像素
        }
