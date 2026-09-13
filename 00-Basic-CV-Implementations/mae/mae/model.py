"""Stage 7 模块: ToyMAE —— 最小完整组装 (phase 1 交付物)。

数据流总览 (对应《00_MAE模块化学习路线与接口契约.md》§1.4):
    images -> patch embedding -> random masking -> encoder -> decoder embed
           -> append mask token -> gather(ids_restore) -> decoder -> pred -> masked MSE

phase-1 学习版: forward 返回 dict, 把每个中间张量暴露出来, 便于 test_mae_flow.py 逐步打印。
"""
import torch
import torch.nn as nn

from .decoder import MAEDecoder
from .encoder import MAEEncoder
from .modules import init_weights
from .patch import patchify
from .target import patch_normalize


def masked_mse(pred, target_norm, mask):
    """只在被遮位置算 MSE: (B,N,T) vs (B,N,T) -> 标量。

    mask: (B, N) bool, True = 被遮 (即要惩罚的位置)
    """
    loss = (pred - target_norm) ** 2
    loss = loss.mean(dim=-1)  # (B, N) 每个 patch 内部先平均
    loss = (loss * mask.float()).sum() / mask.sum()
    return loss


class ToyMAE(nn.Module):
    def __init__(self,
                 image_size=32, patch_size=8, in_chans=3,
                 encoder_embed_dim=192, encoder_depth=4, encoder_num_heads=3,
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (image_size // patch_size) ** 2

        self.encoder = MAEEncoder(image_size, patch_size, in_chans,
                                  encoder_embed_dim, encoder_depth, encoder_num_heads, mask_ratio)
        self.decoder = MAEDecoder(self.num_patches,
                                  encoder_embed_dim, decoder_embed_dim,
                                  decoder_depth, decoder_num_heads, patch_size, in_chans)

        # 初始化: Linear/Conv2d 全部 trunc_normal(std=0.02)
        self.apply(init_weights)
        # mask_token 是 nn.Parameter 不是 module, apply 不会碰它, 手动初始化
        nn.init.trunc_normal_(self.decoder.mask_token, std=0.02)

    def forward(self, imgs, mask_ratio=None):
        enc = self.encoder(imgs, mask_ratio)  # dict
        dec = self.decoder(enc["latent"], enc["ids_restore"])  # dict
        out = {**enc, **dec}

        # ---- target 与 masked MSE ----
        target = patchify(imgs, self.patch_size)  # (B, N, T) 原始像素 target
        target_norm, target_mean, target_var = patch_normalize(target)
        out.update({
            "target": target,  # (B, N, T) 归一化之前的原始像素
            "target_norm": target_norm,  # (B, N, T) 归一化后的回归目标
            "target_mean": target_mean,  # (B, N, 1)
            "target_var": target_var,  # (B, N, 1)
            "loss": masked_mse(dec["pred"], target_norm, enc["mask"]),  # 标量
        })
        return out
