# -*- coding: utf-8 -*-
"""A fixed ViT implementation based on the handwritten vit.py.

This file intentionally does not modify vit.py.  It keeps the same main
building blocks, but fixes the runtime errors, output shape, initialization,
and the public interface expected by train_vit.py / test_vit.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DROPOUT = 0.1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch, num_heads, seq_len, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


# Compatibility with the name used by vit_solution.py.
ScaleDotAttention = ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=DROPOUT):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        x = x.transpose(1, 2)
        batch_size, seq_len = x.shape[:2]
        return x.reshape(batch_size, seq_len, -1)

    def forward(self, q, k, v, mask=None):
        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))

        output, attn = self.attention(q, k, v, mask)
        output = self._merge_heads(output)
        output = self.w_o(output)
        output = self.dropout(output)
        return output, attn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_size=128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @property
    def patch_embedding(self):
        return self.proj

    def forward(self, x):
        x = self.proj(x)          # (B, embed_size, H/P, W/P)
        x = x.flatten(2)          # (B, embed_size, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_size)
        return x


class MLP(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=DROPOUT):
        super().__init__()
        hidden_size = int(hidden_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, mlp_ratio=4.0, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, int(embed_size * mlp_ratio), dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_size=128,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=DROPOUT,
        pool="cls",
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert pool in ("cls", "mean"), "pool must be 'cls' or 'mean'"
        self.img_size = img_size
        self.patch_size = patch_size
        self.pool = pool

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_size=embed_size,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_size))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            EncoderBlock(embed_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @property
    def patch_embedding(self):
        return self.patch_embed

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)
        return self.head(x)


# Compatibility with the class name used in the handwritten vit.py.
Vit = ViT


if __name__ == "__main__":
    model = ViT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_size=128,
        num_heads=8,
        num_layers=4,
    )
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"input shape: {tuple(x.shape)}")
    print(f"patches: {model.patch_embed.num_patches}")
    print(f"logits shape: {tuple(logits.shape)}")
