import torch
import torch.nn as nn
import math

DROPOUT = 0.1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # qkv[batch,num_heads,seq_len,head_dim]
        # k.T[batch,num_heads,head_dim,seq_len]
        # q*k.T[batch,num_heads,q_len,k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, seq_len = x.shape[:2]
        x = x.reshape(B, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _merge_heads(self, x):
        x = x.transpose(1, 2)
        B, seq_len = x.shape[:2]
        x = x.reshape(B, seq_len, -1)
        return x

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        output, attn = self.attention(q, k, v, mask)
        output = self._merge_heads(output)
        output = self.w_o(output)
        output = self.dropout(output)
        return output, attn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_size=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        x = self.patch_embedding(img)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        # x.shape =[batch,num_patches,embed_size]
        return x


class ClassTokenPosEmbed(nn.Module):
    def __init__(self, num_patches, embed_size, dropout=DROPOUT):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_size))
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, mlp_ratio=4, dropout=0.1):
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
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_size=128, num_heads=8, num_layers=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # 1.patch_embed
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                              embed_size=embed_size)
        num_patches = self.patch_embedding.num_patches

        self.cls_pos_embed = ClassTokenPosEmbed(self.patch_embedding.num_patches,
                                                embed_size, dropout)

        # 4.encoder block
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)
        ])
        # 5.
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embedding(x)

        x = self.cls_pos_embed(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


if __name__ == "__main__":
    # 快速自检：构造一个 32x32、4 类的小 ViT，跑一次前向看形状
    print("=" * 60)
    print("ViT 快速自检（python vit_solution.py）")
    print("=" * 60)
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10,
                embed_size=128, num_heads=8, num_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")

    x = torch.randn(2, 3, 32, 32)  # 假装是 2 张 32x32 的 RGB 图
    print(f"输入形状: {tuple(x.shape)}")
    print(f"patch 数量: {model.patch_embedding.num_patches} (32/4 x 32/4)")
    logits = model(x)
    print(f"输出 logits 形状: {tuple(logits.shape)}  # (batch, num_classes)")

    # 🧪 动手实验建议见 `02_验证与修改方向.md`
