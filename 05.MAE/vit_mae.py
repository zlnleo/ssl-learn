import torch
import torch.nn as nn


# embed编码
class PatchEmbedding(nn.Module):
    """
    in_channels:通道数
    embed_dim:embed多少维
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# 位置编码
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out


# MLP
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# DropPath
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# transform - encoder
class Encoderblock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout=dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MAETransformer(nn.Module):
    def __init__(self, depth, dim, heads, drop_path=0.0):
        super().__init__()
        dpr = torch.linspace(0, drop_path, depth)
        self.blocks = nn.ModuleList(
            [Encoderblock(dim, heads, drop_path=dpr[i].item()) for i in range(depth)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=6, heads=8):
        super().__init__()
        self.encoder = MAETransformer(depth, embed_dim, heads, drop_path=0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = self.encoder(x)
        x = self.norm(x)
        return x


def random_masking(x, mask_ratio=0.75):
    """
    MAE核心
    输入:
    x:
    [B,N,D]
    CIFAR:

    [B,64,256]
    mask 75%
    保留:
    16 tokens
    删除:
    48 tokens
    """
    B, N, D = x.shape

    keep_num = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)

    # 随机排序

    ids_shuffle = torch.argsort(noise, dim=1)

    # 恢复原顺序

    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 保留token索引

    ids_keep = ids_shuffle[:, :keep_num]

    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # mask:

    # 0 = visible

    # 1 = masked

    mask = torch.ones(B, N, device=x.device)

    mask[:, :keep_num] = 0

    mask = torch.gather(mask, dim=1, index=ids_restore)

    return (x_masked, mask, ids_restore)


class MAEDecoder(nn.Module):
    def __init__(
        self, embed_dim=256, decoder_dim=128, depth=2, heads=4, num_patches=64
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.pos = PositionEmbedding(num_patches, decoder_dim)

        self.decoder = MAETransformer(depth, decoder_dim, heads)

        self.norm = nn.LayerNorm(decoder_dim)

        self.head = nn.Linear(decoder_dim, 4 * 4 * 3)

    def forward(self, x, ids_restore):
        # encoder dim

        x = self.decoder_embed(x)

        B, N, D = x.shape

        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - N, 1)

        x = torch.cat([x, mask_tokens], dim=1)

        # 恢复patch顺序

        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        x = self.pos(x)

        x = self.decoder(x)

        x = self.norm(x)

        x = self.head(x)

        return x


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding(32, 4, 3, 256)
        self.pos_embedding = PositionEmbedding(64, 256)
        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder()

    def forward(self, img):
        # encoder
        x = self.patch_embedding(img)
        x = self.pos_embedding(x)
        x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)

        latent = self.encoder(x_masked)

        # decoder
        pred = self.decoder(latent, ids_restore)

        return (pred, mask)
