import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """
    实现切 patch + 投影的功能
    Input : x [B, C, H, W]，ImageNet 通常是 [B, 3, 224, 224]
    对 x 切成 196 个 patch，每个 patch 映射成 768 维向量
    最终得到 [B, 196, 768]
    """
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        # 224x224 / 16x16 → 14x14 → 196 个 patch
        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d 做 patch 切分 + embedding
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        x = self.projection(x)
        # [B, 768, 14, 14]

        x = x.flatten(2)
        # [B, 768, 196]

        x = x.transpose(1, 2)
        # [B, 196, 768]

        return x


class CLS_Token(nn.Module):
    """
    在序列前面加入一个 [CLS] token，并将它拼接到 patch 序列最前面
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # [1, 1, embed_dim] 的可学习参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        # CLS 必须在序列最前面
        x = torch.cat((cls, x), dim=1)
        # [B, 197, 768]
        return x


class PositionEmbedding(nn.Module):
    """
    可学习位置编码：
    为序列中的每个 token（包括 patch token 和 CLS token）提供一个位置向量
    """
    def __init__(self, num_tokens: int, embed_dim: int) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_tokens, embed_dim]
        return x + self.position_embedding  # 广播到 batch 维度


class MultiHeadAttention(nn.Module):
    """
    标准 Transformer 的多头自注意力（Self-Attention）
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, embed_dim]
        out, _ = self.attention(x, x, x)
        # out: [B, seq_len, embed_dim]
        return out


class MLP(nn.Module):
    """
    对每个 token 的 embedding 做非线性变换，增强模型表达能力
    """
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, dim]
        return self.net(x)


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block：
    Pre-Norm + Multi-Head Attention + Residual
    Pre-Norm + MLP + Residual
    """
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(embed_dim=dim, num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention 子层
        x = x + self.attn(self.norm1(x))
        # MLP 子层
        x = x + self.mlp(self.norm2(x))
        return x


class TransformEncoder(nn.Module):
    """
    把多个 EncoderBlock 顺序堆叠起来，形成完整的 Transformer Encoder
    """
    def __init__(self, depth: int, dim: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(dim, heads) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ClassificationHead(nn.Module):
    """
    ViT 的最终分类层：
    取 CLS token → 线性映射到 num_classes
    """
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = x[:, 0]  # [B, dim]
        return self.fc(cls)  # [B, num_classes]


class ViT(nn.Module):
    """
    完整的 Vision Transformer
    """
    def __init__(self, num_classes: int = 1000,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 heads: int = 12) -> None:
        super().__init__()
        self.patch = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        self.cls = CLS_Token(embed_dim)
        # num_tokens = num_patches + 1 (CLS)
        self.pos = PositionEmbedding(self.patch.num_patches + 1, embed_dim)
        self.encoder = TransformEncoder(depth, embed_dim, heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        x = self.patch(x)      # [B, 196, 768]
        x = self.cls(x)        # [B, 197, 768]
        x = self.pos(x)        # [B, 197, 768]
        x = self.encoder(x)    # [B, 197, 768]
        x = self.norm(x)       # [B, 197, 768]
        x = self.head(x)       # [B, num_classes]
        return x
