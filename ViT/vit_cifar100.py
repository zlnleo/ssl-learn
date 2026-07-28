import torch
from torch import nn
import  math

class PatchEmbedding(nn.Module):
    """
    CIFAR100 输入为 [B, 3, 32, 32]
    patch_size=4 → 8×8 = 64 个 patch
    每个 patch 映射到 embed_dim
    输出 [B, 64, embed_dim]
    """
    def __init__(self, image_size=32, patch_size=4,
                 in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # [B, 3, 32, 32]
        x = self.projection(x)
        # [B, embed_dim, 8, 8]

        x = x.flatten(2)
        # [B, embed_dim, 64]

        x = x.transpose(1, 2)
        # [B, 64, embed_dim]

        return x


class CLS_Token(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        # [B, 65, embed_dim]
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim)
        )

    def forward(self, x):
        return x + self.position_embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        out, _ = self.attention(x, x, x)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim,drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)

# DropPath 实现（timm 的简化版）
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(embed_dim=dim, num_heads=heads)
        self.drop1 = nn.Dropout(drop)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.drop1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # warmup：线性增长
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        # cosine
        return [
            base_lr * 0.5 * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs) /
                (self.max_epochs - self.warmup_epochs)
            ))
            for base_lr in self.base_lrs
        ]
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = \
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]



class TransformEncoder(nn.Module):
    def __init__(self, depth, dim, heads):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(dim, heads) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        cls = x[:, 0]
        return self.fc(cls)


class ViT(nn.Module):
    """
    CIFAR100 版本的 ViT
    """
    def __init__(self, num_classes=100):
        super().__init__()

        # CIFAR100 输入为 32×32
        self.patch = PatchEmbedding(
            image_size=32,
            patch_size=4,
            in_channels=3,
            embed_dim=256
        )

        embed_dim = 256
        num_tokens = self.patch.num_patches + 1  # +CLS

        self.cls = CLS_Token(embed_dim)
        self.pos = PositionEmbedding(num_tokens, embed_dim)

        self.encoder = TransformEncoder(
            depth=6,      # CIFAR100 不需要太深
            dim=embed_dim,
            heads=8
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.cls(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.head(x)
        return x
