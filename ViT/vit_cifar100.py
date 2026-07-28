import torch
from torch import nn


# ============================
# Patch Embedding
# ============================

class PatchEmbedding(nn.Module):
    """
    Image:
        [B,3,32,32]

    Patch:
        4x4

    Output:
        [B,64,embed_dim]
    """

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256
    ):
        super().__init__()

        self.num_patches = (
            image_size // patch_size
        ) ** 2


        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self,x):

        # [B,3,32,32]
        x = self.projection(x)

        # [B,embed_dim,8,8]
        x = x.flatten(2)

        # [B,embed_dim,64]
        x = x.transpose(1,2)

        # [B,64,embed_dim]
        return x



# ============================
# CLS Token
# ============================

class CLSToken(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.randn(1,1,embed_dim)
        )


    def forward(self,x):

        B = x.shape[0]

        cls = self.cls_token.expand(
            B,-1,-1
        )

        x = torch.cat(
            [cls,x],
            dim=1
        )

        return x



# ============================
# Position Embedding
# ============================

class PositionEmbedding(nn.Module):

    def __init__(
        self,
        num_tokens,
        embed_dim
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                num_tokens,
                embed_dim
            )
        )


    def forward(self,x):

        return x + self.pos_embedding



# ============================
# Multi Head Attention
# ============================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1
    ):
        super().__init__()


        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )


    def forward(self,x):

        out,_ = self.attention(
            x,
            x,
            x
        )

        return out



# ============================
# Feed Forward Network
# ============================

class MLP(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                dim
            ),

            nn.Dropout(dropout)

        )


    def forward(self,x):

        return self.net(x)



# ============================
# DropPath
# ============================

class DropPath(nn.Module):

    def __init__(
        self,
        drop_prob=0.
    ):
        super().__init__()

        self.drop_prob = drop_prob


    def forward(self,x):

        if (
            self.drop_prob == 0.
            or not self.training
        ):
            return x


        keep_prob = (
            1-self.drop_prob
        )


        shape = (
            x.shape[0],
        ) + (
            1,
        ) * (
            x.ndim-1
        )


        random_tensor = (
            keep_prob
            +
            torch.rand(
                shape,
                device=x.device
            )
        )


        random_tensor.floor_()


        return (
            x
            /
            keep_prob
            *
            random_tensor
        )



# ============================
# Transformer Encoder Block
# ============================

class EncoderBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        mlp_ratio=4,
        dropout=0.1,
        drop_path=0.
    ):
        super().__init__()


        self.norm1 = nn.LayerNorm(dim)


        self.attn = MultiHeadAttention(
            dim,
            heads,
            dropout
        )


        self.drop_path1 = DropPath(
            drop_path
        )


        self.norm2 = nn.LayerNorm(dim)


        self.mlp = MLP(
            dim,
            dim * mlp_ratio,
            dropout
        )


        self.drop_path2 = DropPath(
            drop_path
        )



    def forward(self,x):

        x = x + self.drop_path1(
            self.attn(
                self.norm1(x)
            )
        )


        x = x + self.drop_path2(
            self.mlp(
                self.norm2(x)
            )
        )


        return x



# ============================
# Transformer Encoder
# ============================

class TransformEncoder(nn.Module):

    def __init__(
        self,
        depth,
        dim,
        heads,
        drop_path=0.1
    ):
        super().__init__()


        dpr = torch.linspace(
            0,
            drop_path,
            depth
        )


        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    dim,
                    heads,
                    drop_path=dpr[i].item()
                )
                for i in range(depth)
            ]
        )


    def forward(self,x):

        return self.blocks(x)



# ============================
# ViT Encoder
# ============================

class ViTEncoder(nn.Module):

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        depth=6,
        heads=8
    ):
        super().__init__()


        self.patch = PatchEmbedding(
            image_size,
            patch_size,
            in_channels,
            embed_dim
        )


        num_tokens = (
            self.patch.num_patches
            +
            1
        )


        self.cls = CLSToken(
            embed_dim
        )


        self.pos = PositionEmbedding(
            num_tokens,
            embed_dim
        )


        self.encoder = TransformEncoder(
            depth,
            embed_dim,
            heads
        )


        self.norm = nn.LayerNorm(
            embed_dim
        )


        self.out_dim = embed_dim



    def forward(self,x):

        x = self.patch(x)

        x = self.cls(x)

        x = self.pos(x)

        x = self.encoder(x)

        x = self.norm(x)


        # CLS token
        return x[:,0]



# ============================
# Classification Head
# ============================

class ClassificationHead(nn.Module):

    def __init__(
        self,
        dim,
        num_classes
    ):
        super().__init__()

        self.fc = nn.Linear(
            dim,
            num_classes
        )


    def forward(self,x):

        return self.fc(x)



# ============================
# ViT
# ============================

class ViT(nn.Module):

    def __init__(
        self,
        num_classes=100,
        image_size=32,
        patch_size=4,
        embed_dim=256,
        depth=6,
        heads=8
    ):
        super().__init__()


        self.encoder = ViTEncoder(
            image_size,
            patch_size,
            3,
            embed_dim,
            depth,
            heads
        )


        self.head = ClassificationHead(
            embed_dim,
            num_classes
        )


    def forward(self,x):

        feature = self.encoder(x)

        out = self.head(feature)

        return out