"""Stage 4 模块: PatchEmbed / Attention / Mlp / Block —— 从你的 vit.py 移植。

与 ViT 唯一区别: 这里没有任何 [CLS] 逻辑, Block 只吃 (B, L, D) 吐 (B, L, D)。
"""
import torch
import torch.nn as nn


def init_weights(m):
    """MAE 论文的初始化约定: Linear/Conv2d 用 trunc_normal(std=0.02), bias 置 0。"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class PatchEmbed(nn.Module):
    """(B, 3, H, W) -> (B, N, D), 与 vit.py 完全相同。"""

    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D), N 行主序 (与 patchify 约定一致)
        return x


class Attention(nn.Module):
    """多头自注意力: qkv 一次投影, 再切成 num_heads 份。"""

    def __init__(self, dim, num_heads=3, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"embed_dim={dim} 必须能被 num_heads={num_heads} 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 (B, heads, L, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class Mlp(nn.Module):
    """标准 MLP: Linear -> GELU -> Dropout -> Linear, 隐层 4 倍。"""

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class Block(nn.Module):
    """pre-norm TransformerBlock, 与 ViT 完全一致。

    (toy 阶段无 stochastic depth; 无 mask 传入, 因为 decoder 的可见/被遮位置
     信息已经由"token 内容"携带, 注意力本身不需要 mask)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
