# -*- coding: utf-8 -*-
"""
模块 09：完整 Swin Transformer 组装
====================================

学习顺序：这是 Swin Transformer 从零学习项目的第 9 个（也是收尾）模块。
前面 01-08 个模块分别拆解了 PatchEmbed、窗口划分/还原、W-MSA / SW-MSA、
相对位置偏置、attention mask、PatchMerging、stochastic depth 等机制，
本模块把它们**总装**成一个可训练的完整视觉骨干 SwinTransformer。

本文件自包含全部依赖，按如下顺序组织（便于边读边对照）：
    小部件（窗口函数 / 相对位置索引 / attn mask / Mlp / DropPath / WindowAttention）
    -> PatchEmbed -> PatchMerging -> SwinBlock -> BasicLayer
    -> SwinTransformer -> swin_tiny 工厂函数

运行环境约定：默认 CPU；前向验证用 (2, 3, 224, 224)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 第一部分：小部件（窗口划分/还原、相对位置索引、注意力掩码、MLP、DropPath、窗口注意力）
# =============================================================================
def window_partition(x, window_size):
    """把 (B, H, W, C) 切成一个个 window_size×window_size 的小窗，返回 (nW, ws, ws, C)。

    形状：输入 (B, H, W, C) -> 输出 (B*H/ws*W/ws, ws, ws, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """window_partition 的逆操作：把 (nW, ws, ws, C) 还原成 (B, H, W, C)。"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def build_relative_position_index(window_size):
    """构造相对位置索引表 (ws², ws²)。

    对窗口内任意两个 patch 坐标 (h1,w1) 与 (h2,w2)，其相对位移
    (Δh, Δw) 被平移到 [0, 2*ws-1) 区间后编码成唯一标量，用于查
    relative_position_bias_table。
    """
    coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
    coords = coords.reshape(2, -1)                    # (2, ws²)
    rel = coords[:, :, None] - coords[:, None, :]     # (2, ws², ws²) 相对位移
    rel = rel.permute(1, 2, 0).contiguous()           # (ws², ws², 2)
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1
    return rel.sum(-1)                                # (ws², ws²) 一维索引


def build_attn_mask(H, W, window_size, shift_size, device="cpu"):
    """为 SW-MSA 构造 attention mask，屏蔽移位后跨窗（本不属于同一窗）的 token 对。

    思路：给整图每个像素打上“原窗口编号”，移位后再切窗，若同一窗内出现
    不同编号，则它们本不相邻，其注意力 logits 被置为 -100（softmax 后为 0）。
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class Mlp(nn.Module):
    """两层 MLP：in -> hidden(默认 4x) -> in，中间 GELU，首尾 Dropout。"""

    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic Depth：以概率 drop_prob 把整条残差分支置零（训练时），推理时直通。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)      # 只按 batch 维随机
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * noise.floor()


class WindowAttention(nn.Module):
    """窗口多头自注意力 + 相对位置偏置。

    输入 (B_, N, C) 中 N = ws² 为单个窗口内的 token 数；
    相对位置偏置表形状 (2*ws-1)² × num_heads，按索引查表加到 attention logits 上。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 一次线性投影得到 q,k,v
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 可学习相对位置偏置表 + 不可学习的索引 buffer
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer("relative_position_index",
                             build_relative_position_index(window_size), persistent=False)
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # (B_, N, 3, h, d) -> (3, B_, h, N, d)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)       # (B_, h, N, N) 缩放点积
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)  # (1, h, N, N)
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


# =============================================================================
# 第二部分：PatchEmbed（图 -> patch 序列）
# =============================================================================
class PatchEmbed(nn.Module):
    """patch_size×patch_size 卷积把图像切成 patch：(B,3,H,W) -> (B,(H/p)*(W/p),embed_dim)。"""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)                    # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, (H/p)*(W/p), embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# 第三部分：PatchMerging（下采样，通道翻倍）
# =============================================================================
class PatchMerging(nn.Module):
    """把 2×2 相邻 patch 拼起来再线性降到 2*dim：(B,H,W,C) -> (B,H/2,W/2,2C)。"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))   # 奇数尺寸补边
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)        # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))            # (B, H/2*W/2, 2C)


# =============================================================================
# 第四部分：SwinBlock（W-MSA / SW-MSA + MLP 的 Transformer block）
# =============================================================================
class SwinBlock(nn.Module):
    """shift_size=0 -> W-MSA；shift_size=window_size//2 -> SW-MSA。

    任意尺寸先 pad 到 window_size 整数倍，移位/mask 在 pad 坐标系进行，最后 crop 回原尺寸。
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self._mask_cache = None
        self._mask_key = None

    def _get_mask(self, H, W, device):
        """按 (H, W, device) 缓存 SW-MSA 的 attention mask，避免每步重复构造。"""
        key = (H, W, str(device))
        if key != self._mask_key:
            self._mask_cache = build_attn_mask(H, W, self.window_size, self.shift_size, device=device) \
                if self.shift_size > 0 else None
            self._mask_key = key
        return self._mask_cache

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)                    # (B, H, W, C)
        # pad 到 window_size 整数倍
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = window_partition(x, self.window_size)             # (nW, ws, ws, C)
        x = x.view(-1, self.window_size ** 2, C)              # (nW, ws², C)
        x = self.attn(x, mask=self._get_mask(Hp, Wp, x.device))
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, Hp, Wp)       # (B, Hp, Wp, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :H, :W, :].contiguous().view(B, L, C)        # crop 回原尺寸
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# 第五部分：BasicLayer（一个 stage：depth 个 SwinBlock + 可选 downsample）
# =============================================================================
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.depth = depth
        # 偶数块 W-MSA、奇数块 SW-MSA；drop_path 可为逐块列表或标量
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                      norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2        # H/W 随 stage 折半（向上取整）
        return x, H, W


# =============================================================================
# 第六部分：SwinTransformer（顶层组装）
# =============================================================================
class SwinTransformer(nn.Module):
    """完整 Swin Transformer。Swin-Tiny: embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7。"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        # 最后一层通道数 = embed_dim * 2^(num_layers-1)，如 Tiny 为 96*8=768
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim,
                                      norm_layer if patch_norm else None)
        # stochastic depth：随块索引从 0 线性增长到 drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i), depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                # dpr 按 stage 切片：第 i 个 stage 分到 sum(depths[:i]) ~ sum(depths[:i+1])
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging(dim=int(embed_dim * 2 ** i)) if i < self.num_layers - 1 else None)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        H, W = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.patch_embed(x)              # (B, L, C)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)                     # (B, L, C)
        return x.mean(dim=1)                 # (B, C) 全局平均池化

    def forward(self, x):
        return self.head(self.forward_features(x))


def swin_tiny(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """Swin-Tiny 工厂：embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7。"""
    return SwinTransformer(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                           window_size=7, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # 快速冒烟：随机输入前向一次，打印输出形状
    model = swin_tiny(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"输入  : {tuple(x.shape)}")
    print(f"输出  : {tuple(y.shape)}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
