# -*- coding: utf-8 -*-
"""
模块 08：BasicLayer（一个 stage：多个 SwinBlock + 可选 PatchMerging）
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

本模块把 06 的 PatchMerging 与 07 的 SwinBlock（含全部小部件）自包含复制进来，
组装成 Swin Transformer 的一个 stage：depth 个 SwinBlock（偶数位 W-MSA、奇数位
SW-MSA）+ 末尾可选 PatchMerging 降采样。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe basic_layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 自包含复制的 PatchMerging（模块 06）
# ----------------------------------------------------------------------
class PatchMerging(nn.Module):
    """把相邻 2x2 patch 拼接降采样: (B, H*W, C) -> (B, (H/2)*(W/2), 2C)。"""
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(self.norm(x))
        return x


# ----------------------------------------------------------------------
# 自包含复制的 SwinBlock 及其小部件（模块 07）
# ----------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """window_partition 的逆运算"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def build_relative_position_index(window_size: int) -> torch.Tensor:
    """构造相对位置偏置查表索引 (M^2, M^2)。"""
    coords = torch.stack(torch.meshgrid(torch.arange(window_size),
                                        torch.arange(window_size), indexing="ij"))
    coords = coords.reshape(2, -1)
    rel = coords[:, :, None] - coords[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1
    return rel.sum(-1)


def build_attn_mask(H: int, W: int, window_size: int, shift_size: int,
                    device: str = "cpu") -> torch.Tensor:
    """构造 SW-MSA 注意力掩码 (nW, M^2, M^2)。"""
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class Mlp(nn.Module):
    """两层 MLP：fc1 升维 -> GELU -> fc2 降维。"""
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
    """stochastic depth：以 drop_prob 概率把整条残差支路置零，除以 keep_prob 保持期望。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = noise.floor()
        return x / keep_prob * noise


class WindowAttention(nn.Module):
    """窗口多头自注意力：含相对位置偏置与 mask 广播。"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer("relative_position_index",
                             build_relative_position_index(window_size), persistent=False)
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


class SwinBlock(nn.Module):
    """Swin 基本块：pre-norm + (W/SW-MSA) + 残差 + pre-norm + MLP + 残差。"""
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
        key = (H, W, str(device))
        if key != self._mask_key:
            self._mask_cache = build_attn_mask(H, W, self.window_size, self.shift_size, device=device) \
                if self.shift_size > 0 else None
            self._mask_key = key
        return self._mask_cache

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size ** 2, C)
        x = self.attn(x, mask=self._get_mask(Hp, Wp, x.device))
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :H, :W, :].contiguous().view(B, L, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------------------------------------------------
# BasicLayer：一个 stage
# ----------------------------------------------------------------------
class BasicLayer(nn.Module):
    """一个 stage：depth 个 SwinBlock（偶数位 W-MSA、奇数位 SW-MSA）+ 可选 PatchMerging。
    drop_path 可为标量或逐块列表（列表时逐块取值）。"""
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W


def linear_drop_path_schedule(depth: int, drop_path_rate: float):
    """按深度线性增长的 stochastic depth 列表：第 i 块 = drop_path_rate * i / (depth-1)。"""
    if depth <= 1:
        return [0.0] * depth
    return [drop_path_rate * i / (depth - 1) for i in range(depth)]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)

    print("=" * 70)
    print("BasicLayer 演示：两 stage 小模型（56x56，window=7，dim 96->192）")
    print("=" * 70)

    B, H, W = 2, 56, 56
    x = torch.randn(B, H * W, 96)
    print(f"输入 x: {tuple(x.shape)}  (H={H}, W={W}, C=96)")

    # stage1: dim=96, depth=2, 末尾 PatchMerging(96->192)
    stage1 = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7,
                        downsample=PatchMerging(96))
    # stage2: dim=192, depth=2, 无降采样
    stage2 = BasicLayer(dim=192, depth=2, num_heads=6, window_size=7)

    x, H, W = stage1(x, H, W)
    print(f"stage1 之后: {tuple(x.shape)}  (H={H}, W={W}, C=192)")
    x, H, W = stage2(x, H, W)
    print(f"stage2 之后: {tuple(x.shape)}  (H={H}, W={W}, C=192)")

    p1 = sum(p.numel() for p in stage1.parameters())
    p2 = sum(p.numel() for p in stage2.parameters())
    print(f"\nstage1 参数量: {p1:,}")
    print(f"stage2 参数量: {p2:,}")

    # drop_path 线性增长演示
    print("\ndrop_path 按深度线性增长示例（depth=6, rate=0.3）：")
    sched = linear_drop_path_schedule(6, 0.3)
    print("  " + ", ".join(f"{v:.3f}" for v in sched))
