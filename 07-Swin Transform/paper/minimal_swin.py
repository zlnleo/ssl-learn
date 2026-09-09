# -*- coding: utf-8 -*-
"""Swin Transformer 论文最小复现参考代码（单文件，只依赖 torch/torchvision）。

论文：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (ICCV 2021, Best Paper)
复现目标（Swin-T）：
  - 参数量  ≈ 28.3M（实测 28,288,354）
  - MACs     ≈ 4.5G @ 224x224（= 论文 4.5G FLOPs，fvcore 口径：把 1 次乘加计为 1 FLOP）
  - ImageNet-1K top-1 = 81.3%（需要完整训练配方；本文件只复现"模型机制"，
    训练只保留最小闭环做正确性验收，不包含 EMA/RandAug/RepeatedAug 等配方）

最小复现原则：
  1. 每个函数标注对应的学习模块编号（01-09），一行一行对应论文公式；
  2. 关键机制（roll 方向、9 宫格 mask、相对偏置索引）保留完整注释——这些是复现的"考点"；
  3. 验收手段 = 对表（参数/FLOPs）+ 过拟合小 batch（梯度通路正确）：
     "模型能迅速过拟合 1 个 batch" 是复现代码正确的标准试金石。

运行（ssl_cv 环境，项目根目录）：
  python paper/minimal_swin.py            # 三个验收实验：参数/FLOPs 对表 + shape 表 + 过拟合
  python paper/minimal_swin.py --train    # 追加：CIFAR-100 小规模训练（本地数据，不下载）
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows 控制台默认 GBK，强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================ ① 窗口工具（模块 02/03/05）
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, M, M, C)：view 重解释 + permute 换索引序 + contiguous。"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """window_partition 的逆：(B*nW, M, M, C) -> (B, H, W, C)。"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def build_relative_position_index(window_size: int) -> torch.Tensor:
    """相对位置偏置的查表索引 (M^2, M^2)。

    窗口内两 token 的相对位移只有 (2M-1)^2 种取值：平移 M-1 归一到 [0,2M-2]^2，
    再按 (dh*(2M-1)+dw) 展平成偏置表的行号。
    """
    coords = torch.stack(torch.meshgrid(torch.arange(window_size),
                                        torch.arange(window_size), indexing="ij"))
    coords = coords.reshape(2, -1)                     # (2, M^2)
    rel = coords[:, :, None] - coords[:, None, :]      # (2, M^2, M^2) 广播相减 = 相对坐标
    rel = rel.permute(1, 2, 0).contiguous()            # (M^2, M^2, 2)
    rel[:, :, 0] += window_size - 1                    # 平移
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1                # 行号 × 表宽（混合进制）
    return rel.sum(-1)


def build_attn_mask(H: int, W: int, window_size: int, shift_size: int,
                    device: str = "cpu") -> torch.Tensor:
    """SW-MSA 掩码 (nW, M^2, M^2)：0 = 允许注意力，-100 = 屏蔽。

    循环移位后把图按 9 宫格切成 3x3 块编号 0..8；同一窗口内来自不同编号块的
    token 是 roll 造成的"伪邻居"（空间不相邻），对应位置加 -100，
    softmax 后 exp(-100)≈0 → 严格屏蔽。掩码与输入内容无关，可缓存。
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # 区域号差
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


# ============================================================ ② 基础部件
class Mlp(nn.Module):
    """两层 MLP + GELU，隐藏层 4x（论文 3.1 节，参数量 8C^2 是 block 参数主体）。"""

    def __init__(self, dim, mlp_ratio=4., drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(drop), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class DropPath(nn.Module):
    """stochastic depth：训练时整条残差支路以 drop_prob 置零，除以 keep_prob 保期望。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        noise = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                                  dtype=x.dtype, device=x.device)
        return x / keep * noise.floor()


# ============================================================ ③ 窗口注意力（模块 01/03/05）
class WindowAttention(nn.Module):
    """窗口内多头自注意力：QKV -> 多头 -> 缩放点积 -> +相对位置偏置 -> +掩码 -> softmax -> AV -> 投影。

    输入 (B_, N, C)，B_ = B*nW；mask (nW, N, N) 在 batch 间共享（窗口几何一致）。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5          # 缩放因子 1/sqrt(d)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 论文 3.1 节：相对位置偏置 B ∈ R^(M^2 x M^2)，从 (2M-1)^2 x h 的小表索引而来
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer("rel_pos_index", build_relative_position_index(window_size),
                             persistent=False)
        with torch.no_grad():
            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]    # 各 (B_, h, N, d)
        attn = q @ k.transpose(-2, -1)                   # (B_, h, N, N)
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        attn = attn + bias.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(attn.softmax(-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C)))


# ============================================================ ④ Swin Block（模块 07）
class SwinBlock(nn.Module):
    """pre-norm + (W-MSA / SW-MSA) + 残差 + pre-norm + MLP + 残差（论文 3.1 节，公式 3）。

    shift_size=0 -> W-MSA；shift_size=window//2 -> SW-MSA（torch.roll 循环移位）。
    本实现对任意尺寸先 pad 到 window_size 整数倍、移位/mask 在 pad 坐标系进行、最后 crop。
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.window_size, self.shift_size = window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop=drop)
        self._mask, self._mask_key = None, None

    def _get_mask(self, H, W, device):
        key = (H, W, str(device))
        if key != self._mask_key:
            self._mask = build_attn_mask(H, W, self.window_size, self.shift_size, device) \
                if self.shift_size > 0 else None
            self._mask_key = key
        return self._mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        # 1) pad 到 window 整数倍（roll 之前，统一坐标系）
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r
        # 2) 循环移位（仅 SW-MSA）
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # 3) 分窗 + 注意力
        x = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        x = self.attn(x, mask=self._get_mask(Hp, Wp, x.device))
        # 4) 还原：reverse -> unshift -> crop
        x = window_reverse(x.view(-1, self.window_size, self.window_size, C), self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :H, :W, :].contiguous().view(B, L, C)
        x = shortcut + self.drop_path(x)                    # 残差 1（注意力支路）
        return x + self.drop_path(self.mlp(self.norm2(x)))  # 残差 2（MLP 支路）


# ============================================================ ⑤ Patch 化与层级（模块 06/09）
class PatchEmbed(nn.Module):
    """patch 4x4 / stride 4 卷积切块 + LN：(B,3,H,W) -> (B,(H/4)(W/4),96)。"""

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class PatchMerging(nn.Module):
    """2x2 相邻 patch 合并（论文 3.1 节）：HxWxC -> H/2 x W/2 x 2C。
    4 路奇偶分组 -> 通道拼接 4C -> LN -> Linear(4C -> 2C)。"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 or W % 2:                                # 奇数尺寸保护
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)  # (B, H/2, W/2, 4C)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


# ============================================================ ⑥ 完整模型（模块 09）
class SwinTransformer(nn.Module):
    """完整 Swin（论文 Figure 3 左图）。Swin-T: C=96, depths=(2,2,6,2), heads=(3,6,12,24)。"""

    def __init__(self, img_size=224, num_classes=1000, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))   # 768
        self.patch_embed = PatchEmbed(3, embed_dim)
        # stochastic depth：随深度线性增长（论文实验配置）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()          # 每 stage 的 blocks
        self.merges = nn.ModuleList()          # 每 stage 末尾的 PatchMerging（末级为 Identity）
        for i in range(self.num_layers):
            self.stages.append(nn.ModuleList([
                SwinBlock(dim=int(embed_dim * 2 ** i), num_heads=num_heads[i],
                          window_size=window_size,
                          shift_size=0 if j % 2 == 0 else window_size // 2,
                          mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[sum(depths[:i]) + j])
                for j in range(depths[i])]))
            self.merges.append(PatchMerging(int(embed_dim * 2 ** i))
                               if i < self.num_layers - 1 else nn.Identity())
        self.norm = nn.LayerNorm(self.num_features)
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

    def forward(self, x):
        H, W = x.shape[2] // 4, x.shape[3] // 4
        x = self.patch_embed(x)                       # (B, 3136, 96)
        for stage, merge in zip(self.stages, self.merges):
            for blk in stage:
                x = blk(x, H, W)
            if not isinstance(merge, nn.Identity):    # 前 3 个 stage 做 PatchMerging
                x = merge(x, H, W)
                H, W = (H + 1) // 2, (W + 1) // 2
        return self.head(self.norm(x).mean(1))        # LN -> 全局平均池化 -> 分类头


def swin_tiny(num_classes: int = 1000, **kwargs) -> SwinTransformer:
    """论文 Table 1 的 Swin-T 配置。"""
    return SwinTransformer(num_classes=num_classes, **kwargs)


# ============================================================ ⑦ 复现验收：解析式 MACs（论文 4.5G 对表）
def analytic_macs(img_size: int = 224, window_size: int = 7, num_classes: int = 1000) -> float:
    """按公式手算 Swin-T 总 MACs（1 MAC = 1 次乘加 = 论文/fvcore 口径的 1 FLOP）。

    每个 block：注意力投影 4hwC² + MLP 8hwC² + 注意力点积 2M²hwC = 12hwC² + 2M²hwC
    每个 PatchMerging：(hw/4)·(4C·2C) = 2hwC²
    PatchEmbed：hw·(4²·3)·C
    """
    C, depths = 96, (2, 2, 6, 2)
    hw = (img_size // 4) ** 2
    macs = hw * 16 * 3 * C                                      # PatchEmbed
    for i, d in enumerate(depths):
        Ci, hi = C * 2 ** i, hw // 4 ** i
        macs += d * (12 * hi * Ci * Ci + 2 * window_size * window_size * hi * Ci)
        if i < len(depths) - 1:
            macs += 2 * hi * Ci * Ci                            # PatchMerging
    macs += num_classes * C * 8                                 # 分类头（768 -> 1000）
    return macs


# ============================================================ ⑧ 最小训练闭环（验收用，非论文训练配方）
def overfit_check(model, device, steps: int = 20) -> bool:
    """复现正确性的标准试金石：模型应能迅速过拟合 1 个小 batch。"""
    x = torch.randn(8, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (8,), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ok = losses[-1] < losses[0] * 0.25 and losses[-1] < 1.0
    print(f"  过拟合验收：loss {losses[0]:.3f} -> {losses[-1]:.3f}（20 步）  "
          f"{'PASS ✓（梯度通路正确）' if ok else 'FAIL ✗'}")
    return ok


def train_cifar(model, epochs: int = 3, img_size: int = 64, batch_size: int = 64,
                num_workers: int = 0, lr: float = 3e-4, device: str = "cuda"):
    """CIFAR-100 小规模训练（验证端到端可用；完整 81.3% 需论文训练配方 + ImageNet）。"""
    import torchvision
    import torchvision.transforms as T
    data_dir = next((p for p in (r"D:\project\self_supervised_learning\data", "./data")
                     if os.path.isdir(p)), "./data")
    norm = T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    tf = T.Compose([T.Resize(img_size), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    vf = T.Compose([T.Resize(img_size), T.ToTensor(), norm])
    try:
        train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=False, transform=tf)
        val_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=False, transform=vf)
    except RuntimeError:  # 本地没有才允许下载
        train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=tf)
        val_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=vf)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    head = model.head
    model.head = nn.Linear(model.num_features, 100).to(device)   # CIFAR-100 换分类头
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                      for x, y in val_dl)
        print(f"  epoch {ep + 1}/{epochs}  val_acc={correct / len(val_ds):.4f}  ({time.time() - t0:.0f}s)")
    model.head = head  # 还原


# ============================================================ ⑨ 主程序：三个验收实验
def main():
    ap = argparse.ArgumentParser(description="Swin-T 最小复现验收")
    ap.add_argument("--train", action="store_true", help="追加 CIFAR-100 小规模训练")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print("=" * 72)
    print("验收 1/3：参数与 MACs 对表（论文 Swin-T：28.3M 参数 / 4.5G FLOPs）")
    print("=" * 72)
    model = swin_tiny(num_classes=1000).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    macs = analytic_macs()
    print(f"  参数量：{n_params:,}（论文 28.3M，{'PASS ✓' if abs(n_params - 28_288_354) < 1000 else 'FAIL ✗'}）")
    print(f"  MACs  ：{macs / 1e9:.2f}G（论文 4.5G FLOPs/fvcore 口径，{'PASS ✓' if abs(macs - 4.5e9) / 4.5e9 < 0.02 else 'FAIL ✗'}）")
    print("  口径说明：1 MAC = 1 次乘加；论文与 fvcore 把它计为 1 FLOP。严格口径下 1 MAC = 2 FLOPs（约 9G）。")

    print("\n" + "=" * 72)
    print("验收 2/3：前向 shape 表（224x224，batch=2）")
    print("=" * 72)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224, device=device))
    print(f"  input (2,3,224,224) -> logits {tuple(out.shape)}（{'PASS ✓' if tuple(out.shape) == (2, 1000) else 'FAIL ✗'}）")
    print("  stage 分辨率轨迹：56x56/96 -> 28x28/192 -> 14x14/384 -> 7x7/768 -> 池化(768)")

    print("\n" + "=" * 72)
    print("验收 3/3：过拟合小 batch（梯度通路正确性试金石）")
    print("=" * 72)
    overfit_check(model, device)

    if args.train:
        print("\n" + "=" * 72)
        print(f"追加：CIFAR-100 小规模训练（{args.epochs} epoch，本地数据不下载）")
        print("=" * 72)
        train_cifar(model, epochs=args.epochs, device=device)

    print("\n最小复现验收完成。对照模块学习：01-09 各文件夹；工程版见 swin/ 与 train.py。")


if __name__ == "__main__":
    main()
