# -*- coding: utf-8 -*-
"""
================================================================================
 DeiT on CIFAR-100 —— 论文复现式实现
 论文: Training data-efficient image transformers & distillation through attention
      (Hugo Touvron et al., Facebook AI, arXiv:2012.12877)
================================================================================

 把 DeiT 的两大核心思想完整搬到 CIFAR-100 (32x32, 100 类) 上:

   1. 训练配方 (论文 Sec 4.1/4.2):
      RandAugment + Mixup + CutMix + Random Erasing + 标签平滑
      AdamW + 线性缩放学习率 + 余弦衰减 + warmup

   2. 知识蒸馏 (论文 Sec 3.2):
      class token + distillation token 双头结构
      硬蒸馏:  L = (1-a) * CE(cls_head, y) + a * CE(dist_head, argmax(Z_t))
      软蒸馏:  L = (1-a) * CE(cls_head, y) + a * tau^2 * KL(softmax(Z_s/tau) || softmax(Z_t/tau))

 教师: 本文件自带的小型卷积网络 TeacherCNN (先在 CIFAR-100 上训练, 约 65~70%)。

 快速使用 (本机推荐 conda 环境 ssl_cv, 数据在 D:\project\self_supervised_learning\data):
   python deit_cifar100.py                                  # 默认: tiny + 硬蒸馏, 100 epochs
   python deit_cifar100.py --model tiny --distill none      # 消融: 不蒸馏
   python deit_cifar100.py --model tiny --distill soft      # 软蒸馏
   python deit_cifar100.py --model micro --epochs 1 --teacher-epochs 1   # 快速冒烟测试

 说明: 本文件刻意不依赖 timm / 预训练权重, 全部从头实现, 便于学习。
================================================================================
"""

import argparse
import copy      # >>>【AI 添加 2026-08-28】EMA 评估用的模型深拷贝
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# =============================================================================
# 0. 工具函数
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# 1. 数据增强 —— 对应论文 Sec 4.2 "Distillation & inductive bias"
#    论文设置: RandAugment rand-m9-mstd0.5-inc1, Mixup a=0.8, CutMix a=1.0,
#              Random Erasing p=0.25
# =============================================================================

class RandAugmentCIFAR:
    """RandAugment (单张图版本): 随机选 n 个操作, 每个操作幅度在
    [m - mstd, m + mstd] 内扰动, 每训练一个 epoch 后 m += inc。
    论文原文: rand-m9-mstd0.5-inc1 -> n=2, m=9, mstd=0.5, inc=1。
    输入输出均为 [C,H,W] 的 float 张量, 像素范围 [0,1] (在 Normalize 之前调用)。"""

    def __init__(self, n: int = 2, m: int = 9, mstd: float = 0.5, inc: int = 1):
        self.n, self.m, self.mstd, self.inc = n, m, mstd, inc
        self.ops = [self.autocontrast, self.equalize, self.posterize, self.solarize,
                    self.contrast, self.color, self.brightness, self.sharpness,
                    self.rotate, self.shear_x, self.shear_y,
                    self.translate_x, self.translate_y]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for op in np.random.choice(self.ops, self.n, replace=False):
            mag = float(np.clip(np.random.uniform(self.m - self.mstd, self.m + self.mstd), 0, 30))
            img = op(img, mag)
        return img.clamp_(0.0, 1.0)

    # ---- 无幅度操作 ----
    @staticmethod
    def autocontrast(img, _):
        lo = img.amin(dim=(1, 2), keepdim=True)
        hi = img.amax(dim=(1, 2), keepdim=True)
        return (img - lo) / (hi - lo + 1e-8)

    @staticmethod
    def equalize(img, _):
        out = []
        for c in img:  # 逐通道直方图均衡
            vals = (c * 255).round().long().clamp(0, 255)
            hist = torch.bincount(vals.flatten(), minlength=256).float()
            cdf = hist.cumsum(0)
            cdf_min = cdf[cdf > 0].min()
            lut = ((cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-8) * 255).round().clamp(0, 255)
            out.append(lut[vals] / 255.0)
        return torch.stack(out)

    # ---- 有幅度操作 ----
    def posterize(self, img, mag):
        bits = 8 - int(mag / 30 * 4)                      # mag 越大, 保留位数越少 (8 -> 4)
        v = (img * 255).round().long()
        v = (v >> (8 - bits)) << (8 - bits)
        return v.float() / 255.0

    def solarize(self, img, mag):
        t = 1.0 - 0.5 * (mag / 30)                        # mag 越大, 阈值越低
        return torch.where(img < t, img, 1.0 - img)

    def contrast(self, img, mag):
        f = 1.0 + (2 * random.random() - 1) * 0.9 * (mag / 30)   # [0.1, 1.9]
        mean = img.mean(dim=(1, 2), keepdim=True)
        return (img - mean) * f + mean

    def color(self, img, mag):
        f = 1.0 + (2 * random.random() - 1) * 0.9 * (mag / 30)
        gray = img.mean(dim=0, keepdim=True)
        return (img - gray) * f + gray

    def brightness(self, img, mag):
        return img + (2 * random.random() - 1) * 0.9 * (mag / 30)

    def sharpness(self, img, mag):
        f = 1.0 + (2 * random.random() - 1) * 0.9 * (mag / 30)
        blurred = F.avg_pool2d(img.unsqueeze(0), 3, 1, 1).squeeze(0)
        return (img - blurred) * f + blurred

    def rotate(self, img, mag):
        return TF.rotate(img, (2 * random.random() - 1) * 30 * (mag / 30), fill=0)

    def shear_x(self, img, mag):
        return TF.affine(img, 0, [0, 0], 1.0, [(2 * random.random() - 1) * 0.3 * (mag / 30), 0], fill=0)

    def shear_y(self, img, mag):
        return TF.affine(img, 0, [0, 0], 1.0, [0, (2 * random.random() - 1) * 0.3 * (mag / 30)], fill=0)

    def translate_x(self, img, mag):
        dx = (2 * random.random() - 1) * 0.45 * (mag / 30) * img.shape[-1]
        return TF.affine(img, 0, [dx, 0], 1.0, [0, 0], fill=0)

    def translate_y(self, img, mag):
        dy = (2 * random.random() - 1) * 0.45 * (mag / 30) * img.shape[-2]
        return TF.affine(img, 0, [0, dy], 1.0, [0, 0], fill=0)


def mixup_data(x, y, alpha):
    """Mixup: x' = lam*x + (1-lam)*x[idx], 返回混合样本与被打乱的标签。"""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam, idx


def cutmix_data(x, y, alpha):
    """CutMix: 随机矩形区域替换成另一张图的对应区域。"""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y_b = y[idx]
    # 采样剪切框 (中心点 + 面积由 lam 决定)
    W, H = x.size(-1), x.size(-2)
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y_b, lam, idx


def mix_target(y_a, y_b, lam, num_classes):
    """把 Mixup/CutMix 的标签混合成软标签 (one-hot 的凸组合)。"""
    return lam * F.one_hot(y_a, num_classes).float() + (1 - lam) * F.one_hot(y_b, num_classes).float()


def build_train_transform(args):
    """训练增强: 随机裁剪+翻转 -> RandAugment -> Random Erasing -> 归一化。
    与论文一致: RandAugment(rand-m9-mstd0.5-inc1) + RandomErasing(p=0.25)。"""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        RandAugmentCIFAR(n=args.ra_n, m=args.ra_m, mstd=args.ra_mstd, inc=args.ra_inc),
        T.RandomErasing(p=args.re_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])


# =============================================================================
# 2. 教师网络: 小型卷积网络 (论文用 RegNetY-16GF, 这里用 CIFAR 规模的小 CNN)
# =============================================================================

class TeacherCNN(nn.Module):
    """VGG 风格 + BatchNorm 的小卷积网络, 约 0.5M 参数, 30 epoch 可到 65~70%。"""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8 -> 4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total, total / (time.time() - t0)          # (acc, img/s)


def train_teacher(args, device):
    """先训练卷积教师: SGD + momentum + 余弦衰减。返回最优 acc 与模型。"""
    print(f"[Teacher] 训练教师网络, {args.teacher_epochs} epochs, batch {args.batch_size}")
    train_set = CIFAR100(args.data_dir, train=True, download=True,
                         transform=T.Compose([T.RandomCrop(32, padding=4),
                                              T.RandomHorizontalFlip(), T.ToTensor(),
                                              T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]))
    test_set = CIFAR100(args.data_dir, train=False, download=True, transform=build_test_transform())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = TeacherCNN(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.teacher_lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.teacher_epochs + 1):
        # 余弦调度
        lr = args.teacher_lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.teacher_epochs))
        for g in optimizer.param_groups:
            g['lr'] = lr
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run_loss += loss.item() * x.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_total += x.size(0)
        model.eval()
        acc, _ = evaluate(model, test_loader, device)
        model.train()
        best_acc = max(best_acc, acc)
        print(f"  [Teacher] epoch {epoch:3d}  loss {run_loss / run_total:.4f}  "
              f"train_acc {run_correct / run_total:.4f}  test_acc {acc:.4f}")
    print(f"[Teacher] 完成, 最优 test acc = {best_acc:.4f}")
    if args.teacher_path:
        os.makedirs(os.path.dirname(args.teacher_path) or '.', exist_ok=True)
        torch.save({'model': model.state_dict(), 'acc': best_acc}, args.teacher_path)
        print(f"[Teacher] 已保存到 {args.teacher_path}")
    return model, best_acc


# =============================================================================
# 3. DeiT 模型 —— 对应论文 Fig. 2
# =============================================================================

class PatchEmbed(nn.Module):
    """图像切块 + 线性投影。CIFAR: 32x32, patch=4 -> 8x8=64 个 token。"""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B,3,32,32) -> (B,embed,8,8) -> (B,64,embed)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """多头自注意力 (qkv 一次线性投影 + 分头)。"""

    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                   # 各 (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """两层 MLP + GELU (ViT/DeiT 默认)。"""

    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DropPath(nn.Module):
    """Stochastic Depth: 按概率丢弃整条残差分支 (论文中 DeiT-B 用 0.1)。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep).div_(keep)
        return x * mask


class Block(nn.Module):
    """Transformer Encoder Block: pre-LN + 注意力残差 + MLP 残差。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class DistilledViT(nn.Module):
    """DeiT 核心: [class_token, distillation_token, patch_tokens] 序列,
    两个分类头分别输出; 训练时返回两个 logits, 推理时取平均。
    论文 Fig. 2 + Sec 3.2。"""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, distilled=True):
        super().__init__()
        self.num_classes = num_classes
        self.distilled = distilled

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 论文 Sec 3.2: 蒸馏 token 与 class token 并列, 一起参与注意力
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 逐层递增
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 双分类头: head 管真值, head_dist 管教师信号
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes) if distilled else None

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        if self.distilled:
            dist = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, dist, x), dim=1)      # [B, 2 + num_patches, dim]
        else:
            x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]        # 无蒸馏时少一个 token
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.distilled:
            return x[:, 0], x[:, 1]                    # (cls 特征, dist 特征)
        return x[:, 0], None

    def forward(self, x):
        x_cls, x_dist = self.forward_features(x)
        logits_cls = self.head(x_cls)
        logits_dist = self.head_dist(x_dist) if self.distilled else None
        if self.training:
            return (logits_cls, logits_dist) if self.distilled else logits_cls
        # 推理: 两头取平均 (论文 Sec 3.2, 官方实现一致)
        if self.distilled:
            return (logits_cls + logits_dist) / 2
        return logits_cls


# >>>【AI 添加 2026-08-28】EMA (指数滑动平均权重): shadow ← m·shadow + (1-m)·θ
# 注意: EMA 不是 DeiT 论文内容, 是 timm 等仓库的工程技巧 (v2 bonus)。
# 直觉: 训练末期参数在最优解附近抖动, "最后一个 checkpoint" 可能恰好在抖动高点;
#       影子权重是整条参数轨迹的平滑平均, 泛化通常更好, 推理/选最优时用影子权重。
class ModelEma:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd['decay']
        self.shadow = sd['shadow']
# <<<【AI 添加结束】


def build_deit(args, num_classes=100):
    cfgs = {
        'micro': dict(embed_dim=64, depth=4, num_heads=2, mlp_ratio=2.0),   # ~0.4M, 快速测试用
        'tiny': dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0),  # ~5.3M, 对齐 DeiT-Ti
        'small': dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0),  # ~21.7M, 对齐 DeiT-S
    }
    cfg = cfgs[args.model]
    return DistilledViT(img_size=32, patch_size=args.patch_size, num_classes=num_classes,
                        distilled=args.distilled, drop_path_rate=args.drop_path, **cfg)


# =============================================================================
# 4. 损失函数 —— 对应论文 Eq. (1) 硬蒸馏 / Eq. (2) 软蒸馏
# =============================================================================

def soft_cross_entropy(logits, target, smoothing=0.1):
    """带标签平滑的交叉熵; target 可以是类别索引 (long) 或软标签 (float)。"""
    if target.ndim == 1:
        with torch.no_grad():
            t = torch.full_like(logits, smoothing / logits.size(-1))
            t.scatter_(1, target.unsqueeze(1), 1.0 - smoothing + smoothing / logits.size(-1))
    else:
        t = (1 - smoothing) * target + smoothing / logits.size(-1)
    return -(t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def distillation_loss(student_out, teacher_logits, targets, args):
    """按论文 Eq.(1)/(2) 组合两个头的损失:
       总损失 = (1-alpha) * CE(分类头, 真值) + alpha * 蒸馏项(蒸馏头, 教师)
    返回 (总损失, 分类损失, 蒸馏损失)。"""
    logits_cls, logits_dist = student_out
    base_loss = soft_cross_entropy(logits_cls, targets, args.smoothing)

    if args.distill == 'hard':
        # Eq.(1): y_t = argmax_c Z_t(x), 学生用 CE 拟合教师硬标签 (温度无影响)
        with torch.no_grad():
            y_t = teacher_logits.argmax(dim=1)
        dist_loss = F.cross_entropy(logits_dist, y_t)
    elif args.distill == 'soft':
        # Eq.(2): tau^2 * KL(softmax(Z_s/tau) || softmax(Z_t/tau)), tau=3.0
        with torch.no_grad():
            p_t = F.softmax(teacher_logits / args.tau, dim=1)
        dist_loss = F.kl_div(F.log_softmax(logits_dist / args.tau, dim=1), p_t,
                             reduction='batchmean') * (args.tau ** 2)
    else:
        raise ValueError(args.distill)

    total = (1 - args.alpha) * base_loss + args.alpha * dist_loss
    return total, base_loss, dist_loss


# =============================================================================
# 5. 训练循环 —— 对应论文 Sec 4.1 (AdamW + 线性缩放 lr + 余弦 + warmup)
# =============================================================================

def train_one_epoch(student, teacher, loader, optimizer, args, device):
    student.train()
    if teacher is not None:
        teacher.eval()

    run_total = run_base = run_dist = 0.0
    run_correct = run_n = 0
    num_classes = 100

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ---- Mixup / CutMix: 先混合 (图像与标签用同一个 lam) ----
        targets = y
        if (args.mixup > 0 or args.cutmix > 0) and np.random.rand() < args.mix_switch:
            if args.mixup > 0 and np.random.rand() < 0.5:
                x, y_a, y_b, lam, idx = mixup_data(x, y, args.mixup)
            else:
                x, y_a, y_b, lam, idx = cutmix_data(x, y, args.cutmix)
            targets = mix_target(y_a, y_b, lam, num_classes)

        # ---- 教师前向: 看混合后的图 (eval + no_grad) ----
        # >>>【AI 修改 2026-08-29】最终方案: 教师直接看混合图 T(x_mixed), 精确;
        #  旧版"干净图 T(x) + 人工混 logits"是线性近似 (教师非线性, CutMix 下误差更大)。
        #  教师前向只发生一次, 不会因混合而翻倍。
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(x)
        # <<<【AI 修改结束】

        out = student(x)
        if args.distilled:
            loss, base, dist = distillation_loss(out, teacher_logits, targets, args)
        else:
            loss = soft_cross_entropy(out, targets, args.smoothing)
            base, dist = loss, torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_total += loss.item() * x.size(0)
        run_base += base.item() * x.size(0)
        run_dist += dist.item() * x.size(0)
        with torch.no_grad():
            pred = out[0] if args.distilled else out
            run_correct += (pred.argmax(1) == y).sum().item()
        run_n += x.size(0)

    return run_total / run_n, run_base / run_n, run_dist / run_n, run_correct / run_n


def main():
    parser = argparse.ArgumentParser(description='DeiT on CIFAR-100 (paper faithful implementation)')
    # 数据 / 硬件
    parser.add_argument('--data-dir', default=r'D:\project\self_supervised_learning\data',
                        help='CIFAR-100 数据目录 (已解压 cifar-100-python 或自动下载)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='./runs', help='checkpoint 输出目录')
    # 学生模型
    parser.add_argument('--model', default='tiny', choices=['micro', 'tiny', 'small'])
    parser.add_argument('--patch-size', type=int, default=4, help='32/4 -> 64 个 token')
    parser.add_argument('--drop-path', type=float, default=0.0, help='stochastic depth (论文 DeiT-B 用 0.1)')
    # >>>【AI 添加 2026-08-28】EMA 开关与衰减系数 (v2 bonus, 非论文内容)
    parser.add_argument('--ema', action='store_true', help='启用 EMA 权重 (选最优/推理用影子权重)')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA 衰减系数 m')
    # <<<【AI 添加结束】
    # 训练 (论文 Sec 4.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4, help='峰值学习率 (论文: 1e-3 * batch/512)')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--smoothing', type=float, default=0.1)
    # 增强 (论文 Sec 4.2)
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--mix-switch', type=float, default=0.5, help='启用 mixup/cutmix 的批比例')
    parser.add_argument('--ra-n', type=int, default=2)
    parser.add_argument('--ra-m', type=int, default=9)
    parser.add_argument('--ra-mstd', type=float, default=0.5)
    parser.add_argument('--ra-inc', type=int, default=1)
    parser.add_argument('--re-prob', type=float, default=0.25)
    # 蒸馏 (论文 Sec 3.2)
    parser.add_argument('--distill', default='hard', choices=['hard', 'soft', 'none'],
                        help='hard=Eq1, soft=Eq2, none=只训分类头 (消融)')
    parser.add_argument('--alpha', type=float, default=0.5, help='蒸馏权重')
    parser.add_argument('--tau', type=float, default=3.0, help='软蒸馏温度')
    # 教师
    parser.add_argument('--teacher-epochs', type=int, default=30)
    parser.add_argument('--teacher-lr', type=float, default=0.1)
    parser.add_argument('--teacher-path', default='./runs/teacher_cnn_cifar100.pth',
                        help='教师 checkpoint; 存在则直接加载, 否则训练并保存')
    args = parser.parse_args()

    args.distilled = args.distill != 'none'          # distill=none -> 纯 ViT 基线
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 模型: DeiT-{args.model} | 蒸馏: {args.distill}")
    print(f"学生参数: {sum(p.numel() for p in build_deit(args).parameters()) / 1e6:.2f}M")

    # ---- 教师: 加载或训练 ----
    teacher = None
    if args.distilled:
        teacher = TeacherCNN(num_classes=100).to(device)
        if os.path.exists(args.teacher_path):
            ckpt = torch.load(args.teacher_path, map_location=device)
            teacher.load_state_dict(ckpt['model'])
            print(f"[Teacher] 从 {args.teacher_path} 加载 (acc={ckpt['acc']:.4f})")
        else:
            teacher, _ = train_teacher(args, device)

    # ---- 数据 ----
    train_set = CIFAR100(args.data_dir, train=True, download=True, transform=build_train_transform(args))
    test_set = CIFAR100(args.data_dir, train=False, download=True, transform=build_test_transform())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              worker_init_fn=seed_worker, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # ---- 学生模型与优化器 ----
    student = build_deit(args).to(device)
    # >>>【AI 添加 2026-08-28】EMA: 影子权重对象 + 一个深拷贝模型 (每轮灌入影子权重后做评估)
    ema = ModelEma(student, args.ema_decay) if args.ema else None
    ema_model = copy.deepcopy(student) if args.ema else None
    # <<<【AI 添加结束】
    # AdamW: 权重衰减只作用于 1D 权重, 不作用于 bias / LayerNorm (论文 Sec 4.1)
    decay, no_decay = [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=args.lr)

    # ---- 训练主循环 ----
    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        # 学习率: 5 epoch 线性 warmup + 余弦衰减 (论文 Sec 4.1)
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr
        # RandAugment 幅度逐 epoch 递增 (rand-m9-inc1)
        for t in train_set.transform.transforms:
            if isinstance(t, RandAugmentCIFAR):
                t.m += args.ra_inc

        t0 = time.time()
        total, base, dist, train_acc = train_one_epoch(student, teacher, train_loader,
                                                       optimizer, args, device)
        test_acc, img_per_s = evaluate(student, test_loader, device)
        # >>>【AI 添加 2026-08-28】EMA: 每轮更新影子权重并单独评估; 开启 --ema 时以影子权重精度选最优
        if ema is not None:
            ema.update(student)
            ema_model.load_state_dict(ema.shadow)
            ema_acc, _ = evaluate(ema_model, test_loader, device)
            track_acc = ema_acc
        else:
            ema_acc, track_acc = None, test_acc
        # <<<【AI 添加结束】
        is_best = track_acc > best_acc
        best_acc = max(best_acc, track_acc)

        msg = (f"epoch {epoch:3d}/{args.epochs}  lr {lr:.2e}  "
               f"loss {total:.4f} (base {base:.4f} | dist {dist:.4f})  "
               f"train_acc {train_acc:.4f}  test_acc {test_acc:.4f}"
               f"{'' if ema_acc is None else f'  ema_acc {ema_acc:.4f}'}"
               f"{'  *' if is_best else ''}  {img_per_s:.0f} img/s  {time.time() - t0:.1f}s")
        print(msg)

        if is_best:
            ckpt = {'model': student.state_dict(), 'acc': best_acc, 'args': vars(args)}
            # >>>【AI 添加 2026-08-28】EMA: 最优时把影子权重一并存进 checkpoint, 推理时可选加载
            if ema is not None:
                ckpt['ema'] = ema.state_dict()
            # <<<【AI 添加结束】
            torch.save(ckpt, os.path.join(args.out_dir, f'deit_{args.model}_{args.distill}_best.pth'))

    print(f"\n完成! 最优测试精度: {best_acc:.4f}  "
          f"(checkpoint: {os.path.join(args.out_dir, f'deit_{args.model}_{args.distill}_best.pth')})")


if __name__ == '__main__':
    main()
