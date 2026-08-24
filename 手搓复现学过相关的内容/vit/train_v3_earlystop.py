# -*- coding: utf-8 -*-
"""train_v3_earlystop.py —— 版本 3：初始版 + 早停（early stopping）

【使用指南】
    - 本版做什么：在初始版 train.py 的基础上，只加"早停"机制，
      其余内容与初始版逐行一致（详细注释见 train.py 或 04 文档）。
    - 运行：
        python train_v3_earlystop.py --dataset toy --epochs 50 --patience 5
        # 当验证准确率连续 5 个 epoch 不创新高时，自动提前停止
    - 和初始版的差异（全文 3 处，都有【本版新增】标记）：
        ① 新增参数 --patience（默认 10，传 0 表示禁用早停）；
        ② main() 里加 bad_epochs 计数器；
        ③ 训练循环末尾判断：创新高就清零计数，否则 +1，达到阈值就 break。
    - 本版移除了 wandb 钩子。

【核心概念（详见 10 文档 §三）】
    早停解决两个问题：
    1. 防止过拟合——验证指标开始不涨甚至变差时，继续训练只是在背训练集；
    2. 省算力——不用傻等 epochs 跑满。
    patience 的直觉：允许"憋"几个 epoch 不涨（指标会波动），超过就认为到头了。
    注意：早停的"验证集"被反复用来做停止决策，会轻微"泄漏"信息——
    严格做法是验证集调参/早停，测试集只测一次（科研红线，见 10 文档）。
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT

# ---- 以下到 train_one_epoch/evaluate 为止，与初始版 train.py 完全一致 ----
DATA_DIR = "../../data"
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class ToyVisionDataset(torch.utils.data.Dataset):
    """合成"象限亮块"4 分类任务（同初始版）。"""

    def __init__(self, num_samples=128, img_size=32, noise=0.1, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.images, self.labels = [], []
        block = 8
        offset = img_size // 2 - block
        for i in range(num_samples):
            label = i % 4
            img = torch.rand(3, img_size, img_size, generator=g) * noise
            r, c = (label // 2) * offset, (label % 2) * offset
            img[:, r:r + block, c:c + block] = 1.0
            self.images.append(img)
            self.labels.append(label)
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def build_cifar100_loaders(args):
    """CIFAR-100 loader 工厂（同初始版）。"""
    from torchvision import datasets, transforms

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = datasets.CIFAR100(root=args.data_dir, train=True,
                                 download=True, transform=train_transform)
    test_ds = datasets.CIFAR100(root=args.data_dir, train=False,
                                download=True, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader, 3, 32, 100


def build_toy_loaders(args):
    """toy loader 工厂（同初始版）。"""
    train_ds = ToyVisionDataset(num_samples=128, img_size=32, seed=0)
    val_ds = ToyVisionDataset(num_samples=64, img_size=32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, 3, 32, 4


def build_fashionmnist_loaders(args):
    """FashionMNIST loader 工厂（同初始版）。"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, 1, 28, 10


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    """单轮训练（同初始版，含 AMP 四步）。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """验证集评估（同初始版）。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss / len(loader), correct / total


def parse_args():
    """命令行参数（同初始版，删掉 wandb，新增 patience）。"""
    parser = argparse.ArgumentParser(description="训练手写 ViT（CIFAR-100）+ 早停版")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "toy", "fashionmnist"], help="数据集")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据集缓存目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW 权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--embed-size", type=int, default=192, help="token 维度")
    parser.add_argument("--num-heads", type=int, default=6, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=6, help="层数")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="dropout")
    parser.add_argument("--amp", action="store_true", default=True, help="混合精度（GPU 上默认开启）")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoint",
                        help="checkpoint 输出目录（best.pt=最优模型, last.pt=续跑存档）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：从 ckpt-dir/last.pt 恢复完整训练状态")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="实验记录目录（每次运行生成 run_时间戳/config.txt+train.log）")
    # 【本版新增 ①】早停参数
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值：验证准确率连续多少轮不创新高就停止（0 = 禁用早停）")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 实验记录（同初始版）----
    run_dir = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key} = {value}\n")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device: {device}")
    log(f"run dir: {run_dir}")

    # ---- 数据 ----
    if args.dataset == "cifar100":
        train_loader, test_loader, in_channels, img_size, num_classes = build_cifar100_loaders(args)
    elif args.dataset == "toy":
        train_loader, test_loader, in_channels, img_size, num_classes = build_toy_loaders(args)
    else:
        train_loader, test_loader, in_channels, img_size, num_classes = build_fashionmnist_loaders(args)
    log(f"dataset: {args.dataset}, classes: {num_classes}, "
        f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # ---- 模型 / 损失 / 优化器 / 调度器（同初始版）----
    model = ViT(
        img_size=img_size, patch_size=4, in_channels=in_channels,
        num_classes=num_classes, embed_size=args.embed_size,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = dict(img_size=img_size, patch_size=4, in_channels=in_channels,
                  num_classes=num_classes, embed_size=args.embed_size,
                  num_heads=args.num_heads, num_layers=args.num_layers,
                  dropout=args.dropout)

    # ---- 断点续跑（同初始版）----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    start_epoch, best_acc = 1, 0.0
    if args.resume:
        if os.path.exists(last_path):
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt["best_acc"]
            log(f"[resume] 已从 {last_path} 恢复：上次跑到 epoch {ckpt['epoch']}，"
                f"best_acc {best_acc:.4f}，本轮从 epoch {start_epoch} 继续")
        else:
            log(f"[resume] 未找到 {last_path}，从头开始训练")

    # ---- 训练主循环 ----
    # 【本版新增 ②】bad_epochs：记录"验证集已连续多少轮没创新高"
    bad_epochs = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # 【本版新增 ②③】早停逻辑：创新高就清零计数，否则计数 +1，
        # 连续 patience 轮不涨就 break。必须用 if/else 配对——
        # 若写成独立的 `if test_acc <= best_acc`，创新高那轮会因为
        # best_acc 刚被更新成 test_acc 而被误判成"不涨"，计数就乱了。
        if test_acc > best_acc:
            best_acc = test_acc
            bad_epochs = 0  # 创新高 -> 清零"不涨"计数
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "best_acc": best_acc,
                "epoch": epoch,
            }, best_path)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                log(f"[early stop] 验证集连续 {bad_epochs} 轮未提升，"
                    f"提前停止于 epoch {epoch}")
                break

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "config": config,
        }, last_path)

        log(f"epoch {epoch:>3}/{args.epochs}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} "
            f"(best: {best_acc:.4f}, 不涨轮数: {bad_epochs}/{args.patience})")

    log(f"training finished in {time.time() - start:.1f}s, "
        f"best test acc: {best_acc:.4f}")
    log(f"checkpoint: {args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log_file.close()


if __name__ == "__main__":
    main()
