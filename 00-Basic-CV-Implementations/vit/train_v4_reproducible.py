# -*- coding: utf-8 -*-
"""train_v4_reproducible.py —— 版本 4：初始版 + 完整可复现性工程

【使用指南】
    - 本版做什么：在初始版 train.py 的基础上，只补"可复现性"三件套，
      其余内容与初始版一致（详细注释见 train.py 或 04 文档）。
    - 运行：
        python train_v4_reproducible.py --dataset toy --epochs 3
        # 同样的命令跑两次，两次结果应该完全一致（toy 任务可验证）
    - 和初始版的差异（3 处，都有【本版新增/修改】标记）：
        ① 新增 set_seed()：初始版只做了 manual_seed，本版补齐
           random/numpy 种子 + cudnn 确定性开关（牺牲约 5-10% 速度换复现）；
        ② 三个 loader 工厂的 DataLoader 都加了 worker_init_fn：
           num_workers>0 时子进程的随机数不受主进程种子控制，
           不固定它的话数据增强顺序每次都不一样——这是"复现不了"的头号隐藏原因；
        ③ main() 里改用 set_seed()。
    - 本版移除了 wandb 钩子。

【核心概念（详见 13 文档）】
    可复现性四层：代码(种子+确定性) / 数据(划分固定) / 环境(requirements) / 记录(config)。
    本版负责"代码层"；跑两次结果一致后，用 pip freeze > requirements.txt 补上环境层。
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT

DATA_DIR = "../../data"
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# 【本版新增 ①】完整版固定种子函数。
# 初始版只做了 torch.manual_seed + cuda.manual_seed_all，缺三层：
# 1) Python 自带的 random 和 numpy 的种子（数据增强里可能用到）；
# 2) cudnn 确定性开关（部分卷积算法结果不确定）；
# 3) DataLoader 子进程的种子（见 worker_init_fn）。
def set_seed(seed: int):
    random.seed(seed)                      # Python 自带随机数
    np.random.seed(seed)                   # numpy（transforms 底层常用）
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed_all(seed)       # 所有 GPU
    # 确定性开关：让 cudnn 只走结果确定的算法。代价是训练略慢，
    # 复现优先的场景（科研实验）建议开启。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 【本版新增 ②】DataLoader 子进程的种子函数。
# num_workers>0 时，每个 worker 是独立进程，随机状态不受主进程控制；
# 不固定它，数据增强每次跑都不一样 -> 两次运行结果对不上。
# 用法：DataLoader(..., worker_init_fn=worker_init_fn)
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # 从主进程种子派生，保证可复现
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    """CIFAR-100 loader 工厂（【本版修改】加了 worker_init_fn）。"""
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
                              num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)   # 【本版修改】
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)    # 【本版修改】
    return train_loader, test_loader, 3, 32, 100


def build_toy_loaders(args):
    """toy loader 工厂（【本版修改】加了 worker_init_fn）。"""
    train_ds = ToyVisionDataset(num_samples=128, img_size=32, seed=0)
    val_ds = ToyVisionDataset(num_samples=64, img_size=32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              worker_init_fn=worker_init_fn)   # 【本版修改】
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            worker_init_fn=worker_init_fn)     # 【本版修改】
    return train_loader, val_loader, 3, 32, 4


def build_fashionmnist_loaders(args):
    """FashionMNIST loader 工厂（【本版修改】加了 worker_init_fn）。"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              worker_init_fn=worker_init_fn)   # 【本版修改】
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            worker_init_fn=worker_init_fn)     # 【本版修改】
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
    """命令行参数（同初始版，删掉 wandb）。"""
    parser = argparse.ArgumentParser(description="训练手写 ViT（CIFAR-100）+ 可复现性版")
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
    return parser.parse_args()


def main():
    args = parse_args()

    # 【本版修改 ③】用完整版 set_seed 替代初始版的两行种子代码
    set_seed(args.seed)

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
    log(f"seed: {args.seed}（可复现性版：含 numpy/random/cudnn 确定性开关）")

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

    # ---- 训练主循环（同初始版）----
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "best_acc": best_acc,
                "epoch": epoch,
            }, best_path)

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
            f"(best: {best_acc:.4f})")
    log(f"training finished in {time.time() - start:.1f}s, "
        f"best test acc: {best_acc:.4f}")
    log(f"checkpoint: {args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log_file.close()


if __name__ == "__main__":
    main()
