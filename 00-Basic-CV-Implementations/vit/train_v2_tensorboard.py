# -*- coding: utf-8 -*-
"""train_v2_tensorboard.py —— 版本 2：初始版 + TensorBoard

【使用指南】
    - 本版做什么：在初始版 train.py 的基础上，只加 TensorBoard 可视化，
      其余内容与初始版逐行一致（详细注释见 train.py 或 04 文档）。
    - 依赖：pip install tensorboard（本机已装好）
    - 运行训练：
        python train_v2_tensorboard.py --dataset toy --epochs 5
    - 启动看板（另开一个终端）：
        tensorboard --logdir runs
        浏览器打开 http://localhost:6006，勾选不同 run 叠加对比
    - 和初始版的差异（全文只动 4 处，都有【本版新增】标记）：
        ① import SummaryWriter；
        ② main() 建 writer，事件文件写到 runs/run_时间戳/tfboard/；
        ③ 每个 epoch 记录 5 条曲线（train/test loss+acc 和 lr）；
        ④ 训练结束写 hparams 对比视图 + close。
    - 本版移除了 wandb 钩子（暂时用不到，需要时按 08 文档加回）。

【核心概念（详见 09 文档）】
    SummaryWriter("目录") 建一个"事件文件写入器"；
    add_scalar("曲线名", 数值, 横轴坐标) 往看板打一个点；
    曲线名里带 "/"（如 train/loss）会自动分组；
    TensorBoard 是完全本地的看板，不需要账号和联网。
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 【本版新增】TensorBoard 写入器

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
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
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
    """命令行参数（同初始版，删掉了 wandb 开关）。"""
    parser = argparse.ArgumentParser(description="训练手写 ViT（CIFAR-100）+ TensorBoard 版")
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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 实验记录：runs/run_时间戳/ 下存 config.txt + train.log（同初始版）----
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

    # 【本版新增 ②】TensorBoard 写入器：事件文件写到同一次 run 的子目录 tfboard/，
    # 这样 runs/ 下每次运行 = config.txt + train.log + tfboard 三件套
    writer = SummaryWriter(os.path.join(run_dir, "tfboard"))

    log(f"device: {device}")
    log(f"run dir: {run_dir}")
    log(f"tensorboard: 训练时另开终端执行 `tensorboard --logdir runs` 查看曲线")

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

        # 【本版新增 ③】每个 epoch 记录 5 条曲线。
        # tag 里带 "/" 会在看板左侧自动分组（train/test 两组）。
        # lr 也记一条：你能直接在看板上看到余弦调度曲线在下降。
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

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

    # 【本版新增 ④】超参数对比视图 + 收尾。
    # add_hparams 只调一次：之后看板的 HPARAMS 页能按 test_acc 给各次运行排序。
    writer.add_hparams(vars(args), {"best_acc": best_acc})
    writer.close()  # 关闭写入器（不关的话进程结束时也会自动落盘，显式更稳）

    log(f"training finished in {time.time() - start:.1f}s, "
        f"best test acc: {best_acc:.4f}")
    log(f"checkpoint: {args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log(f"查看曲线: tensorboard --logdir {args.log_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
