"""Swin Transformer 手写复习版 —— 训练脚本（骨架对照 00-Basic-CV-Implementations/vit/reviewlearn.py）。

工程能力与 reviewlearn 对齐：
  - argparse 全参数化 + 固定随机种子（含 cudnn 确定性开关 / DataLoader worker 种子）
  - 数据集：CIFAR-100（32x32）/ FashionMNIST（28x28），原始尺寸直接训练
  - 模型：study-handwrite/model 里手写的迷你 Swin（小图专用配置，见 build_model）
  - 混合精度 AMP（GPU 自动开启）、梯度裁剪、CosineAnnealingLR
  - checkpoint：best.pt（仅模型）/ last.pt（完整状态），--resume 断点续跑
  - 早停 patience + TensorBoard + 每次运行独立 run_时间戳 日志目录

用法（在 study-handwrite 目录下运行）：
  python train.py --epochs 100 --batch-size 128           # CIFAR-100（默认）
  python train.py --dataset fashionmnist --epochs 30      # FashionMNIST
  python train.py --resume                                # 从 checkpoint/last.pt 续跑
  tensorboard --logdir runs

（本文件初版只有 parse_args()，main() 已按 vit/reviewlearn.py 骨架补全；
  本次全部改动见《修改记录.md》，跨文件夹 import/数据传递机制见《学习笔记_文件夹之间如何传递.md》）
"""
import argparse
import os
import random
import sys
import time

# Windows 控制台默认 GBK，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 保证从任意目录运行都能 import 到 study-handwrite 下的 model/ 与 utils/ 包
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.model import SwinTransformer
from utils.save import save_checkpoint, load_checkpoint

DROPOUT = 0.1

# ---- 迷你 Swin 配置（CIFAR-100 32x32 / FashionMNIST 28x28 小图专用）----
# patch_size=4 -> token 网格 8x8 / 7x7；window=4 使 8->4->2 每级都能完整成窗；
# num_heads 由 embed_size 推导（head_dim 恒为 32），保证各 stage 的 dim % num_heads == 0。
PATCH_SIZE = 4
WINDOW_SIZE = 4
SWIN_DEPTHS = (2, 2, 2)          # 3 个 stage：8x8 ->(merge) 4x4 ->(merge) 2x2
EMBED_MULTIPLE = 32              # embed_size 需为其倍数（head_dim = 32）


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性开关：只走结果确定的算法，代价是略慢，复现优先建议开启
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def work_init_fn(worker_id):
    """给 DataLoader 每个 worker 固定种子（与主进程解耦，保证可复现）。"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------- 数据集 ----------
def build_cifar100_loader(args):
    from torchvision import datasets, transforms
    CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR100_STD = (0.2673, 0.2564, 0.2762)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    train_ds = datasets.CIFAR100(root=args.data_dir, train=True,
                                 download=True, transform=train_transforms)
    test_ds = datasets.CIFAR100(root=args.data_dir, train=False,
                                download=True, transform=test_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=work_init_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=work_init_fn)
    return train_loader, test_loader, 3, 32, 100     # in_channels, img_size, num_classes


def build_fashionmnist_loader(args):
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=work_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, worker_init_fn=work_init_fn)
    return train_loader, val_loader, 1, 28, 10       # in_channels, img_size, num_classes


# ---------- 模型 ----------
def build_model(args, img_size, in_channels, num_classes):
    """按小图（32/28）实例化手写 SwinTransformer：embed_size 决定宽度，head 数自动推导。

    head 推导：num_heads[i] = embed_size/32 * 2**i（head_dim 恒为 32，各 stage 通道
    embed*2**i 必能被整除，例如 --embed-size 128 -> heads (4, 8, 16)）。
    """
    embed_size = args.embed_size
    assert embed_size % EMBED_MULTIPLE == 0, \
        f"--embed-size {embed_size} 需为 {EMBED_MULTIPLE} 的倍数（保证 dim 可被 num_heads 整除）"
    num_heads = tuple(int(embed_size / EMBED_MULTIPLE * 2 ** i) for i in range(len(SWIN_DEPTHS)))
    return SwinTransformer(img_size=img_size, patch_size=PATCH_SIZE, in_channels=in_channels,
                           num_classes=num_classes, embed_dim=embed_size,
                           depths=SWIN_DEPTHS, num_heads=num_heads,
                           window_size=WINDOW_SIZE, mlp_ratio=4., qkv_bias=True,
                           drop_rate=args.dropout, attn_drop_rate=0.0,
                           drop_path_rate=0.0, patch_merging=True)


# ---------- 训练 / 评估 ----------
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            scores = model(images)
            loss = criterion(scores, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (scores.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        loss = criterion(scores, labels)
        total_loss += loss.item()
        correct += (scores.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss / len(test_loader), correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Swin Transformer")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮次")
    parser.add_argument('--dataset', type=str, default="cifar100",
                        choices=["cifar100", "fashionmnist"], help="数据集")
    parser.add_argument("--data-dir", type=str,
                        default=r"D:\project\self_supervised_learning\data", help="数据目录")
    parser.add_argument('--batch-size', type=int, default=128, help="一批")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdaW权重衰退")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子默认42")
    parser.add_argument("--embed-size", type=int, default=128, help="token维度")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="dropout")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="混合精度（GPU上默认开启）")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoint",
                        help="checkpoint输出目录")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑，从ckpt-dir/last.pt恢复完整的训练状态")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="实验记录目录（每次运行生成 run_时间戳/config.txt+train.log）")
    # 早停新加入patience
    parser.add_argument("--patience", type=int, default=10,
                        help="验证准确率连续多少轮不创新高就停止")
    return parser.parse_args()


def main():
    args = parse_args()
    # 固定种子
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实验记录：每次运行独立的 run_时间戳 目录
    run_dir = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write("{} = {}\n".format(key, value))

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # tensorboard
    writer = SummaryWriter(os.path.join(run_dir, "tfboard"))
    log(f"device: {device}")
    log(f"run_dir: {run_dir}")
    log(f"seed: {args.seed}（含 numpy/random/cudnn 确定性开关）")
    log(f"tensorboard: 另开终端执行 `tensorboard --logdir {args.log_dir}` 查看曲线")

    # 数据集
    if args.dataset == "cifar100":
        train_loader, test_loader, in_channels, img_size, num_classes = \
            build_cifar100_loader(args)
    else:
        train_loader, test_loader, in_channels, img_size, num_classes = \
            build_fashionmnist_loader(args)
    log(f"dataset: {args.dataset}, classes: {num_classes}, "
        f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # 模型
    model = build_model(args, img_size, in_channels, num_classes).to(device)
    log(f"model: SwinTransformer(embed={args.embed_size}, depths={SWIN_DEPTHS}, "
        f"window={WINDOW_SIZE}), params: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失 / 优化器 / 调度器 / 混合精度
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # checkpoint 配置（写入 best.pt / last.pt 的模型结构信息）
    config = dict(img_size=img_size, patch_size=PATCH_SIZE, in_channels=in_channels,
                  num_classes=num_classes, embed_dim=args.embed_size,
                  depths=SWIN_DEPTHS, window_size=WINDOW_SIZE, dropout=args.dropout)

    # 断点续跑：从 last.pt 恢复完整训练状态
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    start_epoch, best_acc = 1, 0.0
    if args.resume:
        if os.path.exists(last_path):
            ckpt = load_checkpoint(last_path, model, optimizer, scheduler, scaler,
                                   map_location=device)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("best_acc", 0.0)
            log(f"[resume] 已从 {last_path} 恢复：上次跑到 epoch {ckpt.get('epoch', 0)}，"
                f"best_acc {best_acc:.4f}，本轮从 epoch {start_epoch} 继续")
        else:
            log(f"[resume] 未找到 {last_path}，从头开始训练")

    # 主训练循环
    bad_epoch = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, scaler, device, epoch, args)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # tensorboard 写入
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # 早停：创新高清零计数，连续 patience 轮不提升就 break
        if test_acc > best_acc:
            best_acc = test_acc
            bad_epoch = 0
            save_checkpoint(best_path, model, epoch, best_acc, config=config)
        else:
            bad_epoch += 1
            if args.patience > 0 and bad_epoch >= args.patience:
                log(f"[early stop] 验证集连续 {bad_epoch} 轮未提升，"
                    f"提前停止于 epoch {epoch}")
                break

        # 保存断点（完整训练状态，供 --resume 使用）
        save_checkpoint(last_path, model, epoch, best_acc, config=config,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        log(f"epoch {epoch:>3}/{args.epochs}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} "
            f"(best: {best_acc:.4f})")

    writer.add_hparams(vars(args), {"best_acc": best_acc})
    writer.close()
    log(f"training finished in {time.time() - start:.2f}s, best test acc: {best_acc:.4f}")
    log(f"checkpoint: {args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log(f"查看曲线: tensorboard --logdir {args.log_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
