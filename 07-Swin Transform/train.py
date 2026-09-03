"""Swin-Tiny 模块化工程训练脚本。

工程能力（按需使用，均有命令行开关）：
  - argparse 全参数化：模型配置（含 window_size / patch_merging 消融开关）、数据集、优化器、调度
  - checkpoint：每轮保存 best/last，含 optimizer/scaler/scheduler 完整状态
  - resume：--resume 恢复训练（epoch/指标/随机状态全部续上）
  - TensorBoard：--output-dir/runs 下记录 loss/lr/acc
  - 日志：logging 同时输出到控制台与 --output-dir/train.log
  - 混合精度：--amp
  - 评估模式：--eval-only

示例（项目根目录，ssl_cv 环境）：
  python train.py --dataset cifar100 --img-size 224 --epochs 100 --batch-size 64 --output-dir output/tiny
  python train.py --dataset cifar10 --img-size 64 --epochs 50                # 快速小图实验
  python train.py --resume output/tiny/last.pt
  python train.py --resume output/tiny/best.pt --eval-only
  python train.py --no-patch-merging            # 实验 3 的消融开关
  python train.py --window-size 14              # 实验 4 的消融开关
"""
import argparse
import json
import logging
import math
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
from torch.utils.tensorboard import SummaryWriter

from swin import build_swin, SWIN_CONFIGS
import data


def parse_args():
    ap = argparse.ArgumentParser(description="Swin-Tiny 模块化训练脚本")
    # 数据
    ap.add_argument("--dataset", default="cifar100",
                    choices=["cifar10", "cifar100", "imagenet", "folder"])
    ap.add_argument("--data-dir", default=None,
                    help="数据集根目录；默认自动探测本地已有目录（D:/project/.../data 或 ./data）")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    # 模型（Swin-T/S/B/L + 消融开关）
    ap.add_argument("--model", default="tiny", choices=list(SWIN_CONFIGS))
    ap.add_argument("--num-classes", type=int, default=None, help="默认随数据集")
    ap.add_argument("--window-size", type=int, default=7, help="实验 4 消融开关")
    ap.add_argument("--no-patch-merging", action="store_true", help="实验 3 消融开关")
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--drop-rate", type=float, default=0.0)
    # 训练
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-freq", type=int, default=50)
    ap.add_argument("--amp", action="store_true", help="混合精度（CUDA）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"],
                    help="训练设备；cuda/auto 在无 GPU 时自动回退 cpu")
    # 工程
    ap.add_argument("--output-dir", default="./output/train")
    ap.add_argument("--resume", default=None, help="checkpoint 路径，恢复训练")
    ap.add_argument("--eval-only", action="store_true")
    return ap.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    for h in logger.handlers:
        logger.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def build_optimizer_scheduler(model, args, train_loader):
    steps_per_epoch = len(train_loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step):  # linear warmup + cosine decay
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.min_lr / args.lr + 0.5 * (1 - args.min_lr / args.lr) * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def save_checkpoint(path, model, opt, sched, scaler, epoch, best_acc, args):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_acc": best_acc,
        "args": vars(args),
    }, path)


def train_one_epoch(model, loader, opt, sched, criterion, epoch, args, device, scaler, logger, writer):
    model.train()
    total = correct = 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        sched.step()
        total += y.numel()
        with torch.no_grad():
            correct += (logits.argmax(1) == y).sum().item()
        step = epoch * len(loader) + i
        if i % args.print_freq == 0:
            lr = sched.get_last_lr()[0]
            logger.info(f"ep {epoch} [{i}/{len(loader)}] loss={loss.item():.4f} lr={lr:.2e}")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
    return time.time() - t0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(args.seed)
    if args.device in ("cuda", "auto") and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[warn] 请求 cuda 但不可用，回退到 cpu")
    logger = setup_logging(args.output_dir)
    logger.info(f"args: {json.dumps(vars(args), indent=2, ensure_ascii=False)}")
    logger.info(f"device: {device}")

    train_loader, val_loader, num_classes = data.build_dataset(args)
    if args.num_classes is not None:
        num_classes = args.num_classes

    model = build_swin(args.model, num_classes=num_classes, window_size=args.window_size,
                       patch_merging=not args.no_patch_merging, drop_path_rate=args.drop_path,
                       drop_rate=args.drop_rate)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model={args.model} params={n_params/1e6:.2f}M window={args.window_size} "
                f"patch_merging={not args.no_patch_merging}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.eval_only and args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        acc = evaluate(model, val_loader, device)
        logger.info(f"eval-only: acc={acc:.4f}")
        return

    start_epoch, best_acc = 0, 0.0
    scaler = torch.amp.GradScaler("cuda") if args.amp and device == "cuda" else None
    opt, sched = build_optimizer_scheduler(model, args, train_loader)

    if args.resume:  # 完整恢复训练状态
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        logger.info(f"resume from {args.resume}: epoch={start_epoch} best_acc={best_acc:.4f}")

    writer = SummaryWriter(os.path.join(args.output_dir, "runs"))
    for epoch in range(start_epoch, args.epochs):
        dt = train_one_epoch(model, train_loader, opt, sched, criterion, epoch, args,
                             device, scaler, logger, writer)
        acc = evaluate(model, val_loader, device)
        is_best = acc > best_acc
        best_acc = max(best_acc, acc)
        logger.info(f"epoch {epoch}  val_acc={acc:.4f}  best={best_acc:.4f}  ({dt:.1f}s)")
        writer.add_scalar("val/acc", acc, epoch)
        writer.add_scalar("val/best_acc", best_acc, epoch)
        save_checkpoint(os.path.join(args.output_dir, "last.pt"),
                        model, opt, sched, scaler, epoch, best_acc, args)
        if is_best:
            save_checkpoint(os.path.join(args.output_dir, "best.pt"),
                            model, opt, sched, scaler, epoch, best_acc, args)
            logger.info(f"  -> 保存 best checkpoint (acc={best_acc:.4f})")
    writer.close()
    logger.info(f"训练完成。最佳 val_acc = {best_acc:.4f}，checkpoint 在 {args.output_dir}")
    logger.info(f"TensorBoard: tensorboard --logdir {os.path.join(args.output_dir, 'runs')}")


if __name__ == "__main__":
    main()
