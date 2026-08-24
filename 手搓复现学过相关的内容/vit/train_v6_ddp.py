# -*- coding: utf-8 -*-
"""train_v6_ddp.py —— 版本 6：初始版 + DDP 多进程分布式训练

【使用指南】
    - 本版做什么：在初始版 train.py 的基础上，只加 DDP 分布式改造。
      不传 torchrun 时它和初始版行为完全一样（单进程单卡），
      用 torchrun 启动时自动变成多进程数据并行。
    - 运行：
        # 单进程（和初始版一样用）：
        python train_v6_ddp.py --dataset toy --epochs 3
        # 多进程（本机单卡可以用 CPU+gloo 模拟 2 个进程练手）。
        # 本机 Windows 版 torch 缺 libuv，torchrun 会报 use_libuv 错误，
        # 用 file:// 集合点 + 手动起两个进程（rank/world_size 写 URL 查询参数）：
        #   进程0: $env:RANK=0; $env:LOCAL_RANK=0; $env:DDP_INIT_METHOD="file://_ddp_store?rank=0&world_size=2"
        #   进程1: 同上但 RANK=1、LOCAL_RANK=1、?rank=1&world_size=2
        #   然后分别执行 python train_v6_ddp.py --dataset toy --epochs 1 --cpu --num-workers 0
        # 多卡 GPU（Linux 服务器上）：
        torchrun --nproc_per_node=4 train_v6_ddp.py --batch-size 32
    - 和初始版的差异（6 处，对应 12 文档的"六处改动模板"，都有【本版新增】标记）：
        ① init_process_group + local_rank 设备选择；
        ② 只有 rank 0 打印日志和写文件；
        ③ 训练集换 DistributedSampler（数据按进程均分）；
        ④ 模型包 DDP；
        ⑤ 每个 epoch 调 sampler.set_epoch；
        ⑥ 保存/评估只在 rank 0 做。
    - 本版移除了 wandb 钩子。

【核心概念（详见 12 文档）】
    DDP = 每张卡一份完整模型 + 一份数据切片，反向时梯度 all-reduce 平均。
    关键换算：总 batch = 每卡 batch × 卡数；lr 随总 batch 线性缩放；
    每 rank 的种子 = 基础种子 + rank。
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT

DATA_DIR = "../../data"
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# 【本版新增】判断是否处于 torchrun 启动的多进程模式：
# torchrun 会给每个进程注入 LOCAL_RANK 环境变量，没有它就是普通单进程。
def is_dist() -> bool:
    return "LOCAL_RANK" in os.environ


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


# 【本版修改】三个 loader 工厂：多进程模式下训练集改用 DistributedSampler。
# 返回多了第 6 个值 train_sampler（单进程时为 None），供 set_epoch 使用。
def build_cifar100_loaders(args):
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
    if is_dist():
        # 【本版新增 ③】DistributedSampler：按 rank 把数据集均分成不重叠的 n 份
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=True)   # 有 sampler 时不能再传 shuffle
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader, train_sampler, 3, 32, 100


def build_toy_loaders(args):
    train_ds = ToyVisionDataset(num_samples=128, img_size=32, seed=0)
    val_ds = ToyVisionDataset(num_samples=64, img_size=32, seed=1)
    if is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, train_sampler, 3, 32, 4


def build_fashionmnist_loaders(args):
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    if is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, train_sampler, 1, 28, 10


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    """单轮训练（同初始版，含 AMP 四步；model 可能是 DDP 包装后的）。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()   # DDP 会在反向时自动 all-reduce 梯度
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
    """命令行参数（同初始版，删掉 wandb，新增 --cpu 供单机模拟多进程）。"""
    parser = argparse.ArgumentParser(description="训练手写 ViT（CIFAR-100）+ DDP 版")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "toy", "fashionmnist"], help="数据集")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据集缓存目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="每张卡的批大小")
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
    # 【本版新增】--cpu：强制 CPU + gloo 后端，单机模拟多进程练手用
    parser.add_argument("--cpu", action="store_true",
                        help="强制 CPU + gloo 后端（单机模拟多进程 DDP 练手）")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoint",
                        help="checkpoint 输出目录（best.pt=最优模型, last.pt=续跑存档）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：从 ckpt-dir/last.pt 恢复完整训练状态")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="实验记录目录（每次运行生成 run_时间戳/config.txt+train.log）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 【本版新增 ①】DDP 初始化：torchrun 启动时走多进程分支，否则退回单进程
    if is_dist():
        if args.cpu:
            backend = "gloo"                                  # CPU 通信用 gloo
            device = torch.device("cpu")
        else:
            backend = "nccl"                                  # GPU 通信用 nccl
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)                 # 每个进程占一块卡
            device = torch.device("cuda", local_rank)
        # init_method 默认 env://（torchrun 的标准方式，靠 TCP 做进程集合）。
        # 注意：本机 Windows 版 torch 的 TCPStore 缺 libuv 编译支持，
        # torchrun 和 env:// 都会报 "use_libuv was requested"。
        # 解决办法：改用 file:// 共享文件做集合点（进程间不建 TCP 连接）：
        #   设置环境变量 DDP_INIT_METHOD=file://_ddp_store 再手动启动各进程；
        # Linux 服务器上用标准 torchrun（env:// + nccl），不需要这个参数。
        init_method = os.environ.get("DDP_INIT_METHOD", "env://")
        dist.init_process_group(backend=backend, init_method=init_method)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = dist.get_rank() if is_dist() else 0                # 本进程编号
    world_size = dist.get_world_size() if is_dist() else 1    # 进程总数
    is_master = (rank == 0)                                   # 只有 master 做全局事务

    # 每 rank 的种子 = 基础种子 + rank：数据增强在各进程间不同，但整体可复现
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    # ---- 实验记录：只有 master 写（【本版新增 ②】），否则 n 个进程打 n 份 ----
    run_dir = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    log_file = None
    if is_master:
        os.makedirs(run_dir, exist_ok=True)
        log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
        with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
            for key, value in vars(args).items():
                f.write(f"{key} = {value}\n")

    def log(msg):
        if is_master:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

    log(f"device: {device}, rank: {rank}/{world_size}, 总 batch = "
        f"{args.batch_size} x {world_size} = {args.batch_size * world_size}")
    log(f"run dir: {run_dir}")

    # ---- 数据 ----
    if args.dataset == "cifar100":
        train_loader, test_loader, train_sampler, in_channels, img_size, num_classes = build_cifar100_loaders(args)
    elif args.dataset == "toy":
        train_loader, test_loader, train_sampler, in_channels, img_size, num_classes = build_toy_loaders(args)
    else:
        train_loader, test_loader, train_sampler, in_channels, img_size, num_classes = build_fashionmnist_loaders(args)
    log(f"dataset: {args.dataset}, classes: {num_classes}, "
        f"本 rank 训练 batches: {len(train_loader)}")

    # ---- 模型 / 损失 / 优化器 / 调度器（同初始版）----
    model = ViT(
        img_size=img_size, patch_size=4, in_channels=in_channels,
        num_classes=num_classes, embed_size=args.embed_size,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # 【本版新增 ④】模型包 DDP：反向时自动做梯度 all-reduce（各卡参数保持一致）
    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

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

    # ---- 断点续跑（同初始版；每个 rank 各自读同一份文件，参数保持一致）----
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
        # 【本版新增 ⑤】每个 epoch 前调 set_epoch：让各 rank 每个 epoch 的数据划分不同
        if is_dist():
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args)

        # 【本版新增 ⑥】评估/保存只在 master 做（简化版；严格的分布式评测要 all_gather）
        if is_master:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    "model_state": model.module.state_dict() if is_dist() else model.state_dict(),
                    # DDP 包过的模型要 .module 才是原始模型（否则 key 带 "module." 前缀）
                    "config": config,
                    "best_acc": best_acc,
                    "epoch": epoch,
                }, best_path)

            torch.save({
                "model_state": model.module.state_dict() if is_dist() else model.state_dict(),
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
    if log_file is not None:
        log_file.close()
    if is_dist():
        dist.destroy_process_group()  # 结束通信群组，进程才能正常退出


if __name__ == "__main__":
    main()
