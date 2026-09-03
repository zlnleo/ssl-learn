# -*- coding: utf-8 -*-
"""train_v6_ddp.py —— DDP 完整版 = reviewlearn.py 的全部功能 + 分布式训练

【使用指南】
    本版 = 你的 reviewlearn.py（argparse + AMP + 调度器 + checkpoint/续跑 +
          runs 日志 + TensorBoard + 早停） + DDP 八处改动。
    学习方式：先看本文件里所有【DDP】标记（共 8 处），其余部分就是
    reviewlearn.py，你已经会了。

【三种运行方式】
    1. 单进程（和 reviewlearn.py 行为完全一样）：
       python train_v6_ddp.py --dataset fashionmnist --epochs 1 --num-workers 0

    2. 本机多进程练手（Windows 缺 libuv，torchrun 不可用，用 file:// 集合点）：
       先开终端 A（rank0）：
         $env:RANK=0; $env:LOCAL_RANK=0; $env:WORLD_SIZE=2
         $env:DDP_INIT_METHOD="file://_ddp_store?rank=0&world_size=2"
         python train_v6_ddp.py --dataset fashionmnist --epochs 1 --cpu --num-workers 0
       再开终端 B（rank1）：
         $env:RANK=1; $env:LOCAL_RANK=1; $env:WORLD_SIZE=2
         $env:DDP_INIT_METHOD="file://_ddp_store?rank=1&world_size=2"
         python train_v6_ddp.py --dataset fashionmnist --epochs 1 --cpu --num-workers 0

    3. Linux 多卡（实验室服务器上）：
       torchrun --nproc_per_node=4 train_v6_ddp.py --batch-size 32

【DDP 的 8 处改动（对照 12 文档的六处模板 + 两个补充）】
    【DDP 1】is_dist() + init_process_group（gloo/nccl，支持 file:// 集合点）
    【DDP 2】只有 rank0 打印日志 / 写 runs / 写 TensorBoard / 评估 / 保存
    【DDP 3】训练集用 DistributedSampler（每个进程分到不重叠的 1/n 数据）
    【DDP 4】模型包 DDP（反向时自动 all-reduce 梯度，各进程参数保持一致）
    【DDP 5】每个 epoch 调 sampler.set_epoch（让数据划分随 epoch 变化）
    【DDP 6】保存时用 model.module 拿到原始模型（否则 key 带 "module." 前缀）
    【DDP 7】每进程种子 = 基础种子 + rank（各进程增强不同但整体可复现）
    【DDP 8】早停用 broadcast_object_list 广播决定——所有进程必须同时退出，
             否则其余进程会在下一轮 all-reduce 时永远等待（死锁）

【三个必须记住的换算】
    - 总 batch = 每卡 batch × 卡数；lr 随总 batch 线性缩放；
    - optimizer/scheduler 在【所有进程】上执行（参数必须同步更新），
      只有 evaluate/log/save 是 rank0 专属；
    - 单卡机器上 DDP 没有收益，本文件的双进程模式只用于学习机制。
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
from torch.utils.tensorboard import SummaryWriter

try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT

DATA_DIR = "../../data"


# 【DDP 1a】torchrun / 手动启动都会给进程注入 LOCAL_RANK 环境变量，
# 有它就是多进程模式，没有就是普通单进程。
def is_dist() -> bool:
    return "LOCAL_RANK" in os.environ


def build_cifar100_loader(args):
    """CIFAR-100 loader（同 reviewlearn.py，多进程时训练集换 DistributedSampler）。"""
    from torchvision import datasets, transforms
    CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR100_STD = (0.2673, 0.2564, 0.2762)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = datasets.CIFAR100(root=args.data_dir, train=True,
                                 download=True, transform=train_transforms)
    test_ds = datasets.CIFAR100(root=args.data_dir, train=False,
                                download=True, transform=test_transforms)

    # 【DDP 3】多进程时：DistributedSampler 按 rank 均分数据，互不重复；
    # 单进程时：维持 reviewlearn.py 的普通 shuffle 写法
    if is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=True)   # 有 sampler 不能再传 shuffle
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader, train_sampler, 3, 32, 100


def build_fashionmnist_loader(args):
    """FashionMNIST loader（同 reviewlearn.py，多进程时训练集换 DistributedSampler）。"""
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
                                  sampler=train_sampler, num_workers=args.num_workers)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, train_sampler, 1, 28, 10


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args):
    """单轮训练（同 reviewlearn.py 的 AMP 四步；model 可能是 DDP 包装后的）。

    注意：这一整个函数在【所有进程】上都执行——optimizer.step() 各进程
    都跑，因为反向时 DDP 已经把梯度 all-reduce 成一致的了。
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            scores = model(images)
            loss = criterion(scores, labels)
        scaler.scale(loss).backward()   # DDP 在这一步自动做梯度 all-reduce
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
    """验证集评估（同 reviewlearn.py；只有 rank0 会调用它）。"""
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
    """命令行参数（同 reviewlearn.py 全集 + 新增 --cpu）。"""
    parser = argparse.ArgumentParser(description="DDP 完整版：reviewlearn 全部功能 + 分布式")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮次")
    parser.add_argument('--dataset', type=str, default="cifar100",
                        choices=["cifar100", "fashionmnist"], help="数据集")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据目录")
    parser.add_argument('--batch-size', type=int, default=128, help="每张卡的批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW 权重衰退")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--embed-size", type=int, default=192, help="token 维度")
    parser.add_argument("--num-heads", type=int, default=6, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=6, help="encoder 层数")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="dropout")
    parser.add_argument("--amp", action="store_true", default=True, help="混合精度（GPU 上默认开启）")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoint", help="checkpoint 输出目录")
    parser.add_argument("--resume", action="store_true", help="断点续跑：从 ckpt-dir/last.pt 恢复")
    parser.add_argument("--log-dir", type=str, default="./runs", help="实验记录目录")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值（0=禁用）")
    # 【DDP】--cpu：强制 CPU + gloo，本机单卡模拟多进程练手用
    parser.add_argument("--cpu", action="store_true", help="强制 CPU + gloo（本机练手 DDP）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ===================== DDP 初始化（【DDP 1】） =====================
    if is_dist():
        if args.cpu:
            backend = "gloo"                                   # CPU 通信用 gloo
            device = torch.device("cpu")
        else:
            backend = "nccl"                                   # GPU 通信用 nccl
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)                  # 每个进程占一块卡
            device = torch.device("cuda", local_rank)
        # init_method 默认 env://（torchrun 标准，靠 TCP 集合）。
        # 本机 Windows 版 torch 缺 libuv，改用 file:// 共享文件集合点
        # （见文件头使用指南第 2 种启动方式）；Linux 上用 torchrun 不需要管它。
        init_method = os.environ.get("DDP_INIT_METHOD", "env://")
        dist.init_process_group(backend=backend, init_method=init_method)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = dist.get_rank() if is_dist() else 0                 # 本进程编号
    world_size = dist.get_world_size() if is_dist() else 1     # 进程总数
    is_master = (rank == 0)                                    # 全局事务只让 rank0 做

    # 【DDP 7】每进程种子 = 基础种子 + rank：各进程的数据增强不同，但整体可复现
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)

    # ===================== 实验记录（【DDP 2】只有 rank0 写） =====================
    run_dir = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    writer = None
    log_file = None
    if is_master:
        os.makedirs(run_dir, exist_ok=True)
        log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
        with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
            for key, value in vars(args).items():
                f.write("{} = {}\n".format(key, value))
        writer = SummaryWriter(os.path.join(run_dir, "tfboard"))   # TensorBoard 也是 rank0 专属

    def log(msg):
        if is_master:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

    log(f"device: {device}, rank: {rank}/{world_size}, "
        f"总 batch = {args.batch_size} x {world_size} = {args.batch_size * world_size}")
    log(f"run_dir: {run_dir}")
    log(f"tensorboard: 训练时另开终端执行 `tensorboard --logdir runs` 查看曲线")

    # ===================== 数据 =====================
    if args.dataset == "cifar100":
        train_loader, test_loader, train_sampler, in_channels, img_size, num_classes = build_cifar100_loader(args)
    else:
        train_loader, test_loader, train_sampler, in_channels, img_size, num_classes = build_fashionmnist_loader(args)
    log(f"dataset: {args.dataset}, classes: {num_classes}, "
        f"本 rank 训练 batches: {len(train_loader)}（每个 rank 各分 1/{world_size} 数据）")

    # ===================== 模型 / 损失 / 优化器 / 调度器 =====================
    model = ViT(img_size=img_size, patch_size=4, in_channels=in_channels,
                num_classes=num_classes, embed_size=args.embed_size,
                num_heads=args.num_heads, num_layers=args.num_layers,
                dropout=args.dropout).to(device)

    # 【DDP 4】DDP 包装：反向时自动 all-reduce 梯度 -> 各进程参数永远一致。
    # 只有多进程时才包；包装后 model 的 state_dict key 会带 "module." 前缀，
    # 保存时要用 model.module（见【DDP 6】）。
    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    log(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 注意：optimizer 和 scheduler 在【所有进程】上都执行 step——
    # 只有所有进程用相同的 lr 更新，参数才能保持一致。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = dict(img_size=img_size, patch_size=4, in_channels=in_channels,
                  num_classes=num_classes, embed_size=args.embed_size,
                  num_heads=args.num_heads, num_layers=args.num_layers,
                  dropout=args.dropout)

    # ===================== 断点续跑（每个 rank 读同一份文件） =====================
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

    # ===================== 主训练循环 =====================
    bad_epoch = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        # 【DDP 5】每个 epoch 前调 set_epoch：让每个进程每轮的数据划分随机化
        if is_dist():
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        scheduler.step()   # 所有进程都执行（lr 必须一致！）

        # ---------- 以下整块只有 rank0 执行（评估/画图/保存/早停） ----------
        stop = False
        if is_master:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/acc", test_acc, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

            # 早停：创新高清零，否则计数 +1（同 reviewlearn.py）
            if test_acc > best_acc:
                best_acc = test_acc
                bad_epoch = 0
                torch.save({
                    "model_state": model.module.state_dict() if is_dist() else model.state_dict(),
                    # 【DDP 6】DDP 包装过的模型要 .module 才拿得到原始模型的参数
                    "config": config,
                    "best_acc": best_acc,
                    "epoch": epoch,
                }, best_path)
            else:
                bad_epoch += 1
                if args.patience > 0 and bad_epoch >= args.patience:
                    stop = True

            # 保存断点（四件套，同 reviewlearn.py）
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

        # 【DDP 8】早停决定要广播给所有进程：只有大家一起退出循环，
        # 才不会出现"rank0 停了、其他进程还在等 all-reduce"的死锁。
        if is_dist():
            stop_list = [stop]
            dist.broadcast_object_list(stop_list, src=0)
            stop = stop_list[0]
        if stop:
            if is_master:
                log(f"[early stop] 验证集连续 {bad_epoch} 轮未提升，提前停止于 epoch {epoch}")
            break

    # ===================== 收尾（只有 rank0） =====================
    if is_master:
        writer.add_hparams(vars(args), {"best_acc": best_acc})
        writer.close()
        log(f"training finished in {time.time() - start:.2f}s, best test acc: {best_acc:.4f}")
        log(f"checkpoint: {args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
        log(f"查看曲线: tensorboard --logdir {args.log_dir}")
        log_file.close()
    if is_dist():
        dist.destroy_process_group()   # 结束通信群组，进程才能正常退出


if __name__ == '__main__':
    main()
