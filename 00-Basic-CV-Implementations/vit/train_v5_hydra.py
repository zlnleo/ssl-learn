# -*- coding: utf-8 -*-
"""train_v5_hydra.py —— 版本 5：初始版 + yaml/hydra 配置管理

【使用指南】
    - 本版做什么：把初始版 train.py 的 argparse 配置整体换成 hydra + yaml，
      训练逻辑（数据/模型/AMP/续跑）与初始版完全一致。
    - 依赖：pip install hydra-core omegaconf（本机已装好）
    - 运行：
        python train_v5_hydra.py                         # 用 configs/train_cifar100.yaml
        python train_v5_hydra.py dataset=toy epochs=5    # 命令行覆盖任意配置项
        python train_v5_hydra.py -m lr=1e-3,3e-3 dataset=toy   # 网格搜索（multirun）
    - 和初始版的差异（核心只有 3 处，都有【本版修改】标记）：
        ① 入口从 parse_args() 变成 @hydra.main 装饰器，参数从 cfg 取；
        ② 参数定义从 argparse 搬进了 configs/*.yaml 文件；
        ③ hydra 会自动切换工作目录，所以 data/checkpoint/runs 路径要
           用 get_original_cwd() 解析回项目目录（否则产物会散落到 outputs/ 里）。
    - 本版移除了 wandb 钩子。

【核心概念（详见 11 文档）】
    hydra 三件套：配置文件(yaml) + 命令行覆盖(dataset=toy) + 网格搜索(-m)。
    每次运行 hydra 自动建 outputs/日期/时间/ 目录存配置快照——实验留档自动化。
"""
import os
import time

import hydra
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# 【本版修改】hydra 运行时会把工作目录切到 outputs/.../ 下，
# 相对路径（../../data、./checkpoint）会因此指错位置。
# get_original_cwd() 返回启动命令时所在的目录，用它把路径解析回项目根。
def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(get_original_cwd(), path))


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


def build_cifar100_loaders(cfg):
    """CIFAR-100 loader 工厂（同初始版，参数改从 cfg 取）。"""
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
    train_ds = datasets.CIFAR100(root=resolve(cfg.data_dir), train=True,
                                 download=True, transform=train_transform)
    test_ds = datasets.CIFAR100(root=resolve(cfg.data_dir), train=False,
                                download=True, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader, 3, 32, 100


def build_toy_loaders(cfg):
    """toy loader 工厂（同初始版，参数改从 cfg 取）。"""
    train_ds = ToyVisionDataset(num_samples=128, img_size=32, seed=0)
    val_ds = ToyVisionDataset(num_samples=64, img_size=32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    return train_loader, val_loader, 3, 32, 4


def build_fashionmnist_loaders(cfg):
    """FashionMNIST loader 工厂（同初始版，参数改从 cfg 取）。"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=resolve(cfg.data_dir), train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=resolve(cfg.data_dir), train=False,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    return train_loader, val_loader, 1, 28, 10


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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


# 【本版修改 ①】入口：@hydra.main 替代 argparse。
# config_path 指向 yaml 所在目录（相对本文件），config_name 是默认配置文件名。
# 命令行 `python train_v5_hydra.py dataset=toy` 会覆盖 yaml 里的对应字段。
@hydra.main(version_base=None, config_path="configs", config_name="train_cifar100")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 实验记录：和初始版一样写 runs/run_时间戳/（路径解析回项目目录）----
    log_dir = resolve(cfg.log_dir)
    run_dir = os.path.join(log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write(OmegaConf_to_str(cfg))   # hydra 的配置快照也存一份

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device: {device}")
    log(f"run dir: {run_dir}")

    # ---- 数据 ----
    if cfg.dataset == "cifar100":
        train_loader, test_loader, in_channels, img_size, num_classes = build_cifar100_loaders(cfg)
    elif cfg.dataset == "toy":
        train_loader, test_loader, in_channels, img_size, num_classes = build_toy_loaders(cfg)
    else:
        train_loader, test_loader, in_channels, img_size, num_classes = build_fashionmnist_loaders(cfg)
    log(f"dataset: {cfg.dataset}, classes: {num_classes}, "
        f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # ---- 模型 / 损失 / 优化器 / 调度器（同初始版，参数改从 cfg 取）----
    model = ViT(
        img_size=img_size, patch_size=4, in_channels=in_channels,
        num_classes=num_classes, embed_size=cfg.embed_size,
        num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = dict(img_size=img_size, patch_size=4, in_channels=in_channels,
                  num_classes=num_classes, embed_size=cfg.embed_size,
                  num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                  dropout=cfg.dropout)

    # ---- 断点续跑（同初始版）----
    ckpt_dir = resolve(cfg.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    start_epoch, best_acc = 1, 0.0
    if cfg.resume:
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
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg)
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

        log(f"epoch {epoch:>3}/{cfg.epochs}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} "
            f"(best: {best_acc:.4f})")
    log(f"training finished in {time.time() - start:.1f}s, "
        f"best test acc: {best_acc:.4f}")
    log(f"checkpoint: {ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log_file.close()


def OmegaConf_to_str(cfg) -> str:
    """把配置序列化成文本（写进 runs 的 config.txt，和初始版格式一致）。"""
    from omegaconf import OmegaConf
    return OmegaConf.to_yaml(cfg)


if __name__ == "__main__":
    main()
