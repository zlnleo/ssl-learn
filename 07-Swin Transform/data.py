"""数据集构建：CIFAR-10 / CIFAR-100（默认，本地已有不下载）/ ImageNet / 自定义文件夹。

数据目录解析顺序：--data-dir 显式指定 > D:\\project\\self_supervised_learning\\data
（本地已有数据集）> ./data。已有数据时绝不触发联网下载。
"""
import os

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

DATA_ROOT_CANDIDATES = [r"D:\project\self_supervised_learning\data", "./data"]


def resolve_data_dir(data_dir):
    """优先使用本地已有数据集目录。"""
    if data_dir:
        return data_dir
    for p in DATA_ROOT_CANDIDATES:
        if os.path.isdir(p):
            return p
    return "./data"


def build_dataset(args):
    """按 args 构建 (train_loader, val_loader, num_classes)。"""
    if args.dataset in ("cifar10", "cifar100"):
        return _cifar(args)
    if args.dataset == "imagenet":
        return _imagenet(args)
    if args.dataset == "folder":
        return _folder(args)
    raise ValueError(f"未知数据集 {args.dataset}")


def _cifar(args):
    data_dir = resolve_data_dir(args.data_dir)
    name, num_classes = (("CIFAR-10", 10) if args.dataset == "cifar10" else ("CIFAR-100", 100))
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if args.dataset == "cifar10" else (CIFAR100_MEAN, CIFAR100_STD)
    print(f"[data] {name}, img_size={args.img_size}, 数据目录 {data_dir}（已有数据不下载）")
    norm = T.Normalize(mean, std)
    train_tf = T.Compose([T.Resize(args.img_size), T.RandomHorizontalFlip(),
                          T.RandomCrop(args.img_size, padding=4), T.ToTensor(), norm])
    val_tf = T.Compose([T.Resize(args.img_size), T.ToTensor(), norm])
    ds_cls = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
              else torchvision.datasets.CIFAR100)

    def make(train, transform):
        try:  # 已有数据：绝不触发下载
            return ds_cls(data_dir, train=train, download=False, transform=transform)
        except RuntimeError:  # 本地数据缺失才允许联网下载
            return ds_cls(data_dir, train=train, download=True, transform=transform)

    train_loader = DataLoader(make(True, train_tf), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(make(False, val_tf), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes


def _imagenet(args):
    print(f"[data] ImageNet, 数据目录 {args.data_dir}")
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = T.Compose([T.RandomResizedCrop(args.img_size), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    val_tf = T.Compose([T.Resize(int(args.img_size * 256 / 224)), T.CenterCrop(args.img_size),
                        T.ToTensor(), norm])
    train_ds = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "train"),
                                                transform=train_tf)
    val_ds = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "val"),
                                              transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, 1000


def _folder(args):
    print(f"[data] 自定义文件夹（仅评估）: {args.data_dir}")
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    val_tf = T.Compose([T.Resize(int(args.img_size * 256 / 224)), T.CenterCrop(args.img_size),
                        T.ToTensor(), norm])
    val_ds = torchvision.datasets.ImageFolder(args.data_dir, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return None, val_loader, len(val_ds.classes)
