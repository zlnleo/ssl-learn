# -*- coding: utf-8 -*-
"""ViT 训练脚本 —— CIFAR-100 版（100 类 32x32 彩色小图分类）。

【这个脚本讲了什么】一个完整的训练脚本按"五段式"组织：
    1. 配置层：argparse 超参数 + 数据集常量（改配置不改代码）；
    2. 数据层：torchvision 数据集 -> transforms 增强 -> DataLoader；
    3. 模型层：你的 ViT（自动优先导入 vit.py，缺省退回参考答案）；
    4. 训练层：train_one_epoch（前向/反向/裁剪/AMP）+ evaluate（验证）；
    5. 主流程：main() 里把上面四层装配起来，循环训练 + 余弦调度 + 保存最优。

【与原论文训练配方的对照】本脚本实现的是 ViT 论文 + 常用工业配方：
    - AdamW + weight decay（ViT 对 weight decay 敏感）；
    - 余弦学习率衰减（比固定 lr 收敛更好）；
    - 梯度裁剪 1.0（防早期大梯度爆炸）；
    - trunc_normal 初始化（在 vit.py 里已实现）；
    - AMP 混合精度（fp16 加速，显存减半，精度基本无损）——阶段一"必做 4 项"之一。

【用法】
    python train.py                          # 默认 CIFAR-100，训练 100 轮
    python train.py --epochs 5               # 快速试跑几轮看流程
    python train.py --dataset toy            # 合成"象限亮块"任务（离线自检）
    python train.py --dataset fashionmnist   # 10 类灰度图（需联网下载）
    python train.py --resume                 # 断点续跑：从 checkpoint/last.pt 恢复

【产物】
    checkpoint/best.pt   只保留测试准确率最高的模型（权重 + config，部署用）
    checkpoint/last.pt   每个 epoch 更新：模型+优化器+调度器+scaler 完整状态（续跑用）
    runs/run_时间戳/     本次运行记录：config.txt（全部超参数）+ train.log（逐轮指标）
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 优先导入你自己写的 vit.py；如果还没写，退回参考答案 vit_solution.py，
# 保证脚本随时能跑通（两者的 ViT 接口完全一致，见 01_引导文档的规格表）。
try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT

# ---------------------------------------------------------------------------
# 1. 配置层：路径与数据集常量
# ---------------------------------------------------------------------------
# CIFAR-100 缓存在 D:\\project\\self_supervised_learning\\data。
# vit/ 位于 ...\\self_supervised_learning\\手搓复现学过相关的内容\\vit 下，
# 所以要向上两级（../../data）才到数据根目录；其中已包含解压好的
# cifar-100-python/（train/test/meta），download=True 时会自动跳过下载（离线可用）。
DATA_DIR = "../../data"

# CIFAR-100 训练集统计量：归一化用 (x - mean) / std，
# 把像素值从 [0,1] 拉到以 0 为中心、方差约 1 的分布，训练更稳。
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# ---------------------------------------------------------------------------
# 2a. 合成玩具数据集："亮块在哪个象限" 4 分类任务
#     （保留给 test_vit.py 的消融实验用，也作为离线快速自检）
# ---------------------------------------------------------------------------
class ToyVisionDataset(torch.utils.data.Dataset):
    """生成 4 类合成图片：背景是微弱噪声，四个象限之一放一个 8x8 亮块。

    这个任务必须"知道亮块在哪"才能分类，正好逼模型去用位置编码和注意力。
    Args:
        num_samples: 样本数（每类 num_samples//4 张）
        img_size:    图片尺寸（正方形）
        noise:       背景噪声幅度
        seed:        控制每张图噪声的随机种子（换种子 = 换一批新图）
    """

    def __init__(self, num_samples=128, img_size=32, noise=0.1, seed=0):
        g = torch.Generator().manual_seed(seed)  # 固定生成器保证可复现
        self.images, self.labels = [], []
        block = 8  # 亮块边长
        offset = img_size // 2 - block  # 亮块在象限内的偏移
        for i in range(num_samples):
            label = i % 4  # 轮流生成 4 类，保证类别均衡
            img = torch.rand(3, img_size, img_size, generator=g) * noise
            r = (label // 2) * offset  # 行：0 -> 上，1 -> 下
            c = (label % 2) * offset  # 列：0 -> 左，1 -> 右
            img[:, r:r + block, c:c + block] = 1.0  # 放亮块
            self.images.append(img)
            self.labels.append(label)
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# ---------------------------------------------------------------------------
# 2b. 三种数据集的 DataLoader 工厂
# ---------------------------------------------------------------------------
def build_cifar100_loaders(args):
    """CIFAR-100：50000 训练 / 10000 测试，100 类 32x32 RGB。

    训练集增强（ViT 数据饥渴，增强收益大）：
    - RandomCrop(32, padding=4)：四周 pad 4 像素再随机裁回 32x32，
      制造"平移"变化，防过拟合；
    - RandomHorizontalFlip：水平翻转，只对"左右对称"类别安全（物体类一般没问题）。
    测试集只归一化，不增强——评估要用"干净的分布"。
    """
    from torchvision import datasets, transforms

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # (H, W, C) uint8 -> (C, H, W) float32 [0,1]
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,  # 训练集必须打乱，防止模型记住样本顺序
        num_workers=args.num_workers,  # 多进程读图；Windows 上依赖 __main__ 守卫
        pin_memory=True,  # 锁页内存，GPU 传输更快
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,  # 评估顺序无所谓，不打乱
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # CIFAR-100：32x32 彩色图、100 类、patch 用 4（28 灰度图那个配置不适用）
    return train_loader, test_loader, 3, 32, 100


def build_toy_loaders(args):
    """合成数据：128 张训练 / 64 张验证（不同随机种子 => 互不相同的图）。"""
    train_ds = ToyVisionDataset(num_samples=128, img_size=32, seed=0)
    val_ds = ToyVisionDataset(num_samples=64, img_size=32, seed=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, 3, 32, 4  # (in_channels, img_size, num_classes)


def build_fashionmnist_loaders(args):
    """FashionMNIST：28x28 灰度图 10 分类，需要联网下载（约 30MB）。"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 单通道归一化
    ])
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    # 28 = 4 * 7，patch_size 用 4 可以整除；灰度图 in_channels=1
    return train_loader, val_loader, 1, 28, 10


# ---------------------------------------------------------------------------
# 3. 训练层：单轮训练 + 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    """训练一轮，返回平均 loss 和准确率。

    混合精度（AMP）要点：
    - autocast 里只包前向：matmul/conv 自动用 fp16 加速，显存减半；
    - softmax / LayerNorm / 损失这类"对精度敏感"的算子 torch 会自动保持 fp32，
      不用手动指定；
    - loss.backward() 要换成 scaler.scale(loss).backward()：梯度乘回放大系数，
      防止 fp16 下小梯度下溢成 0；
    - 梯度裁剪前必须先 scaler.unscale_(optimizer)：把梯度还原成真实尺度再裁；
    - optimizer.step() 换成 scaler.step()：如果这轮出现 inf/nan，scaler 会
      自动跳过本次更新并把放大系数减半（这是 AMP 的自我调节机制）。
    """
    model.train()  # 训练模式：打开 dropout（BN 层则会用 batch 统计量）
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 前向（autocast 上下文里跑 fp16）
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)

        # 标准三步，但每一步都被 AMP 包装
        optimizer.zero_grad(set_to_none=True)  # set_to_none：置 None 比填 0 快且省显存
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # 先还原梯度再裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在测试集上评估：返回 (平均 loss, 准确率)。

    和训练循环的三个关键区别：
    1. @torch.no_grad()：装饰器写法，等价于把整个函数体包进
       `with torch.no_grad():`。作用是不构建计算图、不保存梯度——
       验证阶段只需要结果、不需要反向传播，省显存还更快。
       忘了它会怎样：显存占用变大、跑得慢（结果通常不变，但浪费资源）。
    2. model.eval()：把模型切到"推理模式"。当前模型里它只影响 dropout
       （关闭随机丢弃，保证每次验证结果一致）；以后加了 BatchNorm 还会
       切换成"用训练期累计的全局统计量"而非"当前 batch 统计量"。
       忘了 eval() 的典型症状：验证准确率每次跑都不一样 / 明显偏低。
    3. 不调用 optimizer.zero_grad() 和 optimizer.step()——验证阶段
       绝不动参数。
    """
    model.eval()  # 推理模式：关闭 dropout
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        # loss.item()：把 0 维张量转成 Python 的 float，
        # 否则 total_loss 里会一直挂着计算图，越攒越占显存
        total_loss += loss.item()
        # argmax(-1)：每个样本取 logits 最大的类别下标；
        # 和真实标签比较得到 bool 张量，sum() 数出本批预测对的个数
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()  # 本批样本总数（分母）
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def parse_args():
    """用 argparse 定义命令行参数——"改配置不改代码"。

    add_argument 的四个要素：参数名 / 类型 / 默认值 / 帮助文字。
    - type=int/float/str：自动把命令行传进来的字符串转成对应类型；
    - default=...：不传这个参数时用的值；
    - choices=[...]：白名单校验，传了列表外的值会直接报错（拼写保护）；
    - action="store_true"：开关型参数——命令行里出现这个 flag 就是 True，
      不出现就是 False。所以 --amp 配 default=True 表示"默认开启"，
      --use-wandb 没有 default 表示"默认关闭"。

    用法：python train.py --epochs 50 --lr 3e-4 --batch-size 256
    返回值 args 是个命名空间对象，用 args.epochs、args.lr 取值。
    """
    parser = argparse.ArgumentParser(description="训练手写 ViT（CIFAR-100）")
    # 数据集
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "toy", "fashionmnist"], help="数据集")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据集缓存目录")
    # 训练超参
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW 权重衰减（ViT 配方）")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # 模型超参（CIFAR-100 用 192 维 / 6 层；toy 可改小跑得快）
    parser.add_argument("--embed-size", type=int, default=192, help="token 维度")
    parser.add_argument("--num-heads", type=int, default=6, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=6, help="层数")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="dropout")
    # 工程
    parser.add_argument("--amp", action="store_true", default=True,
                        help="混合精度（GPU 上默认开启）")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoint",
                        help="checkpoint 输出目录（best.pt=最优模型, last.pt=续跑存档）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：从 ckpt-dir/last.pt 恢复完整训练状态")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="实验记录目录（每次运行生成 run_时间戳/config.txt+train.log）")
    parser.add_argument("--use-wandb", action="store_true", help="用 wandb 记录实验")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 0. 固定随机种子：让实验可复现（科研底线）----
    torch.manual_seed(args.seed)  # 定住 CPU 端的随机数（dropout、rand 等）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  # 定住所有 GPU 端的随机数

    # 设备选择：有 GPU 用 GPU，否则 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 0.5 实验记录：每次运行在 runs/ 下建一个时间戳目录 ----
    # 里面存 config.txt（本次全部超参数）和 train.log（逐轮指标），
    # 相当于轻量版 wandb：训练记录不丢、随时回查。以后要接 tensorboard
    # 或 wandb，这个目录结构可以平滑迁移过去。
    run_dir = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in vars(args).items():  # vars(args)：把参数命名空间转成 dict
            f.write(f"{key} = {value}\n")

    def log(msg):
        """屏幕打印 + 写入本次 run 的日志文件（flush 保证崩溃/断电也不丢行）。"""
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device: {device}")
    log(f"run dir: {run_dir}")

    # ---- 1. 数据：根据 --dataset 选一个 loader 工厂 ----
    # 三个工厂的返回值统一为 (训练loader, 测试loader, 通道数, 图片尺寸, 类别数)，
    # 后面构建模型时直接用，不用关心具体是哪个数据集。
    if args.dataset == "cifar100":
        train_loader, test_loader, in_channels, img_size, num_classes = build_cifar100_loaders(args)
    elif args.dataset == "toy":
        train_loader, test_loader, in_channels, img_size, num_classes = build_toy_loaders(args)
    else:
        train_loader, test_loader, in_channels, img_size, num_classes = build_fashionmnist_loaders(args)
    log(f"dataset: {args.dataset}, classes: {num_classes}, "
        f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # ---- 2. 模型 / 损失 / 优化器 / 调度器 ----
    # .to(device)：把模型所有参数和 buffer 一次性搬到 GPU
    model = ViT(
        img_size=img_size, patch_size=4, in_channels=in_channels,
        num_classes=num_classes, embed_size=args.embed_size,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    # 参数量统计：p.numel() 是每个参数张量的元素个数，sum 完除以 1e6 就是百万单位(M)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()  # 分类标配损失：内部先 log_softmax 再算负对数似然
    # AdamW：weight decay（权重衰减）与梯度更新解耦，ViT 原论文配方。
    # 直观理解：每次更新参数时，额外把参数朝 0 的方向"缩"一点点，防止过拟合。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # 学习率调度器：每个 epoch 结束时调用一次 scheduler.step()，
    # 学习率会按余弦曲线从峰值平滑衰减到 0（前期大步学、后期小步精调）。
    # T_max=args.epochs 表示一个完整余弦周期正好覆盖全部训练轮数。
    # 进阶配方是 warmup+cosine：前几个 epoch 先线性升到峰值再开始衰减。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP 的"放大系数管理器"（GradScaler）。
    # 为什么需要它：fp16 能表示的极小值约 6e-5，比这还小的梯度会"下溢"
    # 成 0，参数就学不动了。scaler 的做法是把 loss 乘以一个大系数再反向
    # （梯度随之变大，落回 fp16 可表示范围），更新前用 unscale_ 还原。
    # 若某轮出现 inf/nan（fp16 溢出），scaler.step() 会自动跳过本次更新
    # 并把系数减半——这是 AMP 的自我调节机制。
    use_amp = args.amp and device.type == "cuda"  # CPU 上 AMP 无意义，自动关闭
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # config 提前建好一份，保存 checkpoint 时复用（best/last 两个文件里内容一致）
    config = dict(img_size=img_size, patch_size=4, in_channels=in_channels,
                  num_classes=num_classes, embed_size=args.embed_size,
                  num_heads=args.num_heads, num_layers=args.num_layers,
                  dropout=args.dropout)

    # wandb（可选）：网页版实验记录平台，自动画 loss/acc 曲线、记超参数。
    # 阶段规划的铁律是"每个实验进 wandb"。init 只做一次，之后在循环里
    # wandb.log({"loss": ..., "acc": ...}) 就能实时看到曲线。
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="vit-cifar100", config=vars(args))
        except ImportError:
            log("[提示] 未安装 wandb（pip install wandb），本次不记录实验")

    # ---- 2.5 断点续跑 ----
    os.makedirs(args.ckpt_dir, exist_ok=True)            # 目录不存在就创建
    best_path = os.path.join(args.ckpt_dir, "best.pt")   # 最优模型（部署用）
    last_path = os.path.join(args.ckpt_dir, "last.pt")   # 最新完整状态（续跑用）
    start_epoch, best_acc = 1, 0.0
    if args.resume:
        if os.path.exists(last_path):
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            # 恢复"四件套"，缺一不可：
            # 只恢复模型 = 等于重新训练；不恢复 optimizer = 动量/二阶矩清零；
            # 不恢复 scheduler = lr 曲线从头走；不恢复 scaler = AMP 系数重置。
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

    # ---- 3. 训练主循环：一个 epoch = 训练一轮 + 验证一轮 ----
    start = time.time()  # 计时起点（秒级时间戳）
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()  # 每个 epoch 结束调一次，更新学习率

        # 只保存"目前最好"的模型（工业惯例：硬盘上只留最优，其余丢弃）
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                # state_dict()：一个字典，记录所有参数名 -> 参数张量。
                # 它不含网络结构，所以 config 也要一起存，加载时才能重建模型。
                "model_state": model.state_dict(),
                "config": config,
                "best_acc": best_acc,  # 方便以后知道这个模型多强
                "epoch": epoch,
            }, best_path)

        # 每个 epoch 更新一次"最新完整状态"——断点续跑的存档。
        # 它比 best.pt 大：额外存了 optimizer/scheduler/scaler 三个状态，
        # 为的是恢复时训练过程（学习率、动量、AMP 系数）能和中断前无缝衔接。
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "config": config,
        }, last_path)

        # f-string 格式化：{epoch:>3} 右对齐占 3 位（1 -> "  1"），
        # {train_loss:.4f} 保留 4 位小数。打印成对齐的表格方便肉眼看曲线。
        log(f"epoch {epoch:>3}/{args.epochs}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} "
            f"(best: {best_acc:.4f})")
    log(f"training finished in {time.time() - start:.1f}s, "
        f"best test acc: {best_acc:.4f}")
    log(f"checkpoint: {args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log_file.close()  # 记得关日志文件，否则内容可能滞留在缓冲区


if __name__ == "__main__":
    # 入口守卫：只有"直接运行本文件"时才会执行 main()；
    # 被别的文件 import 时（比如 test_vit.py import 了本文件的
    # ToyVisionDataset）不会触发训练。
    # Windows 上尤其重要：num_workers>0 时 DataLoader 会开子进程，
    # 子进程会重新 import 本文件，没有这个守卫就会递归开进程直到死循环。
    main()
