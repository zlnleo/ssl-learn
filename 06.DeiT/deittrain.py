# -*- coding: utf-8 -*-
"""
train.py —— DeiT-Tiny (CIFAR-100) 训练入口, v1 干净版

流水线 (论文核心路径):
    Dataset -> DataLoader -> 教师(deitteacher.py) -> DeiT-Tiny(model.py) -> 蒸馏损失(loss.py)
            -> AdamW + 余弦 + warmup -> AMP -> TensorBoard -> checkpoint

v1 刻意不包含 (与 GPT 讨论结论, 但注意它们属于论文 Sec 4.2, v2 再加回):
    Mixup / CutMix / RandAugment / 重复增强 / EMA / DDP / LayerScale
v1 的目标只有一个: 「结构对、loss 降、能收敛、蒸馏比不蒸馏好」。精度是 v2 的事。

运行: python train.py --epochs 100 --distill hard

过关检查:
    1~2 epoch: 初始 loss 约 ln(100)=4.6 量级并开始下降, 无 NaN;
    100 epochs (无增强, DeiT-Tiny): 预期 60~68% (比完整增强版低, 正常)。

三个必须自己回答"为什么"的细节 (学习点, 不是工程噪音):
    1) 教师前向必须 teacher.eval() + torch.no_grad() —— 教师只出 logits, 不进反向图;
    2) 学生训练模式 forward 返回 (cls_logits, dist_logits), eval 模式返回双头平均;
    3) AMP: 前向+loss 放进 autocast, backward 用 scaler.scale(loss).backward()。
"""

import argparse
import math
import os
import random
import time

# >>>【AI 添加 2026-08-28】tqdm 进度条 (可选依赖: 没装也照常运行, 只是没有进度条)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda it, **kw: it
# <<<【AI 添加结束】

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

from deitmodel import DistilledVit          # 你自己的实现 (这个骨架不提供 model.py)
from deitteacher import TeacherCNN, train_teacher
from deitloss import Distillation_loss,soft_cross_entropy

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_args():
    parser = argparse.ArgumentParser(description='DeiT-Tiny on CIFAR-100 (v1 clean)')
    # TODO(你来写): 建议至少这几个参数
    #   --epochs (100)  --batch-size (128)  --lr (5e-4)  --warmup-epochs (5)
    #   --weight-decay (0.05)  --distill (hard|soft)  --alpha (0.5)  --tau (3.0)
    #   --smoothing (0.1)  --teacher-epochs (30)  --out-dir (./runs)  --seed (42)
    # 数据目录写死: D:\project\self_supervised_learning\data
    parser.add_argument("--epochs", type=int, default=100,help="训练轮数")
    parser.add_argument("--batch-size",type=int,default=128,help="一个批次")
    parser.add_argument("--lr",type=float,default=5e-4,help="学习率")
    parser.add_argument("--weight-decay",type=float,default=0.05,help="权重衰退")
    parser.add_argument("--distill",type=str,default="hard",choices=["hard","soft"],help="蒸馏方式")
    parser.add_argument("--alpha",type=float,default=0.5,help="比例")
    parser.add_argument("--tau",type=float,default=3.0,help="比例系数让分布更柔和")
    parser.add_argument("--smoothing",type=float,default=0.1,help="loss计算时候平滑系数")
    parser.add_argument("--ckpt-dir",type=str,default="./checkpoint",help="断点保存（best.pt最好模型,last.pt最后运行参数）")
    parser.add_argument("--seed",type=int,default=42,help="随机种子默认42，方便复现")
    parser.add_argument("--log-dir",type=str,default="./runs",help="训练保存内容")
    parser.add_argument("--patience",type=int,default=10,help="多少轮正确率不增加，早停")
    parser.add_argument("--resume",action="store_true",help="继续跑")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num-workers",type=int,default=2,help="loader并行数量")
    parser.add_argument("--data-dir",type=str,default="../data",help="训练数据来源")
    parser.add_argument("--no-amp",action="store_true",help="amp混合训练关闭")

    parser.add_argument("--teacher-dir",default="./runs/teacher",help="教师模型位置")
    parser.add_argument("--teacher-epochs",type=int,default=30,help="教师训练轮次")
    parser.add_argument("--teacher-lr",type=float,default=0.1,help="教师学习率")
    parser.add_argument("--warmup-epochs",type=int,default=5,help="warmup")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    """v1 增强: 只有 RandomCrop(32,pad4) + RandomHorizontalFlip + ToTensor + Normalize。
    学生和教师共用同一套 (v1 没有 Mixup/RandAugment, 学生教师看到的分布一致)。"""
    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_datasets = CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transforms)
    test_datasets =CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transforms)
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,pin_memory=True)
    test_loader = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,pin_memory=True)
    return train_loader, test_loader,32,3,100


@torch.no_grad()
def evaluate(student, loader, criterion,device,args):
    """测试集 top-1。注意: 蒸馏模型 eval 模式下 forward 已返回双头平均, 直接 argmax 即可。"""
    student.eval()
    eval_total_loss,correct,total = 0.0,0,0
    for X,y in loader:
        X , y = X.to(device), y.to(device)
        logits = student(X)
        total_loss=soft_cross_entropy(logits,y,args.smoothing)
        eval_total_loss += total_loss.item()*y.numel()
        correct+=(logits.argmax(dim=1)==y).sum().item()
        total += y.numel()
    return eval_total_loss/total,correct/total



def train_one_epoch(student, teacher, loader, criterion, optimizer, scaler, device, args, desc='train'):  # >>>【AI 添加】desc: 进度条标题
    """一个 epoch 的训练, 返回 (平均 total loss, 平均 base, 平均 dist, train_acc)。

    步骤提示:
      1) student.train(); teacher.eval()
      2) 对每个 batch:
           x, y 搬到 device
           with torch.no_grad(): teacher_logits = teacher(x)     # 教师只推理
           with torch.amp.autocast('cuda'):                      # torch 2.11 新 API
               out = student(x)                                  # (cls_logits, dist_logits)
               loss, base, dist = criterion(out, teacher_logits, y)
           optimizer.zero_grad()
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
      3) 统计各项均值返回
    """
    student.train(),teacher.eval()
    one_epoch_loss,correct,total = 0.0,0,0
    # >>>【AI 添加 2026-08-28】tqdm 进度条: leave=False 训完不占屏幕, set_postfix 实时显示 loss/acc
    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    # <<<【AI 添加结束】
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(X)

        with torch.amp.autocast('cuda',enabled=not args.no_amp):
            student_logits = student(X)
            total_loss,base_loss,dist_loss = criterion(student_logits, teacher_logits, y,args)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        one_epoch_loss += total_loss.item()*y.numel()
        correct += (student_logits[0].argmax(dim=1)==y).sum().item()
        total += y.numel()
        pbar.set_postfix(loss=f"{total_loss.item():.3f}", acc=f"{correct/total:.3f}")  # >>>【AI 添加】进度条后缀
    return one_epoch_loss/total,correct/total




def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device} | 蒸馏: {args.distill}')

    #实验记录
    run_dir = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, 'train.log'), 'w',encoding='utf-8')
    with open(os.path.join(run_dir, 'config.txt'), 'w',encoding='utf-8') as f:
        for key, value in vars(args).items():
            f.write(f'{key} = {value}\n')
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()


    # ---- 数据 ----
    train_loader, test_loader,img_size,in_channels,num_classes= build_loaders(args)


    # ---- 教师: 有 checkpoint 就加载, 没有就现训 ----
    # TODO(你来写):
    #   teacher = TeacherCNN(num_classes=100).to(device)
    #   teacher_path = os.path.join(args.out_dir, 'teacher_cnn_cifar100.pth')
    #   if os.path.exists(teacher_path): 加载 state_dict 并打印其 acc
    #   else: teacher, teacher_acc = train_teacher(...);  (deitteacher.py 已实现后)
    teacher = TeacherCNN(num_classes).to(device)
    teacher_path = os.path.join(args.teacher_dir, 'teacher_cnn_cifar100.pth')
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path))
    else:
        teacher, teacher_acc=train_teacher(args.data_dir,args.teacher_epochs,args.batch_size,args.teacher_lr,device,teacher_path)
    log(f"teacher_model parameters:{sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M")

    # ---- 学生: DeiT-Tiny ----
    student =DistilledVit(img_size,4,in_channels,num_classes,dropout=0.1,attn_drop=0.1,drop_path=0.1).to(device)
    log(f"student_model-DistlledVit parameters:{sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")

    # ---- 损失 / 优化器 / AMP ----
    #损失函数
    criterion = Distillation_loss
    #优化器
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0,
                                              total_iters=args.warmup_epochs),  # 近似从 0 爬坡
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=args.epochs - args.warmup_epochs),
        ],
        milestones=[args.warmup_epochs],
    )
    #使用amp
    use_amp = not args.no_amp and device == "cuda"
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    # ---- TensorBoard ----
    writer = SummaryWriter(os.path.join(run_dir, 'tfboard'))

    #断点续跑
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, 'best.pth')
    last_path = os.path.join(args.ckpt_dir, 'last.pth')
    start_epoch,best_acc= 0,0.0
    if args.resume:
        if os.path.isfile(last_path):
            ckpt = torch.load(last_path, map_location=device,weights_only=False)
            student.load_state_dict(ckpt['student_dict'])
            optimizer.load_state_dict(ckpt['optimizer_dict'])
            scaler.load_state_dict(ckpt['scaler_dict'])
            scheduler.load_state_dict(ckpt['scheduler_dict'])
            start_epoch = ckpt['epoch']
            best_acc = ckpt['best_acc']
            log(f"[resume] 已从 {last_path} 恢复：上次跑到 epoch {ckpt['epoch']}，"
                f"best_acc {best_acc:.4f}，本轮从 epoch {start_epoch} 继续")
        else:
            log(f"[resume] 未找到 {last_path}，从头开始训练")

    #主训练开始
    log(f"开始时间{time.strftime('%Y-%m-%d %H:%M:%S')}")
    bad_epoch=0
    start=time.time()
    for epoch in range(start_epoch+1,args.epochs+1):
        train_loss,train_acc=train_one_epoch(student, teacher, train_loader, criterion, optimizer, scaler, device, args,
                                             desc=f'train {epoch}/{args.epochs}')   # >>>【AI 添加】进度条标题带 epoch
        test_loss,test_acc =evaluate(student, test_loader, criterion, device,args)
        scheduler.step()

        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('train/acc',train_acc,epoch)
        writer.add_scalar('test/loss',test_loss,epoch)
        writer.add_scalar('test/acc',test_acc,epoch)
        #保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            bad_epoch = 0
            torch.save({
                "student_dict": student.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
                # >>>【AI 添加 2026-08-28】把本次运行的完整配置塞进 best.pth
                #     理由: best.pth 是跨实验共享的单例文件(会被不同配置覆盖),
                #     runs/ 里的 config.txt 与 checkpoint 目录是脱钩的,
                #     时间一久就无法从 best.pth 反推它是哪组参数训出来的。
                #     代价只是一行字典, 加载时 torch.load(..., weights_only=False) 即可读出。
                "config": vars(args),
                # <<<【AI 添加结束】
            },best_path)
        else:
            bad_epoch += 1
            if args.patience > 0 and bad_epoch >= args.patience:
                log(f"[early stop] 验证集连续 {bad_epoch} 轮未提升，"
                    f"提前停止于 epoch {epoch}")
                break
        #保存模型
        torch.save({
            "student_dict": student.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "scaler_dict": scaler.state_dict(),
            "scheduler_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            # >>>【AI 添加 2026-08-28】与 best.pth 同理, last.pth 也存一份 config,
            #     续跑时顺便打印出来, 防止"恢复的训练"和"当时启动的训练"配置不一致。
            "config": vars(args),
            # <<<【AI 添加结束】
        },last_path)
        log(f"epoch {epoch:>3}/{args.epochs}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} "
            f"(best: {best_acc:.4f})")
    writer.add_hparams(vars(args), {"best_acc": best_acc})
    writer.close()
    log(f"training finished in {time.time() - start:.2f}s, "f"best test acc: {best_acc:.4f}")
    log(f"checkpoint:{args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log(f"查看曲线: tensorboard --logdir {args.log_dir}")
    log_file.close()



if __name__ == '__main__':
    main()
