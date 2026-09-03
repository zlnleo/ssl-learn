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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda it, **kw: it

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR100

from deitmodel import DistilledVit          # 你自己的实现 (这个骨架不提供 model.py)
from deitteacher import TeacherCNN, train_teacher
from deitloss import Distillation_loss,soft_cross_entropy

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

#Mixup
def Mixup(x,y,alpha=0.8):
    #具体详解在desktop/学习问题解决.md文件又详细的解释
    #以下仅做简单的解释
    #lam通过使用beta分布选出来更加适合叠加的lamda参数（论文中数学证明）
    lam = np.random.beta(alpha,alpha)
    #randperm用法是生成长度为x.size(0)的随机序列,[2,3,4,1,0,5]
    idx =torch.randperm(x.size(0),device=x.device)
    #混合得到x
    mixed_x = lam * x + (1 - lam) * x[idx]
    #把他们和原来的标号对应起来
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b,lam,idx
#CutMix
def CutMix(x,y,alpha=1.0):
    #lam通过使用beta分布选出来更加适合叠加的lamda参数（论文中数学证明）
    lam = np.random.beta(alpha,alpha)
    #randperm用法是生成长度为x.size(0)的随机序列,[2,3,4,1,0,5]
    idx = torch.randperm(x.size(0),device=x.device)
    #获取配对的目标标签。
    y_a,y_b=y,y[idx]
    #长，高，然后裁剪比例和裁剪宽度和裁剪高度
    W,H=x.size(-1),x.size(-2)
    cut_ratio=math.sqrt(1-lam)
    cut_w,cut_h=int(cut_ratio*W),int(cut_ratio*H)
    #中心点位置
    center_W,center_H = np.random.randint(W),np.random.randint(H)
    #框的左上角（x1,y1）右下角（x2，y2）
    x1 = max(center_W-cut_w//2,0)
    y1 = max(center_H-cut_h//2,0)
    x2 = min(center_W+cut_w//2,W)
    y2 = min(center_H+cut_h//2,H)
    #替换--把x[idx]对应的图像覆盖到x上面进行替换
    x[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    lam=1-(x2-x1)*(y2-y1)/(W*H)
    return x,y_a,y_b,lam,idx
#小工具
def mix_target(y_a, y_b, lam, num_classes):
    """把 Mixup/CutMix 的标签混合成软标签 (one-hot 的凸组合)。"""
    return lam * F.one_hot(y_a, num_classes).float() + (1 - lam) * F.one_hot(y_b, num_classes).float()

# =============================================================================
# RandAugment —— 论文 Sec 4.2: rand-m9-mstd0.5-inc1
# >>>整段加了逐行学习注释, 代码逻辑与原版完全一致 (只加了注释)
# =============================================================================
class RandAugmentCIFAR:
    """
    RandAugment 学习版 (输入输出都是 [C,H,W] float 张量, 像素范围 [0,1])。
    核心机制, 记住这张图就懂了整个类:
        一张图 img
            ↓
        np.random.choice: 从 13 个操作里随机抽 n 个 (不重复)
            ↓
        每个操作依次执行: img = op(img, mag)
            ↓
        返回增强后的图 (clamp 回 [0,1])

    两个核心超参数, 一定分清:
        n  —— 做"几种"增强   (数量, 论文 n=2)
        m  —— 每种增强"多狠"  (强度, 统一标尺 0~30, 论文 m=9)
    m 是 0~30 的统一标尺: 旋转用它算出角度、对比度用它算出系数、海报化用它算出
    位数——每个操作自己把 m 映射到自己的物理量。这就是"统一幅度"的含义。
    """
    def __init__(self, n=2, m=9, mstd=0.5, inc=1):
        self.n = n          # 每张图随机做几种增强
        self.m = m          # 当前整体增强强度
        self.mstd = mstd    # 强度抖动: 实际幅度取 [m-0.5, m+0.5] 内的随机数
        self.inc = inc      # 每个 epoch 后 m += inc (论文 inc=1; 第一版可先不用)
        # 操作池: 装的是"方法"本身 (可调用对象), 不是调用结果!
        # self.rotate 是可调用的, 后面统一用 op(img, mag) 调用它
        self.ops = [self.autocontrast, self.equalize, self.posterize, self.solarize,
                    self.contrast, self.color, self.brightness, self.sharpness,
                    self.rotate, self.shear_x, self.shear_y,
                    self.translate_x, self.translate_y]

    def __call__(self, img):
        """
        为什么用 __call__? —— 让"实例"可以像函数一样被调用: ra(img)。
        torchvision 的 Compose 就是通过 t(img) 调用每个 transform 的,
        所以自定义增强必须实现 __call__ 才能塞进 Compose 流水线。
        (nn.Module 的 forward 也是被 __call__ 包了一层, 同一个套路)
        """
        # np.random.choice(列表, n, replace=False): 无放回随机抽 n 个操作
        #   replace=False 保证 "Rotate + Rotate" 这种重复选择不会发生
        for op in np.random.choice(self.ops, self.n, replace=False):
            # 幅度: 不直接固定用 m, 而是在 [m-mstd, m+mstd] 里随机取一个
            #   np.random.uniform(a, b): [a,b) 均匀分布     (m=9, mstd=0.5 → 8.5~9.5)
            #   np.clip(x, 0, 30): 把越界的值夹回 0~30
            #   float(...): 转成 Python float (防止与 float32 张量运算时 dtype 升级)
            mag = float(np.clip(np.random.uniform(self.m - self.mstd, self.m + self.mstd), 0, 30))
            img = op(img, mag)          # 依次执行: 上一轮的输出是下一轮的输入
        return img.clamp_(0.0, 1.0)     # 原地夹回 [0,1], 防多步操作叠加后越界

    # ---- 无幅度操作 (mag 传进来但不使用; 形参名写作 _ 表示"这个参数我不用") ----
    @staticmethod
    def autocontrast(img, _):
        """自动对比度: 每通道线性拉伸, 最暗像素→0、最亮像素→1。教模型忽略光照强度差异。"""
        low = img.amin(dim=(1, 2), keepdim=True)   # 每通道最小值; dim=(1,2) 沿 H,W 求; keepdim 保持形状以便广播
        high = img.amax(dim=(1, 2), keepdim=True)   # 每通道最大值
        return (img - low) / (high - low + 1e-8)      # 线性拉伸; +1e-8 防除 0

    @staticmethod
    def equalize(img, _):
        """直方图均衡: 重新分配像素让直方图接近均匀, 暗部提亮/亮部压暗。教曝光不变性。
        四步: 像素转 0~255 整数 → 统计 256 桶直方图 → 累积分布函数 → 查表映射。"""
        out = []
        for c in img:  # 逐通道处理
            vals = (c * 255).round().long().clamp(0, 255)   # [0,1] → 0~255 整数
            hist = torch.bincount(vals.flatten(), minlength=256).float()  # 每级灰度的像素个数
            cdf = hist.cumsum(0)                           # 累积分布 (前缀和)
            cdf_min = cdf[cdf > 0].min()                   # 跳过为 0 的空桶
            lut = ((cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-8) * 255).round().clamp(0, 255)  # 查找表
            out.append(lut[vals] / 255.0)                  # 用查找表重映射每个像素
        return torch.stack(out)

    # ---- 有幅度操作 (每个都把 mag∈[0,30] 映射到自己的物理量) ----
    def posterize(self, img, mag):
        """色调分离: 每通道量化到 4~8 位 (位数随 mag 减少)。教模型忽略精细色彩差异。
        mag=0→8位, mag=30→4位: bits = 8 - int(mag/30*4)"""
        bits = 8 - int(mag / 30 * 4)                      # mag 越大, 保留位数越少 (8 -> 4)
        v = (img * 255).round().long()
        v = (v >> (8 - bits)) << (8 - bits)               # 位运算: 右移再左移 = 低位清零 = 量化
        return v.float() / 255.0

    def solarize(self, img, mag):
        """过曝反转: 高于阈值的像素取反 1-x (照片过曝效果)。教极端光照不变性。
        mag 越大 → 阈值越低 → 被翻转的像素越多。"""
        t = 1.0 - 0.5 * (mag / 30)                        # 阈值: mag=0→1.0(几乎不翻), mag=30→0.5
        return torch.where(img < t, img, 1.0 - img)       # where(条件, 真, 假): 按条件逐元素选择

    def contrast(self, img, mag):
        """对比度: 围绕均值拉伸差异 (x-mean)*f+mean。f>1 更锐利, f<1 发灰。教对比度不变性。"""
        f = 1.0 + (2 * random.random() - 1) * 0.9 * (mag / 30)   # 系数 [0.1, 1.9]
        #   (2*random()-1): [-1,1] 均匀 = 随机"方向"(增强或减弱);
        #   *0.9*(mag/30): mag 映射到最大变化幅度 0.9
        mean = img.mean(dim=(1, 2), keepdim=True)         # 每通道均值
        return (img - mean) * f + mean

    def color(self, img, mag):
        """饱和度: 与灰度图插值 (img-gray)*f+gray。f=0 变黑白, f>1 色彩更浓。教色彩强度不变性。"""
        f = 1.0 + (2 * random.random() - 1) * 0.9 * (mag / 30)
        gray = img.mean(dim=0, keepdim=True)              # 灰度近似 = 三通道均值 (dim=0 沿通道)
        return (img - gray) * f + gray

    def brightness(self, img, mag):
        """亮度: 全体像素加偏移, 随机变亮或变暗。教亮度不变性。
        注意思想: mag 决定"最多改多少", 随机数决定"往哪个方向改"。"""
        return img + (2 * random.random() - 1) * 0.9 * (mag / 30)

    def sharpness(self, img, mag):
        """锐度: 与模糊图插值 (img-blurred)*f+blurred。f>1 锐化, f<1 模糊。教细节/模糊不变性。"""
        f = 1.0 + (2 * random.random() - 1) * 0.9 * (mag / 30)
        blurred = F.avg_pool2d(img.unsqueeze(0), 3, 1, 1).squeeze(0)  # 3x3 均值池化当廉价模糊核
        #   unsqueeze(0): [C,H,W] → [1,C,H,W] (池化需要 batch 维); squeeze(0) 再删掉
        return (img - blurred) * f + blurred

    def rotate(self, img, mag):
        """旋转: ±30° × (mag/30)。教旋转不变性。fill=0: 转出来的空白填黑色。"""
        return TF.rotate(img, (2 * random.random() - 1) * 30 * (mag / 30), fill=0)

    def shear_x(self, img, mag):
        """水平错切: 图像横向倾斜(推扑克牌效果)。教仿射形变不变性。
        TF.affine(img, angle, translate, scale, shear, fill): shear 必须显式传 (版本坑)。"""
        return TF.affine(img, 0, [0, 0], 1.0, [(2 * random.random() - 1) * 0.3 * (mag / 30), 0], fill=0)

    def shear_y(self, img, mag):
        """垂直错切: 同上, 方向换到 Y。"""
        return TF.affine(img, 0, [0, 0], 1.0, [0, (2 * random.random() - 1) * 0.3 * (mag / 30)], fill=0)

    def translate_x(self, img, mag):
        """水平平移: 最多 ±45% 边长 (mag=30 时)。教位置不变性 —— Transformer 最缺的一项。"""
        dx = (2 * random.random() - 1) * 0.45 * (mag / 30) * img.shape[-1]   # img.shape[-1] = 宽度 W
        return TF.affine(img, 0, [dx, 0], 1.0, [0, 0], fill=0)

    def translate_y(self, img, mag):
        """垂直平移: 同上, img.shape[-2] = 高度 H。"""
        dy = (2 * random.random() - 1) * 0.45 * (mag / 30) * img.shape[-2]
        return TF.affine(img, 0, [0, dy], 1.0, [0, 0], fill=0)


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

    #v2增强数据使用
    parser.add_argument("--mixup",type=float,default=0.8,help="Mixup的alpha，0表示关闭")
    parser.add_argument("--cutmix",type=float,default=1.0,help="CutMix的alpha，0表示关闭")
    parser.add_argument("--mix-switch",type=float,default=0.5,help="启用混合批的比例")

    parser.add_argument('--ra-n', type=int, default=2)
    parser.add_argument('--ra-m', type=int, default=9)
    parser.add_argument('--ra-mstd', type=float, default=0.5)
    parser.add_argument('--ra-inc', type=int, default=1)
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
        RandAugmentCIFAR(n=args.ra_n, m=args.ra_m, mstd=args.ra_mstd, inc=args.ra_inc),
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
    #  添加 2026-08-28tqdm 进度条: leave=False 训完不占屏幕, set_postfix 实时显示 loss/acc
    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)

    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        # ---- v2 数据增强: 先混合, 教师和学生都看混合后的图 ----
        #  教师直接看混合图 T(X_mixed) —— 精确。
        #  之前的做法"干净图 T(X) + 人工混 logits"是线性近似: 教师是非线性网络,
        #  T(λx₁+(1-λ)x₂) ≠ λT(x₁)+(1-λ)T(x₂), Mixup 下近似尚可, CutMix 下误差更大。
        #  注意: 教师前向无论看干净图还是混合图都只有一次, 并不会"翻倍"。
        X_mixed, target = X, y                      # 兜底: 本批不混合时与 v1 完全一致
        if (args.mixup > 0 or args.cutmix > 0) and np.random.rand() < args.mix_switch:#先确定开启任意一种增强之后，再判断能不能增强
            #如果做增强选Mixup还是CutMix
            if args.mixup > 0 and (args.cutmix <= 0 or np.random.rand() < 0.5):
                #args.mixup > 0表示开启了Mixup如果没开直接走下面那条路线
                #args.cutmix <= 0表示的是没开，就只能走mixup
                X_mixed, y_a, y_b, lam, idx = Mixup(X, y, args.mixup)
            else:
                X_mixed, y_a, y_b, lam, idx = CutMix(X, y, args.cutmix)
            target = mix_target(y_a, y_b, lam, 100)                      # 软标签 (B,100)

        with torch.no_grad():
            teacher_logits = teacher(X_mixed)        # 教师看混合后的图, 只推理一次, 不进反向

        with torch.amp.autocast('cuda',enabled=not args.no_amp):
            student_logits = student(X_mixed)        # 学生看同一张混合图
            # criterion 签名: (student_out, teacher_logits, targets, args) —— 顺序不可乱
            total_loss,base_loss,dist_loss = criterion(student_logits, teacher_logits, target, args)

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
        # >>>【AI 修改 2026-08-31】幅度递增 (论文 rand-m9-mstd0.5-inc1)
        #  你原来的写法 t.m += args.ra_inc 新鲜跑一遍是对的, 但断点续跑时 m 会从
        #  args.ra_m 重新开始(比如 50 轮中断时 m 已爬高, 续跑又掉回 9, 训练动态不一致)。
        #  改成无状态公式 m = ra_m + (epoch-1)*inc: 幅度只由 epoch 号决定, 续跑天然一致。
        #  注意: 做 M 消融(固定 m=0/5/15/20)时要传 --ra-inc 0, 否则 m 会从设定值往上爬。
        for t in train_loader.dataset.transform.transforms:  # 遍历 Compose 找 RandAugment
            if isinstance(t, RandAugmentCIFAR):
                t.m = args.ra_m + (epoch - 1) * args.ra_inc
        # <<<【AI 修改结束】
        train_loss,train_acc=train_one_epoch(student, teacher, train_loader, criterion, optimizer, scaler, device, args,
                                             desc=f'train {epoch}/{args.epochs}')   # >>>【AI 添加】进度条标题带 epoch
        test_loss,test_acc =evaluate(student, test_loader, criterion, device,args)
        scheduler.step()

        writer.add_scalar('train/loss',train_loss,epoch)
        writer.add_scalar('train/acc',train_acc,epoch)
        writer.add_scalar('test/loss',test_loss,epoch)
        writer.add_scalar('test/acc',test_acc,epoch)
        writer.add_scalar("optim/lr",optimizer.param_groups[0]["lr"],epoch)
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
