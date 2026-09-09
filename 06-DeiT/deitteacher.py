# -*- coding: utf-8 -*-
"""
deitteacher.py —— 卷积教师 + 教师训练

论文里的教师是 RegNetY-16GF (ImageNet ~84.2%)。CIFAR-100 上我们退而求其次, 但保留论文的
两条核心原则:
  1) 教师必须是卷积网络 (论文消融: 卷积教师 > Transformer 教师, 蒸馏传递的是卷积的归纳偏置);
  2) 教师只要比学生"先学一步"即可 —— 30~40 epoch 达到 65%+ 就够当师傅。

教师训练只用简单增强 (RandomCrop + RandomHorizontalFlip), 与 v1 学生保持一致。
TODO 由你补全; 写完后的过关门槛: test acc >= 0.65。
"""

import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

# >>>【AI 添加 2026-08-28】tqdm 进度条 (可选依赖)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda it, **kw: it
# <<<【AI 添加结束】

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class TeacherCNN(nn.Module):
    """VGG 风格 + BatchNorm 的小卷积网络, 约 0.5M 参数, 30 epoch 可到 65~70%。"""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 8 -> 4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def evaluate(test_loader,model, device):
    """测试集 top-1 精度。模型会被切到 eval 模式。"""
    model.eval()
    correct ,total= 0,0
    for X,y in test_loader:
        X , y = X.to(device), y.to(device)
        output = model(X)
        correct+=(output.argmax(-1)==y).sum().item()
        total += y.numel()

    return correct/total



def build_cifar100_loader(data_dir, batch_size):
    train_T = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_T = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_dataset = CIFAR100(data_dir, train=True, transform=train_T)
    test_dataset = CIFAR100(data_dir, train=False, transform=test_T)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=2)
    return train_loader, test_loader

def train_one_epoch( train_loader,model, optimizer, criterion,device, desc='train'):  # >>>【AI 添加】desc: 进度条标题
    model.train()

    total_loss,correct,total = 0.0 , 0 , 0
    # >>>【AI 添加 2026-08-28】tqdm 进度条
    pbar = tqdm(train_loader, desc=desc, leave=False, ncols=100)
    # <<<【AI 添加结束】
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        correct += (output.argmax(dim=-1)==y).sum().item()
        total += y.numel()
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")   # >>>【AI 添加】进度条后缀
    return total_loss / len(train_loader), correct/total



def train_teacher(data_dir: str, epochs: int = 30, batch_size: int = 128,
                  lr: float = 0.1, device=None, save_path: str = None):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_loader, test_loader = build_cifar100_loader(data_dir, batch_size)

    model = TeacherCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 23],
        gamma=0.1
    )
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss,train_acc=train_one_epoch(train_loader, model, optimizer, criterion, device,
                                             desc=f'teacher {epoch+1}/{epochs}')   # >>>【AI 添加】进度条标题带轮次
        test_acc=evaluate(test_loader, model, device)
        print(f"{epoch+1}/{epochs}epochs:train_loss={train_loss:.4f}, train_acc={train_acc:.4f},test_acc={test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        scheduler.step()
    return model,best_acc



if __name__ == '__main__':
    # 教师可以单独跑: python deitteacher.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model, acc = train_teacher(
        data_dir=r'D:\project\self_supervised_learning\data',
        epochs=30, batch_size=128, lr=0.1, device=device,
        save_path='runs/teacher/teacher_cnn_cifar100.pth')
    print(f'教师最优 acc = {acc:.4f}')
