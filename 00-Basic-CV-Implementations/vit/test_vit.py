# -*- coding: utf-8 -*-
"""vit.py 的自动化检测脚本。

验证内容：
[1] 前向：输出形状 = (batch, num_classes)；
[2] 反向：分类交叉熵能正常回传梯度，优化器能更新参数；
[3] 学习能力：在"象限亮块"合成任务上训练，模型应该能学到 100% 正确分类；
[4] 位置编码消融实验（重点！）：同样的任务，把位置编码去掉再训练——
    验证集准确率会掉到接近随机猜测（4 类 => 25%）。
    这从实验上证明：注意力本身对顺序无感知，位置信息全靠 pos_embed 提供。

运行：python test_vit.py
退出码：全部通过为 0，否则为 1。
"""
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import ToyVisionDataset

# 优先验证你自己写的 vit.py；文件还没建时退回参考答案。
# 注意：没写 vit.py 时测试全绿只是在测参考答案，不算数——
# 以你真正写出来并通过验收为准。
try:
    from vit import ViT
except ImportError:
    from vit_solution import ViT


def _get_pos_embed(model):
    """取出模型的位置编码参数（不关心它藏在哪一层）。

    兼容两种结构：
    - 扁平结构（你自己写的 vit.py）：model.pos_embed；
    - 拆分结构（vit_solution.py）：model.cls_pos_embed.pos_embed。
    测试验证的是"行为"（位置信息被移除），不是内部布局。
    """
    if hasattr(model, "pos_embed"):
        return model.pos_embed
    return model.cls_pos_embed.pos_embed


def build_tiny_vit(use_pos_embed=True, pool="cls"):
    """构建一个很小的 ViT，训练几轮就能学会象限任务。"""
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=4,
                embed_size=64, num_heads=4, num_layers=2, dropout=0.0)
    if not use_pos_embed:
        pos_embed = _get_pos_embed(model)
        # 消融：把位置编码参数清零。注意不能把参数删掉（shape 变了会报错），
        # 清零等价于"模型完全没有位置信息"。
        with torch.no_grad():
            pos_embed.zero_()
        # 冻结住，防止训练中又被学出来
        pos_embed.requires_grad_(False)
    return model


def train_until_fit(model, train_loader, val_loader, device,
                    epochs=40, lr=1e-3, quiet=False):
    """小模型在象限任务上训练若干轮，返回 (训练准确率, 验证准确率)。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(1, epochs + 1):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if not quiet and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:>3}/{epochs}, loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        def acc(loader):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(-1) == labels).sum().item()
                total += labels.numel()
            return correct / total
        return acc(train_loader), acc(val_loader)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- [1] 前向形状检查 ----
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10,
                embed_size=128, num_heads=8, num_layers=4).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    logits = model(x)
    assert logits.shape == (2, 10), f"输出形状错误: {logits.shape}"
    print("[1] forward OK: (2, 3, 32, 32) -> (2, 10)")

    # ---- [2] 反向 + 参数更新检查 ----
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([3, 7], device=device))
    loss.backward()
    # 检查模型第一个参数（patch 嵌入层的权重，构造顺序上它排第一）是否收到梯度。
    # 刻意不写死内部属性名（patch_embed/proj 等），你自己的命名风格也能通过验收——
    # 测试应该验证"行为"，而不是强制"命名"。
    first_param = next(iter(model.parameters()))
    assert first_param.grad is not None, "梯度没有回传到模型第一层参数"
    print("[2] backward OK: 梯度已回传到模型第一层参数（patch 嵌入权重）")

    # ---- 数据：训练 128 张 / 验证 64 张，验证集是"没见过的图" ----
    train_loader = DataLoader(ToyVisionDataset(128, seed=0), batch_size=32, shuffle=True)
    val_loader = DataLoader(ToyVisionDataset(64, seed=1), batch_size=64)

    # ---- [3] 学习能力：带位置编码的模型应该 100% 学会 ----
    print("[3] 训练带位置编码的 ViT（象限亮块任务）...")
    m1 = build_tiny_vit(use_pos_embed=True).to(device)
    train_acc, val_acc = train_until_fit(m1, train_loader, val_loader, device)
    print(f"    带位置编码: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
    assert val_acc >= 0.99, f"带位置编码的模型验证准确率过低: {val_acc}"
    print("[3] PASS: 带位置编码 -> 验证集几乎满分，模型确实学会了'看位置'")

    # ---- [4] 位置编码消融：去掉位置编码，验证集准确率应接近随机(25%) ----
    print("[4] 训练去掉位置编码的 ViT（同样的任务）...")
    m2 = build_tiny_vit(use_pos_embed=False).to(device)
    train_acc2, val_acc2 = train_until_fit(m2, train_loader, val_loader, device)
    print(f"    去掉位置编码: train_acc={train_acc2:.3f}, val_acc={val_acc2:.3f}")
    # 没有位置信息时，模型只能靠死记训练集的噪声模式，
    # 换一批没见过的图（验证集）就抓瞎 -> 准确率掉到随机水平附近
    assert val_acc2 < 0.60, f"消融实验异常：无位置编码的验证准确率过高: {val_acc2}"
    print("[4] PASS: 去掉位置编码 -> 验证集准确率接近随机猜测")

    print("\n全部测试通过 [OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
