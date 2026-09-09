# DINO — 自监督 Vision Transformer（Distillation with NO labels）

> 复现论文 *Emerging Properties in Self-Supervised Vision Transformers*（DINO）：通过「学生-教师蒸馏 + 动量 EMA + 中心化对比损失」做自监督表征学习。本目录同时包含基于 ResNet18（CNN）的教学版/完整版，以及基于 ViT 的 CIFAR-100 完整版。

## 概述

DINO 的核心思想是不用负样本、只用同一张图的不同视图做「自蒸馏」：教师网络（EMA 更新、不参与梯度）给出软目标，学生网络去拟合它；再引入一个可学习的中心向量 `center` 防止表示坍塌。本目录有 4 个脚本：

- `dino-simple.py`：精简教学版，用 ResNet18（CNN）在 CIFAR-10 上跑通整个流程，增强简单、默认只跑 2 个 epoch。
- `dino.py`：完整版（CNN），同样的 ResNet18 结构，但补全了 DINO 论文里的完整数据增强并接入 TensorBoard，默认 100 epoch。
- `dino_vit_cifar100.py`：完整版（ViT），复用 `../ViT/vit_cifar100.py` 的 `ViTEncoder`，在 CIFAR-100 上做自监督预训练，默认 100 epoch。
- `linear_eval_cifar100.py`：线性评估，加载 `dino_vit_last.pth` 中教师编码器的权重，冻结后训练一个 `Linear(256 → 100)` 分类头。

## 文件说明

| 文件 | 定位 | 说明 |
| --- | --- | --- |
| `dino-simple.py` | 精简教学版（CNN） | ResNet18 编码器 + MLP projector；CIFAR-10；仅随机水平翻转增强；默认 2 epoch；无 TensorBoard |
| `dino.py` | 完整版（CNN） | ResNet18 编码器 + MLP projector；CIFAR-10；完整 DINO 增强（RandomResizedCrop / ColorJitter / RandomGrayscale）；默认 100 epoch；TensorBoard + 断点续跑 |
| `dino_vit_cifar100.py` | 完整版（ViT） | 复用 `../ViT/vit_cifar100.py` 的 `ViTEncoder`；CIFAR-100；AMP 混合精度；默认 100 epoch |
| `linear_eval_cifar100.py` | 线性评估 | 加载 `checkpoint/dino_vit_last.pth` 的 `teacher_encoder.*` 权重，冻结后训练线性分类头并打印最优准确率 |

> `checkpoint/` 为训练产物（如 `dino_vit_last.pth`），不属于源码。

## 快速开始

运行环境：conda 环境 `ssl_cv`（Python 3.10）。数据默认放在本目录上级的 `../data`（即仓库根 `data/`），需含 CIFAR-10 / CIFAR-100。

```bash
# 精简教学版（CNN，CIFAR-10，默认 2 epoch）
python dino-simple.py

# 完整版（CNN，CIFAR-10，默认 100 epoch）
python dino.py

# 完整版（ViT，CIFAR-100，默认 100 epoch）
python dino_vit_cifar100.py

# 线性评估（需先运行 dino_vit_cifar100.py 得到 checkpoint）
python linear_eval_cifar100.py
```

注意：`dino_vit_cifar100.py` 与 `linear_eval_cifar100.py` 都通过 `sys.path.append("../ViT")` 导入 `ViTEncoder`，请在本目录下运行。

## 实现要点

- **多视图**：每张图采样 2 个全局视图 + 4 个局部视图，共 6 个 view。教师只看 2 个全局视图，学生看全部 6 个视图；损失遍历所有 (teacher, student) 组合，但跳过同一个全局视图的组合（`i == j`）。
- **数据增强**（`dino_vit_cifar100.py`）：全局 crop 用 `RandomResizedCrop(32, scale=(0.5, 1.0))`，局部 crop 用 `RandomResizedCrop(32, scale=(0.2, 0.5))`，均含随机水平翻转。
- **教师更新**：教师参数 `requires_grad=False`，用 EMA 更新 `teacher = β·teacher + (1-β)·student`，`β=0.996`。
- **损失**：教师输出做 `softmax((z - center) / τ_t)`（`τ_t=0.04`），学生输出做 `log_softmax(z / τ_s)`（`τ_s=0.1`），取交叉熵并求均值；教师与 center 均在 `torch.no_grad()` 下计算。
- **中心化**：`center` 初始化为 0，用动量 `momentum=0.9` 按 batch 均值更新，防止坍塌。
- **优化**：AdamW，`lr=1e-3`、`weight_decay=0.04`，batch size 128；ViT 版用 `GradScaler`（AMP）。
- **断点续跑**：CNN 版存 `checkpoint/dino_last.pth`（每轮覆盖，另每 10 轮存 `dino_epoch_*.pth`）；ViT 版存 `checkpoint/dino_vit_last.pth`（每轮覆盖）。

## 备注

代码中未硬编码任何精度结果；本目录仅保留训练产物 `checkpoint/dino_vit_last.pth`。

[⬆ 返回仓库总览](../README.md)
