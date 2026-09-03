# ViT — Vision Transformer

> 手写复现 *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*（Vision Transformer，ViT）：把图像切成 patch 序列交给 Transformer 编码器，用 CLS token 做分类，并在 CIFAR-100 上做监督训练与线性评估。

## 概述

本目录实现了两套 ViT 模型定义：一套面向 ImageNet 尺度的通用/教学版（`ViT-image.py`），一套面向 CIFAR-100 的小型化适配版（`vit_cifar100.py`）。适配版额外提供 `ViTEncoder`（输出 CLS token 特征，无分类头），被本目录的训练/评估脚本以及 `04-DINO`、`05-MAE` 复用。训练采用监督分类 + WarmupCosineLR 学习率调度，并提供冻结编码器的线性评估脚本。

- `ViT-image.py`：通用/教学版模型定义（ImageNet 尺度，模块注释详尽）。
- `vit_cifar100.py`：CIFAR-100 适配版模型定义（小尺寸 + Dropout/DropPath + 可分离的 `ViTEncoder`）。
- `train_vit_cifar100.py`：CIFAR-100 监督训练入口。
- `linear-eval.py`：冻结编码器 + 线性分类头的评估脚本。
- `scheduler.py`：`WarmupCosineLR` 学习率调度器。
- `vit.md`：学习笔记。

## 文件说明

| 文件 | 定位 | 说明 |
| --- | --- | --- |
| `ViT-image.py` | 通用/教学版模型定义 | `image_size=224, patch_size=16, embed_dim=768, depth=12, heads=12, num_classes=1000`；PatchEmbedding / CLS_Token / PositionEmbedding / EncoderBlock / ClassificationHead 等模块带详细 docstring |
| `vit_cifar100.py` | CIFAR-100 适配版 | `image_size=32, patch_size=4, embed_dim=256, depth=6, heads=8, num_classes=100`；加入 Dropout、DropPath（stochastic depth）；提供 `ViTEncoder`（输出 CLS token 特征）与 `ViT.forward_features` |
| `train_vit_cifar100.py` | 监督训练入口 | CIFAR-100，默认 200 epoch，WarmupCosineLR + RandAugment + label smoothing + AMP |
| `linear-eval.py` | 线性评估 | 加载 `checkpoints/vit_cifar100_last.pth`，冻结编码器，训练 `Linear(256 → 100)`，默认 100 epoch |
| `scheduler.py` | 学习率调度 | `WarmupCosineLR`：前 `warmup_epochs` 线性升温，之后按 cosine 衰减 |
| `vit.md` | 学习笔记 | ViT 各模块（PatchEmbedding / CLS token / 位置编码 / EncoderBlock / 分类头）的结构与原理总结 |

> `checkpoints/`、`runs/` 为运行产物，不属于源码。

## 快速开始

运行环境：conda 环境 `ssl_cv`（Python 3.10）。数据默认放在本目录上级的 `../data`（即仓库根 `data/`），需含 CIFAR-100。

```bash
# 监督训练（默认 200 epoch，断点续跑 checkpoints/vit_cifar100_last.pth）
python train_vit_cifar100.py

# 线性评估（需先运行 train_vit_cifar100.py 得到 checkpoint）
python linear-eval.py
```

`ViT-image.py` 与 `vit_cifar100.py` 仅提供模型定义，无独立训练入口，直接 `import` 使用即可。

## 实现要点

- **PatchEmbedding**：用 `Conv2d(kernel_size=patch_size, stride=patch_size)` 同时完成「切 patch + 线性投影」，再 flatten/transpose 成 `[B, num_patches, embed_dim]`。
- **CLS token 与位置编码**：可学习的 CLS token 拼在序列最前；位置编码为可学习参数，直接相加。
- **编码器块**：Pre-Norm + 多头自注意力（`nn.MultiheadAttention`，batch_first）+ MLP（`dim → 4·dim → dim`）+ 残差连接；`vit_cifar100.py` 版在注意力/MLP 后加入 Dropout 与 DropPath。
- **特征输出**：`vit_cifar100.py` 的 `ViTEncoder` 取 CLS token `x[:, 0]` 输出 `[B, 256]`，作为特征供线性评估与 `04-DINO` 复用；`ViT.forward_features` 暴露该特征。
- **训练配置**（`train_vit_cifar100.py`）：AdamW，`lr=5e-4`、`weight_decay=0.05`，batch size 128；`WarmupCosineLR(warmup_epochs=10, max_epochs=200)`；`CrossEntropyLoss(label_smoothing=0.1)`；数据增强含 `RandomCrop(32, padding=4)` + 随机水平翻转 + `RandAugment`；AMP；checkpoint 存 `checkpoints/vit_cifar100_last.pth`。
- **线性评估**（`linear-eval.py`）：冻结 `vit.encoder` 全部参数，只训练 `Linear(256, 100)`（AdamW，`lr=1e-3`、`weight_decay=0`），默认 100 epoch。

## 学习笔记

- [`vit.md`](./vit.md) — ViT 代码结构与原理总结，逐模块解释 PatchEmbedding、CLS token、位置编码、EncoderBlock、分类头及 forward 流程。

## 备注

代码中未硬编码任何精度结果；本目录仅保留训练产物 `checkpoints/vit_cifar100_last.pth` 与 TensorBoard 日志 `runs/`。

[⬆ 返回仓库总览](../README.md)
