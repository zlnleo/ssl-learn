# 01 · SimCLR

> SimCLR: A Simple Framework for Contrastive Learning of Visual Representations —— 通过同一张图片的两个数据增强视图构造正样本对，用 NT-Xent 对比损失学习视觉表征。本项目使用 CIFAR-10 + ResNet18 复现。

## 概述

SimCLR 的核心思想是**对比学习**：不依赖任何人工标签，而是对同一张图片做两次随机数据增强得到两个视图 `xi`、`xj`（正样本对），把同一个 batch 里的其他图片当作负样本。模型让同一张图片的两个视图在表示空间中尽量接近，与其他图片尽量远离。

本项目实现包含：

- 针对 CIFAR-10 修改过的 **ResNet18 编码器**（去掉分类头，输出 512 维特征）。
- 一个 **Projection Head**（512→512→ReLU→128），把编码器特征映射到适合对比损失的空间。
- **NT-Xent 损失**（temperature=0.5），对 batch 内所有视图做余弦相似度 + 交叉熵。
- 训练使用 **AMP 混合精度** 与 **断点恢复**，并用 TensorBoard 记录 loss。
- 训练完成后只保存 encoder，再用 `linear_eval.py` 做 **线性评估** 验证表征质量。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `simCLR.py` | SimCLR 主训练脚本（增强、数据集、编码器、投影头、NT-Xent 损失、AMP 训练循环） |
| `linear_eval.py` | 线性评估：冻结 encoder，只训练一个 `Linear(512, 10)` 分类头，测试 CIFAR-10 精度 |
| `1.md` | SimCLR 学习总结笔记（见下文） |
| `data/` | CIFAR-10 数据（`cifar-10-batches-py`，已就位） |
| `runs/` | TensorBoard 事件日志（运行产物） |
| `model.pth`、`checkpoint_latest.pth`、`simclr_encoder.pth` | 训练产生的权重/断点文件（运行产物） |

## 快速开始

数据默认放在本目录 `data/` 下（代码中 `download=False`，需预先准备好 CIFAR-10，仓库已含）。

```bash
# 1) 自监督预训练（默认 100 epochs）
python simCLR.py

# 2) 线性评估（加载 simclr_encoder.pth，默认 50 epochs）
python linear_eval.py
```

训练脚本无 argparse，参数直接写在 `simCLR.py` 的 `train()` 里，默认：

- 数据集：CIFAR-10（`train=True`，root=`./data`）
- batch size：256；优化器：Adam，lr=3e-4
- 轮数：100；温度：0.5
- 设备：`cuda`（无 GPU 时回退 `cpu`）
- 每轮保存 `checkpoint_latest.pth`，结束后保存 `simclr_encoder.pth`；loss 写入 TensorBoard（`runs/simclr`）

## 实现要点

- **数据增强**：`RandomResizedCrop(32)`、`RandomHorizontalFlip`、`ColorJitter(0.8,0.8,0.8,0.2)`（p=0.8）、`RandomGrayscale`（p=0.2）、`Normalize([0.5],[0.5])`。每个样本生成两个视图 `xi`、`xj`。
- **编码器（Encoder）**：ResNet18（`weights=None`），针对 32×32 输入把 `conv1` 改为 3×3 stride=1、`maxpool` 改为 `Identity`、`fc` 改为 `Identity`，输出 512 维特征。
- **投影头（ProjectionHead）**：`Linear(512,512) → ReLU → Linear(512,128)`。
- **NT-Xent 损失**：拼接两个视图做 L2 归一化 → 计算相似度矩阵 → 掩掉对角线（自身相似度）→ 除以温度 0.5 → 用交叉熵拉近正样本对。
- **AMP 混合精度**：使用 `torch.amp.GradScaler` + `autocast("cuda")` 加速训练；掩对角线用 `torch.finfo(...).min` 而非 `-1e9`（FP16 无法表示）。
- **断点恢复**：若存在 `checkpoint_latest.pth` 则自动恢复模型、优化器与 epoch。
- **线性评估**：冻结 encoder，仅训练 `Linear(512,10)`（Adam lr=1e-3），在 CIFAR-10 测试集上打印 accuracy。

## 学习笔记

`1.md` 是 SimCLR 的完整学习总结，从整体目标、数据增强、编码器、投影头、正负样本、NT-Xent 损失，到 AMP、模型保存、线性评估逐一讲解，并对比了 SimCLR → MoCo → BYOL → DINO 的演进关系。一句话总结（出自笔记）：

> 通过数据增强制造正样本，通过对比学习让 encoder 学习具有语义的 feature，projection head 只是训练辅助，最终保留 encoder 用于下游任务。

## 备注

仓库内未记录任何实验精度数字（`runs/` 仅存 TensorBoard 事件，`.pth` 为权重/断点），训练与评估脚本的 loss/accuracy 均只打印到终端，因此此处不列出精度结果。

[⬆ 返回仓库总览](../README.md)
