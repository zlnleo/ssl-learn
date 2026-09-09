# 02 · MoCo

> Momentum Contrast for Unsupervised Visual Representation Learning —— 用**动量编码器** + **特征队列**把负样本规模与 batch 大小解耦，从而在较小 batch 下也能提供大量负样本。本项目使用 CIFAR-10 + ResNet18 复现，提供 `full` 与 `simple` 两个版本。

## 概述

SimCLR 的负样本来自当前 batch，因此需要很大的 batch size；MoCo 则把历史 batch 的特征存进一个**队列（queue）**当作负样本，并用一个**动量更新的 key 编码器**生成这些特征，从而：

- 负样本数量由队列长度 `K` 决定，不再受 batch size 限制；
- key 编码器通过 `θ_k ← m·θ_k + (1-m)·θ_q` 缓慢跟随 query 编码器，保证队列里新旧特征的一致性。

本项目实现包含：CIFAR-10 适配的 ResNet18 编码器、query/key 双编码器 + 双投影头、动量更新、循环队列、InfoNCE 损失，以及线性评估脚本。`full` 与 `simple` 的核心机制完全相同，仅在**队列大小、batch size、训练轮数**上不同。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `MoCo_full.py` | 完整版训练脚本（队列 K=16384，batch=256，200 epochs） |
| `MoCo_simple.py` | 简化版训练脚本（默认队列 K=4096，batch=128，20 epochs） |
| `linear_eval.py` | 线性评估：冻结 encoder，训练 `Linear(512, 10)`，测试 CIFAR-10 精度 |
| `checkpoint/` | 训练权重/断点（运行产物，含 `moco_last.pth`） |
| `data/` | 目录内也存放了一份 CIFAR-10（`cifar-10-batches-py`），但代码未引用它（见下，代码实际指向 `../data`） |

## 快速开始

两个脚本代码内把数据集 root 写为 `../data`（即仓库根 `data/` 目录），`download=True`。

```bash
# 简化版：快速跑通（20 epochs，默认队列 K=4096）
python MoCo_simple.py

# 完整版：更长训练 + 更大队列（200 epochs，K=16384）
python MoCo_full.py

# 线性评估（默认加载 checkpoint/moco_encoder_epoch_20.pth，训练 20 epochs）
python linear_eval.py
```

默认超参数（无 argparse，直接写在脚本里）：

| | MoCo_simple | MoCo_full |
| --- | --- | --- |
| 队列长度 K | 4096 | 16384 |
| batch size | 128 | 256 |
| epochs | 20 | 200 |
| 动量 m / 温度 T | 0.999 / 0.07 | 0.999 / 0.07 |
| 优化器 | SGD lr=0.03, momentum=0.9, weight_decay=1e-4 | 同左 |
| 保存 encoder 的 epoch | 20, 50, 100, 150, 200 | 20, 50, 100, 150, 170, 200 |

## 实现要点

- **双编码器**：`encoder_q`（query，正常梯度更新）与 `encoder_k`（key，`requires_grad=False`，不参与反向传播），初始化时用 `load_state_dict` 使两者权重一致；二者各带一个投影头（512→256→ReLU→128）。
- **动量更新**：每次前向先执行 `momentum_update`，`k.data = m·k.data + (1-m)·q.data`（encoder 与 projection 都更新）。
- **队列 queue**：`register_buffer` 注册 `(dim, K)` 的队列，按列做 L2 归一化；`deque_queue` 用指针循环写入当前 batch 的 key 特征（FIFO），实现负样本的"先进先出"。
- **损失（InfoNCE）**：正样本得分 `l_pos = q·k`，负样本得分 `l_neg = q·queue`，拼接后除以温度 `T=0.07`，标签恒为 0（正样本在 logits 第 0 列），用交叉熵计算。
- **数据增强**：`RandomResizedCrop(32, scale=(0.2,1.0))`、水平翻转、ColorJitter、Grayscale、GaussianBlur，以及 CIFAR 均值/标准差归一化。
- **线性评估**：加载 `checkpoint/moco_encoder_epoch_20.pth` 里的 `encoder_q`，冻结后接 `Linear(512, 10)`（Adam lr=1e-3），训练 20 epochs 并输出测试精度。

## 备注

仓库内未记录任何实验精度数字（`checkpoint/` 仅权重，无结果日志），训练/评估的 loss 与 accuracy 只打印到终端，因此此处不列出精度结果。`resume` 标志在脚本中默认 `False`（断点恢复逻辑存在但需手动打开）。

[⬆ 返回仓库总览](../README.md)
