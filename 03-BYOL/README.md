# 03 · BYOL

> Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning —— 不依赖负样本，通过 online / target 双网络 + 动量（EMA）更新，让 online 网络的预测去逼近 target 网络的投影。本项目使用 CIFAR-10 + ResNet18 复现。

## 概述

BYOL 取消了对比学习中"负样本"这一概念，转而训练两个网络：

- **online network**：`encoder + projector + predictor`，通过梯度正常更新；
- **target network**：`encoder + projector`（无 predictor），参数由 online 网络通过 **EMA（指数滑动平均）** 缓慢更新，不参与梯度。

训练目标是让 online 侧经过 predictor 的输出 `p` 去逼近 target 侧的投影 `z`，损失为两个视图上的对称 MSE（余弦距离）。由于 target 网络提供的是"平滑的自身目标"，整个框架不需要任何负样本即可学到有意义的表征。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `BYOL.py` | BYOL 主训练脚本（增强、数据集、online/target 网络、EMA 更新、损失、断点恢复） |
| `linear_eval.py` | 线性评估：冻结 encoder，训练 `Linear(512, 10)`，测试 CIFAR-10 精度 |
| `checkpoint/` | 训练权重/断点（运行产物，含 `byol_checkpoint.pth` 与 `byol_encoder_20/40/60.pth`） |

## 快速开始

数据 root 在代码中写为 `../data`（即仓库根 `data/` 目录），`download=True`。

```bash
# 1) 自监督预训练（默认 100 epochs，自动断点恢复）
python BYOL.py

# 2) 线性评估（默认加载 checkpoint/byol_encoder_60.pth，训练 50 epochs）
python linear_eval.py
```

默认超参数（无 argparse，直接写在脚本里）：

- 数据集：CIFAR-10（`train=True`，root=`../data`）
- batch size：128；优化器：Adam，lr=1e-3
- 轮数：100；EMA 系数 `beta`：0.996
- 投影头维度：512→4096（BatchNorm）→ReLU→256；predictor 同构（256→4096→256）
- 每 10 epochs 保存 `byol_checkpoint.pth`，每 20 epochs 保存 `byol_encoder_{epoch}.pth`；启动时若存在 checkpoint 则自动续训

## 实现要点

- **网络结构**：
  - `Encoder`：ResNet18（`weights=None`），针对 32×32 输入修改 `conv1`/`maxpool`/`fc`，输出展平后 512 维。
  - `ProjectionHead`：`Linear(512,4096) → BatchNorm1d → ReLU → Linear(4096,256)`。
  - `Predictor`：`Linear(256,4096) → BatchNorm1d → ReLU → Linear(4096,256)`。
- **双网络**：online 侧含 encoder + projector + predictor；target 侧只有 encoder + projector。target 初始化时从 online 拷贝权重并置 `requires_grad=False`。
- **EMA 更新**：`update_target` 对 encoder 与 projector 都执行 `target = beta·target + (1-beta)·online`（`beta=0.996`），且在前向中 target 侧全程 `torch.no_grad`。
- **损失**：`BYOL_loss(p, z)` 先对 `p`、`z` 做 L2 归一化，再算 `2 - 2·(p·z).sum(dim=1)` 的均值（即余弦距离的 2 倍），最终损失为两个视图的对称平均 `(loss(p1,t2) + loss(p2,t1)) / 2`。
- **数据增强**：`RandomCrop(32, padding=4)`、水平翻转、ColorJitter、Grayscale、`Normalize([0.5],[0.5])`；每个样本生成两个视图。
- **线性评估**：加载 `byol_encoder_60.pth` 里的 online encoder，冻结后接 `Linear(512,10)`（Adam lr=1e-3），训练 50 epochs，每个 epoch 后打印测试精度。

## 备注

仓库内未记录任何实验精度数字（`checkpoint/` 仅权重，无结果日志），训练/评估的 loss 与 accuracy 只打印到终端，因此此处不列出精度结果。

[⬆ 返回仓库总览](../README.md)
