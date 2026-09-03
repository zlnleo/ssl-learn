# MAE — Masked Autoencoders

> 复现论文 *Masked Autoencoders Are Scalable Vision Learners*（MAE）：随机掩码 75% 的 patch，只让 ViT 编码器处理可见 patch，再用一个轻量解码器重建被掩码的像素，在 CIFAR-100 上做自监督预训练。

## 概述

MAE 是非对称编码器-解码器的掩码图像建模（Masked Image Modeling）方法：先对图像切成 patch，随机掩码掉 75%，编码器只处理剩余的可见 patch（因此计算量小、可扩展），解码器用「可见 patch 的隐变量 + 可学习 mask token」恢复原始顺序并重建全部像素，损失只在被掩码的 patch 上计算。本目录包含模型定义、预训练入口、校验/测试脚本、可视化与线性探测评估。

- `vit_mae.py`：模型定义（含 ViT 编码器、MAE 解码器、`random_masking` 掩码逻辑）。
- `MAE_cifar100.py`：CIFAR-100 自监督预训练入口。
- `check_mae.py`：校验脚本，加载权重后统计 MAE loss / 掩码 patch MSE / 可见 patch MSE / mask ratio 并自动判读。
- `test_mae.py`：用 toy 例子逐步打印 `random_masking` 与解码器拼接/恢复逻辑（不含真实模型）。
- `visualize_mae.py`：重建可视化，生成 `saveimg/mae_reconstruction.png`。
- `linear_probe.py`：线性探测评估，冻结 MAE 编码器 + 线性分类头。

## 文件说明

| 文件 | 定位 | 说明 |
| --- | --- | --- |
| `vit_mae.py` | 模型定义 | PatchEmbedding / 位置编码 / 多头注意力 / MLP / DropPath / `MAEEncoder` / `MAEDecoder` / `MAE`；含 `random_masking(mask_ratio=0.75)` |
| `MAE_cifar100.py` | 预训练入口 | CIFAR-100 无监督预训练，masked MSE loss，AMP，默认 200 epoch |
| `check_mae.py` | 校验脚本 | 加载 `checkpoint/mae_last.pth`，在 CIFAR-100 测试集上统计并自动检查 shape / mask ratio / loss |
| `test_mae.py` | 掩码逻辑测试 | 用一个 8-token 的 toy 张量逐步打印 random_masking 及 decoder 的 mask token 拼接与 ids_restore 恢复 |
| `visualize_mae.py` | 重建可视化 | 取单张图 forward 后 `unpatchify`，三栏展示 Original / Masked 75% / Reconstruction，存 `saveimg/mae_reconstruction.png` |
| `linear_probe.py` | 线性探测 | 复用 MAE 的 patch embedding / 位置编码 / 编码器，不做 mask，全局平均池化后接 `Linear(256 → 100)`，默认 100 epoch |

> `checkpoint/`、`runs/`、`saveimg/` 为运行产物，不属于源码。

## 快速开始

运行环境：conda 环境 `ssl_cv`（Python 3.10）。数据默认放在本目录上级的 `../data`（即仓库根 `data/`），需含 CIFAR-100。

```bash
# 自监督预训练（默认 200 epoch；注意该脚本 download=False，需预先备好数据）
python MAE_cifar100.py

# 校验模型（加载 checkpoint/mae_last.pth）
python check_mae.py

# 掩码逻辑逐步测试（toy 例子）
python test_mae.py

# 重建可视化（生成 saveimg/mae_reconstruction.png）
python visualize_mae.py

# 线性探测评估（冻结编码器，训练线性分类头）
python linear_probe.py
```

## 实现要点

- **掩码逻辑**（`vit_mae.py::random_masking`）：`mask_ratio=0.75`，64 个 patch 中保留 16 个可见 token；用 `torch.rand + argsort` 随机打乱取前 `keep_num` 个，再用 `argsort` 得到 `ids_restore` 供解码器恢复顺序。mask 中 0 表示可见、1 表示被掩码。
- **编码器-解码器**：patch_size=4 → 64 个 patch；`embed_dim=256`、编码器 depth=6 / heads=8；解码器 `decoder_dim=128`、depth=2 / heads=4，输出头 `Linear(128, 4*4*3=48)` 重建每个 patch 的 48 个像素值。
- **损失**：`(pred - target)²` 逐元素 MSE，先 `mean(dim=-1)` 得到每个 patch 的误差，再按 mask 加权 `(loss * mask).sum() / mask.sum()`，即只监督被掩码的 patch。
- **线性探测**：`linear_probe.py` 复用 MAE 的 `patch_embedding` / `pos_embedding` / `encoder`，前向时**不做 masking**，对 64 个 patch 做全局平均池化得到 `[B, 256]` 特征，再接 `Linear(256, 100)`；只优化分类头（AdamW，`lr=1e-3`、`weight_decay=0.0`），并把 encoder 手动置为 eval 模式。
- **优化与保存**：预训练用 AdamW，`lr=1e-3`、`weight_decay=0.05`，batch size 256，AMP；checkpoint 存 `checkpoint/mae_last.pth`，线性探测最优权重存 `checkpoint/mae_linear_probe_best.pth`。

## 备注

代码中未硬编码任何精度结果；`check_mae.py` 中的阈值（mask ratio ≈ 0.75、loss < 0.1、掩码误差 < 可见误差）仅用于自动判读，不是实验结果。

[⬆ 返回仓库总览](../README.md)
