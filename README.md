# Self-Supervised Learning & Computer Vision

这是我在学习计算机视觉（Computer Vision）与自监督学习（Self-Supervised Learning）过程中进行的代码实现、论文学习与实验记录。

目前主要使用 **Python + PyTorch** 进行模型实现和实验，希望通过自己编写代码、复现经典方法以及分析实验结果，加深对深度学习模型和视觉表征学习方法的理解。

---

## 📂 仓库结构（按学习路线）

仓库按学习方法分为若干独立项目，每个项目自带 `README.md` 说明其代码、运行方式与笔记。学习时建议按下面的路线推进，并在每个项目文件夹内阅读对应文档。

| 阶段 | 文件夹 | 主题 | 说明 |
|---|---|---|---|
| 0️⃣ 基础实现 | [00-Basic-CV-Implementations](00-Basic-CV-Implementations/README.md) | 从零手写基础模型 | 手写 Transformer（翻译任务）与 ViT，配套 16 篇 ViT 工程化学习笔记 |
| 1️⃣ 对比学习 | [01-simCLR](01-simCLR/README.md) | SimCLR | 对比学习入门：Encoder 预训练 + 线性评估，含 SimCLR 学习总结笔记 |
| 2️⃣ 动量对比 | [02-MoCo](02-MoCo/README.md) | MoCo | 动量对比学习：`MoCo_simple.py` 与 `MoCo_full.py` 两个版本 + 线性评估 |
| 3️⃣ 非对比自监督 | [03-BYOL](03-BYOL/README.md) | BYOL | 无需负样本的自监督：在线/目标网络 + EMA + predictor + 线性评估 |
| 4️⃣ 自蒸馏 | [04-DINO](04-DINO/README.md) | DINO | ViT 自蒸馏：简易版/完整版在 CIFAR-100 上的预训练与线性评估 |
| 5️⃣ 掩码图像建模 | [05-MAE](05-MAE/README.md) | MAE | 掩码自编码器（CIFAR-100）：预训练、重建可视化、线性探测 |
| 6️⃣ 蒸馏式 ViT | [06-DeiT](06-DeiT/README.md) | DeiT | 从 ViT 手写推导 DeiT 蒸馏 token + 双头结构与硬/软蒸馏训练（CIFAR-100，含完整消融实验） |
| 7️⃣ 层级 Transformer | [07-Swin Transform](07-Swin%20Transform/README.md) | Swin Transformer | 从零学习 Swin：9 大机制模块逐模块实现 + 模块化工程包 + 4 个核心消融实验 |
| 🔬 ViT 训练实践 | [ViT](ViT/README.md) | ViT | 手写 ViT 在 CIFAR-100 上的训练/评估 + `vit.md` 笔记 |
| 💾 数据 | [data](data/) | 本地数据集 | CIFAR-10 / CIFAR-100 / FashionMNIST（已 gitignore，默认数据源） |

> 说明：`00-Basic-CV-Implementations` 是全仓库的前置基础（Transformer 与 ViT 手写入门）；
> `01-simCLR`～`07-Swin Transform` 与 `ViT` 分别是各论文方法的学习复现项目；
> 各项目数据默认读取仓库根 `data/` 目录（可用参数覆盖）。

## 📚 Learning Roadmap

### 1. Contrastive Learning（对比学习）

- **SimCLR** — A Simple Framework for Contrastive Learning of Visual Representations
- **MoCo** — Momentum Contrast for Unsupervised Visual Representation Learning

### 2. Self-Supervised Representation Learning（非对比 / 自蒸馏表征学习）

- **BYOL** — Bootstrap Your Own Latent
- **DINO** — Emerging Properties in Self-Supervised Vision Transformers

### 3. Vision Transformer（视觉 Transformer 系列）

- **ViT** — An Image is Worth 16x16 Words
- **DeiT** — Training Data-Efficient Image Transformers & Distillation through Attention
- **Swin Transformer** — Hierarchical Vision Transformer using Shifted Windows

### 4. Masked Image Modeling（掩码图像建模）

- **MAE** — Masked Autoencoders Are Scalable Vision Learners

## 🛠️ Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- TensorBoard
- Hydra / OmegaConf
- Git
- Docker

## 🎯 Current Direction

目前主要学习计算机视觉、深度学习以及自监督学习相关方法。

下一阶段希望进一步学习**医学图像分析与脑影像处理**，尝试将计算机视觉和深度学习方法应用于医学领域的实际问题。

## 📌 About This Repository

本仓库主要用于记录个人学习过程中的代码实现和实验。

代码以理解模型原理和实现过程为主要目的，因此部分实现并非针对工程部署进行优化，而是尽可能通过自己编写代码理解论文中的核心思想和模型结构。

欢迎交流与指正。
