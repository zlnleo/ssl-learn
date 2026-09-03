# 00-Basic-CV-Implementations —— 从零手写 Transformer / ViT 基础实现

> 本目录是「从零复现经典模型」的起步层：先手写 NLP 的 Transformer（Encoder–Decoder 机器翻译），再把它改造成视觉版 ViT（Vision Transformer）做图像分类，用同一条注意力主线串起两个子模块。

## 概述

这个文件夹存在的目的，是把「Attention Is All You Need」这条主线从文本一路贯通到图像：

- `transformer/` —— 用 `nn.Module` 从零搭一个 Encoder–Decoder Transformer（词嵌入、正弦位置编码、多头注意力、FFN、残差 + LayerNorm 全部手写，不调用 `torch.nn.Transformer`），在极小的英 → 中玩具语料上跑通「数据 → 训练 → 保存 → 加载 → 推理」完整闭环。
- `vit/` —— 把手写 Transformer 的 Encoder 拿过来，把图片切成 patch 当「词」，加上可学习的分类 token 与位置编码，搭出 ViT 用于 CIFAR-100（100 类）图像分类；并配套一整套「从模型到工业级训练脚本」的学习笔记与多版本训练脚本。

两者共享同一套注意力实现思路（多头缩放点积注意力 + mask 约定 + 残差/归一化）。学完 transformer 再看 vit，真正的新概念只有「图片怎么变成 token」这一个。

## 目录结构

| 路径 | 说明 |
|---|---|
| `transformer/` | 从零手写 Transformer，英 → 中玩具翻译（详见其 README） |
| `vit/` | 从零手写 ViT + 训练脚本系列 + 编号学习笔记（详见其 README） |
| `_data_probe/` | FashionMNIST 数据探测用的辅助目录（仅存放下载/解压的原始数据，非学习模块） |
| `.idea/` | PyCharm 工程配置目录（IDE 自动生成，与学习内容无关） |
| `__pycache__/` | Python 字节码缓存（自动生成，可忽略） |

## 子模块速览

| 子模块 | 学的模型 | 核心任务 | 数据来源 |
|---|---|---|---|
| `transformer/` | Transformer（Encoder–Decoder） | 序列到序列（英 → 中机器翻译） | 脚本内 4 条玩具句对，无需下载 |
| `vit/` | ViT（Vision Transformer） | 图像分类 | CIFAR-100（默认 `../../data`，即仓库根 `data/`）；另有 toy 合成数据、FashionMNIST |

## 快速开始

环境：conda 环境 `ssl_cv`（Python 3.10，PyTorch）。数据集默认放在仓库根 `data/`。

```bash
# 1) 先跑通 Transformer（玩具语料极小，几十轮内即可记住）
cd 00-Basic-CV-Implementations/transformer
python test_transformer_simple.py        # 冒烟测试：验证前向/反向/掩码/生成
python train.py                          # 默认 100 轮训练英→中玩具模型
python test_translation.py               # 加载 checkpoint 做端到端验收

# 2) 再进入 ViT（CIFAR-100 分类）
cd ../vit
python test_vit.py                        # 模型验收 + 位置编码消融实验
python train.py --dataset toy --epochs 5  # 先用离线合成数据快速跑通流程
```

## 备注

- 本目录是仓库根 `README.md` 统一索引的项目之一，两个子模块各自维护更细的 `README.md`。
- 各子模块内的 `checkpoint.pt` / `checkpoint/`、`runs/`、`__pycache__/` 均为运行产物，可忽略或按需清理。

---

[← 返回仓库根 README](../README.md)
