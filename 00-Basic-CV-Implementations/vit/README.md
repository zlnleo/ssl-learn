# vit —— 从零手写 ViT（Vision Transformer）+ 工业级训练脚本系列

> 把手写 Transformer 的 Encoder 拿来做图像分类：图片切 patch 当「词」、加 [CLS] token 与可学习位置编码，从零搭出 ViT 并在 CIFAR-100（100 类）上训练；配套 16 篇编号学习笔记与 6 个递进式训练脚本。

## 概述

本目录是 `transformer/` 的视觉延伸：ViT = 「把图片切成小方块（patch）当作词」+「Transformer 的 Encoder（去掉所有掩码，双向注意力）」+「分类头」。

与手写 transformer 的主要差异（详见 `vit_solution.py` 开头的对照表）：

- 输入：patch 向量序列（不是词 id 序列）；
- 结构：只有 Encoder，无 Decoder；
- 注意力：没有任何掩码（双向注意力）；
- 位置编码：可学习参数（不是正弦函数）；
- FFN 激活：GELU（不是 ReLU）；
- 归一化：pre-LN（先 LN 后注意力，不是 post-LN）；
- 输出：整张图一个分类结果。

模型核心（`vit.py`）：`PatchEmbedding`（用 `Conv2d` 把图片切成 patch 并投影）→ `ClassTokenPosEmbed`（[CLS] 拼接 + 可学习位置编码）→ `EncoderBlock`（pre-LN + 多头注意力 + MLP/GELU）→ LayerNorm → 取 [CLS] → 分类头。默认 `patch_size=4`、`img_size=32`。

## 目录结构

| 路径 | 说明 |
|---|---|
| `*.md` | 16 篇编号学习笔记 + 1 篇 `vit_问题与修改建议.md`（见下方「学习笔记索引」） |
| `vit.py` | 你自己手写的 ViT（train/test 脚本优先导入它，缺省退回 `vit_solution.py`） |
| `vit_solution.py` | 参考答案（拆分结构：把 [CLS] 拼接 + 位置编码 + dropout 抽成独立类） |
| `vit_.py` | 修正版 ViT 实现（不修改 `vit.py`，额外提供 `ScaleDotAttention` 兼容别名） |
| `test_vit.py` | 模型验收脚本（前向 / 反向 / 学习能力 / 位置编码消融） |
| `train.py` … `train_v6_ddp.py` | 6 个递进式训练脚本（见下方「代码文件说明」） |
| `reviewlearn.py` / `reviewlearn_hydra.py` | 综合复习版训练脚本（argparse 版 / hydra 版） |
| `test_tensorboard.py` | TensorBoard 最小演示（写一条 loss 曲线到 `runs/demo`） |
| `configs/` | hydra 训练配置：`train_cifar100.yaml`（默认）、`train_toy.yaml` |
| `checkpoint/` | 训练产物：`best.pt`（最优模型）、`last.pt`（断点续跑完整状态） |
| `runs/` | 每次运行记录：`run_时间戳/` 内含 `config.txt`（或 `config.yaml`）、`train.log`、`tfboard/`（TensorBoard 事件文件） |
| `.idea/`、`__pycache__/` | IDE 配置 / Python 字节码缓存，可忽略 |

## 代码文件说明

| 文件 | 作用 |
|---|---|
| `vit.py` | 手写 ViT 模型（ScaledDotProductAttention / MultiHeadAttention / PatchEmbedding / ClassTokenPosEmbed / MLP / EncoderBlock / ViT），`__main__` 自带前向自检 |
| `vit_solution.py` | 参考答案版（拆分结构，逐行注释），写完全程后对照复盘 |
| `vit_.py` | 修正版实现（与 `vit.py` 并存，修复类名/拼写等问题，见 `vit_问题与修改建议.md`） |
| `test_vit.py` | 4 项自动化验收：[1] 前向形状 [2] 反向/优化器更新 [3] toy 任务学到 100% [4] 去掉位置编码 → 掉到接近随机（证明位置信息全靠 pos_embed） |
| `train.py` | **v1 基准版**：argparse + AMP + CosineAnnealingLR + 梯度裁剪 + checkpoint/续跑 + runs 日志，默认 CIFAR-100 |
| `train_v2_tensorboard.py` | v2：在 v1 基础上只加 TensorBoard 可视化 |
| `train_v3_earlystop.py` | v3：在 v1 基础上只加早停（`--patience`） |
| `train_v4_reproducible.py` | v4：在 v1 基础上只补可复现性（set_seed + worker_init_fn + cudnn 确定性） |
| `train_v5_hydra.py` | v5：把 argparse 换成 hydra + yaml 配置（读 `configs/*.yaml`） |
| `train_v6_ddp.py` | v6：DDP 分布式（单进程模式与 reviewlearn 一致；多卡用 torchrun） |
| `reviewlearn.py` | 综合复习版：argparse + AMP + 调度器 + checkpoint/续跑 + runs + TensorBoard + 早停（把 v2/v3/v4 的功能合并） |
| `reviewlearn_hydra.py` | 综合复习版的 hydra 版本 |
| `test_tensorboard.py` | TensorBoard 最小可用 demo（`SummaryWriter` 写曲线） |

> 训练脚本都优先 `from vit import ViT`，缺省回退 `from vit_solution import ViT`，接口一致。

## 学习笔记索引

| 编号 | 文件 | 主题 |
|---|---|---|
| 01 | `01_引导_从零手写ViT.md` | 无答案引导：跟着规格自己写 `vit.py`（只有引导/提示，答案在 `vit_solution.py`） |
| 02 | `02_验证与修改方向.md` | 怎么验证写对了（test_vit.py / train.py），验证后还能往哪改 |
| 03 | `03_ViT完全讲解_原理细节与torch用法.md` | ViT 逐模块原理 + 易错点 + PyTorch API 精讲 |
| 04 | `04_train.py完全讲解.md` | train.py 逐块精讲（五段式结构、AMP、checkpoint 等） |
| 05 | `05_argparse完全讲解.md` | argparse 专题（配置与逻辑分离） |
| 06 | `06_scaler与scheduler完全讲解.md` | AMP GradScaler + 学习率调度（余弦）专题 |
| 07 | `07_训练骨架速查.md` | optimizer / scheduler / scaler 标准姿势速查模板 |
| 08 | `08_wandb完全讲解.md` | wandb 实验记录平台专题（train.py 已埋 `--use-wandb` 钩子） |
| 09 | `09_tensorboard完全讲解.md` | TensorBoard 本地看板专题 |
| 10 | `10_工业训练流程全景与清单.md` | 工业训练项目 15 个零件总地图 |
| 11 | `11_yaml配置管理完全讲解.md` | yaml + hydra 配置管理（argparse 的升级方案） |
| 12 | `12_DDP分布式训练完全讲解.md` | DDP 多卡并行专题 |
| 13 | `13_可复现性工程完全讲解.md` | 可复现性四层工程 |
| 14 | `14_推理与模型导出完全讲解.md` | eval.py 推理 + ONNX 导出 + TensorRT/量化概念 |
| 15 | `15_Git与Docker工程工具完全讲解.md` | Git 分支/PR 流程 + Docker |
| 16 | `16_训练版本对照与使用指南.md` | 6 个 train 脚本版本总地图 + 用法速查 |
| — | `vit_问题与修改建议.md` | 记录手写 `vit.py` 的常见问题与修改建议（修正版在 `vit_.py`） |

## 快速开始

建议学习顺序：**先读 `01_引导_从零手写ViT.md` 自己写 `vit.py` → `python test_vit.py` 验收 → 对照 `vit_solution.py` 复盘 → 精读 `03`、`04` → 按 `16` 逐个版本跑 train 脚本**。

```bash
# 在 vit/ 目录下执行（conda 环境 ssl_cv）

# 1) 模型验收（无需数据，自动跑 toy 合成任务 + 位置编码消融）
python test_vit.py

# 2) 训练：默认 CIFAR-100（数据缓存在仓库根 data/，即 ../../data）
python train.py                           # 默认 100 轮 CIFAR-100
python train.py --dataset toy --epochs 5  # 离线合成「象限亮块」任务，快速试跑
python train.py --dataset fashionmnist    # 10 类灰度图（需联网下载）
python train.py --resume                  # 断点续跑：从 checkpoint/last.pt 恢复

# 3) 各功能版本（依赖见 16 笔记）
python train_v2_tensorboard.py --dataset toy --epochs 5          # + TensorBoard
python train_v3_earlystop.py --dataset toy --epochs 50 --patience 5  # + 早停
python train_v4_reproducible.py --dataset toy --epochs 3         # + 可复现性
python train_v5_hydra.py                                         # + hydra（用 configs/train_cifar100.yaml）
python train_v5_hydra.py dataset=toy epochs=5                    # 命令行覆盖配置
```

`train.py` 默认超参（CIFAR-100）：`epochs=100`、`batch_size=128`、`lr=1e-3`、`weight_decay=0.05`、`grad_clip=1.0`、`seed=42`、`embed_size=192`、`num_heads=6`、`num_layers=6`、`dropout=0.1`、`amp=True`（GPU）。训练产物写入 `checkpoint/best.pt`、`checkpoint/last.pt` 与 `runs/run_时间戳/`。

数据路径：`--data-dir` 默认 `../../data`（即仓库根 `data/`，CIFAR-100 已解压时离线可用）；`toy` 为内存合成数据无需下载。

## 备注

- `vit_问题与修改建议.md` 记录的是手写 `vit.py` 曾出现的类名/拼写等问题；当前 `vit.py` 已是修正后的 `class ViT`、`transpose`，`vit_.py` 是与它并存的修正版实现。
- 本目录没有单独的 `eval.py` 或 `train_vit_cifar100.py`；`14_推理与模型导出完全讲解.md` 讲解如何写 eval.py（作为学习材料，脚本本身未在本目录落地）。
- `runs/`、`checkpoint/` 为共享运行产物目录；`.idea/`、`__pycache__/` 可忽略。

---

[← 返回仓库根 README](../../README.md) · [↑ 返回 00-Basic-CV-Implementations](../README.md)
