# DeiT on CIFAR-100 —— 从 ViT 手写推导 DeiT

《Training data-efficient image transformers & distillation through attention》
（Hugo Touvron et al., Facebook AI, ICML 2021, arXiv:2012.12877）的 CIFAR-100 复现项目。

**学习目标**：不从官方仓库抄代码，从 ViT 出发自己推导并实现 DeiT 的
蒸馏 token + 双头结构、硬/软蒸馏损失与完整训练管道，并跑通 CIFAR-100 实验。

## 结果（CIFAR-100，单卡，torch 2.11）

| 模型 | 配置 | 最优测试精度 |
|---|---|---|
| TeacherCNN（卷积教师） | 30 epochs, SGD | **69.0%** |
| DeiT-Ti + 硬蒸馏 | 100 epochs（早停于 88） | **63.27%** |

v1 训练不含 Mixup/RandAugment（预期区间 60~68%）；教师 65% 过关门槛达成。

## 文件结构

| 文件 | 说明 |
|---|---|
| `deitmodel.py` | 手写 DeiT 模型：PatchEmbed / Attention / MLP / Block / DropPath / 蒸馏 token 双头 |
| `deitloss.py` | 手写损失：标签平滑 CE + 硬蒸馏 (Eq.1) / 软蒸馏 (Eq.2) |
| `deitteacher.py` | 卷积教师 + 教师训练 |
| `deittrain.py` | 训练入口（v1 baseline）：AdamW + warmup + 余弦、AMP、梯度裁剪、tqdm、早停、断点续跑、TensorBoard |
| `deittrain_v2.py` | v2 训练入口：在 v1 基础上加入 Mixup / CutMix（论文 Sec 4.2 配方的第一步） |
| `deit_cifar100.py` | 完整增强参考版（RandAugment / Mixup / CutMix 等论文 Sec 4.2 配方 + EMA，即"完全体"） |
| `_check_user_files.py` | 模型 / 损失冒烟自检脚本 |

## 快速开始

```bash
conda activate ssl_cv            # Python 3.10 / torch 2.11 / CUDA
python deittrain.py --epochs 100 --distill hard   # 训练（教师自动训练并缓存到 runs/teacher/）
python deittrain.py --resume                       # 断点续跑（从 checkpoint/last.pth）
python _check_user_files.py                        # 冒烟自检（秒级）
tensorboard --logdir runs                          # 查看曲线
```

数据默认读取 `D:\project\self_supervised_learning\data`（可用 `--data-dir` 覆盖）。
`tqdm` 为可选依赖（未安装时自动降级为无进度条）。

## 文档

- [DeiT论文讲解.md](DeiT论文讲解.md) —— 论文完整解读（公式 / 结果表 / 消融）
- [deit手写评价.md](deit手写评价.md) —— 两轮代码评审记录（含接口 bug 复盘与结果分析）
- [deit问题解答.md](deit问题解答.md) —— criterion / 测试口径 / 损失平均等 FAQ
- [warmup教程.md](warmup教程.md) —— 学习率 warmup 与断点续跑教程
- [冒烟测试学习.md](冒烟测试学习.md) —— 最小测试数据的编造方法论
- [step_by_step.md](step_by_step.md) —— 从零实现教程（对应参考版）
- [solution.md](solution.md) —— 参考版解决方案
- [DeiT_v2学习路线.md](DeiT_v2学习路线.md) —— v2 学习地图（Mixup / CutMix / RandAugment / EMA + 消融实验设计）
- [Mixup_CutMix接入教程.md](Mixup_CutMix接入教程.md) —— 把 Mixup / CutMix 融进训练循环的分步教程
- [RandAugment教程.md](RandAugment教程.md) —— RandAugment 14 个操作逐个详解 + 分步实现教程

## 环境

Python 3.10 · PyTorch 2.11 (CUDA) · torchvision 0.26 · tqdm（可选）
