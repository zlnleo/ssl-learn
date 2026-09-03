# DeiT on CIFAR-100 —— 从 ViT 手写推导 DeiT

> [⬆ 返回仓库总览](../README.md)

《Training data-efficient image transformers & distillation through attention》
（Hugo Touvron et al., Facebook AI, ICML 2021, arXiv:2012.12877）的 CIFAR-100 复现项目。

**学习目标**：不从官方仓库抄代码，从 ViT 出发自己推导并实现 DeiT 的
蒸馏 token + 双头结构、硬/软蒸馏损失与完整训练管道，并跑通 CIFAR-100 实验。

## 结果（CIFAR-100，单卡，torch 2.11）

| 模型 | 配置 | 最优测试精度 |
|---|---|---|
| TeacherCNN（卷积教师） | 30 epochs, SGD | **69.0%** |
| DeiT-Ti + 硬蒸馏（v1 baseline） | 100 epochs（早停于 88） | 63.27% |
| DeiT-Ti + 硬蒸馏 + CutMix | 100 epochs | 67.90% |
| DeiT-Ti + 硬蒸馏 + **RandAugment (m=20)** | 100 epochs | **68.36%（最优）** |

> 完整消融（Mixup/CutMix/RandAugment/全配方/M 消融五组）见 `DeiT项目总结.md`。
> 最优命令：`python deittrain_v2.py --mixup 0 --cutmix 0 --ra-m 20 --ra-inc 0`

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

- [DeiT项目总结.md](DeiT项目总结.md) —— **项目总结**：全部实验数据 / 时间线 / 知识清单 / 回访清单（先看这篇）
- [DeiT论文讲解.md](DeiT论文讲解.md) —— 论文完整解读（公式 / 结果表 / 消融）
- [deit手写评价.md](deit手写评价.md) —— 三轮代码评审记录（含全部 bug 复盘与结果分析）
- [deit问题解答.md](deit问题解答.md) —— **问答集**：全程问题与解答 + 注释知识点 + 待补充占位
- [DeiT_v2学习路线.md](DeiT_v2学习路线.md) —— v2 学习地图 + 消融实验总表
- [Mixup_CutMix接入教程.md](Mixup_CutMix接入教程.md) —— Mixup / CutMix 融合教程（含逐行"为什么"）
- [RandAugment教程.md](RandAugment教程.md) —— RandAugment 14 操作详解 + 语法课堂 + M 消融结果
- [warmup教程.md](warmup教程.md) —— 学习率 warmup 与断点续跑教程
- [冒烟测试学习.md](冒烟测试学习.md) —— 最小测试数据的编造方法论

## 环境

Python 3.10 · PyTorch 2.11 (CUDA) · torchvision 0.26 · tqdm（可选）
