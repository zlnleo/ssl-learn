# DeiT on CIFAR-100 —— 我的 Solution

> 配套文件：
> - `DeiT论文讲解.md` —— 论文完整解读
> - `deit_cifar100.py` —— 完整可运行代码（本方案）
> - `step_by_step.md` —— 一步一步写代码的教程
>
> 📌 2026-08-28 项目现状：本文描述的是**完整增强参考版**（`deit_cifar100.py`，已跑通验证）。
> 学习主线已迁移到手写模块化版 `deitmodel.py / deitloss.py / deitteacher.py / deittrain.py`
> （教师 69%，DeiT-Ti + 硬蒸馏 100 轮 best 63.27%）。过程复盘与答疑见 `deit手写评价.md`、
> `deit问题解答.md`、`warmup教程.md`、`冒烟测试学习.md`。

---

## 1. 任务定义

把 DeiT 论文的两大贡献 **（① 无外部数据的高效训练配方；② 蒸馏 token + 硬/软知识蒸馏）** 在 **CIFAR-100**（32×32，100 类，5 万训练图）上完整复现：

- 目标 A：实现 DeiT 模型（class token + distillation token 双头结构）；
- 目标 B：训练一个卷积教师，并分别跑 **无蒸馏 / 硬蒸馏 / 软蒸馏** 三种配置做对照；
- 目标 C：验证"蒸馏在数据有限时提升 Transformer 精度"这一论文核心结论。

## 2. 总体思路与设计决策

| 设计点 | 论文做法 (ImageNet) | 我的 CIFAR-100 方案 | 理由 |
|---|---|---|---|
| 输入 | 224×224, patch 16 → 196 tokens | 32×32, **patch 4 → 64 tokens** | CIFAR 图太小，patch 16 只剩 4 个 token；patch 4 与 ImageNet 的 token 数同量级 |
| 模型规模 | DeiT-Ti: embed 192, 12 层, 3 头, ~5.3M | 完全对齐 **DeiT-Ti 配置**（另提供 micro/small） | 5M 参数在 CIFAR-100 上容量合适，单卡分钟级/epoch |
| 教师 | RegNetY-16GF (~84.2%) | 自研 **TeacherCNN**（VGG+BN 风格, ~0.5M, 约 65~70%） | 无预训练权重依赖；教师必须比学生"先学一步"，小 CNN 30 epoch 即可 |
| 蒸馏 | 硬蒸馏 Eq.(1)，α=0.5 | 默认硬蒸馏，支持 soft / none | 论文结论：硬蒸馏简单且不弱于软蒸馏 |
| 增强 | RandAugment m9+mstd0.5+inc1, Mixup 0.8, CutMix 1.0, RE 0.25 | **原样照搬**（RandAugment 手写实现，无 timm 依赖） | 这是"数据高效"的核心，不能打折扣 |
| 优化器 | AdamW, wd 0.05, 余弦 + 5 epoch warmup, lr=1e-3×batch/512 | AdamW + 分组权重衰减 + 余弦 warmup，lr 5e-4 | CIFAR 用 100~300 epoch 即可收敛 |
| 推理 | 双头输出取平均 | 同样 `(logits_cls + logits_dist) / 2` | 与论文/官方实现一致 |

## 3. 代码结构（`deit_cifar100.py`）

```
0. 工具            set_seed / seed_worker
1. 数据增强        RandAugmentCIFAR / mixup_data / cutmix_data / random erasing   ← 论文 4.2
2. 教师网络        TeacherCNN + train_teacher()                                    ← 论文 3.2 教师
3. DeiT 模型       PatchEmbed → Attention → Mlp → Block(+DropPath) → DistilledViT  ← 论文 Fig.2
4. 损失            soft_cross_entropy + distillation_loss (hard / soft)            ← 论文 Eq.1/2
5. 训练循环        AdamW + warmup + 余弦 + Mixup/CutMix 批内混合 + 最佳 checkpoint  ← 论文 4.1
```

### 关键实现点

1. **蒸馏 token 参与全程注意力**（不是只在输出端拼接）：`DistilledViT.forward_features` 中
   `torch.cat((cls, dist, patches), dim=1)` 后加位置编码一起过 12 层 block——这是 DeiT 与
   其他"输出蒸馏"的本质区别。

2. **双头损失**（严格按论文 Eq.1/2）：
   ```python
   total = (1 - alpha) * CE(分类头, y) + alpha * 蒸馏项(蒸馏头, 教师信号)
   # 硬蒸馏: 蒸馏项 = CE(dist_head, argmax(Z_t))        ← alpha=0.5, 无温度
   # 软蒸馏: 蒸馏项 = tau^2 * KL(softmax(Z_s/tau) || softmax(Z_t/tau))  ← tau=3.0
   ```

3. **Mixup/CutMix 与蒸馏共存**：教师只在**干净图像**上前向一次，然后把教师 logits 用与
   图像相同的 `λ` 做线性混合，作为混合样本的教师目标。这样既保持了"教师看过真实图像"
   的一致性，又避免每个 batch 对教师前向两次（省一半算力）。

4. **权重衰减只作用于权重、不作用于 bias/LayerNorm**（AdamW 分组参数），与论文一致，
   对 Transformer 精度有可感知的影响。

5. **可复现性**：固定所有随机种子 + dataloader worker 种子。

## 4. 如何运行

**环境**：本机 conda 环境 `ssl_cv`（Python 3.10, torch 2.11 + CUDA）；CIFAR-100 数据放在
`D:\project\self_supervised_learning\data`（代码默认值，无需改参数）。教师只训练一次并
保存到 `runs/teacher_cnn_cifar100.pth`，后续实验自动复用（可用 `--teacher-path` 指定其它路径）。

```bash
# 0) 激活环境 (或用绝对路径直接调 python)
conda activate ssl_cv

# 1) 快速冒烟测试 (micro 模型, 1 epoch, 验证代码通路)
python deit_cifar100.py --model micro --epochs 1 --teacher-epochs 1

# 2) 标准训练: DeiT-Ti + 硬蒸馏 (论文默认配置), 100 epochs
python deit_cifar100.py --model tiny --distill hard --epochs 100

# 3) 消融 A: 不蒸馏 (纯 ViT 基线)
python deit_cifar100.py --model tiny --distill none --epochs 100

# 4) 消融 B: 软蒸馏
python deit_cifar100.py --model tiny --distill soft --tau 3.0 --epochs 100

# 5) 更大模型 / 更长训练
python deit_cifar100.py --model small --distill hard --epochs 300 --drop-path 0.1
```

## 5. 预期结果

单卡（如 RTX 3060/3090）实测量级：

| 配置 | 参数 | 预期 CIFAR-100 Top-1 |
|---|---|---|
| TeacherCNN（教师） | ~0.5M | 65 ~ 70% |
| DeiT-Ti 无蒸馏（ViT 基线） | ~5.3M | 66 ~ 72% |
| DeiT-Ti + 硬蒸馏 | ~5.3M | **68 ~ 74%**（比基线高 1~3 点） |
| DeiT-Ti + 软蒸馏 | ~5.3M | 与硬蒸馏接近 |
| DeiT-S + 硬蒸馏 + drop-path | ~21.7M | 74 ~ 78% |

**对照论文结论的预期**：ImageNet 上蒸馏带来 +1.6（81.8→83.4）。CIFAR-100 教师较弱且
师生结构差异大，预期 **+1~3 点**；更长的训练（300 epochs）、更强的教师（如 WRN）可进一步
拉开差距。蒸馏的收益主要出现在**训练后期**（学生自己学不动的地方教师帮它兜底）。

## 6. 如果结果不如预期，怎么排查 / 改进

1. **蒸馏没涨点**：
   - 确认教师精度足够（≥65%）；教师太弱会传噪声——先单独训好教师；
   - 调 `--alpha`（0.3~0.7）、软蒸馏 `--tau`（1~6）；
   - 训练更长（200~300 epochs），蒸馏收益在后期才显现。
2. **模型欠拟合 / 过拟合**：
   - 欠拟合：加大 `--model`、降 `--drop-path`、提高 `--lr`；
   - 过拟合：开 `--drop-path 0.1`、提高 mixup/cutmix、加 `--re-prob`。
3. **训练不稳定**：检查 warmup（默认 5 epoch）与余弦调度；batch 减半时 lr 按比例缩小
   （论文的线性缩放规则 `lr=1e-3*batch/512`）。
4. **显存不足**：`--model tiny --batch-size 64 --workers 2`。

## 7. 后续可扩展方向

- 教师换成更强模型（WideResNet-28-10, 预训练 ResNet18）验证"教师越强学生越强"；
- **EMA（指数滑动平均）权重** ✅ 已加入参考版 `deit_cifar100.py`（`--ema` 开关，2026-08-28），
  通常再 +0.5~1 点；
- 尝试 **Repeated Augmentation（RA3）**，论文消融中它贡献很大；
- 蒸馏温度/权重做小网格搜索，画出"蒸馏收益 vs 训练轮数"曲线；
- 迁移到 ImageNet 官方配置复现论文 Table 1（用 timm 的 `deit_tiny_patch16_224` 对照）；
- 学习主线已进入 v2（Mixup/CutMix/RandAugment/EMA + 消融实验），路线见
  `DeiT_v2学习路线.md`。
