# DeiT v2 学习路线 —— 从 63.27% baseline 到论文配方

> 背景：你的 v1（`deittrain.py`，无 Mixup/CutMix/RandAugment）跑出 best 63.27%。
> 这份文档是下一步的学习地图：**先固化 baseline，再逐个学训练技巧，最后做真正的消融实验**。

---

## 0. 我的态度：GPT 这份计划 90% 采纳

这份计划质量很高，方向、顺序、实验纪律都对。我补充两点修正：

1. **EMA 不是 DeiT 论文内容**——它是 timm 等仓库的工程技巧（论文里没有）。当 bonus 学没问题，
   但别把它算进"论文配方"（论文 Sec 4.2 的配方是 RandAugment/Mixup/CutMix/RE/**重复增强 RA3**）；
2. GPT 清单漏了**重复增强（Repeated Augmentation, RA3）**——它才是论文消融里贡献最大的
   增强之一，v2 后期可以补上。

## 1. 第一步：固化 baseline（先记录，再动手）

```
Baseline (v1, 已完成)
─────────────────────────────
Model:        DeiT-Tiny (5.4M, 蒸馏双头)
Dataset:      CIFAR-100
Distillation: Hard (alpha=0.5)
Augmentation: 仅 RandomCrop + RandomFlip（无 Mixup/CutMix/RandAugment）
Optimizer:    AdamW (lr=5e-4, wd=0.05)
Scheduler:    warmup 5 + 余弦
AMP/梯度裁剪/tqdm/早停/续跑: ✅
Best Test Acc ≈ 63.27%（epoch 79，早停于 89）
```

**你训练末期的真实数据（epoch 88）——这就是 v2 的动机：**

| | train | test | 说明 |
|---|---|---|---|
| loss | 0.73 | 1.41 | 测试损失约为训练的两倍 |
| acc | **97.3%** | **63.2%** | 34 个点的缺口 = 严重过拟合 |

> 💡 把这张表记在心里：v2 每一个技巧加入后，你要观察的是
> **"train_acc 是否下降、test_acc 是否上升"**——如果发生，说明技巧在起作用。

以后所有实验都跟 63.27% 比。每次只改一个变量。

---

## 2. 学习顺序总览

```
① Mixup ──→ ② CutMix ──→ ③ RandAugment ──→ ④ EMA (bonus)
        ↓
⑤ 消融实验 (1~5) ──→ ⑥ 蒸馏对照 (ViT/none/hard/soft) ──→ ⑦ Teacher quality
        ↓
⑧ 回到 SSL 论文复盘
```

---

## 3. ① Mixup（第一优先级）

### 3.1 动机：你的 97%/63% 缺口

模型把训练集"背下来了"（97%），但没学到可迁移的规律（63%）。Mixup 的解法：
**不让模型看到任何"原始"训练样本**，看到的全是两张图的插值，死记硬背就失效了。

### 3.2 公式（论文 Sec 4.2，α=0.8）

```
λ ~ Beta(α, α)                      # α=0.8 时 λ 大多落在 0.3~0.7 附近
x' = λ·x_i + (1-λ)·x_j              # 图像线性混合
y' = λ·y_i + (1-λ)·y_j              # 标签也按同一比例混合 → 软标签 (B, C)
```

- **Beta 分布直觉**：α 越大，λ 越集中在 0.5（两图对半混）；α→0 退化成"几乎不混"；
- 一张"0.7 猫 + 0.3 狗"的图，正确标签就是"0.7 猫、0.3 狗"——模型被迫学会输出**连续的概率**，
  而不是死记 one-hot，决策边界被强制拉平滑。

### 3.3 它正好接上你刚学的 soft_cross_entropy

你 `deitloss.py` 里的注释早写了："(B, C) 软标签（v2 上 Mixup 时才需要软标签）"——**现在就是
用它的时刻**。链路：

```
Mixup 产生混合图 + 混合标签 (B, C)
        ↓
soft_cross_entropy(cls_logits, y_mixed, smoothing)   # 2D 分支自动生效, 不用改 loss!
        ↓
学生分类损失
```

你的 2D 软标签分支在冒烟测试里已经验证过（`_check_user_files.py`），v1 埋的伏笔现在兑现。

### 3.4 自己实现的要点（先写伪代码，再对照参考版）

```python
def mixup_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)              # ① 采样混合比例
    idx = torch.randperm(x.size(0))                 # ② 随机配对 (打乱序)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam, idx

# 标签混合成软标签:
target = lam * F.one_hot(y_a, C) + (1 - lam) * F.one_hot(y_b, C)
```

**蒸馏共存（关键）**：教师和学生看**同一张混合图**——先混合，再让教师前向
`T(X_mixed)`（精确）；不要用旧的"干净图 T(X) + 人工混 logits"做法（那是线性近似，
CutMix 下误差明显）。完整对照见参考版 `deit_cifar100.py` 的
`mixup_data / mix_target / train_one_epoch`。

### 3.5 验收

- 初始 loss 仍在 ≈ln100 附近；训练中 `train_acc` 应该**明显下降**（可能掉到 70~85%），
  `test_acc` 上升——**训练集精度下降是好事**；
- 对照实验：Baseline 63.27% vs Baseline+Mixup ？？%（预期 +1~3 点）。

---

## 4. ② CutMix（紧接着学，非常自然）

### 4.1 与 Mixup 的差异

- Mixup：**全局**混合——两张图各占一个透明度，合成的图可能很"糊"，丢失局部纹理；
- CutMix：**局部拼贴**——从图 A 剪一块矩形贴到图 B，保留两块区域的真实纹理，
  同时强迫模型关注"被遮挡后仍可辨别的部分"（更像真实遮挡）。

```
        A(猫)                B(狗)                CutMix 结果
     ┌──────────┐        ┌──────────┐        ┌──────────┐
     │  🐱🐱🐱  │        │  🐶🐶🐶  │        │  🐱🐱🐶  │
     │  🐱🐱🐱  │   →    │  🐶🐶🐶  │   →    │  🐱🐱🐶  │  标签: λ=贴入面积占比
     │  🐱🐱🐱  │        │  🐶🐶🐶  │        │  🐶🐶🐶  │  y = λ·y_A + (1-λ)·y_B
     └──────────┘        └──────────┘        └──────────┘
```

### 4.2 公式（论文 Sec 4.2，α=1.0）

```
λ ~ Beta(1, 1)                      # α=1 即均匀分布
rx, ry ~ U(0, W), U(0, H)           # 剪切框中心
rw = W·sqrt(1-λ),  rh = H·sqrt(1-λ) # 框的尺寸由 λ 决定
贴完后: λ = 1 - 框面积/图面积        # 修正 λ 为"保留下来的面积比例"
```

### 4.3 实现要点

- 用 `rand_bbox` 函数采框：中心 + 边长，裁剪到图像边界内；
- 标签混合方式与 Mixup **完全相同**（又是 2D 软标签 → 又是 `soft_cross_entropy`）——
  这就是为什么学完 Mixup 再学 CutMix 特别顺；
- 参考实现：`deit_cifar100.py` 的 `cutmix_data`。

### 4.4 验收

- 单独加 CutMix 的对照实验（α=1.0）；再跑 Mixup+CutMix 组合（timm 惯例：每批以 0.5 概率
  二选一，mixup α=0.8、cutmix α=1.0——参考版就是这么做的）。

---

## 5. ③ RandAugment

### 5.1 核心问题（比背操作列表重要得多）

**为什么 Transformer 比 CNN 更需要强增强？** CNN 自带局部性/平移不变性的归纳偏置，
天然"少看一些数据也能泛化"；Transformer 没有这些先验，所有规律都要从数据里学——
数据多样性不足，它就记住训练集的细节（你的 97%/63% 就是证据）。增强 = **人工扩大数据多样性**，
把 5 万张图变成"5 万 × 无数种变换"。

### 5.2 思想（一句话）

从 14 个基础操作（Rotate/Shear/Translate/Color/Contrast/Sharpness/Posterize/Solarize/
Equalize/AutoContrast/Brightness/Identity/...）里**随机选 n 个**、每个用**统一的幅度 m**，
比 AutoAugment 的"每个操作单独搜幅度"简单得多，效果却不差。

### 5.3 论文设置逐项解释：`rand-m9-mstd0.5-inc1`

| 记号 | 含义 | 你实现时 |
|---|---|---|
| n=2 | 每张图随机选 2 个操作 | 可先只实现 5~6 个操作起步 |
| m=9 | 幅度 9（上限 30） | CIFAR 上用 9 即可 |
| mstd=0.5 | 幅度按 ±0.5 扰动 | `np.random.uniform(m-0.5, m+0.5)` |
| inc=1 | 每个 epoch 幅度 +1 | 训练越来越"狠"，对应后期防过拟合 |

### 5.4 验收

- 先单独加 RandAugment 跑对照（不加 Mixup/CutMix）；
- 参考实现：`deit_cifar100.py` 的 `RandAugmentCIFAR`（可直接读，建议自己先写个简版）。

---

## 6. ④ EMA（bonus，不在论文里）

### 6.1 公式与直觉

```
θ_EMA ← m·θ_EMA + (1-m)·θ_t      # m ≈ 0.999 (每步) 或 epoch 级 0.99+
```

每一轮把当前参数和历史平均参数做一次加权平均，得到一条**平滑的参数轨迹**。
直觉：训练末期的参数在最优解附近**抖动**（lr 余弦退到很小前的振荡、个别 batch 的噪声），
"最后一个 checkpoint"可能恰好落在抖动的高点上；EMA 权重是整条轨迹的平均，更稳、泛化通常更好。

### 6.2 要点

- 推理（evaluate）时**用影子权重**，训练梯度照常用原始权重；
- 开启 EMA 后，选最优 checkpoint 也应以 EMA 精度为准；
- 参考实现：`deit_cifar100.py` 已加入 `ModelEma` + `--ema/--ema-decay`（本轮 AI 添加，
  带标记），读它的 `update/state_dict/主循环` 三处即可。

### 6.3 验收

- Baseline+EMA 对照：预期 +0.3~1 点，且训练曲线末端抖动变小。

---

## 7. ⑤ 消融实验（这一步才是真学习，别跳）

一次只改一个变量、同 seed、记录 config（你的 runs/ 已自动记录）。**填空并观察**：

| Experiment | 配置 | train_acc | test_acc | train_loss | test_loss |
|---|---|---|---|---|---|
| 1 | Baseline | 97.3% | 63.27% | 0.73 | 1.41 |
| 2 | + Mixup | 67.74% | 66.45% | 1.2509 | 2.0529 |
| 3 | + CutMix | 72.34% | 67.90% | 1.5192 | 2.0163 |
| 4 | + RandAugment | ? | ? | ? | ? |
| 5 | + Mixup + CutMix | 61.97% | 67.12% | 1.6037 | 2.0166 |
| 6 | + 全部 (≈论文配方) | ? | ? | ? | ? |

> 口径说明：test_acc 列为 **best**（最优轮）；train_acc/loss 为最后 epoch 值。
> 行 3 原填 67.59 是"最后一轮"的 0.6759，best 实为 67.90，已按 runs 日志修正。

**最值得观察的现象**：加正则后 `train_acc` 下降、`test_acc` 上升——看到这个，你就真正理解了
"训练集准确率下降反而是好事"（模型不再背答案，开始学规律）。参考论文：ImageNet 上这些增强
合计贡献几个点，其中 RandAugment 与重复增强贡献最大。

---

## 8. ⑥⑦ 蒸馏对照 + Teacher quality

### 8.1 四格对照（`deit_cifar100.py` 一条命令一个格子）

| Model | Distill | Test Acc |
|---|---|---|
| ViT-Tiny（distilled=False） | ❌ | ? （`--distill none`） |
| DeiT-Tiny | ❌ | ? （同上，无蒸馏基线） |
| DeiT-Tiny | Hard | **63.27%**（你的 v1 等价物，参考版带增强） |
| DeiT-Tiny | Soft | ? （`--distill soft --tau 3.0`） |

回答两个问题：**蒸馏到底有没有用？hard 和 soft 差多少？**（论文答案：+1.6、几乎一样，
你自己测出来的数字才是你的知识。）

### 8.2 Teacher quality（漂亮的小研究）

用**同一个 TeacherCNN 训练不同 epoch 数**（如 10/20/30/40）得到不同档位教师（约 60%/66%/69%/71%），
固定学生配置分别蒸馏，画出"教师精度 → 学生精度"曲线。观察：教师太弱 → 噪声信号；
教师合适 → 有价值；教师继续变强 → 学生可能受容量瓶颈（曲线变平）。**这已经是研究思维了。**

---

## 9. ⑧ 回到 SSL：论文复盘问题清单（GPT 第 4 阶段）

v2 做完后回看 SSL，不再问"这 loss 怎么写"，而是问"为什么这么设计"：

| 方法 | 复盘问题 |
|---|---|
| SimCLR | 为什么大 batch？temperature 为什么关键？为什么 projection head？为什么增强是核心？ |
| MoCo | 为什么 queue？为什么 momentum encoder？ |
| BYOL | 为什么不需要负样本？为什么 EMA teacher？ |
| DINO | 为什么 teacher/student？centering/sharpening 各解决什么？为什么 multi-crop？ |
| MAE | 为什么 mask？为什么 encoder 只吃可见 patch？为什么 decoder 可以很轻？ |

---

## 10. 一句话总结

**把 63.27% 当成起点而不是终点。** 下一步只做一件事：学 Mixup → 自己在 `deittrain.py` 里
实现 → 跑对照 → 看 train_acc 降/test_acc 升。每学一个技巧都回填第 7 节的消融表；
参考版 `deit_cifar100.py`（现已含 EMA 的完全体）是你实现卡住时的对照，不是抄写的对象。
