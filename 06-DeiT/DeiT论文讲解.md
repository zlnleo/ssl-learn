# DeiT 论文完整讲解

> **论文**：《Training data-efficient image transformers & distillation through attention》
> **作者**：Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou（Facebook AI）
> **发表**：ICML 2021（arXiv:2012.12877, 2020.12）
> **官方代码**：https://github.com/facebookresearch/deit
> **huggingface 版**：https://huggingface.co/docs/transformers/model_doc/deit

---

## 0. 一句话概括

DeiT 证明了：**不需要 JFT-300M 这种巨型外部数据集，只用标准 ImageNet-1k（128 万张图），通过「更强的训练策略 + 知识蒸馏」，ViT 结构也能训练出媲美甚至超越卷积网络的效果**，而且在一台 8×V100 的机器上 3 天内就能训完。

| 论文核心矛盾 | ViT 的困境 | DeiT 的回答 |
|---|---|---|
| 数据 | ViT 依赖 JFT-300M 预训练 | 只用 ImageNet-1k，靠更强的数据增强与正则 |
| 计算 | ViT 训练需 TPU 集群 | 单机 8×V100，< 3 天 |
| 归纳偏置 | Transformer 缺少卷积的局部性先验 | 用卷积教师做**知识蒸馏**，把归纳偏置"蒸馏"进学生 |

---

## 1. 背景与动机

### 1.1 ViT 的回顾与问题

[ViT](https://arxiv.org/abs/2010.11929)（An Image is Worth 16x16 Words, 2020.10）把图像切成 16×16 的 patch，当作"单词"序列输入标准 Transformer Encoder：

- **Patch Embedding**：一张 224×224 的图切成 14×14=196 个 patch，每个 patch 线性投影成 768 维向量；
- **Class Token**：在序列最前面拼接一个可学习的 `[cls]` token，其最终输出用于分类；
- **位置编码**：可学习的一维位置编码；
- **12 层 Transformer Encoder**（ViT-B 约 86M 参数），无卷积归纳偏置。

ViT 的问题：**从头在 ImageNet 上训练效果差**，必须先在 JFT-300M（3 亿张 Google 内部图片）上预训练，再在 ImageNet 上微调才能达到 84%+ 的精度。这种巨型数据集普通人拿不到，训练成本极高。

### 1.2 DeiT 要回答的问题

1. **数据效率**：没有外部数据，只有 ImageNet-1k，Transformer 还能训吗？怎么训？
2. **训练效率**：如何让训练成本降低到"单机、几天"的量级？
3. **蒸馏机制**：知识蒸馏怎么和 Transformer 结合？蒸馏能不能代替大数据预训练，把卷积网络的归纳偏置（局部性、平移不变性）注入 Transformer？

### 1.3 DeiT 的核心结论

只用 ImageNet-1k：

- **DeiT-B**（86M 参数）达到 **81.8%** top-1，超越同规模卷积网络（如 RegNetY-8GF 约 83%？——注：卷积网络在此规模普遍 80%+）；
- **DeiT-B⚗（distilled，87M）** 达到 **83.4%**，逼近用 JFT-300M 预训练的 ViT-B（约 84.2%），且推理吞吐量是 ViT-B 的数倍。

---

## 2. 核心贡献（4 点）

1. **蒸馏 token（Distillation Token）**：在 class token 旁新增一个蒸馏 token，全程参与自注意力，最后接第二个分类头专门接收教师信号。测试时两个头取平均。**不改变架构主流程、不加额外卷积、几乎零额外成本**。
2. **蒸馏形式**：提出 **硬蒸馏（hard distillation）**——直接用教师预测的类别作为真值标签训练学生，简单有效，与软蒸馏效果相当甚至更好。
3. **训练配方（recipe）**：把卷积网络里成熟的增强/正则手段系统性地搬到 Transformer 上：RandAugment、Mixup、CutMix、Random Erasing、重复增强（Repeated Augmentation）、Stochastic Depth 等。
4. **效率**：DeiT-B 在 8×V100 单机 3 天内训完，吞吐量比同规模 ViT 高（DeiT-B 224 输入约 292 张/秒）。

---

## 3. 方法详解

### 3.1 更强的训练策略（论文 Sec. 4.1 / 4.2）

DeiT 认为：**ViT 在 ImageNet 上训不好，主要是"欠正则化/欠增强"，而不是架构不行**。于是把卷积网络的训练技巧全部移植过来：

#### (a) 数据增强五件套

| 增强方法 | 论文设置 | 作用 |
|---|---|---|
| **RandAugment** | `rand-m9-mstd0.5-inc1`（2 个操作，幅度 9，逐 epoch 递增 1） | 自动搜索空间里的强随机增强 |
| **Mixup** | α = 0.8 | 样本对线性混合，软化标签 |
| **CutMix** | α = 1.0 | 图像块级混合，更强的空间正则 |
| **Random Erasing** | p = 0.25 | 随机遮挡图像块，防止过拟合局部 |
| **Repeated Augmentation** | RA3（同一张图 3 个增强副本） | 让模型每 epoch 见到更多增强多样性 |

消融实验显示：这些增强缺一不可，其中 **RandAugment 与 Repeated Augmentation 贡献最大**。

#### (b) 优化器与调度

- 优化器：**AdamW**，权重衰减 0.05（注意：不衰减 bias 与 LayerNorm 参数）；
- 学习率：**线性缩放** `lr = 0.001 × batch_size / 512`，DeiT-B 用 batch 1024 → lr = 2e-3；
- 调度：**余弦衰减**，前 **5 个 epoch 线性 warmup**；
- 训练时长：**300 epochs**；
- 标签平滑：smooth CE，ε = 0.1；
- **Stochastic Depth（drop path）**：DeiT-B 用 0.1，即每层以一定概率随机丢弃整层残差分支。

> 💡 对比：ViT 原论文只用"预训练 → 微调"两阶段大管道；DeiT 把这些正则手段组合起来后，ImageNet 上从头训练 Transformer 就能成功。

### 3.2 知识蒸馏（论文 Sec. 3.2，论文最核心的创新）

#### 为什么需要蒸馏？

Transformer 没有卷积的归纳偏置，ImageNet 这种"小数据集"上从头学比较吃力。DeiT 的想法：**让一个强的卷积教师（RegNetY-16GF，ImageNet 约 84.2%）通过蒸馏，把卷积的归纳偏置传递给学生 Transformer**。这样学生"不需要亲自在 3 亿张图上总结规律，直接从教师那里继承"。

#### 蒸馏 token（Distillation Token）

输入序列变成：

```
[class_token] [distillation_token] [patch_1] [patch_2] ... [patch_196]
```

- 蒸馏 token 与 class token 一样是可学习参数（截断正态初始化），拼在序列最前；
- 它**全程参与所有层的自注意力**（不是只在最后接进去），与其它 token 充分交互；
- 最后一层出来后，**class token 走分类头 1**（真值监督），**蒸馏 token 走分类头 2**（教师监督）；
- **推理时**：两个头的 logits **取平均** 作为最终输出；
- 论文观察：随着训练进行，class token 与蒸馏 token 的余弦相似度越来越高——两者在学"互补但趋同"的表征。

> 对比其他蒸馏做法：大多数方法是"把教师输出作为学生最后输出的软目标"，DeiT 则是**在注意力流内部专门留出一条通道**接收教师信号，架构上更优雅。

#### 硬蒸馏 Hard Distillation（论文 Eq. 1）

教师先预测出硬标签 `y_t = argmax_c Z_t(x)`（教师输出最大的一类），然后学生用交叉熵去拟合这个硬标签：

```
L_global_hard = ½ · L_CE( ψ(Z_s), y ) + ½ · L_CE( ψ(Z_s), y_t )
```

- `ψ` = softmax；`Z_s` 学生 logits；`Z_t` 教师 logits；`y` 真值；`y_t` 教师硬标签；
- 一半权重学真值，一半权重学教师标签，**简单得惊人，但效果很好**；
- 论文还指出：把教师的硬标签用温度 τ 软化后再当目标，等价于**一种"教师置信度感知"的标签平滑**——这解释了硬蒸馏为何有效：教师对"哪些类容易混淆"的信息编码在了硬标签与置信度里。

#### 软蒸馏 Soft Distillation（论文 Eq. 2）

```
L_global = (1 − λ) · L_CE( ψ(Z_s), y ) + λ · τ² · KL( ψ(Z_s/τ) ‖ ψ(Z_t/τ) )
```

- 教师与学生的输出都除以温度 **τ = 3.0** 后算 KL 散度，损失乘 τ² 保持梯度尺度；
- 蒸馏权重 **λ = 0.5**；
- 软蒸馏能传递教师完整的类别间相似性分布，但论文发现**硬蒸馏与软蒸馏效果几乎相同（硬蒸馏略好）**，且硬蒸馏实现更简单、无需调温度。

> 📌 工程细节（官方实现）：分类头 1 的 CE 损失乘 (1−α)，蒸馏头 2 的损失乘 α（α=0.5）；Mixup/CutMix 的软标签照常作用于头 1，教师则对混合后的输入前向一次。

### 3.3 Class Token vs 平均池化（论文 Sec. 4.3）

ViT 用 class token 分类；也可以去掉 class token，对全部 patch token 做全局平均池化（GAP）再分类。DeiT 实验结论：

- 无蒸馏时两者接近（class token 81.8 vs GAP 约 81.5）；
- **有蒸馏时 class token 明显更好**（83.4 vs 约 83.0）——蒸馏 token 机制与 class token 协同更好。

---

## 4. 实验结果（论文 Table 1）

ImageNet-1k 验证集 top-1 精度，仅用 ImageNet 训练（除最后一行 ViT 用 JFT-300M）：

| 模型 | 参数 | 分辨率 | 吞吐(张/秒) | Top-1 |
|---|---|---|---|---|
| DeiT-Ti | 5M | 224² | 2536.5 | 72.2 |
| DeiT-S | 22M | 224² | 940.4 | 79.8 |
| DeiT-B | 86M | 224² | 292.3 | **81.8** |
| DeiT-Ti⚗ (蒸馏) | 6M | 224² | 2536.5 | 74.5 |
| DeiT-S⚗ (蒸馏) | 22M | 224² | 940.4 | 81.2 |
| DeiT-B⚗ (蒸馏) | 87M | 224² | 292.3 | **83.4** |
| DeiT-B ↑384 | 86M | 384² | 71.1 | 83.1 |
| DeiT-B⚗ ↑384 | 87M | 384² | 71.1 | **84.5** |
| ViT-B/16（JFT-300M 预训练） | 86M | 384² | 85.9 | ≈84.2 |

关键读数：

1. **同参数下蒸馏白赚 1.6 个点**：81.8 → 83.4，只多一个 token + 一个线性头；
2. **Ti（5M 参数）也能 72.2%**——小模型同样受益，蒸馏版 74.5%；
3. **384 分辨率微调再加 ~1 个点**：84.5% 已逼近 JFT 预训练的 ViT-B，而训练成本低一个量级；
4. **吞吐量**：DeiT-B 在 224 下 292 张/秒，而 JFT 版 ViT-B 在 384 下只有 85.9 张/秒。

---

## 5. 消融实验结论（定性总结）

> 注：以下差异数值为论文各表的近似值，精确数字请查原论文 Table 2–6。

| 消融问题 | 结论 |
|---|---|
| 训练组件逐个移除 | 增强组件（尤其 RandAugment、重复增强）贡献最大；移除蒸馏约掉 1.6 点 |
| 蒸馏 vs 标签平滑 | 真教师（约 +1.6）显著优于"把标签平滑当假教师"（约 +0.4~0.7），说明学到的不是单纯平滑 |
| 教师类型 | **卷积教师 > Transformer 教师**：用 DeiT-B 自己当教师几乎没有提升，说明蒸馏传递的是卷积的归纳偏置 |
| 硬 vs 软蒸馏 | 效果几乎一样（约 83.4 vs 83.2），硬蒸馏更简单，成为默认选择 |
| 教师质量 | 教师越强，学生越强（RegNetY-16GF 好于 RegNetY-8GF 等） |
| class token vs GAP | 蒸馏场景下 class token 更好 |

---

## 6. 讨论与局限

**为什么蒸馏对 Transformer 有效？**
- Transformer 缺少卷积的局部性/平移不变性先验，在小数据上容易学偏；
- 蒸馏把教师（卷积网络）"多年积累"的类别关系、难例模式直接传给学生的注意力流；
- 硬蒸馏可视为教师置信度加权的标签平滑，天然抑制学生过自信。

**局限：**
- 依然需要 ImageNet-1k 这个量级的数据，"数据高效"是相对 JFT-300M 而言；
- 依赖一个强教师；教师的质量决定学生的上限；
- 与同时代最强的卷积网络（EfficientNet 系）仍有差距；
- 蒸馏带来的提升（+1.6）在高分辨率微调后依然保留，但额外蒸馏 token 在无教师可用时是负担（虽然可关掉）。

**后续影响：** DeiT 的训练配方成了 Transformer 视觉模型的标准配方，被 Swin、CaiT、XCiT、BEiT、MAE 等广泛沿用；其 "recipe matters" 的观点直接影响了后续 ViT 训练的默认配置（timm 中 `deit_*` 系列）。

---

## 7. 论文公式与符号汇总

| 符号 | 含义 |
|---|---|
| `Z_s`, `Z_t` | 学生 / 教师输出的 logits |
| `ψ` | softmax |
| `y`, `y_t` | 真值标签 / 教师硬标签 `y_t = argmax_c Z_t(x)` |
| `τ` | 蒸馏温度（软蒸馏用 3.0） |
| `λ` / `α` | 蒸馏损失权重（0.5） |
| `L_CE` | 交叉熵 |
| `KL(p‖q)` | KL 散度 |

- 硬蒸馏：`L = ½·CE(ψ(Z_s), y) + ½·CE(ψ(Z_s), y_t)`
- 软蒸馏：`L = (1−λ)·CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ) ‖ ψ(Z_t/τ))`

---

## 8. 代码结构对应（连接本目录的 CIFAR-100 实现）

| 论文概念 | 本仓库 `deit_cifar100.py` 对应代码 |
|---|---|
| Patch Embedding（patch=4, 32×32 → 64 tokens） | `PatchEmbed` |
| 多头自注意力 | `Attention` |
| MLP + GELU | `Mlp` |
| Transformer Block（pre-LN + 残差） | `Block` |
| Stochastic Depth | `DropPath` |
| class token + 蒸馏 token + 双分类头 | `DistilledViT` |
| 卷积教师 | `TeacherCNN` + `train_teacher()` |
| 硬/软蒸馏损失 | `distillation_loss()` |
| RandAugment / Mixup / CutMix / Random Erasing | `RandAugmentCIFAR` / `mixup_data` / `cutmix_data` / `random_erasing` |
| AdamW + 余弦 + warmup | `main()` 训练循环 |

> 📌 2026-08-28 更新：仓库现在的主体是**手写模块化学习版** `deitmodel.py`（模型）、
> `deitloss.py`（损失）、`deitteacher.py`（教师）、`deittrain.py`（训练），上表对应的是
> 保留的完整增强参考版 `deit_cifar100.py`。两版对照阅读效果最好。

---

## 9. 参考链接

- 论文 arXiv：https://arxiv.org/abs/2012.12877
- 官方代码：https://github.com/facebookresearch/deit
- ar5iv 在线阅读版（含全部表格）：https://ar5iv.labs.arxiv.org/html/2012.12877
- HuggingFace Transformers DeiT 文档：https://huggingface.co/docs/transformers/model_doc/deit
- timm 实现：https://github.com/huggingface/pytorch-image-models（`models/deit.py`）
- ViT 论文：https://arxiv.org/abs/2010.11929
