# 模块 06 · Patch Merging 数学推导

> 学习顺序：`06_patch_merging` → `07_swin_block` → `08_basic_layer`
>
> 本文档给出 Patch Merging 从动机到数值例的完整推导链。

---

## 1. 为什么需要 hierarchical representation（层级化表示）

### 1.1 ViT 的问题：全程固定分辨率

Vision Transformer（ViT）从第一层到最后一块 Transformer block，特征图始终是
$\frac{H}{16} \times \frac{W}{16}$ 个 patch，通道数也始终是 $C$。这带来两个后果：

1. **感受野靠堆注意力，但计算量巨大**：为了让高层 token 能“看到”整图，
   需要全局注意力，复杂度 $O(N^2)$，$N$ 是 patch 数，对大图不可承受。
2. **缺少多尺度特征**：检测、分割等任务需要不同粒度特征，而固定分辨率只给一种粒度。

### 1.2 Swin 的答案：仿照 CNN 的金字塔

CNN（如 ResNet）用 `pooling + channel doubling` 逐级降采样：

| 阶段 | 分辨率（ImageNet 224 输入） | 通道 |
|---|---|---|
| stage1 | 56×56 | 96 |
| stage2 | 28×28 | 192 |
| stage3 | 14×14 | 384 |
| stage4 | 7×7 | 768 |

Swin Transformer 在每个 stage 之后插入 **Patch Merging**，把分辨率减半、通道翻倍：

$$
(H, W, C) \;\longrightarrow\; \left(\frac{H}{2}, \frac{W}{2}, 2C\right)
$$

这等价于 CNN 的 `stride=2 pooling` + 通道翻倍，从而：

- **分辨率逐级减半** → token 数减为 $1/4$，计算量下降。
- **通道逐级翻倍** → 每个 token 表达能力增强，信息“浓缩”进更多通道。
- **感受野逐级扩大** → 同样的窗口大小，在更粗的分辨率上覆盖更大的原图范围，
  形成金字塔感受野。

> 关键直觉：**“分辨率减半、通道翻倍”是对信息量守恒的表达**——空间信息被折叠进
> 通道维，总信息量大致不变，但表示从“空间密集”转为“通道密集”。

---

## 2. 2×2 分组的空间语义

### 2.1 原理

输入特征图大小为 $H \times W$，把每个像素位置按 **(行奇偶, 列奇偶)** 分成四类：

- 行偶、列偶 → 左上（top-left）
- 行奇、列偶 → 左下（bottom-left）
- 行偶、列奇 → 右上（top-right）
- 行奇、列奇 → 右下（bottom-right）

四类各取一半行、一半列，各自形成 $\frac{H}{2} \times \frac{W}{2}$ 的网格，
再把这四个网格**沿通道维拼接**，得到 $\frac{H}{2}\times\frac{W}{2}\times 4C$。

### 2.2 ASCII 图

```
原特征图 H=W=4（每个格子是一个位置）：
      c0    c1    c2    c3
 r0   A0    A1    B0    B1
 r1   A2    A3    B2    B3
 r2   C0    C1    D0    D1
 r3   C2    C3    D2    D3

按 (行奇偶, 列奇偶) 拆成 4 路：
  A(偶偶)=左上     B(偶奇)=右上
      A0 A1            B0 B1
      A2 A3            B2 B3
  C(奇偶)=左下     D(奇奇)=右下
      C0 C1            D0 D1
      C2 C3            D2 D3

沿通道拼接后，每个新位置 (i,j) 的 4C 维 = [A | B | C | D] 的对应位置：
  新(i,j) 通道段 [ x0(左上) | x1(左下) | x2(右上) | x3(右下) ]
```

即新位置 $(i,j)$ 收集了旧图 $2\times2$ 邻域 $\{(2i,2j),(2i{+}1,2j),(2i,2j{+}1),(2i{+}1,2j{+}1)\}$。

### 2.3 与 pixel shuffle（亚像素卷积）互为逆过程

- **Pixel shuffle（sub-pixel convolution）**：上采样时把 $r^2 C$ 通道重排成
  $rH \times rW \times C$，是“通道 → 空间”。
- **Patch Merging**：下采样时把 $2\times2$ 空间邻域折叠进通道，是“空间 → 通道”。

两者互为逆操作（忽略可学习的线性投影）：pixel shuffle 放大分辨率，Patch Merging
缩小分辨率。这也是为什么 Patch Merging 常被称作“反向 pixel shuffle”。

---

## 3. 为什么是 4C → 2C，而不是 4C → C

拼接后每个 token 有 $4C$ 维，之后用一个线性层降维。目标通道数有几种选择：

1. **降到 $C$**：信息被过度压缩，通道没有翻倍，违背“层级化”的目标。
2. **保持 $4C$**：通道翻两番，参数量和计算量爆炸（$4C\cdot4C=16C^2$），且后续 stage
   通道过快膨胀。
3. **降到 $2C$（Swin 的选择）**：在“信息混合”与“通道翻倍”之间取得平衡。

降维到 $2C$ 的含义：

- $4C$ 维输入里既有原通道、又有空间邻域信息，Linear 做**跨通道加权混合**，
  相当于一次可学习的“聚合”。
- 输出 $2C$ 恰好是输入的 2 倍，与 CNN 的 channel doubling 节奏一致，保证各 stage
  通道按 $96 \to 192 \to 384 \to 768$ 演进。

---

## 4. 参数量推导

Patch Merging 包含两个可学习模块：

1. `LayerNorm(4C)`：权重 $\gamma \in \mathbb{R}^{4C}$、偏置 $\beta \in \mathbb{R}^{4C}$，
   共 $2\cdot4C = 8C$ 个参数。
2. `Linear(4C → 2C, bias=False)`：权重 $W \in \mathbb{R}^{2C \times 4C}$，
   共 $4C \cdot 2C = 8C^2$ 个参数。

总参数量：

$$
\#\text{params} = 8C + 8C^2
$$

数值例（$C=96$，Swin-Tiny 第一个 Patch Merging）：

$$
\#\text{params} = 8\cdot96 + 8\cdot96^2 = 768 + 73728 = 74496
$$

其中 Linear 的 $8C^2$ 占绝对主导（$\approx 99\%$），LN 的参数可忽略。

---

## 5. 计算量（FLOPs / MACs）推导

Patch Merging 的可学习计算主要来自 `Linear(4C → 2C)`（LN 的归一化开销相对小，
此处按矩阵乘估算）。记降采样后的 token 数为

$$
h'w' = \frac{H}{2}\cdot\frac{W}{2} = \frac{HW}{4}
$$

对**每个 token** 做一次矩阵乘：输入 $4C$ 维，输出 $2C$ 维。

- 乘法次数：$4C \times 2C = 8C^2$
- 加法次数：约 $4C \times 2C$（矩阵乘的累加）
- 一次 MAC（乘加）≈ 一次乘法 + 一次加法，故每个 token 约 $8C^2$ 次 MAC。

总 MACs（batch $B$）：

$$
\text{MACs} = B \cdot h'w' \cdot 8C^2 = B \cdot \frac{HW}{4} \cdot 8C^2 = 2 B \cdot HW \cdot C^2
$$

通常记 $\text{FLOPs} \approx 2\times\text{MACs}$，即：

$$
\text{FLOPs} \approx 4 B \cdot HW \cdot C^2
$$

> 任务描述中的 “FLOPs $2 \cdot hw \cdot C^2$” 指的是 **MACs（乘加数）**，
> 其中 $hw = \frac{H}{2}\cdot\frac{W}{2}$ 是输出 token 数：
>
> $$
> \text{MACs} = h'w' \cdot 8C^2 = \frac{HW}{4}\cdot 8C^2 = 2\,HW\,C^2
> $$
>
> 两边记法一致（$HW$ 为输入 token 数，$h'w'$ 为输出 token 数）。

数值例：

1. **$C=96$，输入 $56\times56$（Swin-Tiny stage1 → stage2）**：
   输出 $h'w' = 28\times28 = 784$，
   $$
   \text{MACs} = 1 \cdot 784 \cdot 8\cdot96^2 = 784 \cdot 73728 \approx 5.78\times10^{7}
   $$

2. **小例 $C=2$，输入 $4\times4$**：
   输出 $h'w' = 4$，
   $$
   \text{MACs} = 1\cdot4\cdot8\cdot2^2 = 128
   $$

可见 Patch Merging 的计算量与 $C^2$ 成正比，随 stage 通道翻倍，每个 token 的
降采样开销也成 4 倍增长；但由于分辨率同时减半，token 数降为 $1/4$，两者抵消，
总计算量在不同 stage 大致持平。

---

## 6. 为什么先 LayerNorm 再 Linear（LN 作用在 4C 维）

执行顺序是：

$$
\underbrace{\text{4 路拼接}}_{4C} \;\xrightarrow{\;\text{LayerNorm}\;} \;\underbrace{\text{Linear}}_{4C\to2C}
$$

理由：

1. **稳定输入分布**：拼接后的 $4C$ 维由 4 个来源拼成，量纲/分布可能不一致，
   LN 先归一化，让后续 Linear 的输入有稳定均值与方差，利于优化。
2. **LN 作用在 4C 维**：`nn.LayerNorm(4C)` 对每个 token 的 $4C$ 个通道做归一化，
   是 **per-token、per-channel 维** 的归一化，不跨 token，因此与序列长度无关。
3. **先归一化再投影**：Linear 负责“混合 + 降维”，LN 负责“归一化”，顺序上先
   归一化再混合，符合 Swin 整体 **pre-norm** 的设计哲学（在变换前归一化）。

> 与 post-norm 对比：若把 LN 放在 Linear 之后，则投影后的 $2C$ 维才是归一化对象，
> 归一化在信息已经混合、降维之后发生，梯度路径更长、稳定性略差。

---

## 7. 完整推导链小结

```
输入 x: (B, H*W, C)
   │  view(B, H, W, C)                         # 恢复 2D
   │  [可选] 奇数尺寸 pad 到偶数
   │  x0,x1,x2,x3 = 按(行奇偶,列奇偶)切分       # 4 路，各 (B, H/2, W/2, C)
   │  cat([x0,x1,x2,x3], dim=-1)               # (B, H/2, W/2, 4C)
   │  view(B, (H/2)*(W/2), 4C)                 # 展平空间维
   │  LayerNorm(4C)                            # per-token 归一化
   │  Linear(4C -> 2C, bias=False)             # 混合 + 降维 + 通道翻倍
   ▼
输出 y: (B, (H/2)*(W/2), 2C)
```

数学表达式（对第 $i$ 个新 token）：

$$
y_i = W_{\text{red}}\;\text{LayerNorm}\Big(
\big[\; x^{(00)}_{i},\; x^{(10)}_{i},\; x^{(01)}_{i},\; x^{(11)}_{i} \;\big]
\Big)
$$

其中 $x^{(pq)}_i$ 表示新 token $i$ 的 $2\times2$ 邻域中行奇偶为 $p$、列奇偶为 $q$
的旧 token 的 $C$ 维向量，$[\cdot]$ 表示沿通道维拼接，$W_{\text{red}}\in\mathbb{R}^{2C\times4C}$。

---

## 8. 数值例总表

| 项目 | 符号 | $C=96$（56→28） | 小例 $C=2$（4→2） |
|---|---|---|---|
| 输入 token 数 | $HW$ | $56^2=3136$ | $4^2=16$ |
| 输出 token 数 | $h'w'$ | $28^2=784$ | $2^2=4$ |
| 拼接后通道 | $4C$ | 384 | 8 |
| 输出通道 | $2C$ | 192 | 4 |
| LN 参数量 | $8C$ | 768 | 16 |
| Linear 参数量 | $8C^2$ | 73728 | 32 |
| 总参数量 | $8C+8C^2$ | 74496 | 48 |
| MACs | $h'w'\cdot8C^2$ | $\approx5.78\times10^7$ | 128 |
