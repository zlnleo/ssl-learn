# 模块 07 · Swin Block 数学推导

> 学习顺序：`06_patch_merging` → `07_swin_block` → `08_basic_layer`
>
> 本文档给出 Swin Block（W-MSA / SW-MSA 总装）的完整推导链与数值例。

---

## 1. 为什么需要窗口注意力（W-MSA）

### 1.1 全局自注意力的代价

标准 Transformer 自注意力，输入 $N$ 个 token，每个 token 对其它所有 token 计算注意力：

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

其中 $Q,K,V \in \mathbb{R}^{N \times d}$。$QK^\top$ 是 $N\times N$ 矩阵，计算复杂度为：

$$
O(N^2 d)
$$

对 $224\times224$ 图像、patch=4 时 $N = 56^2 = 3136$，$N^2 \approx 9.8\times10^6$；若用更细 patch
或更大图，$N^2$ 迅速爆炸。这就是 ViT 只能在大 patch（16×16）上做全局注意力的原因。

### 1.2 窗口化：把全局切成局部

W-MSA 把特征图切成 $M\times M$ 的窗口，只在**每个窗口内部**做注意力。窗口内 token 数
$M^2$，对每个窗口复杂度 $O(M^4 d)$，共 $\frac{HW}{M^2}$ 个窗口：

$$
O\!\left(\frac{HW}{M^2}\cdot M^4 d\right) = O(HW \cdot M^2 d)
$$

与 token 数 $N=HW$ 呈**线性**关系（$M$ 为常数），而非全局的 $O(N^2)$。这是 Swin 能在
高分辨率（56×56）上运行的根基。

### 1.3 窗口化的代价：跨窗口信息不流动

窗口之间互不交互，token 永远“困”在窗口里。为此引入 SW-MSA（见 §3），在相邻层之间
循环移位窗口，让信息跨窗口流动。

---

## 2. 窗口注意力与相对位置偏置

### 2.1 缩放点积注意力

记单窗口 token 数 $N_w = M^2$，每头维度 $d = C/h$（$h$ 为头数）。对窗口内：

$$
Q = X W_Q,\; K = X W_K,\; V = X W_V \qquad X\in\mathbb{R}^{N_w\times C}
$$

注意力矩阵（加入相对位置偏置 $B$）：

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right), \qquad
\text{out} = A V
$$

其中缩放因子 $\sqrt{d}$ 是为了防止点积随 $d$ 增大而方差变大、softmax 饱和。

### 2.2 为什么用相对位置偏置而不是绝对位置编码

ViT 用绝对位置编码（可学习或正弦），把位置信息加进输入。Swin 选择**相对位置偏置**：

- 窗口内 token 的**相对位置种类有限**：行差 $dx$、列差 $dy$ 都在 $[-(M-1), M-1]$，
  共 $(2M-1)^2$ 种；
- 相对位置天然具有**平移不变性**：同一个“左上 vs 右下”的相对关系在不同窗口里语义一致，
  参数共享更高效；
- 实测（Swin 论文）相对位置偏置比绝对位置编码效果好。

偏置表 $B_{\text{table}}\in\mathbb{R}^{(2M-1)^2 \times h}$，对每对 token $(i,j)$ 查
相对位置索引 $\text{idx}(i,j)$，取 $B = B_{\text{table}}[\text{idx}(i,j)]$。

### 2.3 相对位置索引的构造

设窗口坐标 $p=(p_x,p_y)$，相对坐标：

$$
(dx, dy) = p_i - p_j \in [-(M-1), M-1]^2
$$

做两步量化（对应 `build_relative_position_index`）：

1. 平移：$dx \leftarrow dx + (M-1)$，$dy \leftarrow dy + (M-1)$，落入 $[0, 2M-2]$；
2. 行优先展平：$\text{idx} = dx \cdot (2M-1) + dy$。

数值例（$M=4$）：相对位置 $(-3,-3)$ → $(0,0)$ → 索引 0；$(0,0)$ → $(3,3)$ →
索引 $3\cdot7+3=24$；$(3,3)$ → $(6,6)$ → 索引 $6\cdot7+6=48$。共 $(7)^2=49$ 种。

---

## 3. SW-MSA：循环移位 + 注意力掩码

### 3.1 循环移位

把特征图整体**向左上循环移位** $\lfloor M/2 \rfloor$ 个像素（`torch.roll`），再做常规
分窗。这样原本被窗口边界切开的位置被移到一起，注意力得以跨过旧的窗口边界。做完后再
反向移位还原。

> 为什么移位后窗口数量不变？移位只是坐标的循环平移，窗口仍是不重叠的 $M\times M$ 网格，
> 窗口总数 $\frac{HW}{M^2}$ 不变 → **每层计算量恒定**，这也是 SW-MSA 相对“移动窗口”方案
> 的关键优势。

### 3.2 掩码

移位后，某些窗口内混入了原图**不相邻区域**的 token（例如左上角窗口混入了右下角的
token）。它们之间本无空间邻接关系，不应互相注意。`build_attn_mask` 的做法：

1. 把原图按移位切分线画成 3×3 = 9 块，每块编号 0..8；
2. 分窗后，窗口内编号不同的两 token 掩码为 $-100$（softmax 后注意力≈0），
   编号相同则掩码为 0。

$$
\text{mask}(i,j) =
\begin{cases}
0, & \text{同一区域}\\
-100, & \text{不同区域}
\end{cases}
\implies
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B + \text{mask}\right)
$$

### 3.3 掩码与输入内容无关（惰性缓存）

掩码只取决于 $(H, W, \text{window\_size}, \text{shift\_size})$，**与输入数据无关**。
因此同一形状的多次前向可以复用同一张掩码。`SwinBlock` 用 `_mask_cache` + `_mask_key`
缓存，key 变化（尺寸或设备）才重建。这是重要的工程优化：训练时掩码只建一次。

### 3.4 为什么 pad 在 roll 之前

`pad` 是补 0 填充（`F.pad`），`roll` 是循环移位。若先 roll 再 pad，则 pad 区域会出现在
移位后的不同位置、且填充值与移位内容交错，破坏“补到整数倍再做循环移位”的几何一致性。
先 pad 再 roll 保证：所有后续窗口网格都建立在统一的、整数倍的坐标系上，最后 crop 回原
尺寸时裁剪位置与原始内容严格对应。

---

## 4. SwinBlock 的结构与 pre-norm

### 4.1 结构（pre-norm + 双残差）

$$
\begin{aligned}
\hat x &= x + \text{DropPath}\big(\text{W/SW-MSA}(\text{LN}_1(x))\big) \\
x_{out} &= \hat x + \text{DropPath}\big(\text{MLP}(\text{LN}_2(\hat x))\big)
\end{aligned}
$$

LN 放在注意力/MLP **之前**（pre-norm），残差把归一化后的支路加回主干。

### 4.2 为什么 pre-norm 优于 ViT 的 post-norm

- **post-norm**（ViT）：$x_{out} = \text{LN}(x + \text{SubLayer}(x))$，LN 在残差之后。
  梯度要从 LN 反向穿过残差，深层时梯度易消失/放大，训练不稳定，需要 warmup 等技巧。
- **pre-norm**（Swin）：$x_{out} = x + \text{SubLayer}(\text{LN}(x))$，残差是**恒等捷径**，
  梯度可直接沿主干传播，深层网络也能稳定训练，对学习率不敏感。

### 4.3 两处残差与 DropPath

- 残差 1 跳过注意力、残差 2 跳过 MLP，两条捷径都让梯度与信息直通；
- `DropPath`（stochastic depth）以概率 $p$ 把**整条残差支路**置零，除以 $1-p$ 保持期望不变：

$$
\text{DropPath}(x) =
\begin{cases}
\frac{x}{1-p}, & \text{概率 } 1-p\\
0, & \text{概率 } p
\end{cases}
\implies \mathbb{E}[\text{DropPath}(x)] = x
$$

它等价于随机跳过某些 block，是正则化手段，随网络深度增大 $p$（见模块 08）。

### 4.4 MLP ratio = 4

MLP 先把 $C$ 升到 $4C$ 再降回 $C$：

$$
\text{MLP}(x) = W_2\,\text{GELU}(W_1 x), \quad W_1\in\mathbb{R}^{4C\times C},\; W_2\in\mathbb{R}^{C\times 4C}
$$

ratio=4 是 Transformer 的经验惯例（ViT、Swin 一致），给 FFN 更大的中间容量以混合通道信息。

---

## 5. 参数量推导

### 5.1 注意力投影（WindowAttention）

- `qkv`：`Linear(C → 3C)`，权重 $3C^2$（+偏置 $3C$）；
- `proj`：`Linear(C → C)`，权重 $C^2$（+偏置 $C$）。

小计：$3C^2 + C^2 = 4C^2$（忽略偏置）。

### 5.2 相对位置偏置表

$$
(2M-1)^2 \times h
$$

数值例：$M=7, h=3$ 时 $13^2\times3 = 507$。

### 5.3 MLP

- `fc1`：`Linear(C → 4C)`，$4C^2$（+偏置 $4C$）；
- `fc2`：`Linear(4C → C)`，$4C^2$（+偏置 $C$）。

小计：$8C^2$（忽略偏置）。

### 5.4 每 block 总参数量

$$
\#\text{params} \approx 4C^2 + 8C^2 = 12C^2 \quad(+ \;(2M-1)^2 h)
$$

数值例（$C=96$）：

$$
12 \times 96^2 = 110{,}592
$$

加上相对位置偏置表与各偏置项，单 block 约 11 万参数。Swin-Tiny 共 4 个 stage、深度
$[2,2,6,2]$、通道 $[96,192,384,768]$，总参数约 28M，其中后两个 stage 因通道翻倍贡献大头
（见模块 08 的分布表）。

---

## 6. 计算量（FLOPs）推导

### 6.1 窗口注意力

单窗口 $M^2$ 个 token，$QK^\top$ 为 $M^2\times M^2$，复杂度 $O(M^4 d)$；$\frac{HW}{M^2}$
个窗口：

$$
\text{MACs}_{\text{attn}} \approx 2\cdot\frac{HW}{M^2}\cdot M^4\cdot\frac{C}{h}\cdot h
= 2\,HW\,M^2\,C
$$

### 6.2 线性投影 + MLP

- qkv+proj：每 token $4C^2$ 次 MAC；
- MLP：每 token $8C^2$ 次 MAC；
- 合计每 token $12C^2$ 次 MAC，共 $HW$ 个 token：

$$
\text{MACs}_{\text{proj+MLP}} = 12\,HW\,C^2
$$

### 6.3 合计（每 block）

$$
\text{MACs}_{\text{block}} \approx 12\,HW\,C^2 + 2\,HW\,M^2\,C
$$

数值例（$C=96, M=7, HW=56^2=3136$）：

$$
12\cdot3136\cdot96^2 = 12\cdot3136\cdot9216 \approx 3.47\times10^{8}
$$
$$
2\cdot3136\cdot49\cdot96 \approx 2.95\times10^{7}
$$

可见在标准配置下，**投影+MLP 主导计算**，窗口注意力项相对小（因 $M^2=49 \ll 12C$）。

---

## 7. 数值例总表

| 项目 | 公式 | $C=96, M=7, h=3$ |
|---|---|---|
| qkv 权重 | $3C^2$ | 27648 |
| proj 权重 | $C^2$ | 9216 |
| MLP 权重 | $8C^2$ | 73728 |
| 相对位置偏置表 | $(2M-1)^2 h$ | 507 |
| 每 block 权重（约） | $12C^2$ | 110592 |
| 窗口注意力 MACs | $2HW M^2 C$（$HW{=}3136$） | $\approx2.95\times10^7$ |
| 投影+MLP MACs | $12 HW C^2$ | $\approx3.47\times10^8$ |
| 窗口数 | $HW/M^2$ | 64 |

---

## 8. 完整推导链小结

```
输入 x: (B, H*W, C)
  LN1 -> view(B,H,W,C)                       # pre-norm + 恢复 2D
  pad 到 window_size 整数倍                   # (B, Hp, Wp, C)
  [SW] torch.roll(-s,-s)                    # 循环移位
  window_partition                          # (B*nW, M, M, C) -> (B*nW, M^2, C)
  WindowAttention( QKV -> 缩放点积 -> +相对偏置 -> [+mask] -> softmax -> @V -> proj )
  window_reverse                            # (B, Hp, Wp, C)
  [SW] torch.roll(+s,+s)                    # 反向移位
  crop 回 (B, H, W, C) -> view(B,H*W,C)      # 还原
  + DropPath(注意力支路)                      # 残差 1
  LN2 -> MLP(4C) -> DropPath -> +            # 残差 2
输出 x: (B, H*W, C)  形状与输入一致
```

核心思想一句话：**用窗口化把全局注意力的 $O(N^2)$ 降到 $O(N)$，用循环移位在相邻层之间
交换跨窗口信息，用 pre-norm + 双残差 + 相对位置偏置保证深层稳定与几何归纳偏置。**
