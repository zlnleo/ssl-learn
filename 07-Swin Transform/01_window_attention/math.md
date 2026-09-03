# 模块 01 数学推导 · Window Attention（窗口多头自注意力）

> 本文是 `README.md` 第 ② 节的完整数学化展开：**定义 → 推导 → 数值例子**。
> 记号约定：小写斜体为标量，粗体大写为矩阵/张量；$d$ 为单头维度，$C$ 为总通道，$h$ 为头数，$C = h\cdot d$。

---

## 1. 定义与记号

| 记号 | 含义 |
| --- | --- |
| $B$ | batch 大小（图像张数） |
| $H, W$ | 特征图高、宽 |
| $hw = H\cdot W$ | 总 token 数 |
| $M$ | 窗口边长（token 数） |
| $N = M^2$ | 每个窗口内的 token 数 |
| $nW = \frac{H}{M}\cdot\frac{W}{M}$ | 窗口个数 |
| $B_* = B\cdot nW$ | 所有窗口按 batch 拼接后的「batch」数 |
| $C$ | 通道数（embedding 维度） |
| $h$ | 注意力头数 |
| $d = C / h$ | 单头维度 |
| $X \in \mathbb{R}^{B_*\times N\times C}$ | 输入序列 |
| $W_Q, W_K, W_V, W_O$ | 可学习投影权重 |

**核心恒等式**（后面反复用到）：

$$
B_* \cdot N = (B \cdot nW)\cdot M^2 = B \cdot hw
$$

即「窗口拼起来的 batch」×「窗口内 token 数」=「图像数」×「每图 token 数」。

---

## 2. 单头注意力：定义

对每个注意力单元（一个窗口或整张图），有 $N$ 个 token，输入 $\mathbf{X}\in\mathbb{R}^{N\times C}$：

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V
\qquad
(\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d})
$$

注意力权重矩阵（先不缩放、不 softmax）：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N\times N},
\qquad
S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j = \sum_{t=1}^{d} Q_{it}K_{jt}
$$

softmax 归一化（沿行，即 key 方向）：

$$
A_{ij} = \frac{\exp(S_{ij}/\sqrt{d})}{\sum_{j'=1}^{N}\exp(S_{ij'}/\sqrt{d})},
\qquad
\sum_{j} A_{ij} = 1 \ \ (\forall i)
$$

输出（value 的加权和）：

$$
\mathbf{O} = \mathbf{A}\mathbf{V}\in\mathbb{R}^{N\times d},
\qquad
\mathbf{o}_i = \sum_{j=1}^{N} A_{ij}\mathbf{v}_j
$$

最后并头 + 输出投影（见第 4 节）。

---

## 3. 为什么要除以 $\sqrt{d}$：方差分析

### 3.1 假设

设 $\mathbf{q}$ 与 $\mathbf{k}$ 的各个分量**独立同分布**，均值 $0$、方差 $\sigma^2=1$。
（实际网络里会有 LayerNorm/初始化近似保证这一点。）

### 3.2 点积的方差

$$
s = \mathbf{q}\cdot\mathbf{k} = \sum_{t=1}^{d} q_t k_t
$$

由独立性：

$$
\mathbb{E}[q_t k_t] = \mathbb{E}[q_t]\mathbb{E}[k_t] = 0
$$

$$
\mathrm{Var}(q_t k_t)
= \mathbb{E}[q_t^2 k_t^2] - (\mathbb{E}[q_t k_t])^2
= \mathbb{E}[q_t^2]\mathbb{E}[k_t^2] - 0
= 1\cdot 1 = 1
$$

各 $q_t k_t$ 独立，方差可加：

$$
\mathrm{Var}(s) = \sum_{t=1}^{d}\mathrm{Var}(q_t k_t) = d
$$

**结论**：点积 $s$ 的标准差是 $\sqrt{d}$，即点积数值随维度 $d$ 线性放大。

### 3.3 缩放后回到 1

$$
\mathrm{Var}\!\left(\frac{s}{\sqrt{d}}\right)
= \frac{1}{d}\mathrm{Var}(s)
= \frac{1}{d}\cdot d = 1
$$

### 3.4 为什么这很重要

softmax 对很大/很小的输入会进入**饱和区**，其梯度

$$
\frac{\partial \mathrm{softmax}(z)_i}{\partial z_j}
= A_i(\delta_{ij} - A_j)
$$

在 $A_i\to 1$ 或 $A_i\to 0$ 时趋近于 0。若不缩放，$d$ 一大则 $s$ 很大，softmax 退化成 one-hot 且梯度消失；缩放后 $s/\sqrt{d}$ 的分布不随 $d$ 变化，训练稳定。代码里：

```python
self.scale = self.head_dim ** -0.5   # = 1/sqrt(d)
```

---

## 4. 多头注意力：拆头与并头

把 $C$ 维拆成 $h$ 个头，每头 $d=C/h$ 维：

$$
\mathbf{Q} = [\mathbf{Q}^{(1)},\dots,\mathbf{Q}^{(h)}],\quad
\mathbf{K} = [\mathbf{K}^{(1)},\dots,\mathbf{K}^{(h)}],\quad
\mathbf{V} = [\mathbf{V}^{(1)},\dots,\mathbf{V}^{(h)}]
$$

第 $r$ 个头独立算：

$$
\mathbf{O}^{(r)} = \mathrm{softmax}\!\left(\frac{\mathbf{Q}^{(r)}{\mathbf{K}^{(r)}}^\top}{\sqrt{d}}\right)\mathbf{V}^{(r)}
$$

并头：

$$
\mathbf{O} = [\mathbf{O}^{(1)},\dots,\mathbf{O}^{(h)}]\mathbf{W}_O,\qquad \mathbf{W}_O\in\mathbb{R}^{C\times C}
$$

**意义**：$h$ 个子空间各学一种「相似关系」，最后投影融合。代码里用一个 `Linear(C,3C)` 一次算 $Q,K,V$，
再 `reshape` + `permute` 完成拆头，`transpose` + `reshape` 完成并头（见 README ③）。

---

## 5. 计算量（MACs）推导

> 约定：**1 MAC = 1 次乘加 ≈ 2 FLOPs**。矩阵乘法 $\mathbb{R}^{m\times n}\times\mathbb{R}^{n\times p}$ 的 MACs = $m\cdot n\cdot p$。

设总 token 数为 $hw$，每个注意力单元内的 token 数为 $T$（全局 $T=hw$，窗口 $T=M^2$）。

### 5.1 QKV 投影

单个 token：$\mathbf{X}(1\times C)\cdot\mathbf{W}_{qkv}(C\times 3C)$，共 $C\cdot 3C = 3C^2$ 次 MAC。
$hw$ 个 token：

$$
\text{MACs}_{\text{QKV}} = 3\,hw\,C^2
$$

### 5.2 注意力（QKᵀ 与 AV）

每个 token 与 $T$ 个 token 计算：

- $QK^\top$：每个 token 的点积是 $d$ 维，对 $T$ 个 key，共 $T\cdot d$ 次 MAC；由于 $C = h\cdot d$，$h$ 个头合计 $h\cdot T\cdot d = T\cdot C$ 次 MAC/token。
- $AV$：同理 $T\cdot C$ 次 MAC/token。

$$
\text{MACs}_{\text{attn}} = hw\cdot T\cdot C + hw\cdot T\cdot C = 2\,hw\,T\,C
$$

### 5.3 输出投影

$$
\text{MACs}_{\text{proj}} = hw\cdot C^2
$$

### 5.4 合计

$$
\boxed{\ \text{MACs} = 4\,hw\,C^2 + 2\,hw\,T\,C\ }
$$

即 `msa_macs(hw, C, T) = 4*hw*C*C + 2*hw*T*C`。

---

## 6. 全局 vs 窗口：复杂度对比

### 6.1 全局 MSA（$T = hw$）

$$
\text{MACs}_{\text{global}} = 4\,hw\,C^2 + 2\,hw^2\,C = O\big((hw)^2\big)
$$

### 6.2 窗口 W-MSA（$T = M^2$）

$$
\text{MACs}_{\text{window}} = 4\,hw\,C^2 + 2\,hw\,M^2\,C
$$

当 $M$ 固定为常数时，第二项对 $hw$ 是**线性**的：

$$
\text{MACs}_{\text{window}} = O(hw)
$$

### 6.3 注意力部分（QKᵀ + AV）的比值

$$
\frac{\text{attn}_{\text{global}}}{\text{attn}_{\text{window}}}
= \frac{2\,hw^2\,C}{2\,hw\,M^2\,C}
= \frac{hw}{M^2}
$$

### 6.4 总比值

$$
\frac{\text{MACs}_{\text{global}}}{\text{MACs}_{\text{window}}}
= \frac{4C + 2\,hw}{4C + 2\,M^2}
$$

（分子分母同除以 $hw\,C$ 得到；注意投影项 $4C$ 两边相同，会「稀释」总比值。）

---

## 7. 注意力矩阵的显存

注意力矩阵形状为 $(B_*, h, T, T)$，float32 每元素 4 字节：

$$
\text{bytes} = B_*\cdot h\cdot T^2\cdot 4
$$

- 全局：$B_*=B,\ T=hw$ → $B\cdot h\cdot hw^2\cdot 4$
- 窗口：$B_*=B\cdot nW,\ T=M^2$ → $B\cdot nW\cdot h\cdot M^4\cdot 4 = B\cdot h\cdot hw\cdot M^2\cdot 4$

二者比值同样为 $hw/M^2$。

---

## 8. 数值例子

### 8.1 教科书例子（Swin-T 第一阶段）

取 $H=W=56\Rightarrow hw=3136$，$C=96$，$M=7\Rightarrow M^2=49$。

**全局 MSA：**

$$
\text{MACs}_{\text{global}} = 4\cdot 3136\cdot 96^2 + 2\cdot 3136^2\cdot 96
$$

先算投影项：

$$
4\cdot 3136\cdot 9216 = 115{,}605{,}504 \approx 115.6\text{M}
$$

再算注意力项：

$$
2\cdot 3136^2\cdot 96 = 2\cdot 9{,}834{,}496\cdot 96 = 1{,}888{,}223{,}232 \approx 1888.2\text{M}
$$

合计：

$$
115.6\text{M} + 1888.2\text{M} = 2003.8\text{M} \approx 2.00\text{G}
$$

**窗口 W-MSA（$T=49$）：**

$$
\text{MACs}_{\text{window}} = 4\cdot 3136\cdot 96^2 + 2\cdot 3136\cdot 49\cdot 96
$$

投影项仍为 $115.6\text{M}$；注意力项：

$$
2\cdot 3136\cdot 49\cdot 96 = 29{,}503{,}488 \approx 29.5\text{M}
$$

合计：

$$
115.6\text{M} + 29.5\text{M} = 145.1\text{M}
$$

**比值：**

$$
\frac{\text{attn}_{\text{global}}}{\text{attn}_{\text{window}}} = \frac{1888.2}{29.5} = \frac{3136}{49} = 64
$$

$$
\frac{\text{MACs}_{\text{global}}}{\text{MACs}_{\text{window}}} = \frac{2003.8}{145.1} \approx 13.81
$$

> 与 `experiment.py` 输出完全一致：`2003.8 M / 145.1 M = 13.81x`，`注意力比值 = 64.0x`。

### 8.2 显存数值例（$B=2, h=3$）

- 全局：$2\cdot 3\cdot 3136^2\cdot 4 = 236{,}027{,}904\ \text{bytes} \approx 225.1\ \text{MB}$
- 窗口：$2\cdot 3\cdot 3136\cdot 49\cdot 4 = 3{,}686{,}400\ \text{bytes} \approx 3.5\ \text{MB}$

### 8.3 $hw$ 变化对比表（$C=96, M=7$，单位 M MACs）

由 $6.2$ 的公式直接算：

| $H=W$ | $hw$ | 全局 MSA | 窗口 W-MSA | 总比值 |
| --- | --- | --- | --- | --- |
| 14 | 196 | 14.6 | 9.1 | 1.61× |
| 28 | 784 | 146.9 | 36.3 | 4.05× |
| 56 | 3136 | 2003.8 | 145.1 | 13.81× |
| 112 | 12544 | 30674.0 | 580.4 | 52.85× |
| 224 | 50176 | 485234.8 | 2321.7 | 209.00× |

**读法**：窗口 W-MSA 随 $hw$ 线性增长（$145.1/36.3 \approx 4$，对应 $hw$ 变 4 倍），
全局 MSA 随 $hw$ 二次增长（$2003.8/146.9 \approx 13.6 \approx 4^2\times$ 的一部分），
因此总比值随 $hw$ 近似线性放大（$13.81 \to 52.85 \to 209$，约 4 倍一跳）。

### 8.4 窗口数一致性检查

$$
nW = \frac{56}{7}\cdot\frac{56}{7} = 8\cdot 8 = 64
$$

$$
B_* = B\cdot nW = 2\cdot 64 = 128,\qquad B_*\cdot N = 128\cdot 49 = 6272 = 2\cdot 3136 = B\cdot hw
$$

与「核心恒等式」吻合。

---

## 9. softmax 行和性质（形状追踪里的断言依据）

softmax 定义保证：

$$
\sum_{j=1}^{N} A_{ij}
= \frac{\sum_j \exp(S_{ij}/\sqrt d)}{\sum_{j'} \exp(S_{ij'}/\sqrt d)}
= 1
$$

因此 `shape_tracking.py` 断言 `attn.sum(dim=-1) ≈ 1`，最大偏差仅来自浮点舍入（实测 $\sim 10^{-7}$）。

---

## 10. window_size = 1 的退化

当 $N=1$（$M=1$），注意力矩阵退化为 $1\times 1$：

$$
A = \mathrm{softmax}(s) = 1,\qquad \mathbf{o}_i = 1\cdot \mathbf{v}_i = \mathbf{v}_i
$$

即每个 token 只「看自己」，输出 = 自己的 value 再过输出投影，token 之间零交互。
这解释了测试 `test_window_size_one_is_per_token` 的断言依据：$y = W_O(\mathbf{v})$。

---

## 11. 全局 = 一个大窗口（对拍依据）

全局 MSA 是窗口 MSA 在 $T=hw$ 时的特例。二者若共享同一组权重 $W_Q,W_K,W_V,W_O$，
在**同一批 token** 上输出应逐元素相等：

$$
\mathrm{WindowAttention}(X;\ M=H=W) = \mathrm{GlobalMSA}(X)
$$

这正是测试 `test_global_equals_single_big_window` 的数值断言（`allclose` 到 $10^{-6}$），
它证明了「窗口算子」没有引入任何额外信息丢失——全局只是窗口的特例，而非另一套实现。
