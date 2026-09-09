# 模块 03 数学推导：相对位置偏置（Relative Position Bias）

> 本文是 `03_relative_position_bias` 的完整推导链与数值例子，与 README 互补。
> 阅读目标：从"注意力为什么没有位置信息"一路推导到"行号公式"与"转置对称性"。

---

## 1. 问题定义：注意力为什么没有位置信息

窗口自注意力的注意力分数（未加偏置、未缩放）为：

$$
S_{ab} = \langle \mathbf{q}_a, \mathbf{k}_b \rangle = \sum_{d} q_{a,d} \cdot k_{b,d}
$$

其中 $a, b$ 是窗口内的两个 token，$\mathbf{q}, \mathbf{k}$ 是它们各自的 query/key 向量。

**关键观察**：$S_{ab}$ 只依赖 $a, b$ 的**内容**（向量），完全不依赖 $a, b$ 的**坐标**。
设 $a$ 的坐标为 $(i_1, j_1)$，$b$ 的坐标为 $(i_2, j_2)$，则：

$$
\frac{\partial S_{ab}}{\partial (i_1, j_1)} = 0
$$

即注意力分数对位置坐标的梯度恒为 0。因此注意力对 token 的顺序是**排列等变**的：
对任意排列 $\pi$，若 token 内容不变，则

$$
S_{\pi(a), \pi(b)} = S_{a, b}
$$

这意味着"我左边是谁、右边是谁、离我多远"这类信息模型完全无法感知，必须显式注入。

---

## 2. 三种位置编码方案的数学形式对比

### 2.1 绝对可学习位置编码

给每个绝对位置 $p$ 一个可学习向量 $\mathbf{e}_p$，直接加到输入特征上：

$$
\tilde{\mathbf{x}}_p = \mathbf{x}_p + \mathbf{e}_p
$$

- 优点：实现最简单。
- 缺点：位置向量是**绝对**的，图像分辨率变化时要插值；且它不显式编码"两 token 的相对关系"。

### 2.2 RoPE（旋转位置编码）

用旋转矩阵 $\mathbf{R}_m$ 调制位置 $m$ 的 query/key：

$$
\tilde{\mathbf{q}}_m = \mathbf{R}_m \mathbf{q}_m, \quad \tilde{\mathbf{k}}_n = \mathbf{R}_n \mathbf{k}_n
$$

其内积满足：

$$
\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = f(\mathbf{q}_m, \mathbf{k}_n, m - n)
$$

- 优点：只依赖相对位置 $m-n$，零额外参数。
- 缺点：它通过**旋转 Q/K** 间接影响内积，无法直接施加"某个相对位移必被屏蔽"这类硬约束。

### 2.3 相对位置偏置（Swin 采用）

在 softmax **之前**直接给注意力分数矩阵加一个只依赖相对位移的偏置：

$$
\tilde{S}_{ab} = \frac{\langle \mathbf{q}_a, \mathbf{k}_b \rangle}{\sqrt{d}} + B(\text{rel}(a, b))
$$

其中 $\text{rel}(a, b) = (i_1 - i_2, j_1 - j_2)$ 是相对位移，$B(\cdot)$ 是可学习偏置。

$$
\text{Attn}_{ab} = \frac{\exp(\tilde{S}_{ab})}{\sum_{c} \exp(\tilde{S}_{ac})}
$$

- 优点：**内容相似度与空间关系解耦成可加项**，直观、显式、可学习，且相对位移种类有限（$(2M-1)^2$），参数极少。

---

## 3. 为什么相对位移只有 $(2M-1)^2$ 种

窗口尺寸 $M$，token 坐标 $(i, j)$，其中 $i, j \in \{0, 1, \dots, M-1\}$。

相对位移的两个分量：

$$
dh = i_1 - i_2, \qquad dw = j_1 - j_2
$$

$dh$ 的最小值 $= 0 - (M-1) = -(M-1)$，最大值 $= (M-1) - 0 = M-1$，因此：

$$
dh \in \{-(M-1), -(M-2), \dots, M-1\}, \quad \text{共 } 2M-1 \text{ 个取值}
$$

同理 $dw$ 也有 $2M-1$ 个取值。二者独立，故组合数：

$$
\boxed{|\text{可能的相对位移}| = (2M-1)^2}
$$

**数值例子（$M=3$）**：$dh, dw \in \{-2, -1, 0, 1, 2\}$，共 $5 \times 5 = 25$ 种相对位移。

**数值例子（$M=7$，Swin 默认）**：$dh, dw \in \{-6, \dots, 6\}$，共 $13 \times 13 = 169$ 种。

---

## 4. 从相对位移到行号：平移 + 展平推导

偏置表 `relative_position_bias_table` 是一个二维张量，形状 $((2M-1)^2, h)$。我们要把每种相对位移 $(dh, dw)$ 映射成表的一个行号 $r$。

### 4.1 平移（归一化到非负）

$dh, dw$ 有负值，不能直接当索引。平移 $M-1$：

$$
dh' = dh + (M-1), \qquad dw' = dw + (M-1)
$$

平移后：

$$
dh' \in [0, 2M-2], \qquad dw' \in [0, 2M-2]
$$

都是非负整数。

### 4.2 展平（行主序）

把二维坐标 $(dh', dw')$ 折叠成一维行号（**行主序**，即先沿 dw 方向增长）：

$$
r = dh' \times (2M-1) + dw'
$$

其中表宽为 $2M-1$（因为 $dh'$ 有 $2M-1$ 个取值）。

### 4.3 逆映射（解码）

给定行号 $r$，反解出：

$$
dh' = \left\lfloor \frac{r}{2M-1} \right\rfloor, \qquad dw' = r \bmod (2M-1)
$$

再减去 $M-1$：

$$
dh = dh' - (M-1), \qquad dw = dw' - (M-1)
$$

这构成**双射**：行号与相对位移一一对应（`decode_relative_position_row` 即实现此逆映射）。

### 4.4 数值例子（$M=3$，表宽 $=5$）

相对位移 $(dh, dw) = (-1, 2)$：

- 平移：$dh' = -1 + 2 = 1$，$dw' = 2 + 2 = 4$
- 展平：$r = 1 \times 5 + 4 = 9$

反向验证：$dh' = \lfloor 9/5 \rfloor = 1$，$dw' = 9 \bmod 5 = 4$，再减 2 得 $(dh, dw) = (-1, 2)$ ✔

中心位移 $(0, 0)$：$dh' = dw' = 2$，$r = 2 \times 5 + 2 = 12$ —— 这就是索引表对角线上的值。

---

## 5. 索引矩阵的构造：广播相减

记 $M^2$ 个 token 的坐标展平为向量 $\mathbf{c} = [(i_0, j_0), \dots, (i_{M^2-1}, j_{M^2-1})]$，即形状 $(2, M^2)$。

相对位移矩阵（形状 $(M^2, M^2, 2)$）：

$$
\text{rel}[a, b] = \mathbf{c}[:, a] - \mathbf{c}[:, b] = (i_a - i_b, j_a - j_b)
$$

代码用广播实现：

$$
\text{rel} = \mathbf{c}[:, :, \text{None}] - \mathbf{c}[:, \text{None}, :]
$$

展开看：

$$
\text{rel}[k, a, b] = \mathbf{c}[k, a] - \mathbf{c}[k, b], \qquad k \in \{0, 1\}
$$

即 $k=0$ 是 $dh$ 通道，$k=1$ 是 $dw$ 通道。随后对每个 $(a, b)$ 套用第 4 节的平移 + 展平，得到索引矩阵：

$$
\text{index}[a, b] = (dh_{ab} + M - 1)(2M - 1) + (dw_{ab} + M - 1)
$$

---

## 6. 参数量推导

偏置表形状 $((2M-1)^2, h)$，故参数量：

$$
\boxed{N_{\text{params}} = (2M-1)^2 \cdot h}
$$

**数值例子**：$M=7, h=3$（Swin-T 的典型配置）：

$$
N = (2 \times 7 - 1)^2 \times 3 = 13^2 \times 3 = 169 \times 3 = 507
$$

对比：一个 $7 \times 7$ 窗口有 49 个 token，若用"每对 token 一个标量偏置"需要 $49 \times 49 = 2401$ 个标量；而相对偏置只需 $169 \times h$，**省了一个数量级，且能泛化到任意窗口**（只要窗口大小相同）。

---

## 7. 转置对称性：index[a,b] 与 index[b,a] 对应相反位移

由定义：

$$
\text{index}[a, b] \leftrightarrow (dh_{ab}, dw_{ab}) = (i_a - i_b, j_a - j_b)
$$

交换 $a, b$：

$$
\text{index}[b, a] \leftrightarrow (dh_{ba}, dw_{ba}) = (i_b - i_a, j_b - j_a) = (-dh_{ab}, -dw_{ab})
$$

即 **index[a, b] 与 index[b, a] 对应的相对位移互为相反数**。

**数值例子（$M=3$，表宽 5）**：

- token $a = (0, 0)$，token $b = (2, 1)$：$dh = -2, dw = -1$，平移后 $(0, 1)$，行号 $r = 0 \times 5 + 1 = 1$。
- 交换后：$dh = 2, dw = 1$，平移后 $(4, 3)$，行号 $r = 4 \times 5 + 3 = 23$。

验证相反位移：$(-2, -1)$ 与 $(2, 1)$ 确实互为相反数，而 $1$ 与 $23$ 关于中心行号 $12$ 对称（$1 + 23 = 24 = 2 \times 12$）。

一般地，若 $r \leftrightarrow (dh, dw)$，则 $r_{\text{opp}} = (2M-2 - dh')(2M-1) + (2M-2 - dw')$，满足：

$$
r + r_{\text{opp}} = (2M-2)(2M-1) + (2M-2) = (2M-2)(2M)
$$

即二者关于中心行号对称。`test_transpose_symmetry_opposite_displacement` 用矩阵转置直接数值验证这一性质。

---

## 8. 双射的计数验证

对固定相对位移 $(dh, dw)$，满足 $i_1 - i_2 = dh$ 的 $(i_1, i_2)$ 对数为 $M - |dh|$；同理列方向为 $M - |dw|$。所以该位移在索引表中出现次数：

$$
\text{count}(dh, dw) = (M - |dh|)(M - |dw|)
$$

**数值例子（$M=3$）**：

- 位移 $(0, 0)$：出现 $3 \times 3 = 9$ 次（正好是全部对角线）。
- 位移 $(2, 0)$：出现 $1 \times 3 = 3$ 次。
- 位移 $(2, 2)$：出现 $1 \times 1 = 1$ 次。

全部 $(2M-1)^2$ 种位移的次数之和：

$$
\sum_{dh=-(M-1)}^{M-1} \sum_{dw=-(M-1)}^{M-1} (M-|dh|)(M-|dw|)
= \left(\sum_{dh} (M-|dh|)\right)^2 = M^4
$$

正好等于索引表元素总数 $(M^2)^2 = M^4$，说明**每一种位移都至少出现一次、且总次数守恒**（双射的完整性）。

---

## 9. 偏置如何改变注意力分布（softmax 前加 bias 的数学）

设内容分数矩阵为 $\mathbf{S}$，偏置矩阵为 $\mathbf{B}$。softmax 后：

$$
\text{无偏置: } A_{ab} = \frac{e^{S_{ab}}}{\sum_c e^{S_{ac}}}
$$

$$
\text{有偏置: } \tilde{A}_{ab} = \frac{e^{S_{ab} + B_{ab}}}{\sum_c e^{S_{ac} + B_{ac}}}
$$

由于 $e^{S + B} = e^{S} \cdot e^{B}$，偏置的作用等价于给每个分数乘一个**正的缩放因子** $e^{B_{ab}}$：

$$
\tilde{A}_{ab} \propto e^{S_{ab}} \cdot e^{B_{ab}}
$$

- 若 $B_{ab} > 0$（相对位移被"鼓励"），$e^{B_{ab}} > 1$，该注意力权重被放大；
- 若 $B_{ab} < 0$（被"惩罚"），$e^{B_{ab}} < 1$，权重被缩小；
- $B_{ab}$ 越大，放大/缩小越剧烈（指数级）。

**数值例子（$M=3$ 的中心 token）**：令 $B_{ab} = -0.5 \cdot (|dh| + |dw|)$（曼哈顿距离惩罚）。

- 邻居（距离 1）：$B = -0.5$，缩放 $e^{-0.5} \approx 0.607$；
- 角落（距离 4）：$B = -2.0$，缩放 $e^{-2.0} \approx 0.135$。

因此角落 token 的权重相对邻居被压缩到约 $0.135 / 0.607 \approx 0.22$，注意力明显向近邻集中——这正是 `experiment.py` 实验 4 观察到的现象。

---

## 10. 与后面模块的数学衔接

- **模块 04**：移位（`torch.roll`）是坐标的循环平移，**不改变窗口大小 $M$**，因此相对位置偏置表 $(2M-1)^2$ 行保持不变，可跨 W-MSA / SW-MSA 复用。
- **模块 05**：注意力掩码在数学上是把"伪邻居"的分数直接压到 $-\infty$（实现上用 $-100$ 近似），使 $\exp(-100) \approx 0$，权重严格归零。它与相对偏置正交：

$$
\tilde{S}_{ab} = \frac{QK^T}{\sqrt{d}} + \underbrace{B_{ab}}_{\text{相对偏置(管距离)}} + \underbrace{M_{ab}}_{\text{掩码(管可见性, 0 或 -100)}}
$$

---

## 小结（公式速查）

| 概念 | 公式 |
|---|---|
| 相对位移取值数 | $(2M-1)^2$ |
| 平移 | $dh' = dh + M - 1$ |
| 展平行号 | $r = dh'(2M-1) + dw'$ |
| 解码 | $dh = \lfloor r/(2M-1)\rfloor - (M-1)$ |
| 参数量 | $(2M-1)^2 \cdot h$ |
| 位移出现次数 | $(M-\|dh\|)(M-\|dw\|)$ |
| 偏置的 softmax 缩放 | $\tilde{A}_{ab} \propto e^{S_{ab}} e^{B_{ab}}$ |
