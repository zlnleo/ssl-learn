# 模块 04 数学推导：移位窗口与 torch.roll 循环移位

> 本文是 `04_shifted_window` 的完整推导链与数值例子，与 README 互补。
> 阅读目标：从"固定窗口感受野锁死"推导到"循环移位的数学语义、可逆性与最优移位量"。

---

## 1. 问题定义：固定窗口的感受野锁死

设特征图为 $\mathbf{X} \in \mathbb{R}^{B \times H \times W \times C}$，窗口大小 $M$。
W-MSA 把图切成 $n_W = \lfloor H/M \rfloor \times \lfloor W/M \rfloor$ 个窗口，每个窗口独立做注意力。

对第 $l$ 层，窗口 $\omega$ 内 token $t$ 的输出只依赖同窗口的输入：

$$
\mathbf{y}_t^{(l)} = \text{Attn}\big(\{ \mathbf{x}_u^{(l-1)} : u \in \omega(t) \}\big)
$$

由于窗口划分逐层**相同**（$\omega(t)$ 不随 $l$ 变化），递推得：

$$
\text{感受野}_t^{(L)} \subseteq \omega(t), \quad \forall L
$$

即无论堆多少层，token $t$ 的感受野**永远不会超出它所在的窗口**。这是固定窗口的根本缺陷。

---

## 2. 全局注意力的复杂度瓶颈（为什么不能直接全局化）

全局自注意力的注意力矩阵是 $(HW) \times (HW)$，其计算/存储复杂度：

$$
O\big((HW)^2 \cdot C\big)
$$

窗口注意力的复杂度（$n_W$ 个窗口，每个 $M^2$ 个 token）：

$$
O\big(n_W \cdot (M^2)^2 \cdot C\big) = O\big(\tfrac{HW}{M^2} \cdot M^4 \cdot C\big) = O\big(HW \cdot M^2 \cdot C\big)
$$

当 $M \ll HW$ 时，窗口注意力随 $HW$ **线性**增长，远优于全局的二次增长。所以我们**保留窗口**，只改变窗口的**划分方式**。

---

## 3. 循环移位的数学语义

### 3.1 torch.roll 的定义

`torch.roll(x, shifts=s, dims=d)` 沿维度 $d$ 循环滚动：位置 $i$ 的元素移动到 $(i+s) \bmod n$。
等价地，输出在位置 $i$ 取的是输入在 $(i - s) \bmod n$ 的元素：

$$
\text{roll}(x, s)[i] = x[(i - s) \bmod n]
$$

Swin 的 `cyclic_shift(x, s)` 对高、宽两个维度同时取 `shifts = -s`：

$$
\text{cyclic\_shift}(x, s)[i, j] = x[(i - (-s)) \bmod H,\ (j - (-s)) \bmod W]
= x[(i + s) \bmod H,\ (j + s) \bmod W]
$$

这就是"向左上移位 $s$"（越界从另一侧绕回）。

### 3.2 数值例子（$H=W=8, s=2$）

取位置 id $= i \cdot 8 + j$。移位后：

$$
\text{out}[0, j] = x[2, (j+2) \bmod 8] = 2 \cdot 8 + (j+2) \bmod 8
$$

- $j=0$：$x[2,2] = 18$
- $j=1$：$x[2,3] = 19$
- $j=6$：$x[2,0] = 16$（列越界绕回）
- $j=7$：$x[2,1] = 17$

与 `experiment.py` 实验 1 打印的 roll 后第 0 行 `18 19 20 21 22 23 16 17` 完全一致。

---

## 4. 可逆性推导

`cyclic_unshift` 取 `shifts = +s`：

$$
\text{cyclic\_unshift}(x, s)[i, j] = x[(i - s) \bmod H,\ (j - s) \bmod W]
$$

复合：

$$
\text{unshift}\big(\text{shift}(x, s), s\big)[i, j]
= \text{shift}(x, s)[(i-s) \bmod H, (j-s) \bmod W]
= x[((i-s)+s) \bmod H, ((j-s)+s) \bmod W] = x[i, j]
$$

因此：

$$
\boxed{\text{cyclic\_unshift} = \text{cyclic\_shift}^{-1}}
$$

**物理含义**：循环移位是位置的一个**置换（permutation）**，置换必然可逆；正因为中间"不丢任何元素"，分窗计算完再滚回来才能精确还原。这是循环移位优于普通平移的关键。

---

## 5. 为什么 shift = M/2 最优

设移位量为 $s$。新窗口 $\omega'$ 相对旧窗口网格的错位程度由 $s$ 决定。

**边界条件分析**：

1. $s = 0$：$\text{shift}$ 是恒等置换，窗口完全对齐，跨窗交流为零。
2. $s = M$：由于窗口以 $M$ 为周期，$\text{shift}(x, M)$ 后的窗口划分与 $s=0$ **等价**（每个新窗口依然精确对齐某个旧窗口），同样零交流。
3. $s \in (0, M)$：$s$ 越大错位越大。错位在 $s = M/2$ 时达到最大——此时新窗口的边界正好落在旧窗口的**中心**，新窗口的四个 $M/2 \times M/2$ 象限分别来自**四个不同的旧窗口**。

**证明（以 $M=4, s=2$ 为例）**：新窗口覆盖原图行区间 $[(4r+2) \bmod 8, (4r+5) \bmod 8]$。

- $r=0$：行 $[2,5]$，恰好横跨旧窗口行 $[0,3]$ 与 $[4,7]$ 的边界（行 3/4）。
- $r=1$：行 $[6,9] \bmod 8 = \{6,7,0,1\}$，跨越边界并从底部绕回顶部。

一般地，$s = M/2$ 时新窗口边界与旧窗口边界**完全不重合**，每个新窗口都最大化地切割旧网格，跨窗交流最充分。

---

## 6. window_partition 的维度重排数学

设 $nH = H/M$（窗口行数），$nW = W/M$（窗口列数）。`view` 把 $(H, W)$ 拆成 $(nH, M, nW, M)$：

$$
x[b, h, w, c] \mapsto x'[b, h // M, h \bmod M, w // M, w \bmod M, c]
$$

其中 $h = (h//M)\cdot M + (h \bmod M)$。`permute(0,1,3,2,4,5)` 交换第 2、3 维：

$$
x'[b, nH, r, nW, q, c] \mapsto x''[b, nH, nW, r, q, c]
$$

最后 `view(-1, M, M, C)` 把 $(b, nH, nW)$ 合并成窗口序号：

$$
w_{idx} = b \cdot nH \cdot nW + r \cdot nW + c
$$

这就是"batch 优先、窗口行优先"的编号规则，`window_reverse` 逐维逆序还原。`experiment.py` 实验 2 中窗口 `(r,c)` 的 `id = r*2+c` 正是此规则的实例。

---

## 7. 跨窗交流的严格证明（数值例子）

$H=W=8, M=4, s=2$。新窗口 $(1,1)$（第 1 窗口行、第 1 窗口列，id=3）覆盖移位后图的行 $[4,7]$、列 $[4,7]$，对应原图：

$$
\text{行}: \{ (4+2)\bmod 8, (5+2)\bmod 8, (6+2)\bmod 8, (7+2)\bmod 8 \} = \{6, 7, 0, 1\}
$$

$$
\text{列}: \{6, 7, 0, 1\}
$$

因此窗口 $(1,1)$ 内 token 的原图坐标为 $\{0,1,6,7\} \times \{0,1,6,7\}$，即**四个 $2\times2$ 角块**：

| 角块 | 原图行 | 原图列 | 原图 id |
|---|---|---|---|
| 左上 | 0–1 | 0–1 | {0,1,8,9} |
| 右上 | 0–1 | 6–7 | {6,7,14,15} |
| 左下 | 6–7 | 0–1 | {48,49,56,57} |
| 右下 | 6–7 | 6–7 | {54,55,62,63} |

**结论**：第 1 层互不相见的四个角落 token，在第 2 层被装进同一个窗口——跨窗信息流动得以实现。

---

## 8. 伪邻居问题：循环移位的副作用

第 7 节的窗口 $(1,1)$ 里，左上角块 token（原图第 0 行）与右下角块 token（原图第 7 行）在**原图上相距最远**，却被放进了同一个窗口。若直接 softmax 注意力：

$$
\text{Attn}_{ab} = \frac{\exp(S_{ab})}{\sum_c \exp(S_{ac})}
$$

它们会被当作普通邻居加权求和，产生**跨了整个图的错误依赖**。这就是模块 05 要解决的：给"来自不同角块"的 token 对加掩码，令其注意力权重为 0。

---

## 9. 复杂度再核算（移位不增加任何计算量）

移位（`torch.roll`）是 $O(HW)$ 的内存重排，相比窗口注意力的 $O(HW \cdot M^2 \cdot C)$ 可忽略；分窗/还原同样只是 `view/permute` 的零拷贝视图操作。因此 SW-MSA 与 W-MSA 的**计算复杂度完全相同**，却在每一层之间引入了跨窗信息流动——这是移位窗口方案如此优雅的原因。

---

## 小结（公式速查）

| 概念 | 公式 |
|---|---|
| 循环移位 | $\text{out}[i,j] = x[(i+s)\bmod H, (j+s)\bmod W]$ |
| 逆移位 | $\text{unshift} = \text{shift}^{-1}$（取 $+s$） |
| 最优移位 | $s = M/2$（错位最大） |
| 窗口编号 | $w_{idx} = b\cdot nH\cdot nW + r\cdot nW + c$ |
| 窗口注意力复杂度 | $O(HW \cdot M^2 \cdot C)$（随 $HW$ 线性） |
| 副作用 | 新窗口混入不相邻区域 → 需要模块 05 掩码 |
