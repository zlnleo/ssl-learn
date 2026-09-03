# 模块 02 数学推导 · Window Partition / Reverse（窗口划分与还原）

> 本文是 `README.md` 的完整数学化展开：**定义 → 推导 → 数值例子**。
> 核心回答两个问题：① 划分/还原的**索引映射公式**；② 为什么 `permute` 后必须 `contiguous()`。

---

## 1. 定义与记号

| 记号 | 含义 |
| --- | --- |
| $B$ | batch 大小 |
| $H, W$ | 特征图高、宽（token 数） |
| $C$ | 通道数 |
| $M$ | 窗口边长（token 数），要求 $M \mid H$、$M \mid W$ |
| $n_h = H/M$ | 行方向的窗口个数 |
| $n_w = W/M$ | 列方向的窗口个数 |
| $nW = n_h \cdot n_w$ | 总窗口数 |
| $B_* = B \cdot nW$ | 窗口序列的 batch 数 |

**输入输出：**

$$
\text{window\_partition}: \mathbb{R}^{B\times H\times W\times C}
\longrightarrow \mathbb{R}^{B_*\times M\times M\times C}
$$

$$
\text{window\_reverse}: \mathbb{R}^{B_*\times M\times M\times C}
\longrightarrow \mathbb{R}^{B\times H\times W\times C}
$$

---

## 2. 内存模型：行优先（C-order）展平

PyTorch 张量默认按**行优先**在内存中线性存放。四维张量 $\mathbf{X}(B,H,W,C)$ 的元素

$$
X[b, h, w, c]
$$

在内存中的物理偏移（元素个数计）为：

$$
\text{offset}(b,h,w,c) = \big(\,(b\cdot H + h)\cdot W + w\,\big)\cdot C + c
$$

对应的 `stride`（步长）是：

$$
\text{stride}(X) = \big(HWC,\ WC,\ C,\ 1\big)
$$

含义：`stride[k]` = 第 $k$ 个轴每前进 1 格，物理偏移增加多少元素。

---

## 3. 正映射：全局位置 → 窗口编号 → 窗口内位置

对特征图中任一 token 的全局坐标 $(h, w)$（$0\le h<H,\ 0\le w<W$）：

### 3.1 窗口编号 $(i, j)$

$$
i = \left\lfloor \frac{h}{M} \right\rfloor \in [0, n_h),\qquad
j = \left\lfloor \frac{w}{M} \right\rfloor \in [0, n_w)
$$

### 3.2 窗口内坐标 $(m, n)$

$$
m = h \bmod M,\qquad n = w \bmod M
$$

即：

$$
h = i\,M + m,\qquad w = j\,M + n
$$

### 3.3 窗口的行优先展平序号 $k$

$$
k = i \cdot n_w + j \in [0, nW)
$$

### 3.4 窗口内的行优先展平序号 $t$

$$
t = m \cdot M + n \in [0, M^2)
$$

### 3.5 完整链条

$$
(h, w)
\ \xrightarrow{i=\lfloor h/M\rfloor,\ j=\lfloor w/M\rfloor}\
(i, j)
\ \xrightarrow{m=h-iM,\ n=w-jM}\
(m, n)
\ \xrightarrow{k=i\,n_w+j,\ t=m\,M+n}\
(k, t)
$$

最终在窗口序列里的位置是 $(\ b\cdot nW + k,\ m,\ n,\ c\ )$。

---

## 4. 逆映射：窗口序列 → 全局位置

给定窗口序列下标 $(\tilde b, m, n, c)$，其中 $\tilde b \in [0, B_*)$ 是「窗口 batch」：

### 4.1 反推图像 batch 与窗口编号

$$
b = \left\lfloor \frac{\tilde b}{nW} \right\rfloor,\qquad
k = \tilde b \bmod nW
$$

### 4.2 反推窗口坐标 $(i, j)$

$$
i = \left\lfloor \frac{k}{n_w} \right\rfloor,\qquad
j = k \bmod n_w
$$

### 4.3 反推全局坐标 $(h, w)$

$$
h = i\,M + m,\qquad w = j\,M + n
$$

于是

$$
\text{window\_reverse}(\text{window\_partition}(X)) = X
$$

**逐元素成立**，且因为是纯索引重排（无浮点运算），误差恒为 0。

---

## 5. `view` 与 `permute` 的代数

### 5.1 `view` 只是「因数分解」

`view` 不改变内存，只改变「形状的因数分解」。partition 的第一步把

$$
H = n_h \cdot M,\qquad W = n_w \cdot M
$$

代入：

$$
(B,\ H,\ W,\ C) = (B,\ n_h,\ M,\ n_w,\ M,\ C)
$$

形状相乘总数不变：

$$
B\cdot H\cdot W\cdot C = B\cdot n_h\cdot M\cdot n_w\cdot M\cdot C
$$

### 5.2 `permute` 只是「交换轴的 stride」

`permute(0,1,3,2,4,5)` 把轴序 $(0,1,2,3,4,5)$ 变为 $(0,1,3,2,4,5)$：

$$
(B, n_h, M, n_w, M, C) \ \longrightarrow\ (B, n_h, n_w, M, M, C)
$$

每个轴的数据不变，只是「行/列」的解释变了。以第 2、3 轴为例：

- 原来：轴 2 = 窗口内行 $m$（stride $= W\cdot C$），轴 3 = 列窗口编号 $j$（stride $= M\cdot C$）。
- 交换后：轴 2 变成列窗口编号 $j$（沿用 stride $= M\cdot C$），轴 3 变成窗口内行 $m$（沿用 stride $= W\cdot C$）。

物理顺序没动，所以 `is_contiguous()` 变为 `False`。

### 5.3 最后 `view(-1, M, M, C)`

把前三个轴 $(B, n_h, n_w)$ 合并：

$$
B \cdot n_h \cdot n_w = B \cdot nW = B_*
$$

$$
(B, n_h, n_w, M, M, C) \ \longrightarrow\ (B_*, M, M, C)
$$

---

## 6. 为什么 `permute` 后必须 `contiguous()`

### 6.1 连续（contiguous）的定义

张量连续，当且仅当其 `stride` 满足：从最后一维往前，每一维的 stride 恰好等于
「后面所有维度大小的乘积」的累计值。对行优先四维张量 $(a,b,c,d)$，连续条件是

$$
\text{stride} = (bcd,\ cd,\ d,\ 1)
$$

### 6.2 换轴破坏连续性

`permute` 只是把轴的 stride 重新排列，但**没有重新排内存**。换轴后的 stride 不再是
「单调递减的累计乘积」，于是不连续。

### 6.3 `view(-1)` 为何拒绝非连续张量

`view(-1, ...)` 要「把一段内存当作一维线性序列再折叠」。若内存不连续，
「逻辑上的第 0 个元素」与「物理上的第 1 个元素」之间跳的不是 1 个元素，
折叠会发生错位，PyTorch 无法无歧义地完成，故抛 `RuntimeError`。

### 6.4 `contiguous()` 做了什么

`contiguous()` 按**当前逻辑顺序**重新分配一块连续内存并拷贝数据，使得新张量满足连续定义。
这是整个划分流程里**唯一一次真实的内存拷贝**，其余步骤都是「零拷贝视图」。

> 注意：对于 `partition`，`view` 之后本来就是连续的，`permute` 破坏连续性，
> `contiguous()` 修复之。若输入本身由 `view` 得到且连续，则整条链只做一次拷贝。

---

## 7. 数值例子：4×4 图、M=2

取 $B=1, H=W=4, C=1, M=2$，则 $n_h = n_w = 2,\ nW = 4,\ B_* = 4$。

### 7.1 位置 id

$$
X[0,h,w,0] = h\cdot 4 + w
$$

### 7.2 用正映射算每个窗口

以全局 $(h,w)=(2,2)$ 为例：

$$
i = \lfloor 2/2 \rfloor = 1,\quad j = \lfloor 2/2 \rfloor = 1
$$

$$
m = 2 \bmod 2 = 0,\quad n = 2 \bmod 2 = 0
$$

$$
k = 1\cdot 2 + 1 = 3,\quad t = 0\cdot 2 + 0 = 0
$$

即全局 id $10$（$=2\cdot4+2$）落在窗口 $k=3$ 的 $(m,n)=(0,0)$ 处。

### 7.3 四个窗口的 id 集合

| 窗口 $(i,j)$ | $k=i\,n_w+j$ | 覆盖全局 id（`h*W+w`） |
| --- | --- | --- |
| $(0,0)$ | 0 | $\{0,1,4,5\}$ |
| $(0,1)$ | 1 | $\{2,3,6,7\}$ |
| $(1,0)$ | 2 | $\{8,9,12,13\}$ |
| $(1,1)$ | 3 | $\{10,11,14,15\}$ |

与 `experiment.py` 实验 A 输出一致。

### 7.4 逆映射验证

取窗口序列元素 $\tilde b=3$（即窗口 $k=3$）、$(m,n)=(1,1)$：

$$
b = \lfloor 3/4 \rfloor = 0,\quad k = 3 \bmod 4 = 3
$$

$$
i = \lfloor 3/2 \rfloor = 1,\quad j = 3 \bmod 2 = 1
$$

$$
h = 1\cdot 2 + 1 = 3,\quad w = 1\cdot 2 + 1 = 3
$$

全局 id $= 3\cdot 4 + 3 = 15$，恰为窗口 3 的右下角元素，逆映射正确。

---

## 8. 数值例子：8×8 图、M=2 的 stride 演算

取 $B=2, H=W=8, C=3, M=2$，与 `shape_tracking.py` 完全一致。

### 8.1 输入

$$
\text{stride}(X) = (HWC, WC, C, 1) = (192, 24, 3, 1)
$$

### 8.2 `view(2, 4, 2, 4, 2, 3)` 后

$$
\text{stride} = (192, 48, 24, 6, 3, 1)
$$

验证：第 1 轴（$n_h=4$）stride $=48 = M\cdot W\cdot C = 2\cdot 8\cdot 3$，
即「行窗口编号 +1 = 向下跳一个窗口高度 $M$ 行」；第 2 轴（窗口内行 $M$）stride $=24 = W\cdot C$，
即「窗口内行 +1 = 向下跳一行」。

### 8.3 `permute(0,1,3,2,4,5)` 后

$$
\text{stride} = (192, 48, 6, 24, 3, 1)
$$

第 2 轴（列窗口编号 $n_w$）stride $=6 = M\cdot C$，第 3 轴（窗口内行 $M$）stride $=24 = W\cdot C$。
此时 `stride` 不再满足连续条件（第 2 轴 6 < 第 3 轴 24），故 `is_contiguous=False`。

### 8.4 `contiguous()` 后

$$
\text{stride} = (192, 48, 12, 6, 3, 1)
$$

第 2 轴 stride $=12 = M\cdot M\cdot C$（窗口内 4 个 token × 3 通道），满足连续。

### 8.5 `view(-1, 2, 2, 3)` 后

$$
(B_*, M, M, C) = (32, 2, 2, 3),\qquad \text{stride} = (12, 6, 3, 1)
$$

其中 $B_* = B\cdot nW = 2\cdot 16 = 32$。

---

## 9. 可逆性定理

**命题**：对任意满足 $M\mid H$、$M\mid W$ 的张量 $X\in\mathbb{R}^{B\times H\times W\times C}$，

$$
\text{window\_reverse}(\text{window\_partition}(X), M, H, W) = X
$$

**证明**（索引层面）：partition 把 $(b,h,w,c)$ 映射到 $(\tilde b, m, n, c)$，其中

$$
\tilde b = b\cdot nW + \lfloor h/M\rfloor\cdot n_w + \lfloor w/M\rfloor,
\qquad m = h\bmod M,\quad n = w\bmod M
$$

reverse 的逆公式（第 4 节）恰好把它映回 $(b,h,w,c)$。二者一一对应（双射），
且无任何数值运算，故逐元素精确相等。∎

> 这个双射性质是后续「shifted window」正确性的根基：普通窗口必须先能精确还原，
> 平移窗口才能在「先平移 → 划分 → 注意力 → 还原 → 平移回来」的闭环中不丢信息。
