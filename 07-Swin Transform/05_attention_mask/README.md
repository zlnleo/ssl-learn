# 模块 05：注意力掩码（Attention Mask for SW-MSA）

> 模块编号：05　|　学习顺序：04 移位窗口 → **05 注意力掩码**（Swin 核心机制最后一块拼图）
>
> 一句话总结：**循环移位把不相邻区域卷进同一窗口（伪邻居），我们用一张与内容无关的掩码，把伪邻居的注意力分数压到 -100，softmax 后权重严格归零。**

---

## ① 为什么需要（动机）

### 1.1 循环移位的副作用

模块 04 告诉我们：`torch.roll` 循环移位让新窗口跨越旧边界、实现跨窗交流。但它有个**副作用**——新窗口（尤其图像边缘处）会把**空间上不相邻**的区域装进同一个窗口。

以 $H=W=8, M=4, s=2$ 为例，移位后右下角的新窗口 $(1,1)$ 同时包含了原图的**四个角**：

```
   原图 8x8                    移位后右下角窗口(1,1) 装进了 4 个角
   ┌──┬──┬──┬──┬──┬──┬──┬──┐
   │██│  │  │  │  │  │  │██│     ┌──┬──┐
   ├──┼──┼──┼──┼──┼──┼──┼──┤     │左上│右上│  <- 相距最远的区域
   │  │  │  │  │  │  │  │  │     ├──┼──┤     被当成"邻居"
   │  │  │  │  │  │  │  │  │     │左下│右下│
   │  │  │  │  │  │  │  │  │     └──┴──┘
   ├──┼──┼──┼──┼──┼──┼──┼──┤
   │██│  │  │  │  │  │  │██│
   └──┴──┴──┴──┴──┴──┴──┴──┘
```

### 1.2 不屏蔽会怎样

注意力做 softmax 后，同一窗口内所有 token 都会互相加权。若左上角 token 和右下角 token 被装进同一窗口而不加限制，模型会产生**跨整个特征图的错误依赖**——相当于在局部窗口里偷偷做了"全局"注意力，而且这些依赖是**伪装的邻居关系**（它们实际相距最远）。所以必须屏蔽。

---

## ② 核心机制讲解（第一性原理）

### 2.1 mask 机制：加 -100，softmax 后归零

SW-MSA 的注意力分数矩阵 $\mathbf{S} \in \mathbb{R}^{M^2 \times M^2}$ 上，对"伪邻居"位置加 $-100$：

$$
\tilde{S}_{ab} = S_{ab} + M_{ab}, \qquad M_{ab} = \begin{cases} 0 & \text{真邻居(可见)} \\ -100 & \text{伪邻居(屏蔽)} \end{cases}
$$

softmax 后：

$$
\text{Attn}_{ab} = \frac{\exp(\tilde{S}_{ab})}{\sum_c \exp(\tilde{S}_{ac})}
$$

被屏蔽项的分子是 $\exp(S_{ab} - 100) = \exp(S_{ab}) \cdot e^{-100}$。由于

$$
e^{-100} \approx 3.72 \times 10^{-44}
$$

在 float32 下**直接下溢为 0**，所以被屏蔽 token 的注意力权重**严格等于 0**，而其他可见 token 的权重会自动重新归一化（分母只统计可见项）。

> 为什么用 -100 而不是 -∞？-∞ 更"数学纯粹"，但会在某些算子上产生 NaN；-100 已足够让 $\exp(-100)$ 下溢到 0，既安全又简洁。

### 2.2 9 宫格构图

关键问题：**怎么知道一个窗口里哪两个 token 是"伪邻居"？**

答案：把移位后的图按**三行三列 slice** 切成 9 块，每块一个编号 $0\dots8$。两个 token 若来自**不同编号块**，它们就是循环移位造成的伪邻居，必须屏蔽。

以 $H=W=8, M=4, s=2$ 为例，切片为：

```
   行 slice:  [0:4], [4:6], [6:8]       列 slice:  [0:4], [4:6], [6:8]
```

9 宫格编号图（`experiment.py` 实验 1 输出）：

```
        列 0       1       2       3       4       5       6       7
  行 0   0   0   0   0 |  1   1 |  2   2
  行 1   0   0   0   0 |  1   1 |  2   2
  行 2   0   0   0   0 |  1   1 |  2   2
  行 3   0   0   0   0 |  1   1 |  2   2
      ----+-----+----+-----+----+----+-----+-----+----
  行 4   3   3   3   3 |  4   4 |  5   5
  行 5   3   3   3   3 |  4   4 |  5   5
      ----+-----+----+-----+----+----+-----+-----+----
  行 6   6   6   6   6 |  7   7 |  8   8
  行 7   6   6   6   6 |  7   7 |  8   8
```

- 块 0（左上 $4\times4$）：行 $[0:4]$ × 列 $[0:4]$
- 块 1（上中 $4\times2$）：行 $[0:4]$ × 列 $[4:6]$
- 块 2（右上 $4\times2$）：行 $[0:4]$ × 列 $[6:8]$
- 块 3（左中 $2\times4$）：行 $[4:6]$ × 列 $[0:4]$
- 块 4（正中 $2\times2$）：行 $[4:6]$ × 列 $[4:6]$
- 块 5（右中 $2\times2$）：行 $[4:6]$ × 列 $[6:8]$
- 块 6（左下 $2\times4$）：行 $[6:8]$ × 列 $[0:4]$
- 块 7（下中 $2\times2$）：行 $[6:8]$ × 列 $[4:6]$
- 块 8（右下 $2\times2$）：行 $[6:8]$ × 列 $[6:8]$

### 2.3 为什么只有 3×3 = 9 种区域关系

**直觉**：$shift = M/2$ 时，移位量把图沿高、宽各分成三段——左/中/右（列）和上/中/下（行）。任意窗口内的任意 token，相对移位前的位置只可能落在"上/中/下 × 左/中/右"的 9 个区域之一。两个 token 是否"真相邻"，只取决于它们是否在**同一个区域**，与它们具体是谁无关。所以只需 9 个区域编号就能穷举所有"相邻/不相邻"关系。

**更严格的视角**：窗口是 $M\times M$，而 9 宫格的中间切片是 $s \times s$（这里是 $2\times2$）。移位恰好 $M/2$ 时，新窗口的四个象限分别落在四个"角落区域"，中间两个窄条落在"边区域"、正中小块落在"中心区域"。这些区域边界正是移位前后网格的错位线。

### 2.4 为什么 mask 与内容无关、可缓存复用

mask 的构造只用到了 $H, W, M, shift\_size$ 这**四个几何参数**，完全**不看输入特征值**。因此：

1. **nW 个窗口共享一份 mask**（`build_attn_mask` 一次性算好所有窗口的 mask，形状 `(nW, M^2, M^2)`），前向时每个窗口取自己的那份。
2. **batch 之间也共享**：mask 是"单样本"的（$nW$ 是单样本窗口数，与 batch 无关），前向时通过广播 `mask[None, :, None]` 应用到所有样本（见 `swin/attention.py` 的掩码广播）。
3. **跨层复用**：同一 stage 内所有 SW-MSA 层的几何结构相同，mask 可重复使用。

### 2.5 完整数据流 ASCII 图

```
   移位后的图 (H,W) ──9宫格编号──> img_mask (1,H,W,1)
                                        │ window_partition
                                        ▼
                              mask_windows (nW, M, M, 1)
                                        │ view
                                        ▼
                              mask_windows (nW, M^2)      # 每个窗口的区域号向量
                                        │ unsqueeze(1) - unsqueeze(2)
                                        ▼
                              attn_mask (nW, M^2, M^2)    # 区域号之差
                                        │ 差 != 0 -> -100, 差 == 0 -> 0
                                        ▼
                              attn_mask (nW, M^2, M^2)    # 0=允许, -100=屏蔽
```

---

## ③ 逐段代码讲解

### 3.1 `build_attn_mask` 前半：9 宫格编号

```python
img_mask = torch.zeros((1, H, W, 1), device=device)
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1
```

- 三个行切片、三个列切片两两组合，覆盖整图，得到 9 个块，编号 $0\dots8$。
- 注意：切片**不重叠、刚好铺满**（`slice(0,-M)` 与 `slice(-M,-s)` 与 `slice(-s,None)` 首尾相接）。

### 3.2 `build_attn_mask` 后半：分窗 → 区域差 → 掩码

```python
mask_windows = window_partition(img_mask, window_size)          # (nW, M, M, 1)
mask_windows = mask_windows.view(-1, window_size * window_size) # (nW, M^2)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, M^2, M^2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
```

- `view(-1, M^2)`：每个窗口展平成 $M^2$ 长的**区域号向量**。
- `unsqueeze(1) - unsqueeze(2)`：广播相减得到 $(nW, M^2, M^2)$，元素 `[i,j] = 区域号[i] - 区域号[j]`。**同区域差为 0，不同区域差非 0**。
- `masked_fill`：非 0（伪邻居）填 -100，0（真邻居/自己）填 0。

---

## ④ Tensor Shape 跟踪总表

> 与 `shape_tracking.py` 输出完全一致（$H=W=8, M=4, s=2$）。

| 步骤 | 操作 | 形状 | 语义 |
|---|---|---|---|
| 0 | 9 宫格编号 `img_mask` | $(1, 8, 8, 1)$ | 每个像素所属区域编号 0..8 |
| 1 | `window_partition(img_mask, 4)` | $(4, 4, 4, 1)$ | $nW=4$ 个窗口 |
| 2 | `view(-1, M^2)` | $(4, 16)$ | 每个窗口展平成 16 长区域号向量 |
| 3 | `unsqueeze(1) - unsqueeze(2)` | $(4, 16, 16)$ | 区域号之差 |
| 4 | `masked_fill(!=0, -100)` | $(4, 16, 16)$ | 伪邻居屏蔽 |
| 5 | `masked_fill(==0, 0)` | $(4, 16, 16)$ | 真邻居放行 |

**关键断言（shape_tracking.py 内置）**：值域 $\subseteq \{0, -100\}$；对角线全 0；对称 `mask[i,j]==mask[j,i]`；被屏蔽总数 448。

---

## ⑤ debug 实验指南

直接运行 `experiment.py`：

```
D:\env\anaconda\envs\ssl_cv\python.exe "...\05_attention_mask\experiment.py"
```

| 实验 | 内容 | 该看到什么 |
|---|---|---|
| 1 | 9 宫格编号图 | 8×8 图，块 0/1/2 在上三行，3/4/5 在中两行，6/7/8 在下两行 |
| 2 | 每个窗口 mask 字符画 | 窗口 0 全 `.`；窗口 1 左右两半互屏蔽；窗口 3 四个 2×2 角块两两屏蔽 |
| 3 | 屏蔽计数 | 窗口 0/1/2/3 = 0/128/128/192，总计 448 |
| 4 | 值域检查 | 只含 {0, -100}，对角线全 0，对称 |

调试建议：若窗口 mask 与预期不符，先打印 `img_mask`（实验 1）核对 9 宫格编号，再核对 `shift_size` 与 `cyclic_shift` 的移位方向是否一致（mask 的切片假设是"向左上移位"）。

---

## ⑥ 单元测试覆盖点

`test_attention_mask.py`（unittest）：

| 测试 | 断言 |
|---|---|
| `test_mask_shape` | mask 形状 $(nW, M^2, M^2)$ |
| `test_value_range` | 值域只含 {0, -100} |
| `test_diagonal_all_zero` | 对角线全 0（自己永远可见） |
| `test_symmetry` | `mask[i,j]==mask[j,i]` |
| `test_shift_zero_all_zero` | `shift_size=0` 时全 0（等价无 mask） |
| `test_handcrafted_windows` | 手工 9 宫格区域号网格逐元素比对 4 个窗口 |
| `test_window_partition_consistency` | `window_partition` 与模块 04 语义一致 |
| `test_count_masked_total` | 总屏蔽数 448 |

---

## ⑦ 与前后模块的关系

- **上游（04 移位窗口）**：本模块直接解决 04 遗留的"伪邻居"问题——`roll → partition → (加 mask 的注意力) → reverse → unroll` 才是完整的 SW-MSA。
- **下游（swin 完整模型 / 09）**：mask 与相对位置偏置（03）在 `WindowAttention.forward` 里**相加**到注意力分数上：`attn = QKᵀ/√d + bias + mask`。二者正交——**偏置管"有多近"，掩码管"能不能看"**。
- **与 W-MSA 的关系**：W-MSA（`shift_size=0`）**根本不构造 mask**（`mask=None`），本模块的 `build_attn_mask(shift=0)` 恰好返回全 0，正好印证"移位为 0 时无需屏蔽"。
