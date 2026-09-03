# 模块 04：移位窗口（Shifted Window / torch.roll 循环移位）

> 模块编号：04　|　学习顺序：03 相对位置偏置 → **04 移位窗口** → 05 注意力掩码
>
> 一句话总结：**固定窗口让跨窗信息无法流动，Swin 用 `torch.roll` 循环移位把图"错位"后再分窗，让新窗口跨越旧边界，实现跨窗交流。**

---

## ① 为什么需要（动机）

### 1.1 固定窗口 → 感受野被锁死

W-MSA 把特征图切成 $M \times M$ 的窗口（默认 $M=7$），每个窗口**只在自己内部**做自注意力，窗口与窗口之间**零交流**。

```
  第 1 层(固定窗口):         第 2 层(还是固定窗口):
  ┌──────┬──────┐            ┌──────┬──────┐
  │ 窗口A │ 窗口B │            │ 窗口A │ 窗口B │
  │(独立) │(独立) │            │(独立) │(独立) │
  ├──────┼──────┤            ├──────┼──────┤
  │ 窗口C │ 窗口D │            │ 窗口C │ 窗口D │
  └──────┴──────┘            └──────┴──────┘
   A 永远只看到 A 内的 token,  B 永远只看到 B 内的 token ...
```

堆叠 $L$ 层后，窗口 A 里每个 token 的感受野**仍然局限在窗口 A**——因为注意力从未跨过窗口边界。深层网络退化成"一堆彼此独立的小块各自演化"，**失去了全局建模能力**。

### 1.2 全局注意力的代价又太高

如果直接用全局自注意力，复杂度是 $O((HW)^2)$，图像稍大就爆炸（这也是模块 01 引入窗口注意力的原因）。我们既要**窗口的低复杂度**，又要**跨窗的信息流动**——移位窗口正是两全其美的答案。

---

## ② 核心机制讲解（第一性原理）

### 2.1 交替两种窗口划分

Swin 让**相邻两层**使用不同的窗口划分：

```
  第 1 层: W-MSA (标准窗口)        第 2 层: SW-MSA (移位窗口)
  ┌─────┬─────┐                    ┌───┬─────────┬───┐
  │  A  │  B  │                    │ A │         │ B │
  ├─────┼─────┤    --移位后-->     │   │ 新窗口跨 │   │
  │  C  │  D  │                    │   │ 越旧边界 │   │
  └─────┴─────┘                    │ C │         │ D │
                                    └───┴─────────┴───┘
```

第 2 层的窗口**不再对齐**第 1 层的窗口边界，于是第 2 层的某个窗口会同时包含第 1 层多个窗口的 token，跨窗信息开始流动。

### 2.2 循环移位（wrap-around）为什么"合法"

实现"错位"最简单的办法是把整张图平移 `shift` 个像素。但普通平移有两个问题：

1. **越界内容丢失**：向左上平移 2 像素，最上两行、最左两列的内容会被"推出图外"。
2. **尺寸改变**：如果保持尺寸，就得在外面补零（padding），引入无意义的值。

`torch.roll` 的**循环移位（wrap-around）**解决了这两个问题：越界的部分**从另一侧绕回**。

```
  普通平移(丢内容):          循环移位(绕回):
  ┌─────────┐                ┌─────────┐
  │         │  <- 上方两行    │  下半   │  <- 上半滚下来
  │  剩下   │     被裁掉      │         │
  │  内容   │                │  上半   │  <- 下半滚上去
  └─────────┘                └─────────┘
```

**语义**（`cyclic_shift`，向左上移位 `s`）：

$$
\text{out}[i, j] = x[(i + s) \bmod H,\ (j + s) \bmod W]
$$

- 尺寸不变、内容一个不丢（只是位置重排）。
- 移位后照常分窗、算注意力，最后再"滚回来"，**中间不损失任何信息**（可逆，模块 2.4/实验 3 验证）。

### 2.3 为什么 `shift = window_size // 2` 最合适

移位量 `shift` 越大，新窗口与旧窗口的错位越大。但有两个约束：

1. **不能等于 0**：$shift=0$ 等于没移位，窗口完全对齐，跨窗交流为零。
2. **不能太大**：$shift = window\_size$ 时，循环移位又回到"完全对齐"（因为 $M$ 是周期），等于白移。

在 $[1, M-1]$ 之间，**$shift = M/2$（即 `window_size // 2`）错位最大**——每个新窗口切割旧窗口网格的位置最"正中间"，能最大程度地跨越旧的 2×2 窗口交界（对 $shift = M/2$ 而言，新窗口的四个象限分别来自四个不同的旧窗口）。这就是官方取 `shift_size = window_size // 2` 的原因。

```
   M=4, shift=2: 新窗口(红框)正好切在旧网格的正中间
   ┌──┬──┬──┬──┐
   ├──┼──┼──┼──┤
   │  │  │◤ │  │  <- 新窗口左上象限来自旧窗口0
   ├──┼──┼──┼──┤      右上象限来自旧窗口1
   │  │  │  │  │      左下象限来自旧窗口2
   ├──┼──┼──┼──┤      右下象限来自旧窗口3
   │  │  │  │  │
   └──┴──┴──┴──┘
```

### 2.4 完整数据流 ASCII 图

```
   x (B,H,W,C)
     │  cyclic_shift(-s,-s)   # 向左上循环滚动 s
     ▼
   shifted (B,H,W,C)
     │  window_partition      # 切成 M×M 小窗
     ▼
   windows (B*nW, M, M, C)
     │  window attention      # 每个窗口内做注意力(第2层此处需 mask)
     ▼
   windows (B*nW, M, M, C)
     │  window_reverse        # 拼回特征图
     ▼
   shifted_out (B,H,W,C)
     │  cyclic_unshift(+s,+s) # 向右下滚回原位
     ▼
   out (B,H,W,C)
```

### 2.5 伏笔：循环移位引入了"伪邻居"

移位后，新窗口（尤其图像边缘处）会同时包含**空间上不相邻**的区域。例如 $H=W=8, M=4, s=2$ 时，右下角的新窗口同时装进了图像的**四个角**（左上/右上/左下/右下各一个 2×2 块）。如果直接 softmax 做注意力，这些相距最远的 token 会被当成"邻居"加权求和——**这是错误**。解决它的注意力掩码，就是模块 05 的主题。

---

## ③ 逐段代码讲解

### 3.1 `window_partition`

```python
x = x.view(B, H // M, M, W // M, M, C)      # (B, nH, M, nW, M, C)
windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, M, M, C)
```

- `view` 把"窗口行数 nH、窗口内行 M、窗口列数 nW、窗口内列 M"四个维度暴露出来。
- `permute(0, 1, 3, 2, 4, 5)` 把维度顺序从 `(B, nH, M, nW, M, C)` 调整为 `(B, nH, nW, M, M, C)`，让"窗口行 nH / 窗口列 nW"相邻、后面紧跟窗口内空间 `(M, M)`。
- `contiguous().view(-1, M, M, C)` 把 `(B, nH, nW)` 一起展平成 batch 维，得到 $(B \cdot nW, M, M, C)$，**窗口按行优先编号**。

### 3.2 `window_reverse`

`window_partition` 的精确逆过程：`view` 拆回 6 维 → `permute(0,1,3,2,4,5)` 还原维度顺序 → `view(B, H, W, -1)`。

```python
B = windows.shape[0] // ((H // M) * (W // M))   # 由窗口数反推 batch
```

### 3.3 `cyclic_shift` / `cyclic_unshift`

```python
def cyclic_shift(x, shift_size):
    return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
```

- `torch.roll(x, shifts=-s, dims=1)`：沿高维循环移位，`out[i] = x[(i + s) % H]`。
- 两个维度同时 `-s`，即"向左上"滚动。
- `cyclic_unshift` 用 `+s`，是精确逆运算。

---

## ④ Tensor Shape 跟踪总表

> 与 `shape_tracking.py` 输出完全一致（以 $B=1, H=W=8, C=4, M=4, s=2$ 为例）。

| 步骤 | 操作 | 形状 | 语义 |
|---|---|---|---|
| 1 | `cyclic_shift(x, 2)` | $(1, 8, 8, 4)$ | 向左上循环滚动，尺寸不变 |
| 2 | `window_partition(shifted, 4)` | $(4, 4, 4, 4)$ | $B\cdot nW=4$ 个窗口，每窗 $4\times4$ |
| 3 | 单个窗口 `windows[0]` | $(4, 4, 4)$ | $M \times M$ 窗口，$C$ 通道 |
| 4 | `windows.view(B*nW, N, C)` | $(4, 16, 4)$ | 窗口内展平成 $M^2=16$ 个 token |
| 5 | `window_reverse(windows, 4, 8, 8)` | $(1, 8, 8, 4)$ | 拼回特征图 |
| 6 | `cyclic_unshift(shifted_out, 2)` | $(1, 8, 8, 4)$ | 向右下滚回原位 |

**关键断言（shape_tracking.py 内置）**：`unroll(roll(x)) == x`；`reverse(partition(x)) == x`；roll 前后内容多重集一致；分窗后总元素数守恒 $= B\cdot H\cdot W\cdot C$。

---

## ⑤ debug 实验指南

直接运行 `experiment.py`：

```
D:\env\anaconda\envs\ssl_cv\python.exe "...\04_shifted_window\experiment.py"
```

| 实验 | 内容 | 该看到什么 |
|---|---|---|
| 1 | id 张量 roll 前后矩阵 | roll 后每行整体左移 2、越界绕回（如第 0 行 `18 19 20 21 22 23 16 17`） |
| 2 | 各新窗口的原始行/列跨度 | 窗口(1,1) 原始行集合 = {0,1,6,7}、列集合 = {0,1,6,7}，跨了两个边界 |
| 3 | unroll(roll(x))==x | True |
| 4 | 无 mask 时窗口(1,1)混入的区域 | 4 个角块：{0,1,8,9}、{6,7,14,15}、{48,49,56,57}、{54,55,62,63} |

调试建议：实验 2 里"跨边界"的判断是 `orig_rows != 连续 M 个整数`——若你的移位方向或 `window_size` 改了，先确认 `cyclic_shift` 的语义与 `torch.roll` 的 `shifts` 符号对应关系。

---

## ⑥ 单元测试覆盖点

`test_shifted_window.py`（unittest）：

| 测试 | 断言 |
|---|---|
| `test_partition_shape` | 分窗形状 $(B\cdot nW, M, M, C)$（多 batch） |
| `test_partition_shape_non_square` | 非正方形 $H\neq W$ 也能分窗 |
| `test_partition_ordering_matches_manual_slice` | 窗口行优先编号与手工切片一致 |
| `test_partition_reverse_roundtrip` | partition/reverse 往返一致 |
| `test_reverse_infers_batch` | reverse 由窗口数反推 batch |
| `test_shift_semantics_wrap_around` | 逐元素验证 `out[i,j]=x[(i+s)%H,(j+s)%W]` |
| `test_unshift_inverse` | unshift 是 shift 的逆 |
| `test_shift_zero_identity` | shift=0 恒等 |
| `test_shift_preserves_multiset` | roll 只重排不丢内容 |
| `test_shift_keeps_shape` | roll 不改变形状 |

---

## ⑦ 与前后模块的关系

- **上游（03 相对位置偏置）**：移位**不改变窗口大小 $M$**，因此相对位置偏置表 $(2M-1)^2$ 行可以**原样复用**——W-MSA 与 SW-MSA 共用同一份可学习偏置表，这是 Swin 设计上的一个精巧点。
- **本模块（04）**：提供了 `roll → partition → reverse → unroll` 的骨架，是 SW-MSA 的"几何基础"。
- **下游（05 注意力掩码）**：本模块明确指出"移位后同一新窗口内会混入不相邻区域"这一**尚未解决**的问题；模块 05 用 9 宫格编号 + 掩码矩阵，把这些伪邻居的注意力强制归零。
