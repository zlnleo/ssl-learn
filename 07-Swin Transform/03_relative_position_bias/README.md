# 模块 03：相对位置偏置（Relative Position Bias）

> 模块编号：03　|　学习顺序：基础 → **03 相对位置偏置** → 04 移位窗口 → 05 注意力掩码
>
> 一句话总结：**窗口自注意力不知道"谁在谁旁边"，我们在 softmax 之前加一张只依赖相对位移 (dh, dw) 的小偏置表，把空间关系注入注意力。**

---

## ① 为什么需要（动机）

### 1.1 自注意力天然是"排列等变"的

标准自注意力（`softmax(QKᵀ/√d)V`）把输入当成一个**无序集合**：它只看 token 两两之间的内容相似度（点积），完全不知道 token 之间的**空间位置关系**。

一个直观的思想实验：假设窗口里有两个 token A、B，它们的内容完全相同（相同的 key/query）。那么无论 A 在 B 的左边还是右边、距离 1 格还是 5 格，它们之间的注意力分数**一模一样**。也就是说，把窗口内所有 token 的顺序任意打乱，注意力输出（在无位置信息时）不变 —— 这就是**排列等变（permutation equivariance）**。

```
   无位置信息的注意力（对 token 顺序不敏感）:

   tokens 排序:    [猫, 狗, 车]        [车, 猫, 狗]   <- 打乱顺序
   注意力矩阵:       完全一样             完全一样     <- 输出不变(内容相同时)
```

但图像是强结构化的：**边缘在某个方向上的邻居、角点、同一物体的不同部位**，这些"相对方位"信息对视觉理解至关重要。所以我们必须给注意力注入位置信息。

### 1.2 Swin 的特殊约束：窗口很小、token 很多

Swin 把整图切成 $M \times M$ 的小窗口（默认 $M=7$），每个窗口内做自注意力。窗口内的 token 都是**局部**的，它们的相对位移有明确、有限的取值。这给了我们一个比"全局绝对位置编码"更精巧的选择空间。

---

## ② 核心机制讲解（第一性原理）

### 2.1 三种位置编码方案对比

| 方案 | 注入方式 | 参数量 | 特点 | Swin 为何不用 |
|---|---|---|---|---|
| 绝对可学习位置编码 | 输入特征直接 `+` 可学习位置向量 | $O(L \cdot d)$，$L$ 是序列长 | 每个绝对位置一个向量 | 图像尺寸变化时需插值；且窗口内关心的是"相对"而非绝对 |
| RoPE（旋转位置编码） | 用旋转矩阵调制 query/key | 0（纯计算） | 只编码相对位置，且内积天然衰减 | 与窗口机制、移位（模块 04/05）组合时代价高、改动大 |
| **相对位置偏置（本文）** | softmax **前**给注意力矩阵 `+` 偏置 | $O((2M-1)^2 \cdot h)$ | 显式、可学习、直观 | —— **Swin 采用** |

核心区别：**RoPE 通过改变 Q/K 间接影响内积；相对偏置直接加在注意力分数矩阵上**，把"内容相似度"和"空间关系"解耦成两个可加项。

### 2.2 为什么窗口内只需 $(2M-1)^2$ 种相对位移

窗口内 token 的坐标是 $(i, j)$，其中 $i, j \in \{0, \dots, M-1\}$。

两个 token 的相对位移 $(dh, dw) = (i_1 - i_2, j_1 - j_2)$ 的取值范围是：

$$
dh \in \{-(M-1), \dots, M-1\}, \quad dw \in \{-(M-1), \dots, M-1\}
$$

每个分量都有 $2M-1$ 种取值，所以组合起来共有：

$$
(2M-1) \times (2M-1) = (2M-1)^2
$$

种可能的相对位移。**无论窗口内有多少 token（$M^2$ 个），它们两两之间的相对位移都逃不出这 $(2M-1)^2$ 种**。因此一张 $(2M-1)^2$ 行的偏置表就够了。

```
   M=3 的窗口，token 坐标 (i, j):
        j=0   j=1   j=2
   i=0  (0,0) (0,1) (0,2)
   i=1  (1,0) (1,1) (1,2)
   i=2  (2,0) (2,1) (2,2)

   相对位移 dh = i1 - i2 的取值: -2, -1, 0, 1, 2   (共 2*3-1 = 5 种)
   相对位移 dw = j1 - j2 的取值: -2, -1, 0, 1, 2   (共 5 种)
   组合总数 = 5 x 5 = 25 = (2*3-1)^2
```

### 2.3 把相对位移"编成行号"：平移 + 展平

偏置表是一个二维张量，我们要用**一个整数行号**去索引它。做法分两步：

**第 1 步：平移（shift）**。把 $dh, dw$ 从 $[-(M-1), M-1]$ 平移到 $[0, 2M-2]$，这样它们都变成非负索引：

$$
dh' = dh + (M-1), \qquad dw' = dw + (M-1)
$$

**第 2 步：展平（flatten）**。把二维 $(dh', dw')$ 按**行主序**折叠成一维行号：

$$
\text{index} = dh' \times (2M-1) + dw'
$$

`build_relative_position_index` 就是把每个 token 对 $(a, b)$ 的相对位移都转成这个行号，得到一个 $(M^2, M^2)$ 的索引矩阵：

```
                    相对位移 (dh, dw)
    (i1,j1) - (i2,j2)  ──平移──►  (dh', dw')  ──展平──►  index = dh'*(2M-1) + dw'
                                                             │
                                                             ▼
                                          relative_position_bias_table[index, :]  (num_heads,)
```

### 2.4 完整数据流 ASCII 图

```
   coords (2, M, M)                rel (2, M^2, M^2)         rel (M^2, M^2, 2)
   ┌─────────────┐                 ┌─────────────┐           ┌─────────────┐
   │ [i 网格]    │  reshape        │ coords[:,:,None]        │ permute     │
   │ [j 网格]    │ ────────►       │   -            ───────► │ [dh, dw] 在 │
   └─────────────┘  (2, M^2)      │ coords[:,None,:]        │ 最后一维    │
                                  └─────────────┘           └──────┬──────┘
                                                              平移 + 乘 + sum
                                                                   │
   bias (M^2, M^2, h)  ◄── table[index] ◄── index (M^2, M^2) ◄────┘
```

---

## ③ 逐段代码讲解

### 3.1 `build_relative_position_index`

```python
coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij"))
# (2, M, M)：coords[0] 是行坐标 i，coords[1] 是列坐标 j
coords = coords.reshape(2, -1)          # (2, M^2)：展平成 M^2 个 token
rel = coords[:, :, None] - coords[:, None, :]   # (2, M^2, M^2)：广播相减
```

- `coords[:, :, None]` 形状 $(2, M^2, 1)$，`coords[:, None, :]` 形状 $(2, 1, M^2)$，广播相减得到 $(2, M^2, M^2)$，其中 `rel[:, a, b] = 第 a 个 token 坐标 - 第 b 个 token 坐标`。
- 这是"相对位移"的矩阵化实现：**外层是 query，内层是 key**。

```python
rel = rel.permute(1, 2, 0).contiguous()   # (M^2, M^2, 2)
rel[:, :, 0] += M - 1                     # dh: [-(M-1), M-1] -> [0, 2M-2]
rel[:, :, 1] += M - 1                     # dw 同理
rel[:, :, 0] *= 2 * M - 1                 # dh' * (2M-1)，行主序
return rel.sum(-1)                        # (M^2, M^2) long 行号
```

- `permute` 把坐标维挪到最后，方便对 `[..., 0]`（dh）和 `[..., 1]`（dw）分别操作。
- 平移后 `dh', dw'` 都在 $[0, 2M-2]$，再 `dh' * (2M-1) + dw'` 得到唯一行号。
- `sum(-1)` 返回 **long 类型**，可直接用于 `table[index]` 索引。

### 3.2 `RelativePositionBias`

```python
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * M - 1) ** 2, num_heads))     # (2M-1)^2 行, 每行 h 维
self.register_buffer("relative_position_index",
                     build_relative_position_index(M), persistent=False)
```

- 偏置表是**可学习参数**，初始化用 `trunc_normal_(std=0.02)`。
- 索引表是**常量 buffer**，`persistent=False` 表示不写进 `state_dict`（模型保存/加载时不重复存它，需要时重建即可）。

```python
def forward(self):
    return self.relative_position_bias_table[self.relative_position_index]
    # (2M-1)^2 x h 的表 + (M^2, M^2) 的索引 -> (M^2, M^2, h)
```

- 这是 PyTorch 的**高级索引 gather**：索引矩阵每个元素去表里取一行，输出多出一维 `h`。

---

## ④ Tensor Shape 跟踪总表

> 与 `shape_tracking.py` 输出完全一致（以 $M=3$ 为例）。

| 步骤 | 操作 | 形状 | 语义 |
|---|---|---|---|
| 0 | `torch.meshgrid(arange(M), arange(M), indexing="ij")` | 2 × $(M, M)$ | 行/列坐标网格 |
| 1 | `torch.stack(mesh)` | $(2, M, M)$ | 两个坐标通道堆叠 |
| 2 | `coords.reshape(2, -1)` | $(2, M^2)$ | 展平成 $M^2$ 个 token |
| 3 | `coords[:, :, None] - coords[:, None, :]` | $(2, M^2, M^2)$ | 广播相减=相对坐标 |
| 4 | `rel.permute(1, 2, 0).contiguous()` | $(M^2, M^2, 2)$ | 坐标维挪到最后 |
| 5 | `rel[:, :, 0] += M-1`、`rel[:, :, 1] += M-1` | $(M^2, M^2, 2)$ | 平移归一到 $[0, 2M-2]$ |
| 6 | `rel[:, :, 0] *= 2M-1` | $(M^2, M^2, 2)$ | 行主序：dh × 表宽 |
| 7 | `rel.sum(-1)` | $(M^2, M^2)$ | 一维行号（long） |
| — | `table[index]` | $(M^2, M^2, h)$ | forward 输出偏置矩阵 |

**关键断言（shape_tracking.py 内置）**：行号 $\in [0, (2M-1)^2)$；不同行号数 $= (2M-1)^2$；对角线统一映射到中心位移 $(0,0)$ 对应的行号。

---

## ⑤ debug 实验指南

直接运行 `experiment.py`：

```
D:\env\anaconda\envs\ssl_cv\python.exe "...\03_relative_position_bias\experiment.py"
```

| 实验 | 内容 | 该看到什么 |
|---|---|---|
| 1 | 打印 $M=3$ 索引表 + 解码出的 dh/dw | 9×9 矩阵，对角线全为中心行号 12 |
| 2 | 范围检查 | min≥0 且 max<25 |
| 3 | 双射验证 | 25 种位移全部出现；每种出现次数 = $(M-\|dh\|)(M-\|dw\|)$ |
| 4 | 注意力再分配 | 加 bias 后近邻权重上升、远邻权重下降 |
| 5 | 参数量核对 | $M=7, h=3$ → 507 |

调试建议：若 4 的结果不符合预期，检查 `fixed_bias` 的符号 —— **bias 越大 → 注意力越大**，所以要"鼓励近邻"就得让近邻 bias 更大（如 `-0.5 * 曼哈顿距离`）。

---

## ⑥ 单元测试覆盖点

`test_relative_position_bias.py`（unittest）：

| 测试 | 断言 |
|---|---|
| `test_shape` | 索引表形状 $(M^2, M^2)$（多组 M） |
| `test_dtype_long` | 索引表 dtype 为 long |
| `test_value_range` | 行号落在 $[0, (2M-1)^2)$ |
| `test_bijection_full_coverage` | $(2M-1)^2$ 种位移全部出现 |
| `test_diagonal_is_center` | 对角线统一映射到 $(0,0)$ 位移 |
| `test_transpose_symmetry_opposite_displacement` | `index[a,b]` 与 `index[b,a]` 对应相反位移 |
| `test_parameter_shape_and_count` | 偏置表形状、507 参数 |
| `test_parameter_learnable` | 偏置表是 Parameter 且 requires_grad |
| `test_index_buffer_not_persistent` | 索引表不进 state_dict |
| `test_forward_shape` | 输出 $(M^2, M^2, h)$ |
| `test_forward_is_gather_from_table` | forward 等价 `table[index]` |
| `test_grad_flows_to_table` | 梯度能回流到偏置表 |

---

## ⑦ 与前后模块的关系

- **上游（01/02 基础 + 窗口注意力）**：本模块输出的偏置矩阵 `(M^2, M^2, h)` 会在 WindowAttention 里 `permute` 成 `(h, M^2, M^2)`，直接广播加到 `QKᵀ/√d` 上再 softmax。
- **下游（04 移位窗口）**：移位（`torch.roll`）只改变 token 的**空间摆放**，窗口大小 $M$ 不变，所以**相对位置偏置表可以原样复用**——这是 Swin 的一个精妙之处：移位窗口 W-MSA 和 SW-MSA 共用同一份相对位置偏置。
- **再下游（05 注意力掩码）**：相对位置偏置刻画"窗口内任意两 token 的坐标差"，而注意力掩码刻画"移位后这两个 token 是否真的空间相邻"。二者正交：**偏置管"有多近"，掩码管"能不能看"。**
