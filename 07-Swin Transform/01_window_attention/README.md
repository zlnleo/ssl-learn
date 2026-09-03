# 模块 01 · Window Attention（窗口多头自注意力，最简教学版）

> 学习顺序：本项目第 1 步。先彻底搞懂「把注意力限制在窗口内」这件事本身，
> 再去模块 02 学习「怎么把特征图切成窗口」。

---

## ① 为什么需要这个模块（动机）

### 1.1 全局注意力的代价是「平方级」的

Transformer 的标准多头自注意力（MSA）要求**每个 token 与所有 token 两两计算注意力**。
设特征图共有 $hw$ 个 token（例如 $H=W=56$，则 $hw=56^2=3136$），注意力矩阵的形状是
$hw \times hw$，这一项的**计算量**和**显存**都随 $hw$ 呈平方增长： 

$$
\text{注意力计算量（QK}^\top \text{ 与 } AV\text{）} \propto hw^2 \cdot C,
\qquad
\text{注意力矩阵显存} \propto B \cdot h \cdot hw^2
$$

对于计算机视觉，特征图稍大一点 $hw$ 就会到几千甚至几万（$56^2=3136$、$224^2=50176$），
平方级代价立刻爆炸。以 Swin-T 第一阶段为例：

| 方案 | MACs（公式） | 注意力矩阵显存（B=2, h=3, fp32） |
| --- | --- | --- |
| 全局 MSA（$hw=3136$） | $\approx 2.00\ \mathrm{G}$ | $\approx 225\ \mathrm{MB}$ |
| 窗口 W-MSA（$M=7$） | $\approx 145\ \mathrm{M}$ | $\approx 3.5\ \mathrm{MB}$ |
| 比值 | $\approx 13.8\times$ | $\approx 64\times$ |

> 这就是本模块要讲清楚的**第一个结论**：把注意力限制在窗口内，就能把平方级降为线性级。

### 1.2 为什么是「窗口」而不是别的限制方式

朴素地限制注意力范围有很多选择（只看左邻、只看前 K 个……），但 Swin 选择了
「把特征图切成若干不重叠的 $M\times M$ 小窗，窗口内做**完整的**全局注意力」。
理由有三：

1. **局部性先验**：图像的相邻像素相关性最强，窗口恰好覆盖一个局部邻域，视觉归纳偏置天然合理。
2. **实现干净**：每个窗口内仍是标准 MSA，代码复用度极高，只是 $N$ 从 $hw$ 变成 $M^2$。
3. **可扩展**：后续模块 03+ 会用「shifted window（平移窗口）」让窗口之间也能通信，
   而平移仍然建立在「窗口」这个结构上——所以窗口机制是整个 Swin 的地基。

### 1.3 本模块在前后模块中的位置

```
(输入图像) → Patch Partition → (B, H, W, C)
      │
      ▼
[模块 02] window_partition  ──► (B*nW, M, M, C) ──► view 成 (B_, N, C) 序列
      │                                                      │
      │                                          ┌───────────┘
      ▼                                          ▼
                                 [模块 01] WindowAttention  ← 本模块
                                    (B_, N, C) → (B_, N, C)
                                          │
                                          ▼
                                 [模块 02] window_reverse → (B, H, W, C)
```

关键理解：**本模块的注意力算子不知道自己处理的是「窗口」还是「全图」**。
它只看见一串序列 `(B_, N, C)`。是模块 02 负责把 $B$ 张图切成 $B \times nW$ 个窗口序列。
「算子」与「划分」解耦，是 Swin 简洁的根源。

---

## ② 核心机制讲解（从第一性原理）

### 2.1 注意力到底是什么

给定 $N$ 个 token，每个 token 用一个 $C$ 维向量表示。注意力回答一个问题：

> 「当我要更新第 $i$ 个 token 时，应该从其它 token 身上**借多少信息**？」

计算分四步（先忽略多头）：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V \quad \text{（线性投影）}
$$

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right), \qquad
Y = A V \quad \text{（加权求和）}
$$

- $QK^\top$：第 $i$ 行第 $j$ 列 = 第 $i$ 个 query 与第 $j$ 个 key 的**相似度**（内积）。
- $\mathrm{softmax}$：把相似度变成**概率**（每行和为 1）。
- $AV$：第 $i$ 个 token 的新表示 = 所有 token 的 value 按概率加权求和。
- 除以 $\sqrt{d}$：见下文 2.4。

### 2.2 多头注意力：让模型同时关注多种关系

单头只能表达一种「相似度」。多头把 $C$ 维拆成 $h$ 份，每份 $d = C/h$ 维，
各自独立算一次注意力，最后拼回来。数学上等价于在 $h$ 个子空间里并行做注意力：

```
C 维向量  ──拆──►  h 个头，每头 d 维
  x             head_0 : [---- d ----]
                head_1 : [---- d ----]
                ...
                head_{h-1} : [---- d ----]
```

### 2.3 全局 → 窗口：唯一变化是 $N$

| 量 | 全局 MSA | 窗口 W-MSA |
| --- | --- | --- |
| 注意力单元个数 | 1（整张图） | $nW = (H/M)\times(W/M)$ |
| 每个单元的 token 数 $N$ | $hw$ | $M^2$ |
| 注意力矩阵 | $1$ 个 $hw\times hw$ | $nW$ 个 $M^2\times M^2$ |
| 注意力计算量 | $hw^2 C$ | $nW\cdot M^4\cdot C = hw \cdot M^2 \cdot C$ |

因为 $nW \cdot M^2 = hw$，所以窗口注意力的注意力项是 $hw \cdot M^2 \cdot C$：
当窗口大小 $M$ 固定为常数时，它对 $hw$ 是**线性**的。这正是「$O(hw)$（$M$ 固定）」的含义。

ASCII 对比（$4\times4$ 图、窗口 $2\times2$）：

```
全局 MSA：每个 token 看全部 16 个 token
        q0 q1 ... q15
  k0  [ 1  1  ...  1 ]     注意力矩阵 16x16
  k1  [ 1  1  ...  1 ]
  ... [ ...           ]
  k15 [ 1  1  ...  1 ]

窗口 W-MSA：只在窗口内两两互看（4 个窗口，每个 4x4）
  窗口0(左上) 窗口1(右上) 窗口2(左下) 窗口3(右下)
  [4x4 块]   [4x4 块]   [4x4 块]   [4x4 块]
  其余地方 = 0（不交互），对应 attention mask（后续模块再讲）
```

### 2.4 为什么要除以 $\sqrt{d}$（scale）

假设 $q,k$ 的每个分量独立、均值为 0、方差为 1（近似成立）。点积

$$
q \cdot k = \sum_{t=1}^{d} q_t k_t
$$

的方差是 $d$（各分量独立相加）。于是 $d$ 越大，$QK^\top$ 的数值越大，
经过 softmax 后梯度会趋近于饱和区（softmax 的梯度在输入很大时几乎为 0），训练变慢甚至不稳。

除以 $\sqrt{d}$ 后点积方差回到 1，与维度无关：

$$
\mathrm{Var}\!\left(\frac{q\cdot k}{\sqrt{d}}\right)
= \frac{1}{d}\mathrm{Var}(q\cdot k) = \frac{d}{d} = 1
$$

所以代码里 `self.scale = self.head_dim ** -0.5`。

### 2.5 MACs 公式（本模块的「账本」）

约定：**1 MAC = 1 次乘加 ≈ 2 FLOPs**。一次注意力（QKV 投影 + 注意力 + 输出投影）：

$$
\text{MACs} = 4\,hw\,C^2 + 2\,hw \cdot \text{win\_tokens} \cdot C
$$

| 项 | 来历 |
| --- | --- |
| $4hwC^2$ | QKV 投影 $3hwC^2$ + 输出投影 $hwC^2$ |
| $2hw\cdot \text{win\_tokens}\cdot C$ | $QK^\top$ 与 $AV$ 各 $hw\cdot\text{win\_tokens}\cdot C$ |

其中 $\text{win\_tokens}$ 全局时为 $hw$，窗口时为 $M^2$。

---

## ③ 逐段代码讲解

代码在 `window_attention.py`，核心类 `WindowAttention`。

### 3.1 `__init__`：声明参数

```python
def __init__(self, dim, window_size, num_heads, qkv_bias=True,
             attn_drop=0.0, proj_drop=0.0):
    super().__init__()
    assert dim % num_heads == 0
    self.dim = dim
    self.window_size = window_size
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5   # 1/sqrt(d)，见 2.4
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)
```

要点：

- `dim % num_heads == 0`：$C$ 必须能被 $h$ 整除，才能均分成 $d = C/h$ 维的头。
- `self.qkv` 用**一个** `Linear(dim, 3*dim)` 一次算出 $Q,K,V$，而不是三个 Linear。
  好处：一次矩阵乘法、共享同一份权重，计算更快。
- `window_size` 在这个类里**实际上只被存起来，不参与 forward**（因为输入已经是窗口序列）。
  它保留是为了与官方接口一致，也便于阅读者理解语义。
- 两个 `Dropout`：`attn_drop` 作用在注意力权重上，`proj_drop` 作用在输出投影后（正则化）。

### 3.2 `forward`：四步注意力

```python
def forward(self, x):
    B_, N, C = x.shape
    # (1) QKV 投影 + 拆头：(B_,N,3C) -> (B_,N,3,h,d) -> (3,B_,h,N,d)
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                 # 各 (B_, h, N, d)
    # (2) 缩放点积注意力
    attn = (q * self.scale) @ k.transpose(-2, -1)    # (B_, h, N, N)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # (3) 加权求和 + 并头
    out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
    # (4) 输出投影
    return self.proj_drop(self.proj(out))
```

**逐行拆解：**

- `self.qkv(x)`：`(B_, N, C) → (B_, N, 3C)`。输出按 `[Q | K | V]` 三段拼接。
- `.reshape(B_, N, 3, h, d)`：把 `3C` 拆成 `(3, h, d)`，即 `(Q/K/V, head, head_dim)`。
- `.permute(2, 0, 3, 1, 4)`：把第 2 维（Q/K/V）提到最前，得到 `(3, B_, h, N, d)`。
  这样 `qkv[0], qkv[1], qkv[2]` 直接就是 `q, k, v`，形状各为 `(B_, h, N, d)`。
  注意这里用 `permute` 换轴后要接着做 `@` 等操作，PyTorch 会返回连续或非连续张量，
  但后续 `transpose` + `reshape` 的写法保证了结果正确（reshape 会自动处理内存布局）。
- `(q * self.scale) @ k.transpose(-2, -1)`：
  `k` 从 `(B_, h, N, d)` 转置成 `(B_, h, d, N)`，于是 `(B_,h,N,d) @ (B_,h,d,N) = (B_,h,N,N)`。
  语义：第 $i$ 个 query 点积第 $j$ 个 key。
- `.softmax(dim=-1)`：沿**最后一维**（key 方向）归一化 → 每行和为 1。
- `(attn @ v)`：`(B_,h,N,N) @ (B_,h,N,d) = (B_,h,N,d)`，每个 token 得到 value 的加权和。
- `.transpose(1, 2)`：把 `(B_,h,N,d)` 变回 `(B_,N,h,d)`，让 head 维回到倒数第二。
- `.reshape(B_, N, C)`：把 `(h, d)` 合并回 `C`，完成「并头」。

---

## ④ Tensor Shape 跟踪总表

> 与 `shape_tracking.py` 输出**完全一致**（示例：`B_=2, N=4, C=8, heads=2, head_dim=4`）。

| 步骤 | 操作 | 形状 | 说明 |
| --- | --- | --- | --- |
| 输入 | `x` | `(2, 4, 8)` | 2 个窗口、每窗口 4 token、8 通道 |
| 1 | `qkv = qkv(x)` | `(2, 4, 24)` | `3C = 24` |
| 2 | `reshape + permute` | `(3, 2, 2, 4, 4)` | `(3, B_, h, N, d)` |
| 3 | `q / k / v = qkv[0/1/2]` | `(2, 2, 4, 4)` | 各 `(B_, h, N, d)` |
| 4 | `attn = (q*scale) @ k^T` | `(2, 2, 4, 4)` | `(B_, h, N, N)` |
| 5 | `softmax(attn)` | `(2, 2, 4, 4)` | 每行和 = 1 |
| 6 | `out = (attn @ v) → reshape` | `(2, 4, 8)` | 并头回 `(B_, N, C)` |
| 7 | `y = proj(out)` | `(2, 4, 8)` | 输出投影 |

通用形状（任取 `B_, N, C, h`，`d=C/h`）：

$$
x: (B_*,N,C) \to qkv:(B_*,N,3C) \to q,k,v:(B_*,h,N,d)
\to attn:(B_*,h,N,N) \to out:(B_*,N,C) \to y:(B_*,N,C)
$$

---

## ⑤ debug 实验指南（experiment.py）

### 跑法

```powershell
# CPU（默认，快速可复现）
D:\env\anaconda\envs\ssl_cv\python.exe "D:\project\self_supervised_learning\07.Swin Transform\01_window_attention\experiment.py"

# GPU（可选）
D:\env\anaconda\envs\ssl_cv\python.exe "D:\project\self_supervised_learning\07.Swin Transform\01_window_attention\experiment.py" --device cuda
```

### 预期输出（CPU，本机实测）

```
全局 MSA   MACs =     2003.8 M   (2.004 G)
窗口 W-MSA MACs =      145.1 M   (0.145 G)
总比值          =    13.81 x
其中注意力部分比值 = hw / M^2 = 3136/49 = 64.0 x

全局 MSA   :      82.6 ms / forward
窗口 W-MSA :       3.2 ms / forward
实测加速比 :    25.59 x

全局 MSA   :    225.1 MB
窗口 W-MSA :      3.5 MB
```

以及一张 `H=W ∈ {14,28,56,112,224}` 的 MACs 对比表，和一张 16×16 的 attention map 字符热力图。

### 结论（实验要你「看见」的三件事）

1. **公式账对得上**：全局 $\approx 2.00$ G、窗口 $\approx 145$ M，比值 $\approx 13.8\times$；
   而「注意力部分」单独看是 $hw/M^2 = 64\times$（总比值被两边都有的 $4hwC^2$ 投影项稀释了）。
2. **实测耗时同数量级地下降**：本机 CPU 上全局约 82.6ms、窗口约 3.2ms，约 25×。
   实测比值不必等于 MACs 比值（软硬件、内存带宽、softmax 等都有影响），但**下降趋势是确定且巨大的**。
3. **显存差距更大**：注意力矩阵 225MB → 3.5MB（约 64×），这对显存受限的场景是决定性优势。

### 常见坑

- 把「总比值 13.8×」和「注意力比值 64×」混为一谈：前者包含投影项、后者只看注意力项，两个都要会算。
- 直接用 `time.time()` 且不热身：首次调用含编译/分配开销，务必热身 + 多次取均值（脚本已做）。

---

## ⑥ 单元测试覆盖点（test_window_attention.py）

| 测试方法 | 覆盖点 |
| --- | --- |
| `test_output_shape` | 输出形状 `(B_, N, C)` 正确 |
| `test_head_dim_equivalence` | 不同 `num_heads` 下 `head_dim = dim // num_heads` 恒成立 |
| `test_softmax_rows_sum_to_one` | softmax 后每行和 ≈ 1（概率解释） |
| `test_window_size_one_is_per_token` | `window_size=1` 退化为逐 token 线性变换（token 间不交互） |
| `test_grad_flows` | 损失反向传播，`x.grad` 与 `qkv.weight.grad` 均存在且有限 |
| `test_qkv_bias_false` | `qkv_bias=False` 可运行，`qkv.bias` 为 `None` |
| `test_global_equals_single_big_window` | 同一权重下 `WindowAttention(window_size=H)` == 手写全局 MSA |

运行：`D:\env\anaconda\envs\ssl_cv\python.exe test_window_attention.py`，预期 `Ran 7 tests ... OK`。

---

## ⑦ 与前后模块的关系

- **上游（模块 02）**：`window_partition` 把 `(B,H,W,C)` 切成 `(B*nW, M, M, C)`，
  再 view 成 `(B_, N, C)`（$N=M^2$）喂给本模块；本模块输出后由 `window_reverse` 还原。
- **本模块**：只管「序列内部的注意力」，不关心窗口从哪来、到哪去。
- **下游（后续模块）**：
  - **相对位置偏置**：在 `attn` 上额外加一个可学习的 `(M^2, M^2)` 偏置（本模块为最简版，暂缺）。
  - **attention mask**：shifted window 中窗口是错位的，需要 mask 挡住不属于同一窗口的 token 对。
  - **shifted window**：交替使用常规窗口与平移窗口，让信息跨窗口流动（本模块的窗口算子原样复用）。
- **特例关系**：全局 MSA 是本模块在 `window_size = H = W` 时的特例（测试 7 已验证）。
