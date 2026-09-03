# 模块 02 · Window Partition / Reverse（窗口划分与还原）

> 学习顺序：本项目第 2 步。上一模块（01）解决了「窗口内怎么做注意力」，
> 本模块解决「特征图怎么切成窗口、又怎么拼回来」。

---

## ① 为什么需要这个模块（动机）

### 1.1 注意力算子消费的是「序列」，不是「图像」

模块 01 的 `WindowAttention` 的输入输出形状是 `(B_, N, C)`——它是一条扁平的序列：
`B_` 个「注意力单元」、每个单元 `N` 个 token、每个 token `C` 维。

但图像是四维的 `(B, H, W, C)`。要让注意力在窗口内运行，必须先把图像转成
「一堆窗口序列」。这个转换的枢纽就是本模块：

```
(B, H, W, C)  ──window_partition──►  (B*nW, M, M, C)  ──view──►  (B_, N, C)
                                                                      │
                                                                [模块 01] 注意力
                                                                      │
(B, H, W, C)  ◄──window_reverse───  (B*nW, M, M, C)  ◄──view──  (B_, N, C)
```

### 1.2 为什么转换必须「零开销」

如果划分窗口要真的把每个窗口的数据**拷贝**出来，那整个 Swin 的加速优势就被内存搬运吃掉了。
所以本模块的核心诉求是：**只改索引、不搬数据**。PyTorch 的 `view` 和 `permute` 恰好能做到——
它们只改变「如何解释同一块内存」，返回的是同一份数据的「新视图」。

### 1.3 两种「换索引」的本质

- **`view`（reshape）**：重解释内存的**形状**。前提是元素总数不变、且当前内存布局允许直接
  折叠/展开（连续或可「安全广播步长」）。
- **`permute`（transpose 的推广）**：交换**轴的顺序**，即改变每个轴的 `stride`（步长），
  不改变内存中的物理排列。

一句话：`view` 决定「有几个轴、每个轴多大」，`permute` 决定「哪个轴是行、哪个轴是列」。
二者配合就能把一张图的内存**重新解读**成若干个窗口，全程不动一个字节。

---

## ② 核心机制讲解（从第一性原理）

### 2.1 目标：把 4×4 图切成 4 个 2×2 窗口

设输入 `(H=4, W=4, C=1)`，`window_size M=2`。全局 id 编号（每个位置的数值 = 行优先序号 `h*W+w`）：

```
        列 j
      0    1    2    3
 行   ┌────┬────┬────┬────┐
  h 0 │  0 │  1 │  2 │  3 │
    1 │  4 │  5 │  6 │  7 │
    2 │  8 │  9 │ 10 │ 11 │
    3 │ 12 │ 13 │ 14 │ 15 │
      └────┴────┴────┴────┘
```

划分后（窗口按**行优先**编号，`(i,j)` 为窗口的行、列下标）：

```
窗口 (0,0)            窗口 (0,1)
┌─────┬─────┐        ┌─────┬─────┐
│  0  │  1  │        │  2  │  3  │
├─────┼─────┤        ├─────┼─────┤
│  4  │  5  │        │  6  │  7  │
└─────┴─────┘        └─────┴─────┘
   (m,n)=(0,0)(0,1)     (m,n)=(0,0)(0,1)
   (1,0)(1,1)           (1,0)(1,1)

窗口 (1,0)            窗口 (1,1)
┌─────┬─────┐        ┌─────┬─────┐
│  8  │  9  │        │ 10  │ 11  │
├─────┼─────┤        ├─────┼─────┤
│ 12  │ 13  │        │ 14  │ 15  │
└─────┴─────┘        └─────┴─────┘
```

### 2.2 `view` 的拆法：把「高」和「宽」各自拆成两段

关键洞察：`H = (H/M) * M`，`W = (W/M) * M`。所以

$$
(B, H, W, C) = (B,\ H/M,\ M,\ W/M,\ M,\ C)
$$

这一步用 `view` 完成（纯重解释，内存不变）：

```
x: (B, H, W, C)            = (B, H/M, M, W/M, M, C)
   2  4  4  C                   2  2   2  2   2  C
```

现在六个轴的含义依次是：`batch, 行方向的窗口个数, 窗口内行, 列方向的窗口个数, 窗口内列, 通道`。

### 2.3 `permute` 把「窗口编号」提到前面

我们希望窗口编号 `(H/M, W/M)` 相邻、并放在一起，以便最后 `view(-1, M, M, C)` 把前两个
窗口编号轴合并成 `B*nW`。所以把第 3 个轴（列窗口编号）与第 4 个轴（窗口内行）交换：

```
(B, H/M, M, W/M, M, C)  --permute(0,1,3,2,4,5)-->  (B, H/M, W/M, M, M, C)
     ↑      ↑                                        ↑      ↑
  行窗口编号 窗口内行                             行窗口编号 列窗口编号（相邻了）
```

### 2.4 为什么 `permute` 之后必须 `contiguous()` 才能 `view(-1)`

`permute` 只交换轴的 `stride`，**不重新排列内存**。交换后，元素在内存里的物理顺序与新形状
的逻辑顺序不再一致（`is_contiguous() == False`）。

`view(-1)` 要求「把连续一段内存直接折叠成一维」，这在非连续布局下无法安全进行，
PyTorch 会直接抛错。所以必须先 `.contiguous()`：它**真正重新分配一段连续内存**，
把元素按新的逻辑顺序拷贝好（只有这一步才产生一次拷贝），之后 `view(-1)` 才能安全执行。

> 用 `shape_tracking.py` 的输出直观感受 `stride` 的变化（详见 ④ 与实验）：
> `permute` 后 `is_contiguous=False`，`contiguous()` 后 `is_contiguous=True`，stride 也变了。

### 2.5 最后 `view(-1, M, M, C)`：把 `B*nW` 个窗口排成序列

```
(B, H/M, W/M, M, M, C) --view--> (-1, M, M, C) = (B*nW, M, M, C)
```

- `-1` 让 PyTorch 自动推断为 `B * (H/M) * (W/M) = B*nW`。
- 此时每个「样本」就是一个完整的 `M×M` 窗口；再 `view(B*nW, M*M, C)` 就得到模块 01
  需要的 `(B_, N, C)` 序列（$N=M^2$）。

### 2.6 `window_reverse` 是精确的镜像

```
(B*nW, M, M, C)
  --view--> (B, H/M, W/M, M, M, C)     # 把 B*nW 拆回 (B, H/M, W/M)
  --permute(0,1,3,2,4,5)--> (B, H/M, M, W/M, M, C)
  --contiguous().view(B, H, W, C)
```

与 partition 逐轴互逆，因此 `reverse(partition(x)) == x` **精确成立（误差 0）**。

---

## ③ 逐段代码讲解

代码在 `window_partition.py`，两个纯函数（无参数、无状态）。

### 3.1 `window_partition`

```python
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
```

- `H // window_size`：行方向的窗口个数；`window_size` 是窗口内的行数（`M`）。
- `view(...)`：`(B, H, W, C) → (B, H/M, M, W/M, M, C)`，见 ②.2。
- `permute(0, 1, 3, 2, 4, 5)`：把轴序从 `(0,1,2,3,4,5)` 改成 `(0,1,3,2,4,5)`，
  即「列窗口编号」与「窗口内行」互换，让两个窗口编号轴相邻，见 ②.3。
- `.contiguous()`：见 ②.4，让换轴后的张量在内存里连续。
- `.view(-1, window_size, window_size, C)`：`(B, H/M, W/M, M, M, C) → (B*nW, M, M, C)`。

### 3.2 `window_reverse`

```python
def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

- `B` 由「窗口总数 ÷ 每图窗口数」反推：`B = B*nW / nW`。
- `view(B, H/M, W/M, M, M, C)`：把 `B*nW` 拆回 `(B, H/M, W/M)`。
- `permute(0,1,3,2,4,5)`：与 partition 相反方向换轴。
- `view(B, H, W, C)`：合并 `(H/M, M)` → `H`、`(W/M, M)` → `W`。
- 最后一维用 `-1` 保持对 `C` 的通用性（不写死通道数）。

### 3.3 为什么「非整除」会报错

若 `H` 不能被 `M` 整除，`H // M` 是向下取整，`view` 里的元素总数 `B*(H//M)*M*(W//M)*M*C`
就会和真实总数 `B*H*W*C` 对不上，`view` 抛 `RuntimeError`。这实际是一种隐式的
「尺寸必须整除」校验（测试 4 验证了这一点）。

---

## ④ Tensor Shape 跟踪总表

> 与 `shape_tracking.py` 输出**完全一致**（示例：`B=2, H=W=8, C=3, M=2`）。

**Partition：**

| 步骤 | 操作 | 形状 | stride（内存步长） | 连续？ |
| --- | --- | --- | --- | --- |
| 输入 | `x` | `(2, 8, 8, 3)` | `(192, 24, 3, 1)` | 是 |
| 1 | `view(B,H/M,M,W/M,M,C)` | `(2, 4, 2, 4, 2, 3)` | `(192, 48, 24, 6, 3, 1)` | 是 |
| 2 | `permute(0,1,3,2,4,5)` | `(2, 4, 4, 2, 2, 3)` | `(192, 48, 6, 24, 3, 1)` | **否** |
| 3 | `contiguous()` | `(2, 4, 4, 2, 2, 3)` | `(192, 48, 12, 6, 3, 1)` | 是 |
| 4 | `view(-1,M,M,C)` | `(32, 2, 2, 3)` | `(12, 6, 3, 1)` | 是 |

**Reverse：**

| 步骤 | 操作 | 形状 | stride | 连续？ |
| --- | --- | --- | --- | --- |
| 输入 | `windows` | `(32, 2, 2, 3)` | `(12, 6, 3, 1)` | 是 |
| 1 | `view(B,H/M,W/M,M,M,C)` | `(2, 4, 4, 2, 2, 3)` | `(192, 48, 12, 6, 3, 1)` | 是 |
| 2 | `permute(0,1,3,2,4,5)` | `(2, 4, 2, 4, 2, 3)` | `(192, 48, 6, 12, 3, 1)` | 否 |
| 3 | `contiguous().view(B,H,W,C)` | `(2, 8, 8, 3)` | `(192, 24, 3, 1)` | 是 |

**stride 读法**：`stride[k]` 表示「第 k 个轴前进 1 格，内存下标要跳多少元素」。
例如 `view` 后第 2 轴（`H/M`）stride=48 = M·W·C = 2·8·3，说明行窗口编号每 +1，
要跳过整整 `M·W·C` 个元素——这正对应「往下跳一个窗口高度」。

---

## ⑤ debug 实验指南（experiment.py）

### 跑法

```powershell
D:\env\anaconda\envs\ssl_cv\python.exe "D:\project\self_supervised_learning\07.Swin Transform\02_window_partition\experiment.py"
```

### 预期输出与结论

- **实验 A（位置 id 张量）**：`4×4` 图、`M=2` 划分后，4 个窗口的 id 分别为
  `{0,1,4,5}`、`{2,3,6,7}`、`{8,9,12,13}`、`{10,11,14,15}`——证明窗口编号为行优先、划分正确。
- **实验 B（随机张量互逆）**：三种尺寸下 `max|err| = 0.000e+00`、`精确相等 = True`，
  证明 `view+permute` 是纯索引重排，数值零损失。
- **实验 C（ASCII id 图）**：画出原图 id、窗口边界、每个窗口的 id，直观对照。

### 常见坑

- 忘记 `.contiguous()`：`view(-1)` 会抛 `RuntimeError: view size is not compatible ...`。
- 在非整除尺寸上硬跑：`view` 因元素总数对不上而报错——这正是「必须整除」的隐式约束。

---

## ⑥ 单元测试覆盖点（test_window_partition.py）

| 测试方法 | 覆盖点 |
| --- | --- |
| `test_output_shape` | partition 输出 `(B*nW, M, M, C)`、reverse 输出 `(B, H, W, C)` |
| `test_inverse_multiple_sizes` | 多组随机尺寸下 `reverse(partition(x)) == x` |
| `test_window_order_matches_hand_computation` | 窗口编号顺序与手算公式一致（行优先） |
| `test_nondivisible_raises` | `H` 不被 `M` 整除时抛 `RuntimeError` |
| `test_dtype_preserved` | `float32/float64/int64` 下 dtype 均保持 |

运行：`D:\env\anaconda\envs\ssl_cv\python.exe test_window_partition.py`，预期 `Ran 5 tests ... OK`。

---

## ⑦ 与前后模块的关系

- **上游**：来自 Patch Partition / Patch Embedding 的 `(B, H, W, C)` 特征图。
- **本模块**：`window_partition` 把特征图变成窗口序列 → 供模块 01 的注意力消费；
  `window_reverse` 把注意力结果拼回特征图。
- **下游（后续模块）**：
  - **shifted window（模块 03+）**：平移窗口只是在 partition 之前对特征图做一次
    `torch.roll`（循环平移），再复用本模块的 partition/reverse。本模块是它的直接依赖。
  - **attention mask**：shifted window 里窗口错位，需要 mask 标记「哪些 token 属于同一窗口」，
    该 mask 的构造同样依赖本模块的窗口编号语义。
- **数据流闭环**：`window_reverse(window_partition(x)) == x` 精确成立，是后续所有
  shifted-window 正确性的根基——如果连普通划分/还原都不能精确互逆，平移窗口必然出错。
