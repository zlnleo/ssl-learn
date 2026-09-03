# 模块 06 · Patch Merging（相邻 patch 拼接降采样）

> 学习顺序：`06_patch_merging` → `07_swin_block` → `08_basic_layer`
>
> 运行环境：`D:\env\anaconda\envs\ssl_cv\python.exe`（torch 2.11.0+cu128，脚本默认 CPU）

本模块实现 Swin Transformer 中每个 stage 结束后的降采样操作，把特征图分辨率减半、
通道翻倍。它回答一个核心问题：**Transformer 如何像 CNN 一样建立多尺度金字塔？**

---

## ① 为什么需要（动机）

ViT 从头到尾保持 $\frac{H}{16}\times\frac{W}{16}$ 的固定分辨率和固定通道数，缺少
多尺度特征，且全局注意力的 $O(N^2)$ 代价在大图上不可承受。CNN 用
`pooling + channel doubling` 逐级降采样，得到 56→28→14→7 的金字塔。

Patch Merging 就是把 CNN 的这套节奏搬进 Transformer：

- **分辨率减半** → token 数变为 $1/4$，逐级控制计算量；
- **通道翻倍** → 每个 token 携带更浓缩的信息，表达能力增强；
- **逐级扩大感受野** → 同样 7×7 的窗口，在更粗分辨率上覆盖更大原图范围。

结论：**Patch Merging 是“层级化表示（hierarchical representation）”的关键，
让 Swin 既保留 Transformer 的全局建模潜力，又拥有 CNN 的多尺度金字塔。**

---

## ② 核心机制讲解（第一性原理）

### 第一性原理：空间信息折叠进通道

降采样要回答：如何把 $H\times W\times C$ 变成 $\frac{H}{2}\times\frac{W}{2}\times 2C$
而不丢失信息？朴素做法是“丢弃一半位置”（如 stride=2 只取偶数行偶数列），会丢失
奇数位置的信息。

Patch Merging 的答案：**把相邻 $2\times2$ 邻域的 4 个位置都保留，只是把它们从
“空间维”折叠进“通道维”**。空间缩小 4 倍，通道扩大 4 倍，信息无丢弃，再用
Linear 把 $4C$ 压缩成 $2C$（通道净翻倍）。

### ASCII 图：4 个位置流向新 patch 的哪个通道段

```
原特征图 H=W=4，按 (行奇偶, 列奇偶) 分成 4 路：
      c0    c1    c2    c3
 r0   A0    A1    B0    B1        A=左上(偶偶)   B=右上(偶奇)
 r1   A2    A3    B2    B3        C=左下(奇偶)   D=右下(奇奇)
 r2   C0    C1    D0    D1
 r3   C2    C3    D2    D3

每个新位置 (i,j) 收集旧图 2x2 邻域，4 路各自占一段通道：
  新(i,j) 的 4C 维 = [ x0(左上 A) | x1(左下 C) | x2(右上 B) | x3(右下 D) ]
```

### 为什么 4C → 2C 而不是 → C

- 降到 $C$：信息压缩过度、通道不翻倍，违背层级化目标；
- 保持 $4C$：通道翻两番，$16C^2$ 的参数量会让后续 stage 过快膨胀；
- **降到 $2C$**：在“信息混合”与“通道翻倍”之间取平衡，且与 CNN 的
  channel doubling 节奏一致（96→192→384→768）。

### 为什么先 LayerNorm 再 Linear

拼接后的 $4C$ 维由 4 个来源拼成、分布不一，先 `LayerNorm(4C)`（per-token、作用在
通道维）归一化再交给 `Linear(4C→2C)` 混合降维，稳定训练。详见 `math.md` §6。

### 与 pixel shuffle 互为逆过程

pixel shuffle 是“通道 → 空间”（上采样），Patch Merging 是“空间 → 通道”（下采样），
互为逆操作（忽略可学习投影）。

---

## ③ 逐段代码讲解

文件：`patch_merging.py`

```python
class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        self.norm = norm_layer(4 * dim)                 # LN 作用在 4C 维
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 4C -> 2C
```

- `norm`：`LayerNorm(4C)`，对每个 token 的 $4C$ 个通道做归一化（不跨 token）。
- `reduction`：`Linear(4C→2C)`，无偏置，负责通道混合与降维。

```python
def forward(self, x, H, W):
    B, L, C = x.shape
    x = x.view(B, H, W, C)                       # (B, H, W, C)
```

把序列恢复成 2D，才能做按行/列奇偶的切片。

```python
    if H % 2 == 1 or W % 2 == 1:
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2)) # 奇数尺寸保护
```

若 $H$ 或 $W$ 为奇数，先补一行/一列（右下补 0），保证能被 2 整除。
本项目各 stage 设计尺寸均为偶数，此分支不触发，但保留以健壮。

```python
    x0 = x[:, 0::2, 0::2, :]   # 左上 (B, H/2, W/2, C)
    x1 = x[:, 1::2, 0::2, :]   # 左下
    x2 = x[:, 0::2, 1::2, :]   # 右上
    x3 = x[:, 1::2, 1::2, :]   # 右下
    x = torch.cat([x0, x1, x2, x3], dim=-1)      # (B, H/2, W/2, 4C)
```

按 `(行奇偶, 列奇偶)` 拆成 4 路，再沿通道维拼接。`0::2` 取偶数索引、`1::2` 取奇数索引。

```python
    x = x.view(B, -1, 4 * C)                     # (B, (H/2)*(W/2), 4C)
    x = self.reduction(self.norm(x))             # (B, (H/2)*(W/2), 2C)
```

展平空间维后，先 LN 再 Linear，输出 $(B, (H/2)(W/2), 2C)$。

---

## ④ Tensor Shape 跟踪总表

与 `shape_tracking.py` 输出一致（标准偶数尺寸 $B=2,H=8,W=8,C=96$）：

| 步骤 | 操作 | 输出形状 |
|---|---|---|
| 0 | 输入 `x` | $(2, 64, 96)$ |
| 1 | `view(B, H, W, C)` | $(2, 8, 8, 96)$ |
| 2 | （偶数尺寸）无 pad | $(2, 8, 8, 96)$ |
| 3 | `x0`（左上）`0::2, 0::2` | $(2, 4, 4, 96)$ |
| 4 | `x1`（左下）`1::2, 0::2` | $(2, 4, 4, 96)$ |
| 5 | `x2`（右上）`0::2, 1::2` | $(2, 4, 4, 96)$ |
| 6 | `x3`（右下）`1::2, 1::2` | $(2, 4, 4, 96)$ |
| 7 | `cat(..., dim=-1)` | $(2, 4, 4, 384)$ |
| 8 | `view(B, -1, 4C)` | $(2, 16, 384)$ |
| 9 | `LayerNorm(4C)` | $(2, 16, 384)$ |
| 10 | `Linear(4C→2C)` | $(2, 16, 192)$ |

奇数尺寸分支（$H=5,W=5$）：第 2 步 pad 后 $(B,6,6,C)$，最终输出
$(B, (5{+}1)//2 \cdot (5{+}1)//2, 2C)=(B,9,2C)$。

---

## ⑤ Debug 实验指南

运行 `shape_tracking.py` 观察每一步形状是否与上表一致；若发现输出 token 数不对，
优先检查：

1. `x.view(B, H, W, C)` 是否与传入的 `H, W` 匹配（`L == H*W`）；
2. 切片 `0::2`/`1::2` 是否取对（偶行偶列才是“左上”）；
3. `cat` 的 `dim=-1` 是否为通道维；
4. 奇数尺寸时 pad 顺序 `(0,0, 0,W%2, 0,H%2)` 是否只补右下。

运行 `experiment.py` 可直观看到 `(1,4,4,2)` 位置 id 张量经 2×2 分组后每个新位置
由哪些旧位置组成，以及参数量/计算量核对。

---

## ⑥ 单元测试覆盖点

文件：`test_patch_merging.py`

| 测试 | 覆盖点 |
|---|---|
| `test_output_shape_standard` | $(2,16{\times}16,96)\to(2,64,192)$ 形状正确 |
| `test_grouping_content_matches_manual` | 小例逐元素比对 2×2 分组内容与手算一致 |
| `test_odd_size` | $H{=}5,W{=}5$ 输出 $(H{+}1)//2$ 分辨率 |
| `test_linear_equivalence` | 手动构造权重复算 LN+Linear 与模块输出一致 |

---

## ⑦ 与前后模块关系

- **上游（前置）**：Swin 的 patch embedding（`07` 之前把图像切成 $56\times56$、
  通道 96 的序列，本模块假定输入已是 $(B, H{\times}W, C)$ 序列）。
- **本模块**：在 stage 之间做降采样，$(H,W,C)\to(H/2,W/2,2C)$。
- **下游（后继）**：`07_swin_block`（W-MSA/SW-MSA 在**同一个 stage 内**保持分辨率
  不变）；`08_basic_layer` 把若干个 `SwinBlock` + 末尾一个 `PatchMerging` 组装成
  一个 stage，最终串起 56→28→14→7、96→192→384→768 的完整金字塔。
