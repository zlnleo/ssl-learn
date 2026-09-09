# 模块 08 · BasicLayer（Stage）数学推导

> 学习顺序：`06_patch_merging` → `07_swin_block` → `08_basic_layer`
>
> 本文档给出 stage 抽象的动机、Swin-Tiny 各 stage 的 shape 演化与参数量/计算量分布公式。

---

## 1. 为什么需要 stage 抽象

一个 stage = `depth` 个 SwinBlock（同分辨率、同通道）+ 末尾一个可选 PatchMerging。
把网络组织成 stage，本质上是复刻 CNN 的**金字塔**：

| 阶段 | 分辨率（224 输入，patch=4） | 通道 | 深度 | 头数 |
|---|---|---|---|---|
| stage1 | 56×56 = 3136 | 96 | 2 | 3 |
| stage2 | 28×28 = 784 | 192 | 2 | 6 |
| stage3 | 14×14 = 196 | 384 | 6 | 12 |
| stage4 | 7×7 = 49 | 768 | 2 | 24 |

每个 stage 内分辨率与通道**恒定**（SwinBlock 不改变它们），stage 之间用 PatchMerging
做一次“分辨率减半、通道翻倍”。这样得到的层级化特征金字塔可直接被检测/分割头复用
（如 FPN 从不同 stage 取多尺度特征）。

## 2. 为什么偶数块 W-MSA、奇数块 SW-MSA

`BasicLayer` 构造时 `shift_size = 0 if i % 2 == 0 else window_size // 2`：

- 每个 stage 内部按 **W, SW, W, SW, ...** 交替；
- 单独一个 W-MSA 或 SW-MSA 都无法让信息跨窗口流动，只有**成对出现**才能形成
  “窗口内建模 → 跨窗口交换”的完整周期；
- 因为 stage 内分辨率/通道不变，W 与 SW 的窗口数量、每层计算量完全一致，
  交替不增加任何额外开销（详见模块 07 的推导）。

## 3. 为什么 PatchMerging 放在 stage 之间（而非之内）

1. **语义清晰**：stage 是“同一尺度下的特征精炼”，降采样是“切换尺度”。混在一起会
   破坏“层内同构、层间异构”的干净结构。
2. **信息充分混合后再降采样**：先用 depth 个 block 在该尺度充分提取特征，再做
   2×2 折叠降采样，信息损失更小。
3. **计算量节奏**：若在 stage 内随意降采样，分辨率与通道的“减半/翻倍”抵消关系
   （见 §5）会被打乱，无法保证每 block 计算量恒定。

## 4. drop_path 按深度递增（stochastic depth）的动机

stochastic depth 以概率 $p$ 整条跳过某个 block。$p$ 应**随深度线性增大**：

$$
p_i = p_{\max}\cdot \frac{i}{\text{depth}-1}
$$

- **浅层 block**（$i$ 小）提取的是通用低级特征，不应被跳过，$p\approx0$；
- **深层 block**（$i$ 大）特征更特化、冗余更高，可以更大胆地跳过；
- 效果：等效于在训练时随机采样一个“更浅”的子网络，是深度方向的正则化，
  与宽度方向的 Dropout 互补。

数值例（depth=6, rate=0.3）：

$$
p = [0.00,\ 0.06,\ 0.12,\ 0.18,\ 0.24,\ 0.30]
$$

## 5. 每 stage 参数量与计算量分布公式

### 5.1 记号

设 0-indexed stage $i \in \{0,1,2,3\}$：

- 分辨率 token 数：$hw_i = \dfrac{hw}{4^i}$（$hw=56^2=3136$）
- 通道数：$C_i = 96\cdot 2^i$
- 深度：$d_i = [2,2,6,2]$
- 窗口大小 $M=7$，头数 $h_i = 3\cdot 2^i$

### 5.2 每 block 计算量（关键推导）

每 block 计算量由两部分组成（见模块 07 §6）：

$$
\text{MACs}_{\text{block}} \approx 12\,hw_i\,C_i^2 + 2\,hw_i\,M^2\,C_i
$$

投影+MLP 项 $12 hw_i C_i^2$ 主导。代入 $hw_i = hw/4^i$、$C_i = 96\cdot2^i$：

$$
12\,hw_i\,C_i^2
= 12\cdot\frac{hw}{4^i}\cdot(96\cdot 2^i)^2
= 12\cdot hw\cdot 96^2\cdot\frac{4^i}{4^i}
= 12\,hw\cdot 96^2
$$

**结论：分辨率减半（$\div4$）与通道翻倍（$\times4$）精确抵消，每 block 的投影+MLP
计算量与 stage 无关，是个常数。** 因此整个 stage 的计算量只正比于深度：

$$
\text{MACs}_{\text{stage }i} \;\propto\; d_i
$$

> 注：任务描述里 “计算量 $\propto hw\cdot(2^i)^2\cdot d_i/4^i = hw\cdot d_i\cdot 2^i$”
> 的最后一步笔误；正确结果是 $hw\cdot d_i$（与 $i$ 无关的常数乘深度）。这正是 Swin 能
> 在高分辨率上高效运行的根本原因。

### 5.3 数值表（Swin-Tiny）

每 block 投影+MLP 项 $= 12\cdot3136\cdot96^2 = 12\cdot3136\cdot9216 = 346{,}816{,}512$。

| stage | 分辨率 $hw_i$ | 通道 $C_i$ | 深度 $d_i$ | 每 block MACs（约） | stage MACs（约） | 占比 |
|---|---|---|---|---|---|---|
| 1 | 3136 | 96 | 2 | $3.47\times10^{8}$ | $6.94\times10^{8}$ | 16.7% |
| 2 | 784 | 192 | 2 | $3.47\times10^{8}$ | $6.94\times10^{8}$ | 16.7% |
| 3 | 196 | 384 | 6 | $3.47\times10^{8}$ | $2.08\times10^{9}$ | 50.0% |
| 4 | 49 | 768 | 2 | $3.47\times10^{8}$ | $6.94\times10^{8}$ | 16.7% |

（忽略较小的窗口注意力项 $2 hw_i M^2 C_i$ 与 PatchMerging 项；数值保留 3 位有效数字。）

计算量分布：**stage3（depth=6）独占一半计算量**，其余三个 stage 各占约 1/6。

### 5.4 每 stage 参数量

每 block 权重 $\approx 12C_i^2$（+相对位置偏置表 $(2M-1)^2 h_i = 169\cdot h_i$）：

| stage | 通道 $C_i$ | 每 block 权重（约） | 深度 | stage 权重（约） |
|---|---|---|---|---|
| 1 | 96 | 110,592 | 2 | 221,184 |
| 2 | 192 | 442,368 | 2 | 884,736 |
| 3 | 384 | 1,769,472 | 6 | 10,616,832 |
| 4 | 768 | 7,077,888 | 2 | 14,155,776 |

参数量随通道平方增长（$C^2$），因此后两个 stage（尤其 stage4，通道 768）贡献绝大多数
参数，即使其深度只有 2。加上 3 个 PatchMerging（每个 $8C^2$）与 patch embedding，
Swin-Tiny 总参数约 28M。

### 5.5 参数量 vs 计算量的不对称

- **参数量**集中在**通道大**的 stage（stage3/4，$C^2$ 主导）；
- **计算量**集中在**深度大**的 stage（stage3，$d_i$ 主导）。

两者分布不同，这是 Swin（乃至多数分层 ViT）的一个有趣特性：深层高通道 stage 参数多
但深度浅；中层中等通道但深度深、算力大。

## 6. 完整 shape 演化链

```
(B, 3136, 96)   H=W=56
   │ stage1: 2 blocks (W,SW) + PatchMerging
(B, 784, 192)   H=W=28
   │ stage2: 2 blocks (W,SW) + PatchMerging
(B, 196, 384)   H=W=14
   │ stage3: 6 blocks (W,SW,W,SW,W,SW) + PatchMerging
(B, 49, 768)    H=W=7
   │ stage4: 2 blocks (W,SW)（无 PatchMerging）
(B, 49, 768)    H=W=7
```

## 7. 小结

- stage 抽象 = 把“层内精炼（SwinBlock）”与“层间降采样（PatchMerging）”分离；
- W/SW 成对交替保证 stage 内跨窗口信息流动且计算量恒定；
- 通道翻倍与分辨率减半精确抵消 → 每 block 计算量与 stage 无关，stage 计算量 $\propto d_i$；
- 参数量 $\propto C^2$，集中在高通道 stage；计算量 $\propto d_i$，集中在深 stage。
