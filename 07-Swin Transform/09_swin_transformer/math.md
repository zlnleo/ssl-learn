# 模块 09 数学推导：Swin-Tiny 参数量、MACs 与感受野

> 本文件配合 `swin_transformer.py`、`shape_tracking.py`、`experiment.py` 阅读。
> 约定：$C$ 表示当前 stage 的通道维（embed_dim），$h$ 表示注意力头数，
> $M=7$ 表示窗口边长，$N=H \times W$ 表示当前 stage 的 token 数。
> 全部数值以 **Swin-Tiny**（embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7, img=224, patch=4, num_classes=1000）为例。

---

## 1. 参数量明细：为什么最终是 ≈28.3M

### 1.1 单个 SwinBlock 的参数公式

一个 block 由「窗口注意力 + 两个 LayerNorm + 一个 MLP」组成，通道均为 $C$：

| 部件 | 权重项 | 偏置/可学习项 | 合计 |
|---|---|---|---|
| `qkv` Linear(C→3C) | $3C^2$ | $3C$ | $3C^2+3C$ |
| `proj` Linear(C→C) | $C^2$ | $C$ | $C^2+C$ |
| 相对位置偏置表 | $0$ | $(2M-1)^2\,h = 169h$ | $169h$ |
| `norm1` LayerNorm(C) | $C$ | $C$ | $2C$ |
| `norm2` LayerNorm(C) | $C$ | $C$ | $2C$ |
| `fc1` Linear(C→4C) | $4C^2$ | $4C$ | $4C^2+4C$ |
| `fc2` Linear(4C→C) | $4C^2$ | $C$ | $4C^2+C$ |

把二次项、一次项、偏置表分开求和：

$$
\begin{aligned}
\text{二次项} &= 3C^2 + C^2 + 4C^2 + 4C^2 = 12C^2 \\
\text{一次项} &= 3C + C + 2C + 2C + 4C + C = 13C \\
\text{偏置表} &= (2M-1)^2 h = 169h
\end{aligned}
$$

**单块参数量**：

$$
\boxed{\;\text{Params}_{\text{block}} = 12C^2 + 13C + 169h\;}
$$

这就是「blocks 12C²/块」的来源：二次项 $12C^2$ 占绝对主导，$13C$ 是各处 bias 与 LN 的小尾巴，$169h$ 是相对位置偏置表。

### 1.2 逐 stage 的 block 参数量

| Stage | C | h | depth | 单块 $12C^2+13C+169h$ | 该 stage block 合计 |
|---|---|---|---|---|---|
| 1 | 96 | 3 | 2 | $110592+1248+507=112347$ | $224694$ |
| 2 | 192 | 6 | 2 | $442368+2496+1014=445878$ | $891756$ |
| 3 | 384 | 12 | 6 | $1769472+4992+2028=1776492$ | $10658952$ |
| 4 | 768 | 24 | 2 | $7077888+9984+4056=7091928$ | $14183856$ |

**全部 12 个 block 合计**：

$$
224694 + 891756 + 10658952 + 14183856 = 25959258
$$

### 1.3 PatchMerging（3 个）

每个 merging 由 `LayerNorm(4C)` + `Linear(4C→2C, bias=False)` 组成：

$$
\text{Params}_{\text{merge}} = \underbrace{8C}_{\text{LN}(4C)\ 的 w+b} + \underbrace{(4C)(2C)}_{\text{reduction 权重}} = 8C^2 + 8C
$$

这就是「PatchMerging 8C²」的来源（$8C$ 的 LN 项相对很小，常被略去）。

| Merging | 输入 C | $8C^2+8C$ |
|---|---|---|
| 1（stage1→2） | 96 | $73728+768=74496$ |
| 2（stage2→3） | 192 | $294912+1536=296448$ |
| 3（stage3→4） | 384 | $1179648+3072=1182720$ |

$$
\text{Merging 合计} = 74496 + 296448 + 1182720 = 1553664
$$

### 1.4 首尾部件

| 部件 | 计算 | 数量 |
|---|---|---|
| PatchEmbed `proj`（Conv 3→96, k=4, s=4） | $3\cdot96\cdot4\cdot4 + 96$（bias） | $4704$ |
| PatchEmbed `norm`（LN(96)） | $96+96$ | $192$ |
| 末尾 `norm`（LN(768)） | $768+768$ | $1536$ |
| 分类头 `head`（Linear 768→1000） | $768\cdot1000 + 1000$ | $769000$ |

### 1.5 汇总（与 `experiment.py` 实测一致）

$$
\begin{aligned}
\text{Total} &= 4704 + 192 + 25959258 + 1553664 + 1536 + 769000 \\
            &= 28288354 \approx 28.29\text{M}
\end{aligned}
$$

即 **28,288,354 ≈ 28.3M**，与论文报告的 Swin-T 28.3M 一致，也与 `experiment.py` 的精确统计值一致。

> 观察：block（25.96M）贡献了约 92% 的参数；通道每翻倍一次，单块参数就 $\times4$，
> 但 stage3 通道 384 却有 6 个块，所以它是参数最重的 stage（10.66M）。

---

## 2. 全模型 MACs（乘加次数）推导：≈4.5 GFLOPs 量级

### 2.1 单个 block 的 MACs 公式

记 $N=H\times W$ 为 token 数，$d=C/h$ 为每个头的维度：

| 操作 | 计算量 |
|---|---|
| `qkv`：$N$ 个 token × ($C\to3C$) | $N\cdot C\cdot 3C = 3NC^2$ |
| 注意力 QKᵀ：每窗 $M^2$ 个 token，$(M^2,d)\times(d,M^2)$ | $\frac{N}{M^2}\cdot h\cdot M^2\cdot M^2\cdot d = NM^2C$ |
| 注意力 AV：$(M^2,M^2)\times(M^2,d)$ | $NM^2C$ |
| `proj`：$N\times(C\to C)$ | $NC^2$ |
| MLP：$N\times(C\to4C)+N\times(4C\to C)$ | $4NC^2+4NC^2=8NC^2$ |

$$
\text{MACs}_{\text{block}} = 12NC^2 + 2NM^2C = N\,(12C^2 + 2M^2C)
$$

> 与参数公式对比：$12C^2$ 是「每个 token 都要乘一次的权重矩阵量」，
> 而 $2M^2C$ 是窗口注意力里与 token 数无关、只与窗口面积有关的额外项。

### 2.2 逐 stage 计算表（224×224 输入，patch=4）

| Stage | H×W=N | C | depth | 单块 MACs | stage 合计 |
|---|---|---|---|---|---|
| 1 | 56×56=3136 | 96 | 2 | $3136(110592+9408)=376320000$ | $752640000$ |
| 2 | 28×28=784 | 192 | 2 | $784(442368+18816)=361568256$ | $723136512$ |
| 3 | 14×14=196 | 384 | 6 | $196(1769472+37632)=354192384$ | $2125154304$ |
| 4 | 7×7=49 | 768 | 2 | $49(7077888+75264)=350504448$ | $701008896$ |

$$
\text{Blocks 合计} = 4301939712 \approx 4.30\text{ G MACs}
$$

### 2.3 非 block 部分

| 部件 | 计算 | MACs |
|---|---|---|
| PatchEmbed（Conv 3→96, 4×4, s=4） | $56\cdot56\cdot96\cdot(3\cdot16)$ | $14450688$ |
| 3 个 PatchMerging | $N\cdot8C^2$，且 $N\cdot C^2=7225344$ 恒定 | $3\times57780252=173408256$ |
| 分类头 | $768\cdot1000$ | $768000$ |

$$
\text{非 block 合计} = 14450688 + 173408256 + 768000 = 188626944 \approx 0.19\text{ G}
$$

### 2.4 汇总与「FLOPs」口径说明

$$
\text{总 MACs} = 4301939712 + 188626944 = 4490566656 \approx 4.49\text{ G}
$$

**结论：全模型约 4.5G 次乘加（MACs）**。

关于「4.5 GFLOPs」的口径：论文与多数视觉模型的 4.5 GFLOPs 采用的是
「1 次乘加 = 1 FLOP」的合并计数，因此 MACs 总数 ≈ 4.5G 即对应 4.5 GFLOPs。
若改用「乘法、加法各算一次」（1 MAC = 2 FLOPs，例如 fvcore 的计数法），
则等价于约 9.0 GFLOPs。本项目的 `experiment.py` 采用乘加口径，输出 ≈4.5G。

---

## 3. 感受野分析：patch + 3 次 merging 如何覆盖全图

### 3.1 两条增长路径

Swin 的感受野增长靠**两条路径**协同：

1. **PatchEmbed**：一个 token 直接覆盖 $4\times4=16$ 像素。
2. **PatchMerging**：每次 H/W 折半，等效每个 token 的感受野边长 $\times2$。
3. **窗口自注意力**：窗口内每个 token 能汇聚 $M\times M$ 个 token 的信息，
   而「每个 token 代表的像素面积」随 merging 逐级变大，于是窗口的物理覆盖也逐级变大。

### 3.2 逐级表格（224×224，patch=4，M=7）

| 层级 | 每 token 代表像素 | 特征图 H×W | 7×7 窗口的物理覆盖 | 覆盖整图？ |
|---|---|---|---|---|
| PatchEmbed 后 | $4\times4$ | $56\times56$ | $28\times28$ px | ✗ |
| Stage1 结束 | $4\times4$ | $56\times56$ | $28\times28$ px | ✗ |
| Merging1 后 | $8\times8$ | $28\times28$ | $56\times56$ px | ✗ |
| Stage2 结束 | $8\times8$ | $28\times28$ | $56\times56$ px | ✗ |
| Merging2 后 | $16\times16$ | $14\times14$ | $112\times112$ px | ✗ |
| Stage3 结束 | $16\times16$ | $14\times14$ | $112\times112$ px | ✗ |
| Merging3 后 | $32\times32$ | $7\times7$ | $224\times224$ px | ✓ 全图 |
| Stage4 结束 | $32\times32$ | $7\times7$ | $224\times224$ px | ✓ 全图 |

### 3.3 关键结论

- 每次 merging 让「每 token 代表的像素边长」翻倍：$4\to8\to16\to32$ 像素。
- 窗口大小固定为 $7\times7$，所以窗口的物理边长 = $7\times(\text{每 token 边长})$：
  $28\to56\to112\to224$ 像素。
- **3 次 merging 之后**，最后一个 stage 的特征图恰为 $7\times7$，而窗口也是 $7\times7$，
  于是「一个窗口 = 整张图」，最后一个 stage 的每个 token 都能直接与全图交互，
  **等效获得全局感受野**。

这正是 Swin 与 ViT 的核心设计取舍：ViT 从一开始就用全局自注意力（$N^2$ 计算），
Swin 则用「局部窗口 + 层级合并」把全局注意力"分 4 级、逐步放大窗口的物理覆盖"，
从而把复杂度从 $O(N^2)$ 降到 $O(N)$（相对 token 数近似线性），却依然在最后覆盖全图。

### 3.4 同 stage 内的跨窗信息流动（W-MSA / SW-MSA）

窗口注意力本身只在窗内交互，若始终不移动窗口，不同窗口之间永远无法交流。
因此每个 stage 内偶数块用 W-MSA、奇数块用 SW-MSA：窗口整体平移 $\lfloor M/2\rfloor=3$ 个 token，
让相邻窗口的 token 在新窗口里相遇，从而打通跨窗信息流。平移带来的「错位窗口」需要
`build_attn_mask` 屏蔽掉本不相邻的 token 对（详见 `swin_transformer.py` 注释）。

---

## 4. 一句话总结

- **参数量 28.3M**：block 二次项 $12C^2$（共 25.96M）为主，加相对位置偏置表 $169h$、PatchMerging $8C^2$、首尾线形层。
- **计算量 ≈4.5G MACs**：每 token 的权重乘法 $12NC^2$ 为主，窗口注意力项 $2NM^2C$ 相对次要。
- **全局感受野**：patch 4×4 + 3 次 merging（边长 4→8→16→32）+ 固定 7×7 窗口 → 末级窗口覆盖 224×224 全图。
