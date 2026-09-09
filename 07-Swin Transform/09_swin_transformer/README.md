# 模块 09：完整 Swin Transformer 组装

> 从零学习项目的第 9 个（收尾）模块。前面 01-08 模块分别拆解了 PatchEmbed、窗口划分/还原、
> W-MSA/SW-MSA、相对位置偏置、attention mask、PatchMerging、stochastic depth 等「零件」，
> 本模块把它们**总装**成一个可训练、可迁移的完整视觉骨干 `SwinTransformer`。

- 配套代码：`swin_transformer.py`（自包含实现）、`shape_tracking.py`、`experiment.py`、`test_swin_transformer.py`
- 数学推导：`math.md`（参数量 / MACs / 感受野的完整公式与数值例）

---

## ① 为什么需要：把 01-08 总装成一个完整骨干

拆零件是为了看懂每一行，但单个零件没有「可训练、可部署」的价值。组装的意义有三层：

1. **打通数据流**：PatchEmbed 的 `(B,3,224,224)→(B,3136,96)` 之后，token 要依次穿过 4 个 stage，
   每个 stage 里 H/W 折半、通道翻倍，最后经 LN + 全局平均池化变成固定长度的特征向量。
   这些「零件」只有按正确的形状契约串起来，模型才跑得通。
2. **引入「stage 调度」这一新概念**：01-08 学的是单个块内部，而组装要回答
   *「stochastic depth 的 drop 概率怎么按块递增」「H/W 怎么随 stage 更新」
   「最后一层通道数 num_features 怎么算」*——这些是单块里看不到的全局逻辑。
3. **对照 ViT / ResNet 说清设计取舍**：

| 设计点 | ViT | ResNet | Swin（本模块） |
|---|---|---|---|
| 起步粒度 | patch 16×16 | 7×7 卷积 | patch 4×4（更细） |
| 分辨率变化 | 全程不变 | 逐级下采样 | 逐级下采样（4 级） |
| 注意力 | 全局 $O(N^2)$ | 无（卷积） | 窗口局部 $O(N)$，靠 SW-MSA 跨窗 |
| 感受野 | 天生全局 | 逐层扩大 | patch+3 次 merging 后覆盖全图 |
| 归纳偏置 | 弱 | 强（平移等变） | 中等（窗口+层级结构） |

一句话：Swin 把 ViT 的「全局注意力」换成「窗口注意力 + 层级合并」，既保留了 Transformer 的表达能力，
又把复杂度压到近似线性，同时获得类似 CNN 的多尺度特征金字塔——这是它能同时打好分类/检测/分割的原因。

---

## ② 总装结构讲解（含 ASCII 结构图）

### 2.1 顶层数据流

```
输入图像 (B, 3, 224, 224)
        │
        ▼
┌─────────────────────────────────────────────┐
│ PatchEmbed  Conv2d(3→96, k=4, s=4) + LN      │  把图切成 56×56 个 patch
└─────────────────────────────────────────────┘
        │ (B, 3136, 96)
        ▼
┌─────────────────────────────────────────────┐
│ Stage 1  dim=96, heads=3                     │
│   2 × SwinBlock (W-MSA + SW-MSA 交替)        │
│   + PatchMerging(96→192)                     │
└─────────────────────────────────────────────┘
        │ (B, 784, 192)   H/W: 56→28
        ▼
┌─────────────────────────────────────────────┐
│ Stage 2  dim=192, heads=6                    │
│   2 × SwinBlock + PatchMerging(192→384)      │
└─────────────────────────────────────────────┘
        │ (B, 196, 384)   H/W: 28→14
        ▼
┌─────────────────────────────────────────────┐
│ Stage 3  dim=384, heads=12                   │
│   6 × SwinBlock + PatchMerging(384→768)      │
└─────────────────────────────────────────────┘
        │ (B, 49, 768)    H/W: 14→7
        ▼
┌─────────────────────────────────────────────┐
│ Stage 4  dim=768, heads=24                   │
│   2 × SwinBlock（无 downsample）             │
└─────────────────────────────────────────────┘
        │ (B, 49, 768)
        ▼
      LayerNorm(768)
        │
        ▼
  全局平均池化 mean(dim=1)  ──►  (B, 768)
        │
        ▼
  分类头 Linear(768, num_classes)  ──►  (B, num_classes)
```

### 2.2 单个 SwinBlock 内部（01-08 的零件在这里复现）

```
x (B, L, C) ── shortcut ────────────────────────────┐
   │                                                 │
   LN → view(B,H,W,C) → pad → [roll(shift)] →        │
   window_partition → WindowAttention →              │
   window_reverse → [roll 回] → crop → view(B,L,C)   │
   │                                                 │
   └── drop_path ── + ──► x                          │
                         │
   ┌── LN → MLP ── drop_path ── + ──► x'            │
```

要点：`shift_size=0` 是 W-MSA（窗口不移动），`shift_size=window_size//2` 是 SW-MSA（窗口平移 3 个 token），
每个 stage 内偶数块 W-MSA、奇数块 SW-MSA 交替，实现跨窗信息流动。

---

## ③ 逐段代码讲解（重点三处全局逻辑）

代码全部在 `swin_transformer.py`，按「小部件 → PatchEmbed → PatchMerging → SwinBlock →
BasicLayer → SwinTransformer → swin_tiny」顺序组织。这里只讲**组装特有的三处关键逻辑**：

### 3.1 `num_features` 的计算（顶层）

```python
self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
```

- 每个 stage 通道 `dim = embed_dim * 2**i`，最后一个 stage（i = num_layers-1）通道即 `embed_dim * 2**(num_layers-1)`。
- Swin-Tiny：`96 * 2**3 = 768`。这个值同时决定：末尾 `norm` 的维度、全局池化后的特征维度、分类头的输入维度。
- **为什么用公式算而不写死**：换 Swin-B（embed_dim=128）时自动得到 `128*8=1024`，配置一变全链路自动跟随。

### 3.2 `dpr` 列表与按 stage 切片（stochastic depth 的调度）

```python
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
...
drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])]
```

- `torch.linspace(0, drop_path_rate, sum(depths))` 生成「随块索引线性增长」的 drop 概率：
  Swin-Tiny 共 12 块，概率从 0 线性长到 0.1。
- `sum(depths[:i])` 是第 i 个 stage **之前**累计的块数，`sum(depths[:i+1])` 是到第 i 个 stage **结束**的累计块数，
  两者相减正好切出该 stage 的 `depths[i]` 个概率值。
- **为什么这样设计**：浅层块提取的是底层特征，随机丢弃它们伤害更大，所以给更小的 drop 概率；
  深层块冗余度高，可以承受更大的 drop 概率。这是从「单块 DropPath」到「全模型调度」的关键一步。
- 若 `drop_path` 是标量（例如传 0.0），`BasicLayer` 内部会判断 `isinstance(drop_path, (list, tuple))`，
  不是列表就给所有块用同一个标量——保持接口对两种调用方式都兼容。

### 3.3 H/W 随 stage 更新

```python
# BasicLayer.forward
for blk in self.blocks:
    x = blk(x, H, W)          # SwinBlock 需要 H, W 才能 view / pad / 划分窗口
if self.downsample is not None:
    x = self.downsample(x, H, W)
    H, W = (H + 1) // 2, (W + 1) // 2   # 折半，奇数向上取整
return x, H, W
```

- 因为 SwinBlock 的输入始终是 `(B, L, C)` 的「展平序列」，它**不知道**当前 H、W 是多少，
  所以 H、W 必须在外部显式维护并随 downsample 同步更新。
- `(H + 1) // 2` 而非 `H // 2`：奇数分辨率（如 128→64→32→16→8→4 没问题，但 65→33）时向上取整，
  配合 PatchMerging 内部的奇偶 pad，保证任意尺寸都能跑（见 `test_nonstandard_resolution_128`）。
- 这一步把「几何信息（H, W）」和「内容信息（x）」分开传递，是 Swin 与「纯序列 ViT」在工程上的重要差异。

### 3.4 其余零件的落位（简要索引）

| 零件 | 位置 | 作用 |
|---|---|---|
| `window_partition` / `window_reverse` | 小部件 | 窗口划分与还原 |
| `build_relative_position_index` | 小部件 | 相对位置索引（查偏置表用） |
| `build_attn_mask` | 小部件 | SW-MSA 的跨窗屏蔽 |
| `Mlp` / `DropPath` | 小部件 | MLP 与 stochastic depth |
| `WindowAttention` | 小部件 | 窗口注意力 + 相对位置偏置 |
| `PatchEmbed` | 第二部分 | 图 → patch 序列 |
| `PatchMerging` | 第三部分 | 2×2 合并、通道翻倍 |
| `SwinBlock` | 第四部分 | 单块（W-MSA/SW-MSA + MLP） |
| `BasicLayer` | 第五部分 | 一个 stage（depth 个块 + downsample） |
| `SwinTransformer` | 第六部分 | 顶层组装 + 权重初始化 |
| `swin_tiny` | 工厂 | 一行得到 Swin-Tiny |

---

## ④ Tensor Shape 跟踪总表（224×224 全程）

> 与 `shape_tracking.py` 的逐项断言**完全一致**（batch=2，num_classes=1000）。

| 步骤 | 输出形状 | H | W | 说明 |
|---|---|---|---|---|
| 输入图像 | (2, 3, 224, 224) | 224 | 224 | RGB 图 |
| PatchEmbed | (2, 3136, 96) | 56 | 56 | 224/4=56，L=56×56=3136 |
| Stage 1（含 merge） | (2, 784, 192) | 28 | 28 | 通道 96→192，H/W 折半 |
| Stage 2（含 merge） | (2, 196, 384) | 14 | 14 | 通道 192→384 |
| Stage 3（含 merge） | (2, 49, 768) | 7 | 7 | 通道 384→768 |
| Stage 4（无 merge） | (2, 49, 768) | 7 | 7 | 末层不降采样 |
| 末尾 LayerNorm | (2, 49, 768) | 7 | 7 | 仅归一化 |
| 全局平均池化 | (2, 768) | — | — | `x.mean(dim=1)` |
| 分类头 | (2, 1000) | — | — | `Linear(768, 1000)` |

**两条不变量**：① 每个 stage 内部 H/W 不变，只在 stage 末尾的 PatchMerging 处折半；
② 通道每过一个 merging 翻倍：96→192→384→768，且始终满足「H×W 缩小 4 倍、通道扩大 2 倍」的经典 CNN 金字塔规律。

运行 `shape_tracking.py` 可逐段打印上表并自动断言，是理解数据流最直观的方式。

---

## ⑤ debug 实验指南

`experiment.py` 提供 5 个「总装验收」实验，按顺序跑，哪一步挂了就定位到哪一层：

1. **结构摘要**：确认 4 个 stage 的 dim / blocks / heads / window 是否符合预期配置。
2. **参数量与占比**：总参数应为 **28,288,354 ≈ 28.29M**（`math.md` 明细表逐项可对上）。
   若偏差大，先看 stage 数、`num_features`、分类头是否配错。
3. **logits sanity check**：随机权重下输出均值应接近 0、标准差在 0.1~0.5 量级（实测均值 -0.0007、std 0.2082）。
   若出现 NaN/Inf 或标准差极大，通常是注意力 scale、softmax、或初始化问题。
4. **backbone 模式**：`swin_tiny(num_classes=0)` 应输出 `(2, 768)` 特征（用于下游迁移/自监督对比学习）。
5. **确定性验证**：`model.eval()` 下两次前向必须完全一致；若不一致，检查是否有非确定性算子
   （本实现中 DropPath/Dropout 在 eval 下直通，故应完全一致）。

常用排查命令：

```bash
# 单文件冒烟
python swin_transformer.py
# 逐段形状
python shape_tracking.py
# 总装验收
python experiment.py
# 单元测试
python test_swin_transformer.py
```

---

## ⑥ 单元测试覆盖点

`test_swin_transformer.py`（unittest，8 个用例，可直接 `python test_swin_transformer.py` 运行）：

| 用例 | 覆盖点 |
|---|---|
| `test_forward_default_classes` | 默认 (2,3,224,224)→(2,1000) |
| `test_forward_custom_classes` | num_classes=10 →(2,10) |
| `test_forward_feature_mode` | num_classes=0 →(2,768) 特征模式 |
| `test_nonstandard_resolution_128` | 128×128 可跑，H/W 演化 32→16→8→4→4 正确 |
| `test_param_count_in_range` | 参数量落在 [27.5M, 29.5M] |
| `test_gradient_to_patch_embed` | 梯度能回传到 patch_embed（可训练） |
| `test_eval_deterministic_with_zero_drop_path` | drop_path_rate=0 时 eval 两次前向一致 |
| `test_depths_heads_length_mismatch` | depths/num_heads 长度不匹配时报 IndexError |

---

## ⑦ Swin-Tiny / S / B / L 配置表与下游任务衔接

### 7.1 配置表（窗口统一 7、patch 4、img 224、num_classes=1000）

| 配置 | embed_dim C | depths | num_heads | 参数量(约) | FLOPs(约) |
|---|---|---|---|---|---|
| **Swin-T** | 96 | (2, 2, 6, 2) | (3, 6, 12, 24) | 28.3M | 4.5G |
| **Swin-S** | 96 | (2, 2, 18, 2) | (3, 6, 12, 24) | 49.6M | 8.7G |
| **Swin-B** | 128 | (2, 2, 18, 2) | (4, 8, 16, 32) | 87.8M | 15.4G |
| **Swin-L** | 192 | (2, 2, 18, 2) | (6, 12, 24, 48) | 196.5M | 34.0G |

- 规律：S 比 T 只加深 stage3（6→18 块）；B 再加大 embed_dim（96→128）与 heads；
  L 继续加大 embed_dim（→192）与 heads。窗口/patch/img 尺寸全部不变。
- 用本模块的工厂思路，只需改参数即可得到任意配置：

```python
from swin_transformer import SwinTransformer

# Swin-B
swin_b = SwinTransformer(embed_dim=128, depths=(2, 2, 18, 2),
                         num_heads=(4, 8, 16, 32), num_classes=1000)
```

### 7.2 下游任务衔接

- **图像分类**：默认 `num_classes=1000`（ImageNet-1K）；换数据集改 `num_classes` 即可，其余不动。
- **特征提取 / 自监督对比学习**：`swin_tiny(num_classes=0)` 直接输出 `(B, 768)` 的全局特征
  （本项目「自监督学习」场景的常用入口，接 projection head 即可）。
- **检测/分割（多尺度特征）**：Swin 的 4 个 stage 天然输出 4 个尺度
  `C=96/192/384/768`、`H/W=56/28/14/7`，可直接喂给 FPN / UPerNet 等，无需额外改骨干。
  这也是它相对 ViT 的迁移优势——ViT 只有单一尺度特征图。

---

## 附：文件清单与本模块运行结果摘要

| 文件 | 作用 |
|---|---|
| `swin_transformer.py` | 自包含的完整 SwinTransformer + swin_tiny 工厂 |
| `shape_tracking.py` | 逐段形状跟踪 + 断言 + 参数量打印 |
| `experiment.py` | 总装验收（结构/参数量/sanity/backbone/确定性） |
| `test_swin_transformer.py` | 8 个 unittest 用例 |
| `math.md` | 参数量/MACs/感受野完整推导 |
| `README.md` | 本文件 |

实测关键数字（CPU，batch=2）：

- 参数量：**28,288,354 = 28.288M**（与 math.md 推导、论文 28.3M 一致）
- 前向形状：PatchEmbed (2,3136,96) → Stage1 (2,784,192) → Stage2 (2,196,384) → Stage3 (2,49,768) → Stage4 (2,49,768) → 池化 (2,768) → head (2,1000)
- logits：均值 -0.0007，标准差 0.2082；eval 两次前向完全一致
