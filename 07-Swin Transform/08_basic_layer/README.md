# 模块 08 · BasicLayer（Stage：多个 SwinBlock + PatchMerging）

> 学习顺序：`06_patch_merging` → `07_swin_block` → `08_basic_layer`
>
> 运行环境：`D:\env\anaconda\envs\ssl_cv\python.exe`（torch 2.11.0+cu128，脚本默认 CPU）

本模块把模块 06 的 `PatchMerging` 与模块 07 的 `SwinBlock`（含全部小部件）组装成
Swin Transformer 的**一个 stage**：`depth` 个 SwinBlock（偶数位 W-MSA、奇数位 SW-MSA）
+ 末尾可选 PatchMerging。多个 stage 串起来就是完整的 Swin 骨干网。

---

## ① 为什么需要（动机）

单看一个 SwinBlock，分辨率与通道都不变，无法形成多尺度。要让 Swin 像 CNN 一样拥有
56→28→14→7、96→192→384→768 的金字塔，需要一个“容器”把：

- **层内精炼**（同一尺度下反复做注意力 + MLP）与
- **层间降采样**（PatchMerging，分辨率减半、通道翻倍）

分离开。这个容器就是 stage（BasicLayer）。它让网络结构清晰、可配置（每 stage 独立
指定通道、深度、头数、是否降采样），也方便下游任务从不同 stage 抽取多尺度特征。

---

## ② 核心机制讲解（第一性原理）

### stage 的层级递进（ASCII）

```
输入 (B, 56*56, 96)
  │
  ▼
┌──────────────────────────────┐  stage1 (dim=96, depth=2)
│  block0 W-MSA   (同尺度精炼)  │
│  block1 SW-MSA  (跨窗口交换)  │
│  PatchMerging 96 -> 192      │──► (B, 28*28, 192)
└──────────────────────────────┘
  ▼
┌──────────────────────────────┐  stage2 (dim=192, depth=2)
│  block0 W-MSA, block1 SW-MSA │
│  PatchMerging 192 -> 384     │──► (B, 14*14, 384)
└──────────────────────────────┘
  ▼ ... 依此类推，直到 7*7、768 通道
```

### 为什么偶数块 W、奇数块 SW

stage 内 `shift_size = 0 if i % 2 == 0 else window_size // 2`，保证相邻块成对
（W, SW, W, SW, ...）。单独 W 或 SW 都无法跨窗口流动；成对出现才能形成完整周期。
因为 stage 内分辨率/通道不变，W 与 SW 的窗口数量与计算量完全一致，交替零额外开销。

### 为什么 PatchMerging 在 stage 之间（而非之内）

1. 语义清晰：stage 内是“同尺度精炼”，降采样是“切换尺度”，分离后结构干净；
2. 先用 depth 个 block 充分提取，再折叠降采样，信息损失小；
3. 若 stage 内随意降采样，会打乱“通道翻倍抵消分辨率减半”的算力节奏。

### drop_path 随深度递增

stochastic depth 以概率 $p_i$ 整条跳过第 $i$ 个 block，$p_i$ 线性增大
$p_i = p_{\max}\cdot\frac{i}{depth-1}$：浅层不跳（通用特征重要），深层大胆跳（特征冗余）。
这是深度方向的正则化，与宽度方向的 Dropout 互补。

---

## ③ 逐段代码讲解

文件：`basic_layer.py`（自包含复制了 PatchMerging、SwinBlock 及全部小部件）

```python
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, ..., downsample=None):
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,   # 偶数位 W，奇数位 SW
                ...
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample
```

- `shift_size` 按索引奇偶交替；
- `drop_path` 支持**标量**（所有块共享）或**列表/元组**（逐块取值），后者用于
  stochastic depth 的线性递增；
- `downsample` 通常传入 `PatchMerging(dim)`，也可为 `None`（最后一个 stage 或
  需要保持分辨率的场景）。

```python
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)                        # stage 内 H、W、C 均不变
        if self.downsample is not None:
            x = self.downsample(x, H, W)            # 降采样
            H, W = (H + 1) // 2, (W + 1) // 2       # 同步更新分辨率
        return x, H, W
```

要点：forward 返回 `(x, H, W)` 三元组，把分辨率显式带出去，因为 PatchMerging 之后
H、W 变了，后续 stage 需要新的 H、W。

---

## ④ Tensor Shape 跟踪总表

与 `shape_tracking.py` 输出一致（两 stage：56→28→14、96→192→384）：

| 阶段 | 步骤 | 输出形状 | (H, W, C) |
|---|---|---|---|
| 输入 | — | $(1, 3136, 96)$ | (56, 56, 96) |
| stage1 | block0 (W) | $(1, 3136, 96)$ | (56, 56, 96) |
| stage1 | block1 (SW) | $(1, 3136, 96)$ | (56, 56, 96) |
| stage1 | PatchMerging | $(1, 784, 192)$ | (28, 28, 192) |
| stage2 | block0 (W) | $(1, 784, 192)$ | (28, 28, 192) |
| stage2 | block1 (SW) | $(1, 784, 192)$ | (28, 28, 192) |
| stage2 | PatchMerging | $(1, 196, 384)$ | (14, 14, 384) |

`downsample=None` 时：输出形状与输入完全一致，H、W 不变。

---

## ⑤ Debug 实验指南

运行 `shape_tracking.py` 对照上表；运行 `experiment.py` 看参数量/计算量分布与
drop_path 递增。

常见排查点：

1. stage 输出形状不对 → 检查 `(H+1)//2` 更新是否遗漏、`downsample` 是否传入正确；
2. 忘记返回 `(x, H, W)` → 后续 stage 拿到的 H、W 还是旧值，`view(B,H,W,C)` 会报错；
3. drop_path 列表长度 ≠ depth → 索引越界；确保列表逐块对齐；
4. 最后一个 stage 传了 `downsample` → 会多降一次采样，与预期分辨率不符。

---

## ⑥ 单元测试覆盖点

文件：`test_basic_layer.py`

| 测试 | 覆盖点 |
|---|---|
| `test_two_stage_output_shape` | 两 stage 56→28→14、96→192→384 形状正确 |
| `test_downsample_none_preserves_shape` | downsample=None 时形状不变 |
| `test_shift_alternation` | blocks 数量正确、第 0 块 shift=0、第 1 块 shift=window//2 |
| `test_drop_path_scalar_and_list` | drop_path 标量/列表两种传法都能构造 |

---

## ⑦ 与前后模块关系

- **上游**：`06_patch_merging`（提供降采样）、`07_swin_block`（提供基本块与小部件），
  本模块把它们**自包含复制**进来并组装。
- **本模块**：定义 `BasicLayer`（一个 stage）与 `linear_drop_path_schedule`
  （stochastic depth 递增调度）。
- **下游**：多个 `BasicLayer` 串接即完整 Swin 骨干网（stage1→2→3→4），再往上接
  patch embedding（输入侧）与分类头/检测头（输出侧）就是完整模型。
