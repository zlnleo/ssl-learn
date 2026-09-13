# 00_MAE代码学习指南

> 本文档是**当前代码的学习地图**，不是新的设计文档。
> 它只解释此刻 `mae/` 目录里真实存在的 `.py` 文件：每个概念都标注 `文件 + 行号`，你可以随时打开对应文件核对。
> 配套设计契约见《00_MAE模块化学习路线与接口契约.md》，本文档的粒度更细，逐行追踪代码。

**本文档对应的 12 个 Python 文件（Phase 1 交付物）：**

| 文件 | 定位 |
|---|---|
| `mae/config.py` | 固定超参数 TOY_CONFIG |
| `mae/patch.py` | patchify / unpatchify |
| `mae/masking.py` | random_masking（MAE 核心新逻辑） |
| `mae/target.py` | patch_normalize / patch_denormalize |
| `mae/modules.py` | PatchEmbed / Attention / Mlp / Block / init_weights |
| `mae/pos_embed.py` | 2D sincos / 可学习位置编码 |
| `mae/encoder.py` | MAEEncoder |
| `mae/decoder.py` | MAEDecoder |
| `mae/model.py` | ToyMAE 组装 + masked_mse |
| `mae/__init__.py` | 包出口 |
| `test_mae_flow.py` | 完整 forward 冒烟测试（逐步打印 + 6 组断言） |
| `masking_example.py` | N=4, V=2 的 random_masking 数值演示 |

**数值约定（下文所有 Shape 都以这两个配置为准）：**

- 代码配置 `TOY_CONFIG`（`mae/config.py` L3-16）：image_size=32, patch_size=8, in_chans=3 → N=16, T=192；mask_ratio=0.75 → V=4, M=12；encoder D=192/L=4/3 头；decoder Dd=128/Ld=2/4 头。
- 测试用 batch：`x = torch.randn(2, 3, 32, 32)`（`test_mae_flow.py` L46），所以下文例子里的 B=2。

---

## 1. 当前 MAE 工程结构

### 1.1 每个文件的职责、依赖关系与数据流位置

```
输入图像 x: (B, 3, 32, 32)
        │
        ├─(目标分支)──► patch.py:patchify ──► target.py:patch_normalize ──► target_norm
        │               (B,N,T) 原始像素           (B,N,T) 归一化像素
        │
        └─(网络分支)──► modules.py:PatchEmbed ──► pos_embed.py ──► masking.py:random_masking
                        (B,N,D) 投影特征           (1,N,D)          (B,V,D) + mask + ids_*
                                 │
                                 ▼
                        encoder.py:MAEEncoder（L×Block, 只算可见 token）
                                 │ latent (B,V,D)
                                 ▼
                        decoder.py:MAEDecoder（mask_token + gather + Ld×Block）
                                 │ pred (B,N,T)
                                 ▼
                        model.py:masked_mse(pred, target_norm, mask) ──► loss 标量
```

| 文件 | 职责 | 依赖谁 | 被谁依赖 | 数据流位置 |
|---|---|---|---|---|
| `mae/config.py` | 集中存放 TOY_CONFIG 字典，所有模块的 shape 都由它推导 | 无 | test_mae_flow.py、masking_example.py（间接） | 起点 |
| `mae/patch.py` | 图像 ⇄ patch 序列的双向映射，**定义全项目唯一的 patch 顺序约定**（行主序） | 无 | target 分支（model.py L58）、可视化（未来） | 目标分支第一站 |
| `mae/masking.py` | 生成随机掩码、切出可见 token、给出还原索引 | 无 | encoder.py L32 | 网络分支的"分岔口" |
| `mae/target.py` | per-patch 归一化（把像素 target 变成结构信息）与反归一化 | 无 | model.py L59 | 目标分支第二站 |
| `mae/modules.py` | 从你的 vit.py 移植的积木：PatchEmbed / Attention / Mlp / Block / init_weights | 无 | encoder.py L11、decoder.py L14、model.py L14 | 网络分支的"积木库" |
| `mae/pos_embed.py` | 生成位置编码参数（默认固定 2D sincos） | 无 | encoder.py L22、decoder.py L26 | 编码/解码两侧各一份 |
| `mae/encoder.py` | ViT encoder 去掉 [CLS]，只编码可见 token，产出 latent + mask + ids_* | masking、modules、pos_embed | model.py L53 | 网络分支中段 |
| `mae/decoder.py` | 填 mask token → gather 还原顺序 → 重建像素 | modules、pos_embed | model.py L54 | 网络分支后段 |
| `mae/model.py` | ToyMAE 组装（唯一知道全局拓扑的地方）+ masked_mse 损失函数 | encoder、decoder、patch、target | test_mae_flow.py | 两条分支汇合点 |
| `mae/__init__.py` | 对外暴露 8 个名字（TOY_CONFIG、patchify、unpatchify、patch_normalize、patch_denormalize、random_masking、ToyMAE、masked_mse） | 所有子模块 | 外部脚本 | 包出口 |
| `test_mae_flow.py` | 用 x=randn(2,3,32,32) 逐步打印完整数据流 + 6 组硬断言 | mae 包 | 无（入口） | 验收 |
| `masking_example.py` | N=4/V=2 的手工 token 数值演示，验证 shuffle→keep→restore→gather | mae.masking | 无（入口） | 验收 |

### 1.2 两个特殊说明

1. **`model.py` 里没有名为 forward_encoder / forward_decoder / forward_loss 的方法**。当前代码的实际分工是：
   - "forward_encoder" 的角色 = `MAEEncoder.forward`（`mae/encoder.py` L26-48）
   - "forward_decoder" 的角色 = `MAEDecoder.forward`（`mae/decoder.py` L31-56）
   - "forward_loss" 的角色 = `ToyMAE.forward` 里 L57-66 的 target 计算块 + 模块级函数 `masked_mse`（`mae/model.py` L19-27）
2. **代码里没有叫 norm_pix_loss 的函数**。论文官方代码的 `norm_pix_loss` 做的事，在当前代码里被拆成两半：`target.py` 的 `patch_normalize`（归一化）+ `model.py` 的 `masked_mse`（只在被遮位置取均值）。详见 §8。

---

## 2. 整体 MAE 数据流（按当前代码逐步标 Shape）

以下每步都标注了产生它的代码位置，Shape 用 B=2 的实际运行值：

```
imgs: (2, 3, 32, 32)                          [test_mae_flow.py L46]
  │
  │  ┌────────────────── 目标分支 (无参数, 无梯度需求) ──────────────────┐
  │  │ patchify(imgs, patch_size=8)   (mae/patch.py L19-21)            │
  │  ▼                                                                 │
  │ target: (2, 16, 192)  原始像素                        [model.py L58]│
  │  │ patch_normalize(target)   (mae/target.py L16-18)                │
  │  ▼                                                                 │
  │ target_norm: (2, 16, 192)   +   target_mean/var: (2, 16, 1)        │
  │                                                       [model.py L59]│
  └──────────────────────────────────────┬──────────────────────────────┘
                                         │
  ┌────────────────── 网络分支 (全部可学习参数在这里) ────────────────────┐
  │ PatchEmbed: Conv2d(8x8, stride=8)   (mae/modules.py L32, L35-36)   │
  ▼                                                                    │
patch_embed: (2, 16, 192)                              [encoder.py L29]│
  │ + pos_embed (1, 16, 192) 广播        (mae/encoder.py L22, L30)     │
  ▼                                                                    │
tokens_full: (2, 16, 192)                              [encoder.py L30]│
  │ random_masking(x_full, r=0.75)       (mae/masking.py L33-45)       │
  ├──► x_vis:       (2, 4, 192)          [masking.py L41]              │
  ├──► mask:        (2, 16) bool         [masking.py L45]              │
  ├──► ids_restore: (2, 16) int64        [masking.py L38]              │
  ├──► ids_keep:    (2, 4)  int64        [masking.py L40]              │
  └──► ids_shuffle: (2, 16) int64        [masking.py L37]              │
  │ 4 × Block(192, 3 heads) 只作用于 x_vis → LayerNorm(192)            │
  ▼                                (mae/encoder.py L34-37)             │
latent: (2, 4, 192)                                    [encoder.py L37]│
  │ decoder_embed: Linear(192 → 128)     (mae/decoder.py L24, L36)     │
  ▼                                                                    │
decoder_embed: (2, 4, 128)                              [decoder.py L36]│
  │ mask_token (1,1,128) .repeat(2, 12, 1) → mask_tokens: (2, 12, 128) │
  │ torch.cat([x_emb, mask_tokens], dim=1)              [decoder.py L39-40]
  ▼                                                                    │
拼接序列: (2, 16, 128)        ← 注意: 这个顺序不是原始 patch 顺序!     │
  │ torch.gather(dim=1, ids_restore)      (mae/decoder.py L43)         │
  ▼                                                                    │
还原序列: (2, 16, 128)        ← 恢复原始 patch 顺序                    │
  │ + decoder_pos_embed (1, 16, 128)      (mae/decoder.py L26, L44)    │
  ▼                                                                    │
restored: (2, 16, 128)                                  [decoder.py L44]│
  │ 2 × Block(128, 4 heads) → LayerNorm(128) → pred Linear(128 → 192)  │
  ▼                                          (mae/decoder.py L46-50)   │
pred: (2, 16, 192)     每个 patch 重建出的"归一化像素"   [decoder.py L50]│
  └──────────────────────────────────────┬──────────────────────────────┘
                                         │
        masked_mse(pred, target_norm, mask)   (mae/model.py L19-27)
                                         ▼
                               loss: 标量 ≈ 1.0498 (随机初始化时)
```

三个全局观察（先记住，后文会展开）：

1. **两条分支共享同一个 patch 顺序约定**（patch.py 行主序），否则 pred 的 patch i 和 target 的 patch i 对不上，loss 就毫无意义。
2. **网络分支从 N=16 缩到 V=4**，只在 decoder 里才恢复成 N=16。
3. **pred 和 target_norm 都是 (B, N, T) 且都是"归一化像素"**，mask 负责告诉 loss 哪些位置需要惩罚。

---

## 3. 从 forward() 开始逐步追踪

当前代码只有一个入口：`ToyMAE.forward`（`mae/model.py` L52-67）。下面按它的**实际调用顺序**逐行追踪。

### 3.1 ToyMAE.forward(imgs, mask_ratio=None)（model.py L52）

**签名设计**：`mask_ratio` 默认 None，此时沿用构造时的 `self.mask_ratio`（L38）；传入时可在不重建模型的情况下临时改掩码率（例如调试时试 mask_ratio=0.5）。这个默认值逻辑同样出现在 `MAEEncoder.forward`（`encoder.py` L27）。

**输入**：`imgs: (B, 3, 32, 32)`，就是原始图像。

**L53：`enc = self.encoder(imgs, mask_ratio)` → 一个 dict**

这一步对应你问的 "forward_encoder()"。它产出的 8 个键（`encoder.py` L39-48）：

| 键 | Shape | 含义 |
|---|---|---|
| `patch_embed` | (B, 16, 192) | Conv 投影后的完整序列，**尚未**加位置编码 |
| `tokens_full` | (B, 16, 192) | `patch_embed + pos_embed`，shuffle 之前的完整序列 |
| `x_vis` | (B, 4, 192) | gather 出的可见 token（进 blocks 之前的快照） |
| `mask` | (B, 16) bool | True = 被遮 |
| `ids_keep` | (B, 4) int64 | 可见 patch 的原始序号 |
| `ids_restore` | (B, 16) int64 | 逆排列索引 |
| `ids_shuffle` | (B, 16) int64 | 随机排列本身 |
| `latent` | (B, 4, 192) | 4 个 Block + LayerNorm 之后的编码结果 |

**为什么 forward 返回 dict？** 这是 phase-1 的学习版设计（`model.py` L7 注释）：把每个中间张量暴露出来，`test_mae_flow.py` 才能逐段打印。正式实现会收紧成"只返回必要三件套"，但学习阶段"什么都看得见"更重要。

**L54：`dec = self.decoder(enc["latent"], enc["ids_restore"])` → dict**

对应 "forward_decoder()"。注意**只传了两个东西**：encoder 的最终输出 `latent` 和还原索引 `ids_restore`。decoder 完全不知道 mask 是怎么生成的——这就是模块化收益（`decoder.py` 的 docstring 也强调 decoder 只依赖这两个输入）。产出的 3 个键（`decoder.py` L52-56）：

| 键 | Shape | 含义 |
|---|---|---|
| `decoder_embed` | (B, 4, 128) | Linear(192→128) 后的可见 token |
| `restored` | (B, 16, 128) | gather 还原 + decoder_pos_embed 后的完整序列 |
| `pred` | (B, 16, 192) | 每个 patch 重建出的归一化像素 |

**L55：`out = {**enc, **dec}`** —— 两个 dict 合并，共 11 个键。

**L58-59：目标分支（对应 "forward_loss()" 的前半段）**

```python
target = patchify(imgs, self.patch_size)      # (B, N, T) 原始像素
target_norm, target_mean, target_var = patch_normalize(target)
```

注意这里**重新对 imgs 做了一次 patchify**，走的是无参数的 `patch.py:patchify`，不是 `PatchEmbed`（后者的输出是 192 维的投影特征，不能当像素回归目标）。

**L60-66：损失计算（对应 "forward_loss()" 的后半段）**

```python
out.update({...,
    "loss": masked_mse(dec["pred"], target_norm, enc["mask"]),  # 标量
})
```

`masked_mse` 是模块级函数（`model.py` L19-27），不是方法：它接收 `pred (B,N,T)`、`target_norm (B,N,T)`、`mask (B,N) bool`，返回 0 维标量。细节见 §8。

**输出**：`out` dict 共 15 个键 = 11 个中间张量 + `target` / `target_norm` / `target_mean` / `target_var` + `loss`。`test_mae_flow.py` L63-94 正是按这个 dict 逐个打印。

**为什么 loss 在 forward 里算？** 学习版把"推理 + 损失"合成一次调用，保证打印出来的一切都出自同一次前向（同一次 shuffle）。训练版通常会把 loss 拆成独立方法，避免推理时浪费计算——这是后续阶段的事。

### 3.2 每一步的"为什么这样设计"

| 步骤 | 输入 → 输出 | 为什么这样设计 |
|---|---|---|
| patch embedding | (B,3,32,32) → (B,16,192) | 图像不是 token，先用 Conv 切成 16 个 192 维 token；conv stride=P 恰好等于无重叠切块 |
| + pos_embed 再 shuffle | (B,16,192) → (B,16,192) | **先加位置编码再打乱**：打乱时位置编码跟着 token 一起走，每个 token 永远带着自己的"坐标"；这是官方实现顺序（`encoder.py` L30 注释） |
| random masking | (B,16,192) → (B,4,192) | 只留 25% 可见，encoder 计算量降 4 倍；同时生成"还原说明书" ids_restore |
| encoder blocks | (B,4,192) → (B,4,192) | 4 个标准 Block，与 ViT 完全一致，只是序列短 |
| decoder embed | (B,4,192) → (B,4,128) | 降维到 decoder 自己的宽度 Dd=128；decoder 轻量、用完就扔 |
| append mask token | (B,4,128) → (B,16,128) | 被遮的 12 个位置需要"占位符"才能凑成完整序列 |
| ids_restore gather | (B,16,128) → (B,16,128) | 把乱序的 [可见+占位] 还原成原始 patch 顺序，保证 pred 与 target 逐位置对齐 |
| decoder blocks + pred | (B,16,128) → (B,16,192) | 128 维特征映射回 T=192 个像素值（归一化空间） |
| masked MSE | pred/target_norm/mask → 标量 | 只惩罚 12 个被遮 patch，迫使模型"无中生有" |

---

## 4. patchify / unpatchify（mae/patch.py）

### 4.1 图像如何变成 patch 序列

`patchify`（L12-22）只有三步，无任何参数：

```python
n_h, n_w = H // P, W // P              # 32/8=4, 4x4 网格
x = x.reshape(B, C, n_h, P, n_w, P)    # 把 H 拆成 (n_h, P), W 拆成 (n_w, P)
x = x.permute(0, 2, 4, 1, 3, 5)        # (B, n_h, n_w, C, P, P)
x = x.reshape(B, n_h * n_w, C * P * P) # (B, N, T)
```

关键理解：`reshape` 不改变内存顺序，它只是**重新解释维度**。原图内存布局是 `(C, H, W)` 行主序，所以：

- 第一次 reshape 把高度 H 切成"块行 n_h × 块内行 P"、宽度 W 切成"块列 n_w × 块内列 P"；
- `permute` 把"块坐标 (n_h, n_w)"提到前面，把"块内坐标 (P, P)"压到后面；
- 最后一次 reshape 融合成 `(B, N, T)`。

### 4.2 patch 的排列顺序（全项目唯一约定）

`patch.py` L3-5 的 docstring 定义了两条规则，**所有模块都遵守**：

1. **patch 之间**：第 i 个 patch = patch 网格第 `i // n_w` 行、第 `i % n_w` 列（行主序）。所以 patch 0 是左上角、patch 3 是右上角、patch 15 是右下角。
2. **patch 内部**：像素按 `(channel, row, col)` 排列，即先通道 0 的 8×8、再通道 1 的 8×8、再通道 2 的 8×8。

这个顺序在 `test_mae_flow.py` ASSERT 6（L154-162）里用"斜坡图"验证过：构造 `ramp = arange(3*32*32).reshape(1,3,32,32)`，则 patch 0 应等于 `ch*1024 + row*32 + col`（ch,row,col ∈ 0..7），patch 15 应等于右下角 8×8 块。

### 4.3 patch 序列如何恢复图像

`unpatchify`（L25-35）是精确逆操作：

```python
x = x.reshape(B, n_h, n_w, C, P, P)
x = x.permute(0, 3, 1, 4, 2, 5)   # (B, C, n_h, P, n_w, P)  ← 与 patchify 的 permute 互逆
x = x.reshape(B, C, h, w)
```

两个 `permute` 互为逆映射：`(0,2,4,1,3,5)` 与 `(0,3,1,4,2,5)`。`test_mae_flow.py` L151-153 断言往返误差 < 1e-6。当前代码里 unpatchify **暂时还没被模型调用**，它是为后续可视化准备的（把 pred 拼回图片看重建效果）。

### 4.4 target 为什么使用 patchify 后的像素

三个原因（对应 `model.py` L58）：

1. **loss 的粒度是 patch**：pred 是 `(B, N, T)` 的像素向量，target 必须同 shape 才能逐元素相减（`masked_mse` L24）。
2. **不能用 PatchEmbed 的输出当 target**：`PatchEmbed` 是带可学习权重的 Conv 投影，输出是 192 维**特征**；而 patchify 输出的是 T=192 维**原始像素值**（8×8×3）。拿特征回归特征没有意义，重建任务回归的就是像素。
3. **无参数**：`patchify` 是纯 reshape，不引入额外复杂度，也保证 target 和图像像素严格一一对应。

一个容易混淆的点：`PatchEmbed` 的 conv 切块（`modules.py` L32, L35-36）和 `patchify` 的切块产生**同样的 patch 网格顺序**（都是行主序），只是前者多了线性投影。正因为顺序一致，encoder 里第 i 个 token 与 target 里第 i 个 patch 才指向图像上同一块区域。

---

## 5. random_masking 深度解析（mae/masking.py）★ 文档重点

`random_masking`（L19-47）是 MAE 唯一的核心新逻辑。整个函数只有 5 个实质性操作，逐一拆解。

### 5.1 逐行拆解

```python
len_keep = int(N * (1.0 - mask_ratio))          # L34
```

N=16, r=0.75 → `len_keep = int(16 * 0.25) = 4`。这就是 V。**注意是 int() 截断**，不是四舍五入。

```python
noise = torch.rand(B, N, ...)                   # L36
ids_shuffle = torch.argsort(noise, dim=1)       # L37
```

每张图独立生成 B×N 个随机数，按行排序。argsort 返回的是"排序后每个位置对应的原始下标"，所以每一行都是一个 0..15 的随机排列。用 argsort(rand) 生成排列是标准技巧（等价于 randperm，但按批并行）。

```python
ids_restore = torch.argsort(ids_shuffle, dim=1) # L38
```

**对排列再取 argsort，得到逆排列**。为什么？见 §5.4 的证明。

```python
ids_keep = ids_shuffle[:, :len_keep]            # L40
x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # L41
```

取排列的前 V 个下标作为"可见名单"，然后用 gather 按名单挑出可见 token。

`torch.gather` 的语义（沿 dim=1）：

```
out[b, i, :] = x[b, index[b, i], :]
```

index 的形状必须与输出一致，所以 `ids_keep (B,V)` 要先 `unsqueeze(-1)` 成 (B,V,1) 再 `repeat` 成 (B,V,D)。这就是为什么测试输出里 `x_vis` 的每一行正好等于 `tokens_full` 中 `ids_keep` 指到的那些行。

```python
mask = ids_restore >= len_keep                  # L45
```

原始顺序的 mask：位置 i 被遮 ⇔ 还原索引 ids_restore[i] 落在"mask token 区段" [V, N)。这是官方代码"先造 [False]*V + [True]*M 再按 ids_restore gather 回填"（L44 注释）的等价简化写法。

### 5.2 手推例子：N=4, V=2, [x0, x1, x2, x3]

以下例子就是 `masking.py` L9-14 docstring 里的例子（`masking_example.py` 是它的可运行版本，seed=42 时实际跑出的是另一组同样合法的排列 [2,0,1,3]）。

**Step 1 — shuffle**：假设随机排列是

```
ids_shuffle = [3, 1, 0, 2]
```

读法："新序列第 0 位放原来的 x3，第 1 位放 x1，第 2 位放 x0，第 3 位放 x2"。

**Step 2 — keep visible tokens**：V=2，取前两个：

```
ids_keep = [3, 1]        → 可见 patch 是 3 号和 1 号
x_vis    = [x3, x1]
```

**Step 3 — append mask tokens**：decoder 把 M=2 个占位符拼在可见 token 后面（占位符用 m 表示，对应代码里的 `mask_token`）：

```
concat = [x3, x1, m, m]        # 下标 0,1 是可见区段, 下标 2,3 是 mask 区段
```

**注意**：concat 的下标 0、1、2、3 **不是**原始 patch 位置——第 0 位装的是 x3，第 2 位装的是占位符。这就是 §7 要解释的"为什么 concat 顺序不是原始顺序"。

**Step 4 — ids_restore**：

```
ids_restore = argsort([3, 1, 0, 2]) = [2, 1, 3, 0]
```

读法："原始位置 0 的内容现在在 concat 的第 2 位；原始位置 1 在 concat 第 1 位；原始位置 2 在 concat 第 3 位；原始位置 3 在 concat 第 0 位"。验证：原位置 0 是 x0（被遮），concat[2] 确实是 m ✓；原位置 1 是 x1，concat[1] = x1 ✓。

**Step 5 — gather 恢复**：

```
gather(concat, ids_restore) = [concat[2], concat[1], concat[3], concat[0]]
                            = [m, x1, m, x3]
```

恰好等于"原始顺序、被遮位置填 m"：`[m, x1, m, x3]` ✓。同时 `mask = ids_restore >= 2 = [True, False, True, False]`，与"位置 0、2 被遮"一致 ✓。

`masking_example.py` L45-66 用 `-1.0` 当占位符完整复现了这五步，并用 `torch.equal` 断言恢复结果。

### 5.3 四个索引对象的角色分工

| 对象 | Shape | 语义 | 谁用 |
|---|---|---|---|
| `ids_shuffle` | (B, N) | 随机排列：shuffle[k] = 打乱后第 k 位放的原序号 | 教学/调试（代码注释：调试/教学用） |
| `ids_keep` | (B, V) | 可见名单 = shuffle 的前 V 个 | encoder 挑 x_vis（L41） |
| `ids_restore` | (B, N) | 逆排列：restore[i] = 原位置 i 的内容在 concat 里的下标 | decoder gather 还原（decoder.py L43） |
| `mask` | (B, N) bool | True = 被遮，原始顺序 | loss 选位置（model.py L65）、可视化 |

### 5.4 为什么 ids_restore = argsort(ids_shuffle) 是逆排列

设 p = ids_shuffle 是 {0..N-1} 上的排列。argsort(p) 的定义是：

```
argsort(p)[i] = 满足 p[j] = i 的那个 j      （把 p 的值排序后, 每个值 i 对应的下标）
```

于是：

```
p[argsort(p)[i]] = p[j] = i    对任意 i
```

即 `gather(p, argsort(p)) == arange(N)`——这正是逆排列的定义。`test_mae_flow.py` ASSERT 2（L112-116）用 `out["ids_shuffle"].gather(1, out["ids_restore"]) == arange(16)` 精确验证了这一点。

**为什么逆排列恰好能还原 concat？** 对原始位置 i 分两种情况：

- i 可见：存在 k < V 使 ids_shuffle[k] = i。由上面的性质，ids_restore[i] = k。而 concat[k] = x_vis[k] = x[i]（gather 的定义）。所以 gather 后位置 i 放回了自己的 token ✓。
- i 被遮：i 不在前 V 个，所以 ids_restore[i] ≥ V，concat[ids_restore[i]] 落在 mask 区段，位置 i 得到占位符 ✓。

这就是 `mask = ids_restore >= len_keep`（L45）与"被遮位置恰好拿到 mask token"这两件事是同一回事的原因。

### 5.5 一个必须知道的坑（写进函数签名）

`masking.py` L19 的 `generator` 参数：每个 forward 都会重新 `torch.rand`（L36），所以每次前向的 mask 都不同。当前单卡学习无影响；将来 DDP 多卡时，必须给每张卡不同的 generator/seed，否则所有卡看到同一套 mask，梯度去相关就失效了。现在只需要知道这个参数是为了解决这个问题而预留的。

---

## 6. Encoder 为什么只处理 visible patches（mae/encoder.py）

### 6.1 代码证据

三个事实，全部可以在代码里找到：

1. **MAEEncoder 的参数列表里没有 mask_token**。`__init__`（L21-24）只有 patch_embed、pos_embed、blocks、norm 四样。对比 `MAEDecoder.__init__`（`decoder.py` L25）赫然有 `self.mask_token`。
2. **blocks 只作用于 x_vis**：`encoder.py` L34-36：

   ```python
   x_vis = x_vis_in
   for blk in self.blocks:     # 只对 V 个可见 token 做注意力
       x_vis = blk(x_vis)
   ```

   进入循环的是 (B, 4, 192)，不是 (B, 16, 192)。
3. **pos_embed 加在 shuffle 之前**（L30），可见 token 带着自己的位置编码被 gather 走，encoder 完全不接触被遮位置的信息。

### 6.2 mask token 为什么不进 encoder

对照 BERT 思考：BERT 的 encoder 必须吃 mask token，因为**同一个 encoder 要同时服务预训练和下游任务**，下游输入里没有 [MASK]，如果不让 mask token 进 encoder，预训练/下游的分布就错位了。

MAE 反过来：encoder 是**预训练专用**的，下游使用时拿到的就是完整 patch 序列（加个 [CLS] 微调），输入里从来不会出现 mask token。所以 mask token 没有理由进 encoder，把它留给 decoder 即可。这一刀砍掉了 encoder 四分之三的 token。

### 6.3 相比完整 ViT 的计算量变化

Block 里注意力矩阵是 L×L（`modules.py` L57 的 `q @ k.transpose(-2,-1)`），逐 token 开销主要看序列长度：

| | 序列长度 | 注意力矩阵 |
|---|---|---|
| 完整 ViT（toy 配置） | N = 16 | 16×16 = 256 |
| MAE encoder（toy 配置） | V = 4 | 4×4 = 16 |

注意力开销降到 **1/16**。放到论文配置（224×224, P=16 → N=196, V=49），注意力降到约 1/16，再加上 MLP 部分线性缩减，整体训练吞吐约为完整 ViT 的 **3 倍以上**。同时显存也省：KV 张量从 (B, N, D) 变成 (B, V, D)。

另一个隐性收益：encoder 的 tokens 是**从图像里 gather 出来的真实 patch**，decoder 的 mask token 只占 decoder 自己的序列，所以 encoder 侧的注意力不需要任何 padding/mask 处理。

### 6.4 为什么"只看得见 25%"还能编码出有用的东西

这是 MAE 的核心赌注：注意力本身对 token 顺序不敏感（q·k 两两计算），顺序信息完全由 pos_embed 携带；打乱后可见 token 仍然各自带着坐标，encoder 实际在做"25% 采样点上的全局注意力"。加上目标是要重建 75% 的内容，模型被迫从稀疏样本里推断全局结构——这正是我们要的自监督信号。

---

## 7. Decoder 如何恢复完整序列（mae/decoder.py）

按 `MAEDecoder.forward`（L31-56）的真实顺序走一遍，输入 `x_vis (B,4,192)` + `ids_restore (B,16)`。

### 7.1 逐行拆解

**Step 1 — decoder embed（L36）**：

```python
x_emb = self.decoder_embed(x_vis)    # (B, 4, 192) → (B, 4, 128)
```

`decoder_embed` 是 `nn.Linear(192, 128)`（L24），把 encoder 的 192 维特征降进 decoder 自己的 128 维空间。

**Step 2 — append mask tokens（L39-40）**：

```python
mask_tokens = self.mask_token.repeat(B, N - V, 1)   # (1,1,128) → (B, 12, 128)
x = torch.cat([x_emb, mask_tokens], dim=1)          # (B, 4+12, 128)
```

`mask_token` 是**共享**的可学习参数 (1,1,128)（L25），repeat 成 12 份。用 `repeat`（拷贝）而不是 `expand`（视图），避免后续原地操作踩共享内存的坑（官方代码同样用 repeat）。拼接后序列长度恢复成 N=16。

**Step 3 — ids_restore gather（L43）**：

```python
x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, Dd))  # (B, 16, 128)
```

按 §5.2 的机制把乱序序列还原成原始 patch 顺序。`ids_restore.unsqueeze(-1).repeat(...)` 与 `masking.py` L41 的写法同款：索引要扩到和输入同 shape。

**Step 4 — decoder positional embedding（L44）**：

```python
restored = x + self.decoder_pos_embed      # (1, 16, 128) 广播 → (B, 16, 128)
```

decoder 有自己的位置编码（L26），索引与原始 patch 位置一一对应——所以**必须先 gather 再加**，否则位置编码会加到错误的 token 上。

**Step 5 — blocks → norm → pred（L46-50）**：

```python
x = restored
for blk in self.blocks:        # 2 × Block(128, 4 heads)
    x = blk(x)
x = self.norm(x)               # LayerNorm(128)
pred = self.pred(x)            # Linear(128 → 192) → (B, 16, 192)
```

`pred`（L29）把 128 维特征映射回 T=192 个像素值。注意输出是**归一化像素**（与 target_norm 同空间），不是原始 RGB。

### 7.2 为什么 concat 后的顺序不是原始 patch 顺序

两个原因，缺一不可：

1. **可见区段是乱序的**：x_vis 按 ids_keep 的顺序排列，而 ids_keep 来自随机排列的前 V 个（`masking.py` L37, L40），所以 `concat` 的前 V 位是"随机顺序的可见 token"。
2. **mask 区段没有身份**：12 个 mask token 是同一个 (1,1,128) 参数 repeat 出来的，彼此完全相等。它们被放在 concat 的第 V..N-1 位，但这些位置与"哪些原始 patch 被遮"没有任何对应关系。

所以 `concat[i]` 与"原始 patch i"是两个完全不同的坐标系。

### 7.3 为什么必须使用 ids_restore

反证一下就清楚了：**pred 最终要和 target 逐位置对齐**。`target_norm` 的第 i 行是原始 patch i 的归一化像素（`patchify` 行主序），`masked_mse` 里 `pred - target_norm` 是逐行相减（`model.py` L24）。如果 pred 的第 i 行装的是别的 patch 的重建，MSE 的数值可能差不多，但**每个位置学到的对应关系全错**，可视化出来必然是拼图错位。

ids_restore 正是"concat 坐标系 → 原始 patch 坐标系"的唯一映射（§5.4 证明它是逆排列）。gather 之后、加 pos_embed 之前的那一步，是 decoder 里最值得反复读的一行。

顺带解释一个设计细节：为什么 mask token 可以共享？因为 gather 还原后，12 个被遮位置的内容完全相同（同一个参数 + 各自的位置编码）。**区分 12 个被遮位置的唯一信息就是 decoder_pos_embed**——位置编码在这里承担了全部"定位"职责。`test_mae_flow.py` ASSERT 3（L118-133）精确验证了这一点：被遮位置 `restored == mask_token + decoder_pos_embed`，可见位置 `restored == decoder_embed[ids_restore] + pos_embed`。

---

## 8. Loss（mae/model.py + mae/target.py）

### 8.1 target 如何得到（model.py L58-59）

```python
target = patchify(imgs, self.patch_size)              # (B, N, T) 原始像素
target_norm, target_mean, target_var = patch_normalize(target)
```

`patch_normalize`（`target.py` L16-18）对每个 patch 做归一化：

```python
mean = target.mean(dim=-1, keepdim=True)              # (B, N, 1) 对 T=192 个像素求均值
var  = target.var(dim=-1, keepdim=True, unbiased=False)
norm = (target - mean) / (var + 1e-6).sqrt()          # (B, N, T)
```

三个要点：

1. **对每个 patch 的全部 T 个像素一起统计，不分通道**——所以一个 8×8×3 的 patch 只有一个 mean 和一个 var。
2. `1e-6` 防止方差为 0（纯色 patch）时除零。
3. 归一化后每个 patch 的方差恰好为 1、均值为 0——这是后面"初始 loss ≈ 1"的根。

### 8.2 pred 的 shape

`pred: (B, N, T) = (2, 16, 192)`，来自 `decoder.py` L50。它是模型预测的**归一化像素**。所以 loss 的输入 pred 与 target_norm 天然同空间，可以直接相减。

### 8.3 mask 的语义

`mask: (B, N) bool`，**True = 被遮**（`masking.py` L45 与 docstring L28）。注意和某些代码库相反的习惯：这里 True 表示"要重建/要惩罚"的位置。`mask.sum()` 对 bool 张量求 True 的个数 → 每张图 12 个，batch 共 24（`test_mae_flow.py` L109 断言 `== 2 * 12`）。

### 8.4 masked_mse 逐行拆解（model.py L19-27）

```python
loss = (pred - target_norm) ** 2          # (B, N, T) 逐像素平方差
loss = loss.mean(dim=-1)                  # (B, N) 每个 patch 内部平均 → 每 patch 一个标量
loss = (loss * mask.float()).sum() / mask.sum()   # 只留被遮 patch, 再取平均
```

最后一行做了两件事：

- `loss * mask.float()`：可见位置的 loss 被乘 0 清零；
- `/ mask.sum()`：除以被遮 patch 总数（24），即**在被遮 patch 上取均值**（而不是除以 B×N 全体）。

### 8.5 为什么只计算 masked patches 的 loss

1. **防止"抄袭"**：可见 patch 在 encoder 里被看过，模型只需把它们抄到输出就能拿分，这种梯度信号教不会模型任何东西。只惩罚被遮的 75%，模型被迫从 25% 的可见信息里推理出另外 75%。
2. **与论文一致**：官方实现正是 `(loss * mask).sum() / mask.sum()`，这里逐行对应。
3. **高效**：一个乘法就把可见位置的贡献清零，不需要重新切片。

### 8.6 初始 loss ≈ 1.0 的推导（重要 sanity check）

随机初始化时（`init_weights` trunc_normal(0.02)，`model.py` L48-50），`pred` 接近 0；`target_norm` 每个 patch 方差 = 1、均值 = 0。于是被遮位置的期望损失：

```
E[(pred − target)²] ≈ E[(0 − target)²] = var(target) + mean(target)² = 1 + 0 = 1
```

实际运行值 1.0498（`test_mae_flow.py` ASSERT 4，L135-138 断言落在 (0.3, 2.0) 区间）。**如果哪天你改了代码后初始 loss 明显偏离 1，说明 target 归一化或 mask 逻辑出了问题**——这是 MAE 最便宜的自检手段。

### 8.7 关于 norm_pix_loss（当前代码没有这个名字）

论文官方代码把"像素归一化 + masked MSE"打包成一个函数 `norm_pix_loss`。**当前代码里不存在这个函数**，它的职责被拆成两处，对照关系如下：

| 官方 norm_pix_loss 做的事 | 当前代码位置 |
|---|---|
| patch 像素的 mean/var 归一化 | `target.py` L16-18 的 `patch_normalize` |
| 只在被遮位置取均值 | `model.py` L19-27 的 `masked_mse` |

另外 `target.py` L22-24 的 `patch_denormalize` 是归一化的逆操作，当前还没被调用，留作后续可视化：`pred` 在归一化空间，要画回真实图像必须 denorm（`norm * sqrt(var+1e-6) + mean`）。

---

## 9. 与之前实现的 ViT 对比

### 9.1 从 ViT 直接搬来的知识（对应文件与行号）

| ViT 已有概念 | 当前代码位置 | 与 ViT 的差异 |
|---|---|---|
| PatchEmbed（Conv 切块 + 线性投影） | `modules.py` L24-37 | **零差异**：同款 Conv2d(P,P,stride=P) + flatten + transpose |
| Position Embedding | `pos_embed.py` L18-26 | 维度少了 [CLS] 那一行：ViT 是 (1, N+1, D)，MAE 是 (1, N, D)；且默认改成固定 sincos（`learnable=False`，L29-41） |
| Transformer Block（pre-norm MSA + MLP） | `modules.py` L78-95 | **零差异**：`x + attn(norm1(x))`、`x + mlp(norm2(x))`；少了 stochastic depth（注释 L81 说明 toy 阶段不需要） |
| Attention（qkv 合并投影 + 多头切分） | `modules.py` L40-60 | **零差异**：qkv 一次 Linear 再 reshape/permute 成 (3, B, heads, L, head_dim)，scale = head_dim^-0.5 |
| MLP（4 倍隐层 + GELU） | `modules.py` L63-75 | **零差异** |
| 权重初始化 | `modules.py` L9-21 | trunc_normal(0.02) 同款；LayerNorm 也纳入统一 init |

### 9.2 MAE 新增的知识（对应文件与行号）

| MAE 新概念 | 当前代码位置 | 一句话本质 |
|---|---|---|
| Patchify target（无参数切块） | `patch.py` L12-22 | 用纯 reshape 拿到 (B,N,T) 原始像素当回归目标；注意与 PatchEmbed 的区别（§4.4） |
| Random Masking | `masking.py` L36-37 | argsort(rand) 生成逐图独立的随机排列 |
| ids_keep | `masking.py` L40 | 排列前 V 个下标 = 可见名单，gather 的依据（L41） |
| ids_restore | `masking.py` L38 | 逆排列 = 对排列再 argsort（§5.4 证明），decoder 还原顺序的唯一依据 |
| mask（bool） | `masking.py` L45 | `ids_restore >= V` 的等价写法，True = 被遮 |
| Mask Token | `decoder.py` L25, L39 | 共享可学习占位符 (1,1,Dd)，只存在于 decoder |
| 非对称 Encoder/Decoder | `encoder.py` + `decoder.py` | encoder 只算 25% 真实 token；decoder 轻量（2 层/128 维）且用完即弃 |
| Masked Reconstruction Loss | `model.py` L19-27 + `target.py` L16-18 | 归一化像素 + 只在被遮位置取均值 |

### 9.3 从 ViT 里"被删掉"的东西

- **[CLS] token**：encoder 参数里没有 cls_token（`encoder.py` L21-24），pos_embed 也只有 N 行。它会在未来的**微调阶段**加回来（预训练阶段不需要）。
- **分类头**：`pred` Linear（`decoder.py` L29）输出 T=192 个像素而不是类别数。
- **attention mask / padding**：序列要么全是真实 patch（encoder），要么真实+占位混合但长度恒定（decoder），都不需要掩码注意力。
- **dropout 默认 0**：`modules.py` L43, L66, L85 的 dropout 参数默认 0.0，toy 阶段不丢。

---

## 10. 阅读代码推荐顺序（配套动手实验）

按下面的顺序读，每一步都给出"读什么 + 动手改什么验证"：

| # | 读 | 动手实验 |
|---|---|---|
| 1 | `mae/patch.py` 全文（35 行） | 跑 `python test_mae_flow.py`，只看 ASSERT 6 的 ramp 检查输出；自己在 Python 里 `patchify(ramp, 8)[0,0]` 手算前 12 个数 |
| 2 | `mae/masking.py` 全文（47 行），重点 L33-45 | 跑 `python masking_example.py`，把输出里的 [2,0,1,3] 换成 [3,1,0,2] 手推一遍 |
| 3 | 手推 ids_restore（§5.4 的证明） | 验证 `gather(ids_shuffle, ids_restore) == arange(N)`；再验证 `mask == ids_restore >= V` |
| 4 | `mae/encoder.py`（48 行），重点 L29-37 | 把 `mask_ratio=0.5` 传入 `model(x, 0.5)`，观察 x_vis 变 (2,8,192)、mask 每行 8 个 True、loss 仍 ≈ 1。再试 `model(x, 0.0)`：x_vis (2,16,192)、mask 全 False，但 **loss 变成 nan**——因为 `mask.sum()=0`，`masked_mse` 最后一行的除法是 0/0（这是当前代码未防护的边界情况，phase 1 可接受，理解原因即可） |
| 5 | `mae/decoder.py`（56 行），重点 L39-44 | 对照 ASSERT 3 的输出验证：被遮位置 restored == mask_token + pos_embed；把 `decoder_depth=0` 传进 ToyMAE 看 loss 变化 |
| 6 | `mae/target.py` + `model.py` 的 masked_mse（L19-27） | 手工验证初始 loss ≈ 1 的推导（初始时**每个** patch 的 target_norm 方差都是 1、pred≈0，可见和被遮位置都一样）。把 `masked_mse` 里的 `* mask.float()` 去掉再跑：求和包含全部 16 个 patch、但除数仍是 12 个计数 → loss ≈ 16/12 ≈ 1.33，体会 mask 乘法的两个作用（清零 + 配合除数） |
| 7 | `mae/model.py` 的 forward（L52-67） | 对照 §2 数据流图，从 `out` dict 的 15 个键倒推出每个键是图中哪一步 |
| 8 | `test_mae_flow.py` 六个 ASSERT（L96-162） | 逐个断言问自己：它验证的是哪个模块的哪个性质？答不上来的回去重读对应模块 |
| 9 | `mae/modules.py` + `mae/pos_embed.py`（已熟悉的移植代码） | 与你 `vit/vit.py` 逐段对照，确认"零差异"清单（§9.1）属实 |
| 10 | `mae/config.py` + `mae/__init__.py` | 把 `encoder_num_heads` 改成 5 再跑 test，体会 `modules.py` L45 的 assert（192 % 5 ≠ 0）；改回 3 恢复 |

**读完这份指南 + 跑完两个脚本后的自测题**（都能在代码里找到答案）：

1. `x_vis` 的每一行和 `tokens_full` 是什么关系？（`masking.py` L41 gather 语义）
2. 为什么 `pos_embed` 加在 shuffle 之前和之后是等价的，但代码选择了"之前"？（`encoder.py` L30）
3. decoder 里如果**先加 pos_embed 再 gather** 会出什么问题？（§7.2-7.3）
4. loss 为什么除以 `mask.sum()` 而不是 `B*N`？（`model.py` L26）
5. 初始 loss 为什么 ≈ 1？训练一段时间后 loss 会往什么方向走、下限大概是多少？（§8.6）

---

> 学习建议：把 §2 的数据流图抄一遍，并在每个箭头旁写上 Shape 和产生它的 `文件:L行号`。能不看文档完整默写这张图，就说明当前这份代码的 tensor flow 你已经掌握了。 
