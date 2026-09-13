# MAE 模块化学习路线与接口契约

> 学习方式：理解 → 分模块实现 → Smoke Test → 最后组装。
> 本文件是"设计契约"：每个模块只定义输入/输出 Shape、职责、依赖和验收标准，**不包含实现代码**。
> 实现时严格按模块顺序，逐个通过对应 Smoke Test 后再进入下一个。

---

## 0. 符号约定与两套配置

| 符号 | 含义 | 学习配置 (toy) | 论文配置 (ViT-B/L) |
|---|---|---|---|
| B | batch size | 4 | 64~4096 |
| H×W | 输入图像尺寸 | 32×32 (CIFAR) | 224×224 |
| P | patch 边长 | 8（toy）/ 4（CIFAR 训练） | 16 |
| N | patch 数 = HW/P² | 16（toy）/ 64（P=4） | 196 |
| r | mask 比例 | 0.75 | 0.75 |
| V | 可见 patch 数 = int(N·(1−r)) | 4（toy）/ 16 | 49 |
| M | 被 mask patch 数 = N−V | 12（toy）/ 48 | 147 |
| T | 一个 patch 的像素数 = P²·3 | 192（toy） | 768 |
| D | encoder 宽度（embed_dim） | 192 | 768 / 1024 |
| Dd | decoder 宽度 | 128 | 512 |
| L / Ld | encoder / decoder 层数 | 6 / 2~4 | 12·24 / 8 |

---

## 1. MAE 核心思想与整体数据流

### 1.1 一句话

**随机遮住 75% 的 patch，让一个只看得见 25% 的 ViT encoder 编码可见 patch，再由一个轻量 decoder 把被遮住的部分"画"回来——重建 75% 内容迫使模型学会全局语义，而不是靠邻近像素插值偷懒。**

### 1.2 与 ViT 的关键差异

| | ViT（监督分类） | MAE（自监督预训练） |
|---|---|---|
| encoder 输入 | 全部 N 个 patch + [CLS] | 只有 V=N·25% 个可见 patch，**无 [CLS]** |
| mask token | 不存在 | encoder 里**没有**，只出现在 decoder 里 |
| 输出头 | [CLS] → Linear → logits | decoder → 每个 patch 预测 T 个像素值 |
| 损失 | CrossEntropy | **只在被 mask 的位置**算 MSE（归一化像素） |
| positional embedding | (1, N+1, D)，带 CLS | encoder (1, N, D)；decoder 另有 (1, N, Dd) |
| 单张图计算量 | 100% | encoder 注意力开销降为 (V/N)²≈1/16，整体训练快约 3 倍以上 |
| 训练后 | 直接用 | 丢弃 decoder，encoder 拿去做下游（加 CLS 微调 / 线性探测） |

### 1.3 三个核心设计决策（理解它们才算理解 MAE）

1. **75% 高掩码率**：图像的空间冗余远高于语言。低掩码率（20~40%）时模型靠邻域插值就能重建，学不到全局语义（论文消融：线性探测显著变差）；75% 既高效又强迫学语义。
2. **encoder 不含 mask token**：mask token 在 BERT 里必须进 encoder（encoder 要与下游共享），但 MAE 的 encoder 只在预训练阶段用，且下游直接用 patch 输入。把 mask token 从 encoder 剔除后，encoder 只处理 25% 的 token → 训练快 3 倍以上、省显存。
3. **归一化像素 + 轻量 decoder 作 target**：直接回归原始 RGB 会让模型偏向学"平均颜色"。对每个 patch 做 per-patch 的 mean/std 归一化（mean=(B,N,1)、var=(B,N,1)），target 变成"结构信息"，初始 loss ≈ 1.0。decoder 轻量（宽度约 encoder 一半、深度约 1/3，参数量约 encoder 的 10%），因为重建任务本身简单，且预训练完就扔掉。

### 1.4 整体数据流（跟着这个图逐模块实现）

```
输入图像 x: (B, 3, H, W)
      │  PatchEmbed: Conv2d(kernel=P, stride=P) + flatten + transpose
      ▼
tokens: (B, N, D) ──────────────► + pos_embed (1, N, D)
      │
      │  random_masking(r=0.75)
      │       ids_shuffle = rand(B,N).argsort(dim=1)   ← 核心新逻辑
      │       ids_keep    = ids_shuffle[:, :V]
      │       ids_restore = ids_shuffle.argsort(dim=1) ← 逆排列
      ├──────────► x_vis: (B, V, D)      （gather 取前 V 个）
      │            mask: (B, N) bool     （原始顺序，True=被遮）
      │            ids_restore: (B, N) int64
      ▼
Encoder: L × TransformerBlock → LayerNorm
      │
      ▼
x_enc: (B, V, D)
      │  DecoderEmbed: Linear(D → Dd)
      ▼
x_vis_dec: (B, V, Dd)   +   mask_tokens: (B, M, Dd)  ← 共享可学习 (1,1,Dd) 广播
      │  cat → (B, V+M, Dd)
      │  gather(dim=1, ids_restore) → (B, N, Dd)     ← 恢复原始 patch 顺序
      │  + decoder_pos_embed (1, N, Dd)
      ▼
Ld × TransformerBlock → LayerNorm → Linear(Dd → T)
      │
      ▼
pred: (B, N, T)
      │
      ├──► 与 target 在 mask 位置求 MSE
      ▼
target: patchify(x) → per-patch normalize → target_norm: (B, N, T)
            mean/var: (B, N, 1)

loss = mean( (pred[mask] − target_norm[mask])² )     ← 只惩罚被遮的 M 个位置
```

### 1.5 一个容易踩的细节：位置编码加在 shuffle 之前还是之后？

官方实现：encoder 里**先给全部 N 个 token 加 pos_embed，再 shuffle/gather**（等价于加在之后，因为 shuffle 是双射，位置和 token 一一对应）；decoder 里是 **gather 还原之后再加 decoder_pos_embed**。两种写法自洽即可，但**建议照抄官方顺序**，方便以后对拍。

---

## 2. 模块拆分总览

| # | 模块 | 文件 | 输入 | 输出 | 一句话职责 |
|---|---|---|---|---|---|
| 1 | patchify / unpatchify | `mae/patch.py` | (B,3,H,W) ⇄ (B,N,T) | 双向映射 | 建立"图像 ↔ patch 序列"的行主序约定 |
| 2 | random_masking | `mae/masking.py` | (B,N,D) + r | (B,V,D), mask, ids_restore | 生成掩码、切分可见 token、提供逆还原索引 |
| 3 | per-patch normalize | `mae/target.py` | (B,N,T) | (B,N,T)+mean+var | 生成归一化 target 与反归一化（可视化用） |
| 4 | TransformerBlock / PatchEmbed / pos_embed | `mae/modules.py`, `mae/pos_embed.py` | 同 ViT | 同 ViT | **从你的 vit.py 直接移植** |
| 5 | Encoder | `mae/encoder.py` | (B,3,H,W) | (B,V,D)+mask+ids_restore | ViT encoder 去 [CLS]、只算可见 token |
| 6 | Decoder | `mae/decoder.py` | (B,V,D)+ids_restore | (B,N,T) | 拼回 mask token、还原顺序、重建像素 |
| 7 | MAE 组装 + loss | `mae/model.py` | (B,3,H,W) | pred+mask+target | 把 5/6/3 串起来，实现 masked MSE |
| 8 | 可视化 | `mae/visualize.py` | pred/mask/imgs | PIL 网格图 | 人眼验收的最终手段 |
| 9 | 数据 + 训练循环 | `data.py`, `train.py` | — | checkpoint | 复用你 vit 项目的训练骨架 |

---

## 3. 模块接口契约（实现前先背下这些 Shape）

> 每个模块"只承诺自己的输入输出，不关心别人的内部实现"。
> 下面用 toy 配置给具体数字：B=4, 32×32, P=8 → N=16, V=4, M=12, T=192, D=192, Dd=128。

### 3.1 `patch.py` —— patchify / unpatchify（纯 reshape 操作，无参数）

```
patchify  : (B, 3, H, W)          → (B, N, T)        T = P·P·3
unpatchify: (B, N, T)             → (B, 3, H, W)
约定：第 i 个 patch = 第 (i // (W/P)) 行、第 (i % (W/P)) 列（行主序）
```
- 职责：定义**唯一的 patch 顺序约定**。所有后续模块（mask 索引、pos_embed、可视化）都依赖这个约定。
- 接口关系：被 `PatchEmbed` 的语义复刻、被 `target.py` 用来切 raw pixel、被 `visualize.py` 用来拼图。
- 注意：这里实现"无参数版"（等价于 `nn.functional.unfold` 的重排），与 `modules.PatchEmbed`（带 Linear 投影）分开，因为 target 要的是**原始像素**不是投影后的特征。

### 3.2 `masking.py` —— random_masking（MAE 唯一的全新核心逻辑）

```
random_masking(x: (B, N, D), r: float) →
      x_vis      : (B, V, D)      # 按 ids_keep 顺序 gather 出的可见 token
      mask       : (B, N) bool    # 原始 patch 顺序，True = 被遮
      ids_restore: (B, N) int64   # 逆排列索引，decoder 用它把 (V+M) 还原成 N
      ids_keep   : (B, V) int64   # 可见 patch 的原始序号（测试/可视化用）
```
- 职责：① 每张图独立生成随机排列；② 切出前 V 个作可见；③ 提供"还原到原始顺序"的逆索引。
- 接口关系：encoder 用 x_vis；decoder 用 ids_restore；loss 与可视化用 mask；测试用 ids_keep。
- 关键性质：`gather(cat([x_vis, 任意占位], dim=1), dim=1, ids_restore)` 得到的序列中，`~mask` 位置 == 原 x 的可见 patch，`mask` 位置 == 占位。这是 decoder 的数学基础。

### 3.3 `target.py` —— per-patch 归一化（无参数）

```
patch_normalize(target: (B, N, T)) → (norm: (B, N, T), mean: (B, N, 1), var: (B, N, 1))
patch_denormalize(norm, mean, var) → (B, N, T)
norm = (target − mean) / (var + 1e-6)^0.5      # 注意：对每个 patch 的全部 T 个像素一起统计
```
- 职责：把回归 target 从"绝对像素"变成"结构信息"；denorm 仅供可视化恢复真实图。
- 接口关系：`model.py` 的 loss 用 normalize；`visualize.py` 用 denormalize。
- 性质：归一化后每个 patch mean≈0、var≈1 → 随机初始化的网络 pred≈0 时，初始 loss≈1.0（重要 sanity check）。

### 3.4 `modules.py` + `pos_embed.py` —— 从你的 vit.py 移植

```
PatchEmbed: (B,3,H,W) → (B,N,D)          # Conv2d(P,P,stride=P)，与 vit.py 相同
pos_embed  : (1, N, D)  → 广播到 (B,N,D)  # 注意：比 ViT 少一个 CLS 位 (1,N+1,D)→(1,N,D)
Block      : (B, L, D)  → (B, L, D)       # pre-norm MSA + MLP，与 vit.py 完全相同
```
- 职责：全部复用，**唯一改动**是 pos_embed 去掉 CLS 列、Block 数量参数化（L 与 Ld 各自可配）。
- 接口关系：encoder 用 PatchEmbed+pos_embed+L 个 Block；decoder 用 Ld 个 Block + 一个 Linear。

### 3.5 `encoder.py` —— MAEEncoder

```
forward(imgs: (B,3,H,W), r: float) →
      x_vis: (B, V, D), mask: (B, N) bool, ids_restore: (B, N) int64
内部：PatchEmbed → +pos_embed(全部 N) → random_masking → L×Block → LayerNorm
```
- 职责：ViT encoder 的"半截"版本：无 [CLS]、无分类头、只编码可见 token。
- 接口关系：产出三样东西，x_vis 给 decoder，mask/ids_restore 给 model 层（decoder、loss、可视化都要）。

### 3.6 `decoder.py` —— MAEDecoder

```
forward(x_vis: (B,V,D), ids_restore: (B,N)) → pred: (B, N, T)
内部：
  Linear(D→Dd)                        → (B, V, Dd)
  cat([x_vis_dec, mask_token 广播 (B,M,Dd)], dim=1)   → (B, V+M, Dd)
  gather(dim=1, ids_restore)          → (B, N, Dd)     # 还原原始顺序
  + decoder_pos_embed (1,N,Dd)        → (B, N, Dd)
  Ld×Block → LayerNorm → Linear(Dd→T) → (B, N, T)
```
- 职责：① 用共享 mask token 填满缺失位置；② 用 ids_restore 把顺序还原；③ 把特征映射回像素空间。
- 接口关系：只依赖 encoder 的 x_vis 与 ids_restore，**不知道也不关心 mask 是怎么生成的**——这就是模块化收益。
- 可学习新参数：`mask_token (1,1,Dd)`、`decoder_pos_embed (1,N,Dd)`、Linear 头。mask token 是被 mask 位置唯一的"信息入口"。

### 3.7 `model.py` —— MAE 组装 + masked MSE

```
forward(imgs) → pred: (B,N,T), mask: (B,N), target_norm: (B,N,T)   # 推理/可视化用
forward_loss(imgs) → scalar                                          # 训练用
loss = ((pred − target_norm)² 在 dim=-1 上 mean → (B,N)) 中 mask=True 位置的平均
```
- 职责：唯一知道全局拓扑的地方（encoder → decoder → target → loss 的顺序、谁把 ids_restore 传给谁）。
- 接口关系：**只调用** 3.5/3.6/3.3 的公开接口，不触碰任何内部实现。

### 3.8 `visualize.py`

```
输入：imgs (B,3,H,W), pred (B,N,T), mask (B,N), target_norm (B,N,T), mean/var
输出：一张 PIL 网格图：第1行=原图+mask 覆盖(灰)，第2行=重建图，第3行=target 还原图
```
- 职责：把"loss 在下降"翻译成"人眼看得懂的重建质量"。调试 index 顺序 bug 时它比任何单测都直观。

---

## 4. 实现顺序与依赖关系

```
stage1 patch ──────────────┬──► stage5 encoder ──────────┐
stage2 masking ────────────┴──► stage6 decoder ──────────┼──► stage7 model ──► stage9 train
stage3 target ───────────────────────────────────────────┘          │
stage4 modules/pos_embed（移植）────────────────────────────────────┘
stage8 visualize（随时可插，建议在 stage7 后立刻做）
```

| 阶段 | 内容 | 依赖 | 完成判据（全部通过才进下一阶段） |
|---|---|---|---|
| 0 | 工程骨架：`mae/` 包、`tests/`、pytest、种子工具、config | 无 | `pytest` 能跑（0 个测试通过也行） |
| 1 | `patch.py`：patchify/unpatchify | 无 | test_patch 通过（见 §5.1） |
| 2 | `masking.py`：random_masking | 无 | test_masking 通过（§5.2，**最重的测试**） |
| 3 | `target.py`：归一化 | stage1 | test_target 通过（§5.3） |
| 4 | `modules.py`+`pos_embed.py`：从 vit.py 移植 | 无 | test_modules 通过（§5.4，直接复刻你 vit 的测试） |
| 5 | `encoder.py` | 2+4 | test_encoder 通过（§5.5，含 r=0 与 ViT 对齐） |
| 6 | `decoder.py` | 2+4 | test_decoder 通过（§5.6） |
| 7 | `model.py`：组装 + loss | 3+5+6 | test_model 通过（§5.7，含过拟合测试） |
| 8 | `visualize.py` | 1+3+7 | 人眼看到 25% 输入 → 大致重建（§5.8） |
| 9 | `data.py`+`train.py` | 全部 | CIFAR 短训：loss 从 ≈1.0 持续下降，重建图变清晰（§5.9） |
| 10（可选） | 线性探测 / 微调分类 | 9 | encoder 学到可分特征 |

**为什么这个顺序？**
- stage 1~3 是纯张量逻辑，不涉及网络，是 MAE 全部新概念的所在地，先啃。
- stage 4 是"舒适区"，移植你熟悉的代码，顺便把测试基建跑通。
- stage 5~7 只是把已验证的积木按 §1.4 数据流拼起来，此时如果出错，只可能是接口契约没对齐。
- stage 8 可视化是 MAE 这种生成式任务的"终极 smoke test"。

---

## 5. 每个模块的 Smoke Test 方案

> 所有测试用 toy 配置：B=4, 3×32×32, P=8 → N=16, V=4, M=12, T=192, D=192, Dd=128。
> 全部固定 `torch.manual_seed`，浮点断言用 fp32 + 1e-4~1e-5 容差。

### 5.1 test_patch.py
1. Shape：`patchify` → (4,16,192)；`unpatchify` 还原 (4,3,32,32)。
2. **往返一致性**：`unpatchify(patchify(x))` 与 x 逐元素误差 < 1e-5。
3. **顺序正确性（最重要）**：构造一张"斜坡图" `x = arange(3·32·32).reshape(3,32,32)`，断言：
   - patch[0] 等于左上角 8×8 块按 (C, row, col) 展平的行主序值；
   - patch[15] 等于右下角块；patch[1] 等于右上相邻块。
   - 与 `F.unfold(x, kernel_size=8, stride=8)` 交叉核对（这是消灭"patch 顺序 bug"的杀手锏，后面所有 bug 一半源自这里）。
4. 非整除尺寸（如 224 不能整除时）不要求支持——第一阶段先限定 H、W 可被 P 整除，训练数据用 RandomResizedCrop 保证。

### 5.2 test_masking.py（本项目的核心测试）
1. Shape：x_vis (4,4,192)、mask (4,16)、ids_restore (4,16)、ids_keep (4,4)。
2. 数量：`V == int(N·0.75 取整)` 即 4；`mask.sum(dim=1)` 全为 12。
3. **划分完备**：ids_keep 与"mask=True 的位置"合并排序后 == `arange(16)`（无重复、无遗漏）。
4. **逆排列性质**：`ids_shuffle.gather(1, ids_restore) == arange(16)`（即 ids_restore 确为逆排列）。
5. **unshuffle 恒等式（核心）**：`x2 = gather(cat([x_vis, zeros(B,M,D)], dim=1), dim=1, ids_restore)` 后，`x2[~mask] == x[~mask]` 且 `x2[mask] == 0`。
6. 确定性：同 seed 两次结果逐元素相同；不同 seed 结果不同。
7. 统计性：同一 seed 下 batch 内 4 张图的 mask 互不相同；100 次随机抽样中每个位置被 mask 的频率 ∈ [0.70, 0.80]。
8. 边界：r=0 时 V=N、mask 全 False、gather 还原后与输入完全一致；r=0.999 时 V=1 正常。

### 5.3 test_target.py
1. `patch_normalize` 后每个 patch（(4,16,192) 的最后一维）mean≈0（|mean|<1e-4）、var≈1（|var−1|<1e-3）。
2. `patch_denormalize(normalize(x))` 还原误差 < 1e-5。
3. 手工构造"纯色 patch + 单像素扰动"用例，验证归一化确实除掉了颜色/亮度信息。

### 5.4 test_modules.py（移植自你的 vit/test_vit.py）
1. Block：shape 不变、无 NaN、梯度流通（param.grad 非零且有限）。
2. PatchEmbed：输出 (B,N,D)；与 vit.py 同参数时输出一致。
3. pos_embed：shape (1,N,D)，无 CLS 列。
4. 复刻你之前给 ViT 写的"随机输入前向 + 反向"冒烟即可。

### 5.5 test_encoder.py
1. 输出三元组 shape：x_vis (4,4,192)、mask (4,16)、ids_restore (4,16)。
2. 无 NaN，PatchEmbed 权重 grad 非零。
3. **对齐测试（复用你的 ViT 的终极验证）**：r=0（不 mask）时，MAEEncoder 的输出应与"vit.py 中抽出的 PatchEmbed+pos_embed[:,1:]+L×Block+LayerNorm"逐层一致（相同 seed、相同初始化、相同输入）。这一步证明你移植没出错。
4. 参数统计：encoder 参数量与去掉 CLS/分类头的 ViT 相同。

### 5.6 test_decoder.py
1. 随机输入 (4,4,192)+ids_restore → pred (4,16,192)，无 NaN。
2. 梯度检查：**mask_token.grad 非零**（证明梯度能穿过 gather 到达被 mask 位置）；可选进阶：mask_token.grad == 所有 mask 位置输出梯度的和。
3. Linear 头权重 shape (192, 128)。
4. r=0 边界：输入 (4,16,128 维)，输出仍 (4,16,192)，不崩。

### 5.7 test_model.py（组装冒烟）
1. 前向：pred (4,16,192)、mask (4,16)、target_norm (4,16,192)；backward 一次不报错，所有参数 grad 有限。
2. **初始 loss ≈ 1.0**（±0.2）：归一化 target 后随机初始化模型的理论初值。
3. **masked-only 性质**：把 pred 中某个"可见位置"的像素手工改掉，loss 不变；改 mask 位置则变。证明 loss 确实只算 M 个位置。
4. **过拟合测试（最有说服力）**：固定 2 张 32×32 图，Adam lr=1e-3，200 步，loss 从 ≈1.0 降到 < 0.05。过不了这个测试，组装的某个接口一定错了。
5. mask_token.grad 非零。

### 5.8 test_visualize.py（人眼验收）
- 输入原图 + mask 网格图中，被 mask 区域是均匀灰色；
- 重建图与 target 图尺寸正确；
- 训练 50~100 步后，重建图应能看出原图大致轮廓（颜色、结构），而不是灰白噪声。

### 5.9 train.py 训练冒烟
- CIFAR-10：32×32、P=4（N=64, V=16, M=48）、mask 0.75、D=192、L=6、Dd=128、Ld=3；
- AdamW lr=1.5e-4·(batch/256) 线性缩放、warmup 5~10 epoch、cosine 衰减、AMP + TensorBoard——**全部照搬你 vit 的 train_v2/v4 骨架**；
- 判定：1 个 epoch 内 loss 稳定下降；每 50 步存一张重建网格图，20~50 epoch 后重建图明显成形；checkpoint 能 save/load/resume。

---

## 6. 需要重点吃透的 index / shuffle / gather 操作

> 全部集中在 stage 2（masking）和 stage 6（decoder 还原）。其余模块几乎没有索引操作。

### 6.1 五个必须会写的模式

1. **批量随机排列**：`ids_shuffle = torch.rand(B, N).argsort(dim=1)` —— 每行独立随机排列。
2. **切前 V 个**：`ids_keep = ids_shuffle[:, :V]` —— 切片本身也是一种索引。
3. **按索引 gather**：`x_vis = x.gather(dim=1, index=ids_keep.unsqueeze(-1).expand(B, V, D))` —— 语义：`out[b,i,:] = x[b, ids_keep[b,i], :]`。
4. **逆排列 = 对排列再 argsort**：`ids_restore = ids_shuffle.argsort(dim=1)`。
5. **拼回并还原**：`cat([x_vis, mask_tokens], dim=1).gather(dim=1, ids_restore)`。

### 6.2 逆排列的数值例子（建议手推一遍，N=4, V=2）

| 原始位置 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 是否可见 | ✗(遮) | ✓ | ✗(遮) | ✓ |
| ids_shuffle = | 3 | 1 | 0 | 2 |
| ids_keep = [3, 1]；ids_restore = argsort([3,1,0,2]) = | 2 | 1 | 3 | 0 |

concat 后的序列 = `[x3, x1, m, m]`，按 ids_restore `[2,1,3,0]` gather → `[m, x1, m, x3]`，恰好恢复原始顺序（0、2 是 mask，1、3 是可见）。**mask 也可以用 `ids_restore >= V` 直接得到**（0 位 2≥2 ✓、2 位 3≥2 ✓），与官方"gather 回填"写法等价。

### 6.3 四个必须避开的坑

1. **bool 索引会打平**：`x[mask]` 返回 `(B·M, D)` 而不是 `(B, M, D)`（官方代码里 M 每张图恒定所以安全，但要知道它打平了；若你以后做"每张图不同 V"，必须换 gather 写法）。
2. **expand 出的 tensor 不可原地写**：mask_token `(1,1,Dd).expand(B,M,Dd)` 是 view，若后续有 in-place 操作要 `.clone()`；官方用 `.repeat` 绕开这个坑但多占显存。
3. **gather 的 dim 必须与 index 的 dim 一致**：对 `(B, N, D)` 取 patch 索引要 `gather(dim=1, ...)`，`dim=0` 是 batch 维。
4. **shuffle 的随机性**：每个 forward 都要重新生成排列，记得在函数内 `torch.manual_seed(seed)` 或在训练循环里控制，否则 DDP 下每张卡 mask 不同（这是官方代码会额外处理的地方，先单卡理解）。

### 6.4 进阶理解（可选）

- 梯度如何穿过 `gather`：gather 本质是"从输入挑元素"，反向就是"把梯度按 index 累加（scatter-add）回输入"。mask_token 的梯度正是"所有 mask 位置输出梯度的和"——这是 5.6 进阶测试的理论依据。
- `einsum` 与 `F.unfold` 的关系：patchify 的纯 reshape 写法与 unfold 等价，学会用 ramp 图交叉验证，以后遇到任何"顺序错位"bug 都能 5 分钟定位。

---

## 7. 复用清单（对照你已有的代码）

| 来源 | 直接复用 | 需要改 | 不适用 |
|---|---|---|---|
| `vit/vit.py` | Attention/MSA、MLP、**Block（pre-norm 结构完全一致）**、PatchEmbed、pos_embed（去掉 CLS 列）、trunc_normal 初始化、LayerNorm(eps=1e-6)、GELU | encoder 去掉 [CLS] 与分类头；深度/宽度参数化 | 分类头、CLS token（预训练阶段） |
| `vit/test_vit.py` | 测试写法与 seed 工具 | — | — |
| `vit/train_v2~v6.py` | AdamW + warmup + cosine、AMP scaler、TensorBoard/wandb、checkpoint save/resume、yaml/hydra 配置、可复现性种子 | loss 换成 masked MSE；评估换成"每 N 步出重建图" | 分类 accuracy 评估 |
| `transformer/` | attention 的 mask 概念直觉（BERT 式 mask token 的来龙去脉） | — | 语言模型的具体结构 |
| DeiT 学习经验 | 训练基础设施与微调阶段的思路（预训练完加 CLS 微调，蒸馏可跳过） | — | 蒸馏 token |
| Swin 学习经验 | "inductive bias 差异"的直觉：MAE 的非对称设计依赖**全局注意力 + patch 结构**，Swin 的窗口注意力/层次结构不能直接套 | — | 窗口注意力、patch merging |

---

## 8. 建议目录结构

```
mae/
├── 00_MAE模块化学习路线与接口契约.md   ← 本文档（对照逐模块实现）
├── mae/                                # 包（与仓库同名，import mae）
│   ├── __init__.py
│   ├── patch.py                        # stage1: patchify / unpatchify
│   ├── masking.py                      # stage2: random_masking
│   ├── target.py                       # stage3: per-patch 归一化
│   ├── modules.py                      # stage4: Block/PatchEmbed/MLP/Attention（移植）
│   ├── pos_embed.py                    # stage4: 位置编码
│   ├── encoder.py                      # stage5: MAEEncoder
│   ├── decoder.py                      # stage6: MAEDecoder
│   ├── model.py                        # stage7: MAE 组装 + masked MSE
│   └── visualize.py                    # stage8: 重建网格图
├── tests/
│   ├── conftest.py                     # 固定 seed、toy 配置 fixture
│   ├── test_patch.py
│   ├── test_masking.py
│   ├── test_target.py
│   ├── test_modules.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_model.py
│   └── test_visualize.py
├── configs/
│   ├── toy.yaml                        # smoke 用：32×32, P=8
│   └── cifar10.yaml                    # 训练用：32×32, P=4
├── data.py                             # CIFAR-10 Dataset + 变换（照搬 vit 项目）
├── train.py                            # stage9: 预训练循环（照搬 vit 骨架）
├── finetune.py                         # stage10(可选): 线性探测/微调
├── runs/                               # TensorBoard / checkpoint（gitignore）
└── README.md                           # 最后写：使用说明与学习总结
```

---

## 9. 下一步

1. 通读本契约，把 §1.4 数据流图手抄一遍，标注每个箭头上下的 Shape；
2. 从 **stage 0 + stage 1（patch.py）** 开始：先写 `tests/test_patch.py` 的 ramp 顺序测试，再实现 `patchify/unpatchify`，跑通后进入 stage 2；
3. stage 2 的 masking 是全项目最值得慢下来的地方——建议先把 §6.2 的逆排列例子手推两遍再动手。

> 每个 stage 完成后，把测试结果贴出来，我会帮你 review 接口是否与契约一致，再进入下一阶段。

---

## 10. 策略更新（已确认）：最小完整版优先

调整为：**先跑通最小完整 forward，再逐模块补 pytest 与训练**（不做大型工程先行）。

### Phase 1 交付物（已完成，运行验证通过）

- `mae/` 包（保持模块化文件，但每个文件只含最小实现）：
  - `patch.py` patchify / unpatchify（无参数）
  - `masking.py` random_masking（返回 x_vis, mask, ids_restore, ids_keep, ids_shuffle）
  - `target.py` per-patch 归一化 / 反归一化
  - `modules.py` PatchEmbed / Attention / Mlp / Block（移植自 vit.py）
  - `pos_embed.py` 2D sincos / 可学习位置编码
  - `encoder.py` MAEEncoder（无 [CLS]，只编码可见 token）
  - `decoder.py` MAEDecoder（mask token + gather 还原 + 像素预测）
  - `model.py` ToyMAE 组装 + masked MSE
  - `config.py` TOY_CONFIG（image_size=32, patch_size=8, N=16, D=192/L=4, Dd=128/Ld=2, r=0.75）
- `test_mae_flow.py`：`x = torch.randn(2, 3, 32, 32)` 逐步打印完整数据流 + 6 组硬断言
- `masking_example.py`：N=4, V=2 的 shuffle/keep/restore/gather 数值演示

运行方式：

```
python masking_example.py    # 先看这个: 4 个 token 的还原全过程
python test_mae_flow.py      # 再看这个: 完整 forward 数据流
```

Phase 1 刻意不含：train.py / finetune.py / AMP / TensorBoard / YAML / checkpoint / DDP。
Phase 2（后续再做）：按本文档 §5 逐模块补 pytest、可视化、过拟合测试，最后补训练循环。
