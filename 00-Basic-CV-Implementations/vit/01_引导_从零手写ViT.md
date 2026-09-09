# 01 · 引导：从零手写 ViT（无答案版）

> **目标**：跟着本文的引导，自己在 `vit/vit.py` 里写出一个能跑、能学、能通过验收的 ViT。
>
> **规则**：本文只有引导、规格和提示，**没有实现代码**。写不出来时按"先自己想 → 看提示 → 看答案类"的顺序来。完整答案在 `vit_solution.py`（写完全程再对照，别一上来就抄）。
>
> **前置知识**：你已经在 `transformer/` 目录手写过 Encoder-Decoder Transformer，本文会大量引用它。ViT 90% 的零件你都会，真正的新概念只有"图片怎么变成 token"。

---

## 0. 动手前的知识唤醒

**一句话理解 ViT**：把"一句话里的词"换成"图片里的小方块(patch)"，把你的 Transformer 的 **Encoder** 拿来用（去掉所有掩码），最后加一个分类头。

**和你的 transformer 逐项对照**：

| | 你的 transformer (NLP) | ViT (视觉) |
|---|---|---|
| 输入 | 词 id 序列 | 图片切块投影成的向量序列 |
| 结构 | Encoder + Decoder | 只有 Encoder（无 Decoder） |
| 注意力掩码 | 因果掩码 / padding 掩码 | 没有任何掩码（双向注意力） |
| 位置编码 | 正弦函数（固定，不可学习） | 可学习参数（随机初始化） |
| FFN 激活 | ReLU | GELU |
| 归一化位置 | post-LN（先注意力后 LN） | pre-LN（先 LN 后注意力） |
| 输出 | 每个位置一个词的 logits | 整张图一个分类结果 |

**你要实现的完整数据流**（6 个零件拼成一条线）：

```
(B, C, H, W) 图片
      │  ① PatchEmbedding：切块 + 展平 + 线性投影        【唯一的新概念】
      ▼
(B, N, E)  patch token 序列（N = patch 数量，E = 向量维度）
      │  ② 拼接 [CLS] token，加可学习位置编码
      ▼
(B, N+1, E)
      │  ③ × num_layers 个 EncoderBlock
      │     每块 = pre-LN → 多头自注意力 → 残差
      │           pre-LN → MLP(GELU) → 残差
      ▼
(B, N+1, E)
      │  ④ 最后的 LN → 取 [CLS]（或对所有 patch 取平均）
      ▼
(B, E)  →  Linear  →  (B, num_classes) 分类 logits
```

> 句子类比：一张 32×32 的图、patch 大小 4，会切出 (32/4)×(32/4) = **64** 个 token——
> 相当于把图片变成了一句 64 个词的句子；`[CLS]` 相当于句首专门负责"总结全文做分类"的代表。

---

## 1. 接口规格（验收脚本按这个来找你的类，必须一致）

你在 `vit/vit.py` 里要提供以下类和常量。`test_vit.py` 和 `train_vit.py` 会自动优先导入你的 `vit.py`（如果文件不存在才会退回 `vit_solution.py`，**别让这个兜底骗过自己**——以你真正写出来为准）。

| 名称 | 构造参数 | forward 输入 → 输出 | 要点 |
|---|---|---|---|
| 常量 `DROPOUT` | — | — | 值 0.1，`train_vit.py` 会 import 它 |
| `ScaleDotAttention` | `dropout` | `(q,k,v,mask) → (output, attn)` | 和你 transformer 里的一模一样 |
| `MultiHeadAttention` | `embed_size, num_heads, dropout` | `(q,k,v,mask) → (output, attn)` | 切头/并头/四投影 |
| `PatchEmbedding` | `img_size, patch_size, in_channels, embed_size` | `(B,C,H,W) → (B,N,E)` | 还要暴露属性 `num_patches` |
| `MLP` | `embed_size, hidden_size, dropout` | `(B,S,E) → (B,S,E)` | 升维→GELU→降维 |
| `EncoderBlock` | `embed_size, num_heads, mlp_ratio=4.0, dropout` | `(B,S,E) → (B,S,E)` | pre-LN |
| `ViT` | `img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_size=128, num_heads=8, num_layers=4, mlp_ratio=4.0, dropout, pool="cls"` | `(B,C,H,W) → (B,num_classes)` | 组装上面全部 |

---

## 2. 分步引导

每一步：先读"任务"和"形状规格"，自己写；卡住了再展开提示（提示从模糊到具体，逐级解锁）；最后用"自查"确认。

### Step 1 · ScaleDotAttention —— 复制你自己

- **任务**：缩放点积注意力，`Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`。
- **提示 1**：去 `transformer/transformer_simple.py` 找你写过的那个类，几乎原样可用。
- **提示 2**：mask 参数留不留都行——想想 ViT 里 mask 永远该是什么？
- **易错点**：除以的是 `√dₖ`，用 `k.shape[-1]` 而不是写死某个数；softmax 在最后一维。
- **自查**：给定 q、k、v 都是 `(1, 2, 3, 8)`，输出仍是 `(1, 2, 3, 8)`，attn 每行和为 1。

### Step 2 · MultiHeadAttention —— 还是复制你自己

- **任务**：多头注意力，你的 transformer 里写过：四个 Linear 投影、切头、注意、并头、输出投影。
- **提示 1**：切头 = `reshape` 成 `(B, S, heads, d_k)` 再 `transpose(1,2)`；并头是逆操作。
- **提示 2**：ViT 里调用方式永远是 `attn(x, x, x)`，q=k=v。
- **易错点**：`assert embed_size % num_heads == 0`；并头时最后用 `reshape(B, S, -1)`。
- **自查**：输入 `(2, 65, 128)`、8 头，输出形状不变。

### Step 3 · PatchEmbedding —— 全文件唯一的新概念，认真想

- **任务**：把 `(B, C, H, W)` 的图片变成 `(B, N, E)` 的 token 序列，其中 `N = (H/P) × (W/P)`。
- **形状规格**：32×32 图、patch=4 → N = 8×8 = 64；每个 patch 展平是 `C×P×P` 维（RGB 时为 48），再投影到 E。
- **提示 1（朴素思路）**：切成 N 个小块 → 每块展平成 `C*P*P` 维向量 → 过一个 Linear 到 E 维。三个动作分开做能跑就行，先跑通再优化。
- **提示 2（合并思路）**：这三个动作有一个共性——**每个 patch 都乘同一个投影矩阵**。什么算子天生就是"滑窗 + 同一个核"？
- **提示 3**：`kernel_size = stride = P`、输出通道 = E 的 `Conv2d`，恰好一次性完成切块+投影。
- **提示 4**：Conv2d 输出是 `(B, E, H/P, W/P)`，还需要把最后两维并成一维、再挪到第二维（想想 `flatten` 和 `transpose` 的参数）。
- **易错点**：忘了断言 `img_size % patch_size == 0`；flatten/transpose 顺序反了会得到 `(B, E, N)` 而不是 `(B, N, E)`。
- **自查**：输入 `(2, 3, 32, 32)`、patch=4、E=128 → 输出 `(2, 64, 128)`，且类实例有属性 `num_patches == 64`。

### Step 4 · MLP —— FFN 换了个名字

- **任务**：位置级两层 MLP：升维（E → E×mlp_ratio）→ 激活 → 降维（回 E）。
- **提示 1**：你 transformer 里的 `FFN` 就是它，照着写。
- **提示 2**：激活函数换成 GELU（原论文用它），PyTorch 里叫 `nn.GELU()`。
- **易错点**：降维后输出维度必须回到 E，否则残差加不上。
- **自查**：输入 `(2, 65, 128)`、hidden=512 → 输出形状不变。

### Step 5 · EncoderBlock —— 注意和你 transformer 的写法不同

- **任务**：一块 Encoder = 自注意力子层 + MLP 子层，各自带残差和 LayerNorm。
- **提示 1**：子层结构和你 transformer 的 `EncoderLayer` 一样，但 **LayerNorm 的位置要换**。
- **提示 2**：你写的是 post-LN（`x = norm(x + attn(x))`）；ViT 用 pre-LN：先 `h = norm1(x)`，再 `h = attn(h,h,h)`，最后 `x = x + h`。MLP 支路同理，注意两个 LN 要分开（norm1、norm2）。
- **提示 3**：调用注意力时 `attn(h, h, h)[0]`——它返回 (output, attn) 元组，只要 output。
- **易错点**：norm1 和 norm2 必须是两个独立的 `nn.LayerNorm`；没有 mask 参数要传。
- **自查**：输入 `(2, 65, 128)`，输出形状不变。

### Step 6 · ViT 组装 —— 把 6 个零件拼成一张图

按顺序实现 `__init__` 和 `forward`：

1. `PatchEmbedding` 拿到 N；
2. `cls_token`：形状 `(1, 1, E)` 的 `nn.Parameter`（可学习的分类代表）；
3. `pos_embed`：形状 `(1, N+1, E)` 的 `nn.Parameter`——**注意 +1**，那是给 [CLS] 留的位置！这是全文件最容易写错的一个数；
4. `pos_drop`：位置编码相加后的 dropout；
5. `blocks`：`nn.ModuleList` 装 num_layers 个 EncoderBlock（你在 transformer 里用过 ModuleList）；
6. 最后的 `norm`（LayerNorm）+ `head`（Linear E→num_classes）。

`forward` 的五步：

1. `x = patch_embed(x)`；
2. `[CLS]` 拼到序列最前面（batch 维怎么复制？`expand(B, -1, -1)` 或 repeat）；
3. 加 `pos_embed`、过 `pos_drop`；
4. 逐层过 `blocks`；
5. `norm` 之后按 `pool` 聚合：`"cls"` 取 `x[:, 0]`，`"mean"` 取 `x[:, 1:].mean(dim=1)`；最后过 `head`。

- **提示（初始化）**：原论文对所有权重做 `trunc_normal(std=0.02)`、LayerNorm 置 1/0、cls/pos 也做 trunc_normal。ViT 对初始化敏感，别用默认初始化糊弄。不会写就用 `self.apply(...)` 传一个静态方法遍历子模块。
- **易错点**：`pos_embed` 忘了 +1 会在 cat 之后相加时报 shape 错；`cls_token` 忘记按 batch 复制会广播失败；`pool` 参数要断言只能是 `"cls"` 或 `"mean"`。
- **自查**：构造默认 ViT，喂 `(2, 3, 32, 32)` → 输出 `(2, 10)`；参数量约 0.81M（embed=128、8 头、4 层时）。

### Step 7 · 自检入口

- **任务**：加一个 `if __name__ == "__main__":` 块：构造小 ViT、喂随机图、打印输入/输出形状和参数量。
- **自查**：命令行直接 `python vit.py` 能打印出正确形状。

---

## 3. 写完后的排查清单（报错了先看这里）

- **shape 不匹配**：90% 是 `pos_embed` 忘了 +1，或 flatten/transpose 顺序错；
- **matmul 报错**：检查切头后 k 的转置是不是 `(-2, -1)`；
- **除零 / assert 失败**：`embed_size % num_heads`、`img_size % patch_size` 两个断言都加了吗；
- **训练不收敛（loss 卡在 ln(类别数) 附近）**：这是模型完全没学到东西的信号——先查位置编码有没有真的加进去、有没有被 0 初始化盖掉；
- **测试导入了参考答案而不是你的代码**：说明你还没建 `vit.py`，或文件名/类名和规格表不一致。

---

## 4. 验收流程

1. 写完 `vit.py`，先跑 `python test_vit.py`——它会优先验证你的实现，4 项全过即合格；
2. 再跑 `python train.py --dataset toy`（合成"象限亮块"任务），观察准确率能否从 25% 涨到接近 100%；
3. 对照 `vit_solution.py` 逐类 diff，看参考答案是怎么写的、注释里讲了什么；
4. 想继续深入，读 `02_验证与修改方向.md`。

**写在最后**：这个练习里真正该花时间的只有 Step 3（图片→token）和 Step 6 的 pos_embed 的 +1；其余都是在调用你已经会的东西。如果某一步卡了 20 分钟以上，允许自己看答案——但看完要合上答案重写一遍，写不出来就不算会。
