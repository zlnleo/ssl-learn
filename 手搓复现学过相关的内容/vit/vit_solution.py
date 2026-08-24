# -*- coding: utf-8 -*-
"""
手搓 ViT（Vision Transformer）—— 【参考答案版 · 拆分结构】
============================================================

⚠️ 使用顺序：
   1. 先按 `01_引导_从零手写ViT.md` 的引导，自己动手写 `vit.py`；
   2. 写完用 `test_vit.py` 验收；
   3. 卡住了或写完了，再回来对照本文件复盘，不要直接抄。

本文件的注释会把"每一步为什么这么做"讲清楚，适合写完后逐行对照。
对照表和思考题与引导文档一致，这里多了完整实现。

【本版结构】把 "[CLS] 拼接 + 位置编码 + dropout" 三步抽成了独立的
`ClassTokenPosEmbed` 类（第 2 部分）。好处见 `03_ViT完全讲解...md` 第 5 节：
- 顺序（先拼 CLS、再加位置编码）被封装，调用方永远写不错；
- `num_patches + 1` 的坑只在这个类里出现一次；
- ViT.__init__ 变成"装配线"，forward 四五行对应数据流图四个箭头。

一句话理解 ViT：把"一句话里的词"换成"图片里的小方块(patch)"，把你
Transformer 的 Encoder 拿来用（去掉所有掩码），最后加一个分类头。

⚠️ 背景知识：ViT 不像 CNN 那样自带"平移不变性/局部性"等视觉先验，
所以它更依赖数据量（原论文用 3 亿张图预训练）。在小数据集上从头训练
ViT 通常打不过 ResNet——这是正常的，不是你的代码写错了。

【ViT vs 你写过的 transformer 对照表】
┌──────────────┬───────────────────────────┬──────────────────────────────┐
│              │ 你的 transformer (NLP)     │ ViT (视觉)                   │
├──────────────┼───────────────────────────┼──────────────────────────────┤
│ 输入          │ 词 id 序列                 │ 图片切块投影成的向量序列      │
│ 结构          │ Encoder + Decoder         │ 只有 Encoder（无 Decoder）   │
│ 注意力掩码     │ 因果掩码 / padding 掩码    │ 没有任何掩码（双向注意力）    │
│ 位置编码      │ 正弦函数（固定，不可学习）   │ 可学习参数（随机初始化）      │
│ FFN 激活      │ ReLU                      │ GELU                         │
│ 归一化位置    │ post-LN（先注意力后 LN）    │ pre-LN（先 LN 后注意力）      │
│ 输出          │ 每个位置一个词的 logits     │ 整张图一个分类结果            │
└──────────────┴───────────────────────────┴──────────────────────────────┘
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.1  # 默认 dropout 概率（train.py 会 import 这个常量）


# ===========================================================================
# 第 1 部分：把图片变成 token —— PatchEmbedding（ViT 里唯一的新概念）
# ===========================================================================
class PatchEmbedding(nn.Module):
    """把一张图片切成小方块（patch），每个 patch 投影成一个词向量。

    为什么不能把每个像素当成一个 token？注意力是 O(n^2) 的：
    32x32 的图有 1024 个像素，224x224 的图有 50176 个像素——显存直接爆炸。
    所以先切成 16x16 的块：一张 224x224 的图只剩 (224/16)^2 = 196 个
    token，和一句话的长度差不多。

    数学：输入 (B, C, H, W)，patch 大小 P：
        切出 (H/P) x (W/P) 个块，每个块展平成 C*P*P 维向量，
        再线性投影到 embed_size。
        => 输出 (B, num_patches, embed_size)，num_patches = (H/P)*(W/P)

    实现技巧："切块 + 展平 + 线性投影"三个动作恰好等价于一个
    kernel_size = stride = P 的二维卷积（卷积核滑动的过程就是
    "每个 patch 乘同一个投影矩阵"的过程），一行代码搞定。
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_size=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 一张图切出的 patch 数

        # in_channels -> embed_size 的"切块投影"，等价于 patchify + Linear
        self.proj = nn.Conv2d(in_channels, embed_size,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)        # (B, embed_size, H/P, W/P)
        x = x.flatten(2)        # (B, embed_size, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_size) —— 变成"一句话"了
        return x


# ✍ 思考题 1：把 patch_size 从 4 改成 8，token 数量怎么变？注意力计算量怎么变？
#    答案：token 数 = (H/P)^2，P 翻倍则 token 数变成 1/4；注意力是 O(n^2)，
#    计算量变成 1/16。但每个 token 的信息粒度变粗了——patch 太大丢细节、
#    太小则序列太长，都是在找平衡点。


# ===========================================================================
# 第 2 部分：[CLS] 拼接 + 位置编码 —— 拆出来的独立模块 ClassTokenPosEmbed
# ===========================================================================
class ClassTokenPosEmbed(nn.Module):
    """[CLS] token 拼接 + 可学习位置编码 + dropout，三步按固定顺序封装。

    为什么拆成独立类（详见 03 讲解文档第 5 节）：
    1. 单一职责：这个类只回答"怎么给 token 序列补上分类代表和位置信息"；
    2. 顺序被封死：必须先拼 [CLS]、再加位置编码——顺序是高频易错点，
       封装后调用方永远写不错；
    3. 坑被关进笼子：pos_embed 长度 = num_patches + 1（+1 给 [CLS]），
       这个数只在本类里出现一次；
    4. 可独立测试：单独构造本类就能验证形状，不用搭整个 ViT。

    两个参数的初始化都用 trunc_normal(0.02)——和 Linear/Conv2d 同一套
    初始化策略（见 ViT._init_weights），从 0 开始学也行但收敛更慢、
    更吃随机种子（你改 vit.py 时踩过：89% 卡住就是它）。
    """

    def __init__(self, num_patches, embed_size, dropout=DROPOUT):
        super().__init__()
        # [CLS]：一个可学习的"分类专用"向量，永远放在序列最前面。
        # 经过 N 层双向注意力后，它聚合了全图信息，最后取它做分类。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        # 可学习位置编码（注意！和你 transformer 的正弦编码不一样）：
        # patch 的位置关系让模型自己学。
        # 长度 = num_patches + 1：多出来的 1 是给 [CLS] 的位置！
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_size))

        self.pos_drop = nn.Dropout(dropout)  # 位置编码相加之后的 dropout

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, num_patches, embed)  patch token 序列
        B = x.shape[0]

        # 第 1 步：把 [CLS] 拼到序列最前面 (B, 1+num_patches, embed)
        # expand 按 batch 复制，是零拷贝的"只读视图"，不占新内存；
        # 需要独立可写的数据时才用 repeat（真拷贝）。
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # 第 2 步：加上可学习位置编码（顺序不能反：先拼后加）
        x = x + self.pos_embed
        return self.pos_drop(x)


# ✍ 思考题 5：为什么 ViT 的位置编码从"正弦函数"换成了"可学习参数"？
#    答案：图像 patch 之间的空间关系和语言词序很不一样（二维、局部、
#    平移），让模型自己学更灵活。可学习版本也有代价：训练时学的
#    位置数量是固定的，换更大的输入分辨率时需要插值。两个版本
#    论文里都有人用，都是合法选择。


# ===========================================================================
# 第 3 部分：注意力 —— 和你在 transformer 里写的一模一样，只是永远不传 mask
# ===========================================================================
class ScaleDotAttention(nn.Module):
    """缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V。

    这个类和你在 transformer_simple.py 里写的完全相同，mask 参数也保留了
    （为了代码一致）。但在 ViT 里 mask 永远是 None，原因：
    1. 不需要因果掩码——ViT 是双向注意力，每个 patch 可以看所有其他 patch；
       没有"预测下一个 patch"这件事，自然没有"偷看未来"的问题；
    2. 不需要 padding 掩码——所有图片都切成相同数量的 patch，
       序列长度恒等，不存在 PAD。
    """

    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: (B, heads, q_len, d_k)；k、v: (B, heads, k_len, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        if mask is not None:  # ViT 不会走到这个分支
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力：和你 transformer 里的实现一致（切头/并头/四投影）。

    直接把你写过的 MultiHeadAttention 搬过来就能用，ViT 只是
    调用时永远 q = k = v = x 且 mask = None。
    """

    def __init__(self, embed_size, num_heads, dropout=DROPOUT):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.d_k = embed_size // num_heads

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)

        self.attention = ScaleDotAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, seq, embed) -> (B, heads, seq, d_k)"""
        B, seq_len = x.shape[:2]
        x = x.reshape(B, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """(B, heads, seq, d_k) -> (B, seq, embed)"""
        x = x.transpose(1, 2)
        B, seq_len = x.shape[:2]
        return x.reshape(B, seq_len, -1)

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        output, attn = self.attention(q, k, v, mask)

        output = self._merge_heads(output)
        output = self.w_o(output)
        output = self.dropout(output)
        return output, attn


# ✍ 思考题 2：为什么 ViT 不需要因果掩码？
#    答案：因果掩码是给 decoder 用的——防止它在"预测第 i 个词"时看到
#    i 之后的答案。ViT 只做分类：整张图一次性看完，patch 之间没有先后
#    顺序，也不存在"预测下一个 patch"的自回归任务，所以双向注意力即可。


# ===========================================================================
# 第 4 部分：Encoder 块 —— 你熟悉的注意力 + FFN，注意 pre-LN
# ===========================================================================
class MLP(nn.Module):
    """等价于你 transformer 里的 FFN，两处不同：

    1. 激活函数 ReLU -> GELU（更平滑的 S 形激活，原论文用它）；
    2. 多了一个 fc2 之后的 dropout（原论文的配置，和你的 FFN 里
       "只在 fc1 后 dropout"的写法略有差异，两者都行）。
    """

    def __init__(self, embed_size, hidden_size, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)   # 升维
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, embed_size)   # 降回原维度
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EncoderBlock(nn.Module):
    """ViT 的一层：多头自注意力 + MLP，都采用 pre-LN。

    数据流：x -> LN -> 注意力 -> +x -> LN -> MLP -> +x
    和你 transformer 里的 post-LN（x -> 注意力 -> x+attn -> LN）对比着看：
    区别只是 LayerNorm 的位置，pre-LN 的梯度可以从残差支路"绕开归一化"
    直接回流，深网络训练更稳定，所以现代模型普遍用 pre-LN。
    """

    def __init__(self, embed_size, num_heads, mlp_ratio=4.0, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        # mlp_ratio：FFN 隐藏层是 embed_size 的几倍（你 transformer 里写死的 *4）
        # int()：mlp_ratio 默认是浮点，nn.Linear 的维度必须是整数
        self.mlp = MLP(embed_size, int(embed_size * mlp_ratio), dropout=dropout)

    def forward(self, x):
        # 先 LN 再注意力，残差放在最外面（pre-LN 的标准写法）
        h = self.norm1(x)
        h = self.attn(h, h, h)[0]  # q = k = v；没有 mask —— 双向注意力
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


# ✍ 思考题 3：pre-LN 和 post-LN 到底差在哪？
#    答案：只是 LayerNorm 放的位置不同。post-LN = x + sublayer(x) 之后
#    再归一化；pre-LN = 先归一化再进子层。pre-LN 的残差通路更"直"，
#    梯度回传更稳，层数越深优势越明显；你的 transformer 用 post-LN
#    完全没问题（层数少），但要知道这两种写法的存在和区别。


# ===========================================================================
# 第 5 部分：ViT 组装 —— 一条装配线：patch 嵌入 -> cls/位置 -> 块 -> 分类头
# ===========================================================================
class ViT(nn.Module):
    """Vision Transformer：patch 嵌入 + [CLS]/位置 + N 层 EncoderBlock + 分类头。

    拆分结构之后，__init__ 是一张"零件清单"，forward 是一条"装配线"，
    每一行对应数据流图里的一个箭头。

    Args:
        img_size:     输入图片尺寸（正方形，默认 32）
        patch_size:   patch 边长，必须整除 img_size
        in_channels:  图片通道数（RGB 为 3，灰度图为 1）
        num_classes:  分类类别数
        embed_size:   token 向量维度
        num_heads:    注意力头数
        num_layers:   EncoderBlock 层数
        mlp_ratio:    MLP 隐藏层 = embed_size * mlp_ratio
        pool:         "cls" 用 [CLS] token 分类；"mean" 对所有 patch 取平均（GAP）
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_size=128, num_heads=8, num_layers=4, mlp_ratio=4.0,
                 dropout=DROPOUT, pool="cls"):
        super().__init__()
        assert img_size % patch_size == 0, "图片尺寸必须能被 patch 大小整除"
        assert pool in ("cls", "mean"), "pool 只能是 'cls' 或 'mean'"
        self.pool = pool

        # 1. 图片 -> token 序列
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_size)

        # 2. [CLS] + 可学习位置编码（顺序、+1 这些细节都封装在类里了）
        self.cls_pos_embed = ClassTokenPosEmbed(self.patch_embed.num_patches,
                                                embed_size, dropout)

        # 3. N 层 EncoderBlock（你 transformer 里 ModuleList 的用法）
        self.blocks = nn.ModuleList(
            EncoderBlock(embed_size, num_heads, mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )

        # 4. 分类头：最后的 LN（pre-LN 架构惯例）+ 线性层输出各类别 logits
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, num_classes)

        # ---- 初始化：原论文用 trunc_normal(std=0.02)，对 ViT 收敛很关键 ----
        # cls_token/pos_embed 的初始化已经在 ClassTokenPosEmbed 里做过了，
        # 这里 apply 负责剩下的 Linear / Conv2d / LayerNorm。
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """对 Linear/Conv2d 用截断正态初始化，LayerNorm 权重置 1、偏置置 0。"""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, H, W)
        # 第 1 步：图片变成 token 序列 (B, num_patches, embed)
        x = self.patch_embed(x)

        # 第 2 步：拼 [CLS] + 加位置编码 + dropout（封装在 ClassTokenPosEmbed 里）
        x = self.cls_pos_embed(x)

        # 第 3 步：过 N 层 EncoderBlock（无掩码的双向注意力）
        for block in self.blocks:
            x = block(x)

        # 第 4 步：聚合序列 -> 分类
        x = self.norm(x)
        if self.pool == "cls":
            x = x[:, 0]              # 取 [CLS] 位置的向量
        else:
            x = x[:, 1:].mean(dim=1) # 对所有 patch 向量取平均（GAP）
        return self.head(x)          # (B, num_classes) 的 logits


# ✍ 思考题 4：[CLS] token 和全局平均池化（GAP）哪个好？
#    答案：原论文用 [CLS]；后续很多工作证明 GAP 效果相当甚至略好，
#    还省一个 token 的计算。没有绝对优劣——所以本类用 pool 参数把
#    两种都实现了，你可以自己改着对比。


if __name__ == "__main__":
    # 快速自检：构造一个 32x32、10 类的小 ViT，跑一次前向看形状
    print("=" * 60)
    print("ViT 快速自检（python vit_solution.py）")
    print("=" * 60)
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10,
                embed_size=128, num_heads=8, num_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")

    x = torch.randn(2, 3, 32, 32)  # 假装是 2 张 32x32 的 RGB 图
    print(f"输入形状: {tuple(x.shape)}")
    print(f"patch 数量: {model.patch_embed.num_patches} (32/4 x 32/4)")
    print(f"位置编码数量: {model.cls_pos_embed.pos_embed.size(1)} (patch 数 + 1 个 [CLS] 位)")
    logits = model(x)
    print(f"输出 logits 形状: {tuple(logits.shape)}  # (batch, num_classes)")

    # 🧪 动手实验建议见 `02_验证与修改方向.md`
