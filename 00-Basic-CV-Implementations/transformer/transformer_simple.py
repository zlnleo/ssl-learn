# -*- coding: utf-8 -*-
"""手写的 Transformer（encoder-decoder 结构，对应原论文 "Attention Is All You Need"）。

全部组件（Embedding、位置编码、多头注意力、FFN、Encoder、Decoder）都是
用 nn.Module 手写搭出来的，没有调用 torch.nn.Transformer。核心设计思路：

1. 同一个 MultiHeadAttention 类通过传入不同的 mask 复用于三种注意力：
   - Encoder 自注意力：mask = padding 掩码（屏蔽源序列里的 PAD）；
   - Decoder 掩码自注意力：mask = 因果(下三角)掩码 & padding 掩码；
   - Decoder 交叉注意力：q 来自 decoder，k/v 来自 encoder 输出，mask = padding 掩码。
2. 所有 mask 都是 bool 张量，约定 True = "可以看"，False = "不许看"；
   在 ScaleDotAttention 里把 False 的位置 masked_fill 成 -inf，softmax 之后
   该位置的注意力权重自然变成 0。
3. 训练时用 teacher forcing（decoder 一次性看到整个 tgt 序列，靠因果掩码
   挡住未来信息），推理时用 generate 一步一步贪心解码。

运行冒烟测试：python test_transformer_simple.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
PAD, BOS, EOS = 0, 1, 2  # 特殊 token 的 id：填充 / 句首 / 句尾
MAX_LEN = 50             # 位置编码表默认支持的最大序列长度
DROPOUT = 0.1            # 默认 dropout 概率


# ---------------------------------------------------------------------------
# 1. Embedding 与位置编码
# ---------------------------------------------------------------------------
class Embedding(nn.Module):
    """词嵌入：把 token id（整数）映射成稠密向量。

    这里单独包了一层 nn.Embedding 而不是直接用，只是为了和其他模块保持
    一致的 nn.Module 封装风格；功能上等价于 self.embedding = nn.Embedding(...)。
    """

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, embedding_size)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """正弦位置编码（Transformer 原论文的方案）。

    注意力机制本身对 token 的顺序不敏感（打乱顺序注意力结果不变），所以
    需要给每个位置 i 加上一个固定的位置向量 pe[i]：偶数维度填 sin，
    奇数维度填 cos，不同维度的频率按 10000^(-2i/embed_size) 衰减。
    这样相邻位置的编码相近、远处位置的编码差异大，模型能通过点积感知相对位置。

    细节：pe 用 register_buffer 注册，而不是普通属性——
    1) 它会跟着模型一起 .to(device) 搬家；
    2) 它不会出现在 parameters() 里，不参与梯度更新（它是固定常量，不是可学习参数）。
    """

    def __init__(self, embed_size, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: (max_len, embed_size)，一次性算好整张表，forward 时按长度切片
        pe = torch.zeros(max_len, embed_size)
        # position: (max_len, 1)，第 i 行就是位置编号 i
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: 频率项 1/10000^(2i/embed_size)，等价于
        # exp(2i/embed_size * (-ln 10000))，只对偶数下标取值
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        # 偶数维度（0,2,4,...）填 sin，奇数维度（1,3,5,...）填 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 最前面加一个 batch 维，方便和 (batch, seq_len, embed_size) 的输入广播相加
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        # 按当前序列长度从 pe 表里切片，直接加到词向量上（加法，不是拼接）
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. 注意力
# ---------------------------------------------------------------------------
class ScaleDotAttention(nn.Module):
    """缩放点积注意力（Scaled Dot-Product Attention）。

    公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    除以 sqrt(d_k) 的原因：d_k 越大，点积的数值方差越大，softmax 会进入
    梯度极小的饱和区；缩放后分数落在更合适的范围里。

    mask 约定：bool 张量，True = 可以注意，False = 屏蔽。
    形状自动广播到 scores (batch, heads, q_len, k_len)：
    - padding 掩码：  (batch, 1, 1, k_len) —— 两个 1 对齐 batch 和 heads 维；
    - 因果掩码：      (1, 1, q_len, k_len) —— 下三角，q_len == k_len 时用。
    """

    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: (batch, heads, q_len, d_k)；k、v: (batch, heads, k_len, d_k)
        # transpose(-2, -1) 把 k 转成 (batch, heads, d_k, k_len)，
        # 最后两维就是标准矩阵乘 (q_len, d_k) x (d_k, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])

        if mask is not None:
            # ~mask 取反：把"不许看"的位置设成 -inf。
            # softmax(-inf) = 0，所以被屏蔽位置的注意力权重严格为 0
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # 在 k_len 维上做归一化
        attn = self.dropout(attn)         # 对注意力权重做随机丢弃
        output = torch.matmul(attn, v)    # (batch, heads, q_len, d_k)
        return output, attn               # 权重一并返回，方便调试/可视化


class MultiHeadAttention(nn.Module):
    """多头注意力：把 embed_size 切成 num_heads 份，每个头独立做一次缩放点积注意力。

    切头的意义：不同头在不同子空间里投影，能学到不同的注意力模式
    （有的头看相邻词、有的头看全局依赖、有的头盯语法结构……）。

    关键设计：本类不区分自注意力还是交叉注意力——q、k、v 都从外部传入：
    - 自注意力：  q = k = v = x；
    - 交叉注意力：q = decoder 的 x，k = v = encoder 的输出。
    这就是"一个类复用于三种注意力"的由来，差别只在调用参数和 mask 上。
    """

    def __init__(self, embed_size, num_heads, dropout=DROPOUT):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.d_k = embed_size // num_heads  # 每个头分到的维度

        # 四个线性投影：q/k/v 的输入投影 + 多头拼接后的输出投影
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)

        self.attention = ScaleDotAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """切头：把最后一维拆成 (num_heads, d_k)，再把 heads 维提到 seq_len 前面。

        x: (batch, seq_len, embed_size) -> (batch, num_heads, seq_len, d_k)
        """
        B, seq_len = x.shape[:2]
        x = x.reshape(B, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)  # 交换 seq_len 和 num_heads 两维
        return x

    def _merge_heads(self, x):
        """并头：_split_heads 的逆操作。

        x: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, embed_size)
        """
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        B, seq_len = x.shape[:2]
        x = x.reshape(B, seq_len, -1)  # 把 (num_heads, d_k) 拼回 embed_size
        return x

    def forward(self, q, k, v, mask=None):
        # 1. 线性投影（embed_size -> embed_size，随后才切头）
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 2. 切头
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 3. 每个头独立做缩放点积注意力（mask 原样透传）
        output, attn = self.attention(q, k, v, mask)

        # 4. 并头 + 输出投影 + dropout
        output = self._merge_heads(output)
        output = self.w_o(output)
        output = self.dropout(output)
        return output, attn


# ---------------------------------------------------------------------------
# 3. 前馈网络 FFN
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    """位置级前馈网络（Position-wise FFN）。

    对序列里"每个位置"的向量独立地做同样的两层 MLP（位置之间不交互，
    交互全靠注意力层）：FFN(x) = fc2(ReLU(fc1(x)))
    先升到 hidden_size（通常是 embed_size 的 4 倍）再压回 embed_size，
    给模型提供非线性变换能力。
    """

    def __init__(self, embed_size, hidden_size, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)   # 升维
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, embed_size)   # 降回原维度，保证能和残差相加

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 4. Encoder
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Encoder 的一层：多头自注意力 + FFN，各带一个残差连接和 LayerNorm。

    数据流：x -> 自注意力 -> 残差相加 -> LN -> FFN -> 残差相加 -> LN
    （这种"先加残差、后归一化"的顺序是 post-LN 写法。）
    """

    def __init__(self, embed_size, num_heads, d_ff, dropout=DROPOUT):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout=dropout)
        self.ffn = FFN(embed_size, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力：q = k = v = x；src_mask 屏蔽源序列里的 PAD 位置
        attn, _ = self.attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn))  # 残差连接 + LayerNorm

        ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn))
        return x


class Encoder(nn.Module):
    """完整的 Encoder：Embedding + 位置编码 + N 层 EncoderLayer。"""

    def __init__(self, vocab_size, embed_size, num_heads, num_layers,
                 d_ff, max_len, dropout=DROPOUT):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len, dropout=dropout)
        # ModuleList 会把每一层注册成子模块，parameters()/to() 才能正确覆盖它们
        self.layers = nn.ModuleList(
            EncoderLayer(embed_size, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ---------------------------------------------------------------------------
# 5. Decoder
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """Decoder 的一层，比 Encoder 多一个交叉注意力子层。

    三个子层依次是：
    1. 掩码自注意力：只看"当前位置及之前"的 target token（因果掩码），
       防止预测第 i 个词时偷看 i 之后的答案；
    2. 交叉注意力：q 来自 decoder 当前状态，k/v 来自 encoder 输出，
       让 decoder 去"查阅"源序列（src_mask 屏蔽源端 PAD）；
    3. FFN。
    每个子层都跟一个残差连接 + LayerNorm。
    """

    def __init__(self, embed_size, num_heads, d_ff, dropout=DROPOUT):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout=dropout)       # 掩码自注意力
        self.cross_attn = MultiHeadAttention(embed_size, num_heads, dropout=dropout) # 交叉注意力
        self.ffn = FFN(embed_size, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. 掩码自注意力：q = k = v = x；tgt_mask = 因果掩码 & padding 掩码
        attn, _ = self.attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn))

        # 2. 交叉注意力：q = x（decoder），k = v = encoder_output，用源端 padding 掩码
        cross_attn, _ = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        # 3. FFN
        ffn = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn))
        return x


class Decoder(nn.Module):
    """完整的 Decoder：Embedding + 位置编码 + N 层 DecoderBlock + 输出投影。"""

    def __init__(self, vocab_size, embed_size, num_heads, num_layers,
                 d_ff, max_len, dropout=DROPOUT):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            DecoderBlock(embed_size, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        )
        # 最后一层线性变换：把每个位置的向量映射回词表大小，得到 logits
        self.proj = nn.Linear(embed_size, vocab_size)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.proj(x)


# ---------------------------------------------------------------------------
# 6. 组装 Transformer + 掩码工具函数
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    """把 Encoder 和 Decoder 组装成完整的 seq2seq Transformer。"""

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_heads, d_ff,
                 num_layers, max_len=MAX_LEN, dropout=DROPOUT,
                 pad_idx=PAD, bos_idx=BOS, eos_idx=EOS):
        super().__init__()
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.encoder = Encoder(src_vocab_size, embed_size, num_heads,
                               num_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_heads,
                               num_layers, d_ff, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """先编码得到 memory，再解码输出每个位置在词表上的 logits。"""
        memory = self.encoder(src, src_mask)                 # (batch, src_len, embed_size)
        return self.decoder(tgt, memory, src_mask, tgt_mask) # (batch, tgt_len, tgt_vocab)

    @torch.no_grad()
    def generate(self, src, max_len=32, src_mask=None):
        """贪心解码：从 [BOS] 开始一步一步生成，直到遇到 EOS 或达到 max_len。

        注意：这是最简实现，只支持 batch_size = 1。
        """
        memory = self.encoder(src, src_mask)  # 源序列只编码一次，之后每步复用
        tgt = torch.tensor([[self.bos_idx]], device=src.device)  # 从 [BOS] 开始
        outputs = []
        for _ in range(max_len):
            # 每步都要重建掩码：因果掩码（只看已生成的部分）& padding 掩码
            # 注意 make_subsequent_mask 要把 device 传进去，否则 GPU 上会报
            # "mask 在 CPU、scores 在 CUDA" 的设备不一致错误
            tgt_mask = (make_subsequent_mask(tgt.size(1), device=tgt.device)
                        & make_padding_mask(tgt, self.pad_idx))
            logits = self.decoder(tgt, memory, src_mask, tgt_mask)      # (1, cur_len, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 取最后一步概率最大的词
            if next_token.item() == self.eos_idx:  # 遇到 EOS 就停
                break
            outputs.append(next_token.item())
            tgt = torch.cat([tgt, next_token], dim=1)  # 拼到序列末尾，进入下一轮
        return outputs


def make_padding_mask(seq, pad_idx=PAD):
    """padding 掩码：把"不是 PAD"的位置标记为 True。

    seq: (batch, seq_len) -> (batch, 1, 1, seq_len) 的 bool 张量。
    中间两个 1 是为了和注意力分数 (batch, heads, q_len, k_len) 的
    batch、heads 维对齐广播，实际作用在最后的 k_len 维上。
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(sz, device=None):
    """因果（下三角）掩码：第 i 个位置只能看到 0..i，"不能偷看未来"。

    sz: 序列长度 -> (1, 1, sz, sz) 的 bool 张量，True 位于左下三角区域。
    device 参数：掩码默认建在 CPU 上，用在 GPU 时必须显式指定 device。
    """
    mask = torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
