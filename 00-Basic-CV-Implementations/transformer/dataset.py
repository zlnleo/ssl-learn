# -*- coding: utf-8 -*-
"""数据集与分词器。

包含三个部分：
- SimpleTokenizer：按空格切词的极简分词器（词表构建 / 编码 / 解码）；
- TranslationDataset：把 (源句, 目标句) 对包装成 PyTorch Dataset；
- collate_fn：把一个 batch 里不等长的序列 padding 对齐成规整张量。

特殊 token 约定（与 transformer_simple.py 保持一致）：
    PAD = 0, BOS = 1, EOS = 2, UNK = 3
普通词从 4 开始编号，保证所有 id 连续且正好覆盖 Embedding 的 vocab_size。
"""
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from transformer_simple import BOS, EOS, PAD

UNK = 3  # 未登录词 <UNK> 的固定 id，紧跟三个特殊 token 之后


class SimpleTokenizer:
    """基于空格切分的极简分词器。

    工业化项目里这里通常替换成 BPE/WordPiece（如 HuggingFace 的 tokenizers），
    这里为了配合"手搓 Transformer"的教学目标保持最小实现，但对外接口
    （build_vocab / encode / decode / vocab_size）与真实分词器对齐——
    以后换成 SentencePiece 只需要改这一个类，其余代码不用动。
    """

    def __init__(self):
        # word2idx: 词 -> id；idx2word: id -> 词（互为反查表，都要维护）
        self.word2idx = {"<PAD>": PAD, "<BOS>": BOS, "<EOS>": EOS, "<UNK>": UNK}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    @property
    def vocab_size(self) -> int:
        """词表大小，用于初始化模型的 vocab_size 参数。"""
        return len(self.word2idx)

    def build_vocab(self, sentences: List[str]) -> None:
        """扫描语料建立词表（只增不删），普通词从 UNK+1 开始编号。

        注意：必须先 build_vocab 再 encode！
        否则词表里只有特殊 token，所有实词都会被映射成 UNK，
        模型学到的东西毫无意义。
        """
        idx = UNK + 1
        for sentence in sentences:
            for word in sentence.split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """把一句文本编码成 id 序列，两端自动补 [BOS] 和 [EOS]。

        - 词表里没有的词统一映射为 UNK（训练集里不会出现，推理时兜底）；
        - max_len 为 None 时不截断；否则把词序列截到 max_len-2，
          给首尾的 BOS/EOS 留出位置。
        """
        ids = [self.word2idx.get(word, UNK) for word in text.split()]
        if max_len is not None and len(ids) > max_len - 2:
            ids = ids[:max_len - 2]
        return [BOS] + ids + [EOS]

    def decode(self, ids: List[int]) -> str:
        """把 id 序列还原成文本：跳过 BOS，遇到 EOS/PAD 提前结束，未知 id 输出 <UNK>。"""
        words = []
        for idx in ids:
            if idx == BOS:
                continue
            if idx in (EOS, PAD):
                break
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)


class TranslationDataset(Dataset):
    """机器翻译数据集：每一条样本是 (源句, 目标句) 文本对，取用时即时编码。

    采用"即时编码"（在 __getitem__ 里才 encode）而不是一次性预处理缓存：
    对大规模语料更省内存；缺点是每条样本会被重复编码。
    本数据集只有几条样本，两种方式差别可以忽略。
    """

    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        self.data = data                              # List[(src_text, tgt_text)]
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src_text, tgt_text = self.data[index]
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)
        # dtype 必须显式指定为 long：nn.Embedding 的输入要求整数索引，
        # 若不指定，torch.tensor 会推断成 int32/int64，在部分平台上会报 dtype 错误
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]],
               max_len: Optional[int] = None):
    """把一个 batch 的 (src, tgt) 对 padding 成两个规整的二维张量。

    batch: List[(src_tensor, tgt_tensor)]，各样本长度可以不同；
    pad_sequence 按 batch 内最长样本对齐，短序列在末尾补 PAD。

    返回: (src, tgt)，形状均为 (batch, max_seq_len)。
    max_len: 可选，超出部分直接截断（防止极端长句撑爆显存）。
    """
    src_list, tgt_list = [], []
    for src, tgt in batch:
        src_list.append(src)
        tgt_list.append(tgt)

    src = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=PAD,
    )
    tgt = torch.nn.utils.rnn.pad_sequence(
        tgt_list, batch_first=True, padding_value=PAD,
    )

    if max_len is not None:
        src = src[:, :max_len]
        tgt = tgt[:, :max_len]
    return src, tgt
