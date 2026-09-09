# -*- coding: utf-8 -*-
"""训练 / 推理 / 测试三方共用的工程工具。

包含：设备选择、掩码构造、checkpoint 存取。

单独拆出 utils.py 而不是全塞在模型文件里的原因（工业化惯例）：
模型文件只负责定义网络结构；设备、掩码、存取这类"工程胶水"集中放一个
地方，train / 推理 / test 三个脚本共用同一份实现，避免同一个逻辑
抄三遍——改一处就处处生效（单一数据源原则）。
"""
from typing import Dict, Optional

import torch

from transformer_simple import PAD


def get_device():
    """统一设备选择逻辑：有 GPU 用 GPU，否则 CPU。

    所有脚本都从这里拿 device，而不是各自写一遍 cuda.is_available()，
    将来加 MPS（Mac）、指定 GPU 编号等逻辑时只需要改这一个函数。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_padding_mask(seq: torch.Tensor, pad_idx: int = PAD) -> torch.Tensor:
    """padding 掩码：把"不是 PAD"的位置标记为 True。

    seq: (batch, seq_len) -> (batch, 1, 1, seq_len) 的 bool 张量。
    中间两个 1 是为了和注意力分数 (batch, heads, q_len, k_len) 的
    batch、heads 维对齐广播，实际作用在最后的 k_len 维上。
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    """因果（下三角）掩码：第 i 个位置只能看到 0..i，"不能偷看未来"。

    sz: 序列长度 -> (1, 1, sz, sz) 的 bool 张量，True 位于左下三角区域。
    device 参数：掩码默认建在 CPU 上，模型在 GPU 上时必须显式传入 device，
    否则 masked_fill 会报设备不一致错误。
    """
    mask = torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def build_masks(src: torch.Tensor, tgt_input: torch.Tensor):
    """一次构造训练/验证所需的全部掩码。

    - src_mask：屏蔽源端 PAD 位置；
    - tgt_mask：因果掩码 & 目标端 PAD 掩码，
      保证 decoder 既看不到未来信息、也不去注意自己这边的 PAD。

    训练和测试脚本共用这个函数，掩码逻辑只有一份实现。
    """
    src_mask = make_padding_mask(src)
    tgt_mask = (make_padding_mask(tgt_input)
                & make_subsequent_mask(tgt_input.size(1), device=tgt_input.device))
    return src_mask, tgt_mask


def save_checkpoint(path: str,
                    model_state: Dict,
                    config: Dict,
                    src_vocab: Dict,
                    src_idx2word: Dict,
                    tgt_vocab: Dict,
                    tgt_idx2word: Dict) -> None:
    """把"模型权重 + 结构配置 + 两个分词器的词表"打包保存成一个文件。

    为什么词表也要存：推理/测试时必须用和训练时完全一致的词表，
    否则同一个 id 在两边代表不同的词，输出全是乱码。
    一个 checkpoint 文件自包含所有重建信息，方便迁移和部署。
    """
    checkpoint = {
        "model_state": model_state,      # model.state_dict()，纯权重
        "config": config,                # 模型超参数，用于重建网络结构
        "src_vocab": src_vocab,          # 源端词表 word2idx
        "src_idx2word": src_idx2word,    # 源端词表 idx2word
        "tgt_vocab": tgt_vocab,          # 目标端词表 word2idx
        "tgt_idx2word": tgt_idx2word,    # 目标端词表 idx2word
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict:
    """读取 checkpoint 并整体搬到指定设备。

    weights_only=False 的原因：checkpoint 里除了权重张量，还存了
    词表 dict（int/str），纯张量模式会拒绝加载。
    """
    device = device or get_device()
    return torch.load(path, map_location=device, weights_only=False)
