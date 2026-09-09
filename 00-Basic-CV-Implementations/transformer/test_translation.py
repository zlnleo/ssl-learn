# -*- coding: utf-8 -*-
"""训练成果检测：加载 checkpoint，逐条验证翻译结果是否与期望完全一致。

这是"端到端验收测试"，不复用任何训练代码，完全模拟部署时的推理路径：
1. 从 checkpoint 重建模型（结构配置 + 权重）和两个分词器（词表）——不重新训练；
2. 对验收用例逐条走完整推理：encode -> generate -> decode；
3. 断言生成的译文和期望译文逐字一致。模型把训练集记住，说明
   数据集 -> 训练 -> 保存 -> 加载 -> 推理 整条链路是通的；
4. 额外喂一条含未登录词的句子，验证推理不会崩溃（内容不做断言，
   未训练过的组合本来就不保证正确）。

运行：
    python test_translation.py                       # 使用默认 checkpoint.pt
    python test_translation.py --checkpoint xxx.pt   # 指定 checkpoint

退出码：全部通过为 0，有任何一条不一致为 1（可以接进 CI 流水线）。
"""
import argparse
import sys

import torch

from dataset import SimpleTokenizer
from transformer_simple import Transformer
from utils import get_device, load_checkpoint, make_padding_mask

# ---------------------------------------------------------------------------
# 验收用例：和 train.py 里的 DATA 一一对应。
# 测试文件刻意不 import train.py——验收要和训练解耦，期望值是独立写死的，
# 否则"训练改了数据、测试跟着改"就失去了检测意义。
# ---------------------------------------------------------------------------
CASES = [
    ("i love cats", "我 喜欢 猫"),
    ("i love dogs", "我 喜欢 狗"),
    ("hello", "你好"),
    ("i like cats", "我 喜欢 猫"),
]


def rebuild_tokenizer(word2idx, idx2word):
    """用 checkpoint 里保存的词表重建分词器（不需要重新扫描语料）。

    注意：必须用训练时保存的那份词表，自建新词表会导致 id 错位。
    """
    tokenizer = SimpleTokenizer()
    tokenizer.word2idx = word2idx
    tokenizer.idx2word = idx2word
    return tokenizer


def translate(model, src_tokenizer, tgt_tokenizer, text, device, max_len=16):
    """完整推理一条：文本 -> 源端 ids -> 模型贪心生成 -> 目标端文本。"""
    src_ids = torch.tensor(src_tokenizer.encode(text)).unsqueeze(0).to(device)
    src_mask = make_padding_mask(src_ids)
    out_ids = model.generate(src_ids, max_len=max_len, src_mask=src_mask)
    return tgt_tokenizer.decode(out_ids)


def main():
    parser = argparse.ArgumentParser(description="检测训练好的 Transformer 翻译模型")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt",
                        help="checkpoint 文件路径")
    args = parser.parse_args()

    # ---- 1. 加载 checkpoint：权重 + 结构配置 + 两份词表 ----
    ckpt = load_checkpoint(args.checkpoint)
    device = get_device()
    print(f"device: {device}")

    model = Transformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # 推理模式：关闭 dropout，保证结果确定

    src_tokenizer = rebuild_tokenizer(ckpt["src_vocab"], ckpt["src_idx2word"])
    tgt_tokenizer = rebuild_tokenizer(ckpt["tgt_vocab"], ckpt["tgt_idx2word"])

    # ---- 2. 逐条验收：译文必须和期望完全一致 ----
    passed = 0
    for src_text, expected in CASES:
        pred = translate(model, src_tokenizer, tgt_tokenizer, src_text, device)
        ok = (pred == expected)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {src_text!r} -> {pred!r} (期望: {expected!r})")

    # ---- 3. 鲁棒性：含未登录词的句子必须能正常推理、不崩溃 ----
    unknown_pred = translate(model, src_tokenizer, tgt_tokenizer, "i love birds", device)
    print(f"[INFO] 未登录词句子 'i love birds' -> {unknown_pred!r}（只要求能跑通，内容不做断言）")

    # ---- 4. 结论 ----
    if passed == len(CASES):
        print(f"\n全部通过 [OK]: {passed}/{len(CASES)} 条翻译与期望完全一致")
        sys.exit(0)
    else:
        print(f"\n检测失败 [FAIL]: {len(CASES) - passed} 条翻译与期望不一致")
        sys.exit(1)


if __name__ == "__main__":
    main()
