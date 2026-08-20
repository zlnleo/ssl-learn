# -*- coding: utf-8 -*-
"""训练脚本：在小型英 -> 中玩具数据集上训练手写 Transformer。

相对"教学版脚本"，这里补上了几个工业化惯例：
1. 训练逻辑全部收进 main()，配合 `__main__` 守卫——
   文件被 import 时不会意外执行，也方便单元测试；
2. 超参数通过 argparse 从命令行传入（带默认值），改配置不改代码；
3. 固定随机种子，结果可复现；
4. 训练结束后保存 checkpoint（模型权重 + 结构配置 + 两个分词器词表），
   推理/测试脚本直接加载，不必重新训练；
5. 数据、模型、损失、优化器各管一段，训练循环独立成函数。

用法：
    python train.py                                    # 默认配置训练 100 轮
    python train.py --epochs 300 --lr 3e-4             # 调超参数
    python train.py --save-path my_checkpoint.pt       # 指定保存路径
"""
import argparse
import functools
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SimpleTokenizer, TranslationDataset, collate_fn
from transformer_simple import PAD, Transformer
from utils import build_masks, get_device, make_padding_mask, save_checkpoint

# ---------------------------------------------------------------------------
# 玩具数据集：4 条英 -> 中句对（特意选用了"词表极小、模式简单"的语料，
# 保证手写模型在几十轮内就能记住，便于验证整条 pipeline 是否正确）
# ---------------------------------------------------------------------------
DATA = [
    ("i love cats", "我 喜欢 猫"),
    ("i love dogs", "我 喜欢 狗"),
    ("hello", "你好"),
    ("i like cats", "我 喜欢 猫"),
]


def parse_args():
    """命令行参数：训练脚本的配置入口，全部带默认值。"""
    parser = argparse.ArgumentParser(description="训练手写 Transformer 翻译模型")
    # 训练相关
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    # 模型相关
    parser.add_argument("--embed-size", type=int, default=128, help="词向量维度")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN 隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=2, help="Encoder/Decoder 层数")
    parser.add_argument("--max-len", type=int, default=50, help="序列最大长度")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 概率")
    # 输出相关
    parser.add_argument("--save-path", type=str, default="checkpoint.pt",
                        help="checkpoint 保存路径")
    return parser.parse_args()


def build_tokenizers():
    """构建并初始化源端/目标端两个分词器。

    关键步骤：build_vocab 必须先执行！否则词表里只有 4 个特殊 token，
    所有实词都会被 encode 成 UNK——模型拿到的全是垃圾信息，
    这是原版代码直接跑不起来的根源。
    """
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    src_tokenizer.build_vocab([src for src, _ in DATA])  # 扫描源端语料建词表
    tgt_tokenizer.build_vocab([tgt for _, tgt in DATA])  # 扫描目标端语料建词表
    return src_tokenizer, tgt_tokenizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一轮，返回该轮平均 loss。

    teacher forcing（教师强制）：
    - decoder 输入 = 目标序列去掉最后一个词：  [BOS, w1, ..., wn]
    - 预测目标   = 目标序列去掉第一个词：      [w1, ..., wn, EOS]
    即"用位置 i 预测位置 i+1"。虽然把整句一次性喂给了 decoder，
    但有因果掩码挡住未来信息，位置 i 看不到 i+1 之后的词，不会作弊。
    """
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]  # decoder 的输入（带 BOS、去 EOS）
        tgt_label = tgt[:, 1:]   # 每个位置要预测的下一个词（去 BOS、带 EOS）

        # 一次构建两个掩码：源端 padding 掩码 + 目标端(因果 & padding)掩码
        src_mask, tgt_mask = build_masks(src, tgt_input)

        # 前向：输出 (batch, tgt_len-1, tgt_vocab) 的 logits
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # 交叉熵：把 (batch, seq, vocab) 展平成 (batch*seq, vocab) 再算；
        # ignore_index=PAD 让 padding 位置不参与损失
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))

        # 标准三步：清零梯度 -> 反向传播 -> 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def demo_translate(model, src_tokenizer, tgt_tokenizer, device):
    """训练完成后用训练集句子做一次推理演示，肉眼检查效果（不影响训练）。

    eval() 关闭 dropout，保证推理结果确定。
    """
    model.eval()
    for src_text, _ in DATA:
        src_ids = torch.tensor(src_tokenizer.encode(src_text)).unsqueeze(0).to(device)
        src_mask = make_padding_mask(src_ids)
        out_ids = model.generate(src_ids, max_len=16, src_mask=src_mask)
        print(f"  {src_text!r} -> {tgt_tokenizer.decode(out_ids)!r}")


def main():
    args = parse_args()

    # 固定随机种子：同样的种子 + 同样的数据 => 同样的训练结果，方便复现和调试
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f"device: {device}")

    # ---- 1. 数据流水线：分词器 -> Dataset -> DataLoader ----
    src_tokenizer, tgt_tokenizer = build_tokenizers()
    print(f"src_vocab_size: {src_tokenizer.vocab_size}, "
          f"tgt_vocab_size: {tgt_tokenizer.vocab_size}")

    dataset = TranslationDataset(DATA, src_tokenizer, tgt_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 训练集打乱顺序，避免模型记住样本顺序而不是规律
        # partial 把 max_len 固定进 collate_fn，DataLoader 只负责传 batch
        collate_fn=functools.partial(collate_fn, max_len=args.max_len),
    )

    # ---- 2. 模型 / 损失 / 优化器 ----
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 3. 训练主循环 ----
    print(f"start training: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        # 每 10 轮打印一次，避免刷屏；loss 持续下降说明训练正常
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:>3}/{args.epochs}, loss: {loss:.4f}")
    print(f"training finished in {time.time() - start:.1f}s, final loss: {loss:.4f}")

    # ---- 4. 训练后演示 + 保存 checkpoint ----
    print("demo translations:")
    demo_translate(model, src_tokenizer, tgt_tokenizer, device)

    # config 用于在测试/部署时原样重建网络结构，必须和训练时一致
    config = dict(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    save_checkpoint(
        args.save_path,
        model_state=model.state_dict(),
        config=config,
        src_vocab=src_tokenizer.word2idx,
        src_idx2word=src_tokenizer.idx2word,
        tgt_vocab=tgt_tokenizer.word2idx,
        tgt_idx2word=tgt_tokenizer.idx2word,
    )
    print(f"checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
