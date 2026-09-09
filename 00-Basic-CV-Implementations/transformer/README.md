# transformer —— 从零手写 Transformer（Encoder–Decoder 机器翻译）

> 用 `nn.Module` 从零手写一个 Encoder–Decoder Transformer（不调用 `torch.nn.Transformer`），在极小的英 → 中玩具语料上跑通「数据 → 训练 → 保存 → 加载 → 推理」完整闭环，用于学习 Transformer 每个组件的原理与实现。

## 概述

本目录是对论文「Attention Is All You Need」的逐步手写复现。全部组件——词嵌入、正弦位置编码、缩放点积注意力、多头注意力、FFN、Encoder、Decoder——都用 PyTorch 的 `nn.Module` 手写，没有调用 `torch.nn.Transformer`。

关键设计：

- 同一个 `MultiHeadAttention` 类通过传入不同 mask，复用于三种注意力：Encoder 自注意力、Decoder 掩码（因果）自注意力、Decoder 交叉注意力。
- mask 约定：bool 张量，`True = 可以看`、`False = 屏蔽`（在注意力分数里把 False 位置 `masked_fill` 成 `-inf`）。
- 特殊 token：`PAD=0`、`BOS=1`、`EOS=2`（`UNK=3` 定义在 dataset.py）。
- 训练用 teacher forcing（decoder 一次看到整句，靠因果掩码挡未来），推理用 `generate()` 逐 token 贪心解码（仅支持 batch_size=1）。
- 任务：机器翻译。语料是 `train.py` 里硬编码的 4 条英 → 中句对（玩具数据，用于验证整条 pipeline 而非追求翻译质量）。

## 目录结构 / 文件说明

| 文件 | 作用 |
|---|---|
| `transformer_simple.py` | 核心模型：`Embedding`、`PositionalEncoding`（正弦位置编码）、`ScaleDotAttention`、`MultiHeadAttention`、`FFN`、`EncoderLayer`/`Encoder`、`DecoderBlock`/`Decoder`、`Transformer` 组装类，以及 `make_padding_mask` / `make_subsequent_mask` 掩码工具、贪心解码 `generate()` |
| `dataset.py` | 数据层：`SimpleTokenizer`（按空格切词，含 build_vocab / encode / decode）、`TranslationDataset`（(源句, 目标句) 对）、`collate_fn`（把不等长序列 padding 对齐） |
| `utils.py` | 工程工具：`get_device`（设备选择）、`make_padding_mask` / `make_subsequent_mask` / `build_masks`（掩码构造）、`save_checkpoint` / `load_checkpoint`（checkpoint 存取） |
| `train.py` | 训练入口（argparse 参数 + 固定随机种子 + 训练循环 + 训练后演示翻译 + 保存 checkpoint）；语料为内置 4 条英 → 中句对 |
| `test_transformer_simple.py` | 模型冒烟测试：验证 mask 生效、前向 logits 形状、反向回传、优化器更新、因果掩码防偷看、贪心生成、CUDA（如有 GPU） |
| `test_translation.py` | 端到端验收测试：从 checkpoint 重建模型与词表，逐条断言译文与期望完全一致，退出码 0/1 可接 CI |
| `checkpoint.pt` | 训练产物（模型权重 + 结构配置 + 两个分词器词表），由 `train.py` 生成，可被 `test_translation.py` 加载 |

> `__pycache__/` 为 Python 字节码缓存，自动生成，可忽略。

## 快速开始

环境：conda 环境 `ssl_cv`（Python 3.10，PyTorch）。本目录不依赖外部数据集（语料内置在 `train.py` 的 `DATA` 里），无需 `data/` 目录。

```bash
# 在 transformer/ 目录下执行

# 1) 冒烟测试：验证模型各组件正确（无需先训练）
python test_transformer_simple.py

# 2) 训练：默认配置训练 100 轮，结束后打印 demo 翻译并保存 checkpoint.pt
python train.py
python train.py --epochs 300 --lr 3e-4        # 调超参数
python train.py --save-path my_checkpoint.pt  # 指定保存路径

# 3) 端到端验收：加载 checkpoint，逐条验证翻译
python test_translation.py                        # 用默认 checkpoint.pt
python test_translation.py --checkpoint my_checkpoint.pt
```

`train.py` 主要命令行参数（均带默认值）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 2 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--seed` | 42 | 随机种子（可复现） |
| `--embed-size` | 128 | 词向量维度 |
| `--num-heads` | 8 | 注意力头数 |
| `--d-ff` | 512 | FFN 隐藏层维度 |
| `--num-layers` | 2 | Encoder/Decoder 层数 |
| `--max-len` | 50 | 序列最大长度 |
| `--dropout` | 0.1 | dropout 概率 |
| `--save-path` | checkpoint.pt | checkpoint 保存路径 |

训练内置的玩具语料（`train.py` 中 `DATA`，4 条英 → 中句对）：`i love cats`、`i love dogs`、`hello`、`i like cats` → 对应中文译文。

## 备注

- 训练损失用 `CrossEntropyLoss(ignore_index=PAD)`，优化器为 `Adam`。
- `generate()` 是最简贪心解码实现，只支持 batch_size=1（训练时的 teacher forcing 不受此限制）。
- 本目录与 `../vit/` 共享同一套注意力实现思路：ViT 就是「把词换成图片 patch、去掉 Decoder 与掩码的 Transformer Encoder + 分类头」。

---

[← 返回仓库根 README](../../README.md) · [↑ 返回 00-Basic-CV-Implementations](../README.md)
