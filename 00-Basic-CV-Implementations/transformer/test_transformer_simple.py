# -*- coding: utf-8 -*-
"""transformer_simple.py 的冒烟测试。

验证内容：
[0] 屏蔽位置的注意力权重严格为 0（mask 是否真正生效）；
[1] 前向：logits 形状正确；
[2] 反向：teacher-forcing 交叉熵损失能正常回传梯度；
[3] 训练循环：优化器 step 能更新参数；
[4] 因果掩码：位置 i 的 logits 不受 i 之后的 token 影响（防偷看验证）；
[5] 贪心生成：generate 能跑完；
[6] 若有 GPU，验证搬到 CUDA 上也能跑（顺带验证掩码的 device 处理）。

运行：python test_transformer_simple.py
"""
import torch
import torch.nn.functional as F

from transformer_simple import (
    ScaleDotAttention,
    Transformer,
    make_padding_mask,
    make_subsequent_mask,
)


def main():
    torch.manual_seed(0)
    src_vocab, tgt_vocab = 32, 32

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_size=64,   # 必须能被 num_heads 整除
        num_heads=8,
        d_ff=256,
        num_layers=2,
    )

    # ---- [0] mask 是否真正生效：被屏蔽位置权重必须为 0 ----
    attn_module = ScaleDotAttention()
    q = torch.randn(1, 2, 3, 8)  # (batch, heads, q_len, d_k)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    mask = torch.tensor([[[[True, True, False, True]]]])  # 屏蔽 k 的第 3 个位置
    _, attn_w = attn_module(q, k, v, mask)
    assert attn_w[:, :, :, 2].abs().max().item() == 0.0
    print("[0] mask OK: 被屏蔽位置的注意力权重严格为 0")

    # ---- 构造带 padding 的 src / tgt ----
    src = torch.tensor([
        [1, 5, 6, 7, 8, 2, 0, 0, 0, 0],   # BOS ... EOS PAD PAD ...
        [1, 9, 10, 2, 0, 0, 0, 0, 0, 0],
    ])
    tgt = torch.tensor([
        [1, 11, 12, 2, 0, 0],
        [1, 13, 2, 0, 0, 0],
    ])
    src_mask = make_padding_mask(src)
    tgt_mask = make_subsequent_mask(tgt.size(1)) & make_padding_mask(tgt)

    # ---- [1] 前向：logits 形状 = (batch, tgt_len, tgt_vocab) ----
    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (2, 6, tgt_vocab), logits.shape
    print("[1] forward OK, logits shape:", tuple(logits.shape))

    # ---- [2] 反向：用位置 i 的 logits 预测位置 i+1 的 token ----
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, tgt_vocab),
        tgt[:, 1:].reshape(-1),
        ignore_index=0,  # 忽略 PAD 位置，不参与损失
    )
    loss.backward()
    assert model.encoder.embedding.embedding.weight.grad is not None
    print("[2] backward OK, loss =", round(loss.item(), 4))

    # ---- [3] 训练循环：优化器 step 能更新参数 ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    print("[3] optimizer step OK")

    # ---- [4] 因果掩码验证：改了最后一个 token，前面的 logits 必须不变 ----
    model.eval()  # 关掉 dropout，保证两次前向的差异只来自输入差异
    tgt_a = torch.tensor([[1, 4, 5, 6]])
    tgt_b = torch.tensor([[1, 4, 5, 7]])  # 只改了最后一个 token
    src_a = src[:1, :4]                   # [BOS, 5, 6, 7]，无 padding
    m = make_subsequent_mask(tgt_a.size(1))
    logits_a = model(src_a, tgt_a, make_padding_mask(src_a), m)
    logits_b = model(src_a, tgt_b, make_padding_mask(src_a), m)
    # 位置 0..2 看不到位置 3 的 token，它们的 logits 必须完全一致
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6)
    # 位置 3 自己能看到自己，logits 必须不同
    assert not torch.allclose(logits_a[:, 3], logits_b[:, 3], atol=1e-6)
    print("[4] causal mask OK: 前 3 个位置不受未来 token 影响")
    model.train()

    # ---- [5] 贪心生成 ----
    out = model.generate(src[:1], max_len=8, src_mask=make_padding_mask(src[:1]))
    print("[5] generate OK, 输出:", out, "(遇到 EOS 前的 token 序列)")

    # ---- [6] GPU 验证（若有）----
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        src_c = src_a.to(device)
        tgt_c = tgt_a.to(device)
        logits_c = model(
            src_c, tgt_c,
            make_padding_mask(src_c),
            make_subsequent_mask(tgt_c.size(1), device=device),
        )
        print("[6] CUDA forward OK, logits shape:", tuple(logits_c.shape))
    else:
        print("[6] 当前环境无 GPU，跳过 CUDA 测试")

    print("\n全部测试通过 [OK]")


if __name__ == "__main__":
    main()
