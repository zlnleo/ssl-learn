# -*- coding: utf-8 -*-
"""
模块 09 · experiment.py
========================
Swin-Tiny 的「总装验证实验」：结构摘要、精确参数量与占比、前向 sanity check、
backbone（特征）模式、eval 确定性验证。全部输出为 ASCII/文本，无绘图依赖。

运行：
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe 09_swin_transformer\\experiment.py
"""

import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from swin_transformer import swin_tiny


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def ascii_bar(name, value, total, width=44):
    """画一根 ASCII 条形图：名称 + 占比 + 数值。"""
    ratio = value / total
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{name:<18} |{bar}| {ratio * 100:5.1f}%  ({value:>11,})"


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("Swin-Tiny 总装验证实验")
    print("=" * 78)

    # ---------- 1. 结构摘要 ----------
    print("\n[1] 模型结构摘要")
    print("-" * 78)
    model = swin_tiny(num_classes=1000)
    print(f"PatchEmbed: conv 3->{model.embed_dim}, patch={model.patch_size}")
    print(f"总 stage 数: {model.num_layers}, 末层通道 num_features={model.num_features}")
    for i, layer in enumerate(model.layers):
        dim = int(model.embed_dim * 2 ** i)
        heads = layer.blocks[0].attn.num_heads
        down = "有 PatchMerging" if layer.downsample is not None else "无(最后一层)"
        print(f"  Stage {i + 1}: dim={dim:>4}, blocks={layer.depth}, heads={heads:>2}, "
              f"window={layer.blocks[0].window_size}, downsample: {down}")

    # ---------- 2. 精确参数量与占比 ----------
    print("\n[2] 精确参数量与各部件占比（对照 math.md 明细表）")
    print("-" * 78)
    parts = [
        ("PatchEmbed", count_params(model.patch_embed)),
        ("Stage 1", count_params(model.layers[0])),
        ("Stage 2", count_params(model.layers[1])),
        ("Stage 3", count_params(model.layers[2])),
        ("Stage 4", count_params(model.layers[3])),
        ("末尾 LN", count_params(model.norm)),
        ("分类头 head", count_params(model.head)),
    ]
    total = sum(v for _, v in parts)
    for name, v in parts:
        print(ascii_bar(name, v, total))
    print("-" * 78)
    print(f"总计参数量: {total:,} = {total / 1e6:.3f} M   (math.md 推导 28,288,354 ≈ 28.29M)")
    assert 27.5e6 < total < 29.5e6, f"参数量超出预期区间: {total}"

    # ---------- 3. 前向 logits sanity check ----------
    print("\n[3] 随机权重下前向 logits 的均值/方差（sanity check）")
    print("-" * 78)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
    print(f"输出形状: {tuple(logits.shape)}")
    print(f"logits 均值: {logits.mean().item():+.4f}")
    print(f"logits 标准差: {logits.std().item():.4f}")
    print("(随机初始化下 logits 应接近 0 均值、标准差处于合理量级，说明梯度/数值无爆炸)")

    # ---------- 4. backbone 特征模式 ----------
    print("\n[4] num_classes=0 的 backbone（特征）模式")
    print("-" * 78)
    backbone = swin_tiny(num_classes=0)
    backbone.eval()
    with torch.no_grad():
        feat = backbone(torch.randn(2, 3, 224, 224))
    print(f"backbone 输出特征形状: {tuple(feat.shape)}  (期望 (2, 768))")
    assert tuple(feat.shape) == (2, 768)

    # ---------- 5. eval 模式确定性验证 ----------
    print("\n[5] eval 模式前向两次一致性（确定性验证）")
    print("-" * 78)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    same = torch.equal(a, b)
    max_diff = (a - b).abs().max().item()
    print(f"两次前向完全一致: {same}   (最大差异 {max_diff:.2e})")
    assert same, "eval 模式下两次前向应完全一致"

    print("\n" + "=" * 78)
    print("实验完成：结构 / 参数量 / sanity / backbone / 确定性 全部通过 ✓")
    print("=" * 78)


if __name__ == "__main__":
    main()
