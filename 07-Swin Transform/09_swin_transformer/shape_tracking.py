# -*- coding: utf-8 -*-
"""
模块 09 · shape_tracking.py
============================
逐段打印 Swin-Tiny 在 (2, 3, 224, 224) 输入下的张量形状演化，并逐项断言，
用于建立「代码结构与数据流动」的直觉（与 README 的形状总表、math.md 完全对应）。

运行：
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe 09_swin_transformer\\shape_tracking.py
"""

import sys
import torch

# 让中文在任意 stdout 编码下都能安全输出（Windows 管道/重定向下不抛异常）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from swin_transformer import swin_tiny


def count_params(module):
    """统计某个子模块的可训练参数量。"""
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(0)
    model = swin_tiny(num_classes=1000)
    model.eval()                      # 追踪形状时置 eval，去掉 dropout/drop_path 随机性

    x = torch.randn(2, 3, 224, 224)   # batch=2, 3 通道, 224×224

    print("=" * 72)
    print("Swin-Tiny 形状跟踪：输入 (2, 3, 224, 224)")
    print("=" * 72)

    # ---- 1. PatchEmbed ----
    H, W = x.shape[2] // model.patch_size, x.shape[3] // model.patch_size
    x = model.patch_embed(x)          # (B, L, C)
    print(f"[PatchEmbed] 输出 {tuple(x.shape)}   (H={H}, W={W})")
    assert tuple(x.shape) == (2, 3136, 96), f"PatchEmbed 形状错误: {tuple(x.shape)}"

    # ---- 2. 四个 stage（含每个 stage 末尾的 PatchMerging） ----
    expected_stage_shapes = [(2, 784, 192), (2, 196, 384), (2, 49, 768), (2, 49, 768)]
    expected_hw = [(28, 28), (14, 14), (7, 7), (7, 7)]
    for i, layer in enumerate(model.layers):
        x, H, W = layer(x, H, W)
        print(f"[Stage {i + 1}] 输出 {tuple(x.shape)}   (H={H}, W={W})")
        assert tuple(x.shape) == expected_stage_shapes[i], \
            f"Stage{i + 1} 形状错误: {tuple(x.shape)}"
        assert (H, W) == expected_hw[i], f"Stage{i + 1} H/W 错误: ({H}, {W})"

    # ---- 3. 末尾 LayerNorm ----
    x = model.norm(x)
    print(f"[末尾 LN ] 输出 {tuple(x.shape)}")
    assert tuple(x.shape) == (2, 49, 768)

    # ---- 4. 全局平均池化 ----
    pooled = x.mean(dim=1)            # (B, C)
    print(f"[全局池化] 输出 {tuple(pooled.shape)}")
    assert tuple(pooled.shape) == (2, 768)

    # ---- 5. 分类头 ----
    out = model.head(pooled)
    print(f"[分类头  ] 输出 {tuple(out.shape)}")
    assert tuple(out.shape) == (2, 1000)

    # ---- 6. 与整模型前向一致性校验 ----
    with torch.no_grad():
        ref = model(torch.randn(2, 3, 224, 224))
        assert tuple(ref.shape) == (2, 1000)
    print(f"[一致性  ] 整模型 forward 输出 {tuple(ref.shape)}（与手拆路径一致）")

    # ---- 7. 各 stage 参数量 ----
    print("\n" + "-" * 72)
    print("各部件参数量（与 math.md 明细表对照）")
    print("-" * 72)
    total = 0
    print(f"PatchEmbed         : {count_params(model.patch_embed):>10,}")
    total += count_params(model.patch_embed)
    for i, layer in enumerate(model.layers):
        n = count_params(layer)
        total += n
        print(f"Stage {i + 1} (block+merge): {n:>10,}")
    n = count_params(model.norm)
    total += n
    print(f"末尾 LN            : {n:>10,}")
    n = count_params(model.head)
    total += n
    print(f"分类头 head        : {n:>10,}")
    print("-" * 72)
    print(f"总计               : {total:>10,}  ({total / 1e6:.3f} M)")
    print("=" * 72)
    print("全部形状断言通过 ✓")


if __name__ == "__main__":
    main()
