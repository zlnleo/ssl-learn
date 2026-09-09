"""实验 3：Swin without Patch Merging vs Swin with Patch Merging —— 层级表示的价值。

对照：
  with merge    ：标准 Swin-Tiny（PatchMerging 开，56→28→14→7，通道 96→192→384→768）
  without merge ：PatchMerging 全关（4 个 stage 恒为 56x56，通道恒 96，heads 恒 3）

观测：参数量、MACs、速度、显存、（可选）CIFAR-10 精度。
注意：无 merging 变体在所有 stage 保持 56x56 分辨率，token 数 3136 恒定，
其注意力部分计算量不随深度递减——这正是 Patch Merging 存在的理由之一。

用法（项目根目录，ssl_cv 环境）：
  python experiments/exp3_patch_merging_ablation.py              # 静态 + 训练
  python experiments/exp3_patch_merging_ablation.py --skip-train # 只静态
"""
import argparse

import torch

import common
from common import (build_cifar, get_device, model_stats, print_table, seed_all,
                    train_model, fmt)
from swin import swin_tiny


def static_analysis(args):
    print("\n" + "=" * 78)
    print("实验 3-A：静态测量 —— 有无 Patch Merging 的 Swin-Tiny")
    print("=" * 78)
    n_cls = 100 if args.dataset == "cifar100" else 10
    x = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    models = {
        "with merge   ": swin_tiny(num_classes=n_cls),
        "without merge": swin_tiny(num_classes=n_cls, patch_merging=False),
    }
    rows = []
    for name, m in models.items():
        stats = model_stats(m, x, label=name, device=args.device)
        # 各 stage 分辨率轨迹
        H = W = args.img_size // 4
        sizes = []
        for layer in m.layers:
            sizes.append(f"{H}x{W}/{layer.blocks[0].attn.dim}")
            if layer.downsample is not None:
                H, W = (H + 1) // 2, (W + 1) // 2
        rows.append([name, fmt(stats["params"]), f"{stats['gflops']:.2f} G",
                     f"{stats['ms']:.2f} ms", " -> ".join(sizes)])
    print_table(["配置", "参数量", "≈FLOPs/样本(224)", "前向耗时", "各 stage (分辨率/通道)"],
                rows, title=None)
    print("  解读（与实测一致）：")
    print("  - QKV/MLP 投影部分：通道翻倍恰好抵消 token 数 4 倍缩减，两种配置几乎相同；")
    print("  - 注意力点积部分：有 merging 时每 stage 减半（∝ hw·C/4^i·2^i），无 merging 时")
    print("    12 个 block 全部在 56x56 高分辨率上做注意力，这部分更贵；")
    print("  - 参数量：merging 让通道逐级翻倍，MLP 参数 ∝ C²，因此 with merge 参数量反而大得多；")
    print("  - 本质：层级表示用通道容量换取分辨率，以更少的注意力代价获得更大的有效感受野。")


def train_part(args):
    print("\n" + "=" * 78)
    print(f"实验 3-B：端到端精度 —— 有无 Patch Merging（{args.dataset} @ {args.img_size}）")
    print("=" * 78)
    seed_all(args.seed)
    train_loader, val_loader, num_classes = build_cifar(
        args.dataset, args.img_size, args.batch, args.num_workers, args.data_dir)
    results = []
    configs = {
        "with merge   ": dict(patch_merging=True),
        "without merge": dict(patch_merging=False),
    }
    for name, cfg in configs.items():
        print(f"\n--- 训练 {name} ---")
        model = swin_tiny(num_classes=num_classes, **cfg)
        stats = model_stats(model, torch.randn(1, 3, args.img_size, args.img_size, device=args.device),
                            label=name, device=args.device)
        print(f"  参数量 {fmt(stats['params'])} | ≈FLOPs {stats['gflops']:.2f} G | "
              f"前向 {stats['ms']:.2f} ms")
        res = train_model(model, train_loader, val_loader, args.epochs, lr=args.lr,
                          device=args.device, label=name)
        results.append((name, stats, res))
    print("\n[3-B 结论表]")
    print_table(["配置", "参数量", "≈FLOPs/样本", "最佳 val acc", "最终 val acc"], [
        [name, fmt(s["params"]), f"{s['gflops']:.2f} G", f"{r['best_acc']:.3f}", f"{r['final_acc']:.3f}"]
        for name, s, r in results
    ])


def main():
    ap = argparse.ArgumentParser(description="实验 3：有无 Patch Merging")
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"],
                    help="训练数据集（本地已有则直接使用，不下载）")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data-dir", default=None,
                    help="数据集根目录，默认自动探测本地已有目录（D:/project/.../data 或 ./data）")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    args.device = get_device(args.device)
    static_analysis(args)
    if not args.skip_train:
        train_part(args)
    print("\n实验 3 完成。预期结论：Patch Merging 构建的层级表示（分辨率减半 + 通道翻倍）"
          "以更少的注意力计算获得逐级扩大的感受野；无 merging 变体全程高分辨率做注意力，"
          "计算更贵且缺少层级归纳偏置——这正是 Swin 优于 ViT（固定分辨率）的关键设计之一。")


if __name__ == "__main__":
    main()
