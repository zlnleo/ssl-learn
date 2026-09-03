"""实验 2（最重要）：Window Attention vs Shifted Window Attention —— 为什么必须移位。

两层结构对照：
  W-W  ：两层 W-MSA（窗口永不移动，信息困在窗口内）
  W-SW ：第一层 W-MSA + 第二层 SW-MSA（窗口错位，信息跨窗流动）

- 结构分析（无需训练、与权重无关）：两层后每个 token 的"可达 token 集"大小
  与中心 token 的覆盖图（ASCII），证明 W-W 感受野=本窗口，W-SW 感受野显著扩张。
- 端到端精度（--skip-train 可跳过）：CIFAR-10 上两种配置的 val acc 对比。

用法（项目根目录，ssl_cv 环境）：
  python experiments/exp2_window_vs_shifted.py
  python experiments/exp2_window_vs_shifted.py --skip-train
"""
import argparse

import torch

import common
from common import (TokenClassifier, analyze_connectivity, build_cifar, get_device,
                    model_stats, print_table, seed_all, train_model, fmt)


def connectivity_analysis(args):
    print("\n" + "=" * 78)
    print("实验 2-A：结构分析 —— 两层注意力后的可达 token 集（与权重无关）")
    print("=" * 78)
    H = W = args.img_size // 4          # patch 4 -> 特征图尺寸
    for name, shift in [("W-W（两层 W-MSA）", 0), ("W-SW（W-MSA + SW-MSA）", args.window // 2)]:
        reach, cover = analyze_connectivity(H, W, args.window, shift_second=shift)
        n = H * W
        print(f"\n--- {name}，window={args.window}，{H}x{W} 特征图 ---")
        print(f"  可达 token 数：min={min(reach)}  mean={sum(reach)/n:.1f}  max={max(reach)}  "
              f"(全图 {n} 个 token，占比 {sum(reach)/n/n*100:.1f}%)")
        print(f"  中心 token（O）两层后覆盖图（# = 可达）：")
        print("\n".join("    " + line for line in cover.splitlines()))


def train_part(args):
    print("\n" + "=" * 78)
    print(f"实验 2-B：端到端精度 —— Window vs Shifted Window（{args.dataset}）")
    print("=" * 78)
    seed_all(args.seed)
    train_loader, val_loader, num_classes = build_cifar(
        args.dataset, args.img_size, args.batch, args.num_workers, args.data_dir)
    results = []
    configs = {
        "window": ("window", "window"),
        "shifted": ("window", "shifted"),
    }
    for name, modes in configs.items():
        print(f"\n--- 训练 {name} ---")
        model = TokenClassifier(img_size=args.img_size, embed_dim=args.dim, num_heads=args.heads,
                                num_classes=num_classes, modes=modes, window_size=args.window)
        stats = model_stats(model, torch.randn(2, 3, args.img_size, args.img_size, device=args.device),
                            label=name, device=args.device)
        print(f"  参数量 {fmt(stats['params'])} | MACs/样本 {fmt(stats['macs'])} | "
              f"前向 {stats['ms']:.2f} ms")
        res = train_model(model, train_loader, val_loader, args.epochs, lr=args.lr,
                          device=args.device, label=name)
        results.append((name, stats, res))
    print("\n[2-B 结论表]")
    print_table(["配置", "参数量", "MACs/样本", "最佳 val acc", "最终 val acc"], [
        [name, fmt(s["params"]), fmt(s["macs"]), f"{r['best_acc']:.3f}", f"{r['final_acc']:.3f}"]
        for name, s, r in results
    ])


def main():
    ap = argparse.ArgumentParser(description="实验 2：Window vs Shifted Window")
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"],
                    help="训练数据集（本地已有则直接使用，不下载）")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
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
    connectivity_analysis(args)
    if not args.skip_train:
        train_part(args)
    print("\n实验 2 完成。预期结论：W-SW 的可达 token 数远大于 W-W（感受野跨窗口扩张），"
          "端到端精度上 shifted 配置显著更优——这就是 Swin 成对使用 W-MSA/SW-MSA 的根本原因。")


if __name__ == "__main__":
    main()
