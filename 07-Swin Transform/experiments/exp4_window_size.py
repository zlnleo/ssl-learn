"""实验 4：不同 window_size 的影响（4 / 7 / 14）。

window_size 直接决定注意力部分计算量：Ω_attn = 2·M²·hw·C。
- M=4 ：窗口小、窗口多，注意力最便宜，但跨窗口信息流动慢
- M=7 ：Swin-Tiny 标准配置
- M=14：在 56x56 特征图上 4x4=16 个窗口（接近全局注意力），最贵但窗口最大

观测：参数量（相对偏置表大小随 M 变）、MACs、速度、（可选）CIFAR-10 精度。
注：window=14 时最后一个 stage（7x7 特征图）会触发 pad-到-14 的分支（单窗口），
这是工程实现对任意尺寸的健壮性设计，见 swin/block.py 的说明。

用法（项目根目录，ssl_cv 环境）：
  python experiments/exp4_window_size.py
  python experiments/exp4_window_size.py --skip-train
"""
import argparse

import torch

import common
from common import (build_cifar, get_device, model_stats, print_table, seed_all,
                    train_model, fmt)
from swin import swin_tiny


def static_analysis(args):
    print("\n" + "=" * 78)
    print("实验 4-A：静态测量 —— window_size ∈ {4, 7, 14} 的 Swin-Tiny @ 224")
    print("=" * 78)
    n_cls = 100 if args.dataset == "cifar100" else 10
    x = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    rows = []
    for w in args.window_sizes:
        model = swin_tiny(num_classes=n_cls, window_size=w)
        stats = model_stats(model, x, label=f"window={w}", device=args.device)
        # 相对位置偏置表参数量：(2M-1)^2 * h 之和
        bias_params = sum(
            (2 * blk.attn.window_size - 1) ** 2 * blk.attn.num_heads
            for layer in model.layers for blk in layer.blocks)
        rows.append([f"window_size={w}", fmt(stats["params"]), f"其中偏置表 {fmt(bias_params)}",
                     f"{stats['gflops']:.2f} G", f"{stats['ms']:.2f} ms"])
    print_table(["配置", "参数量", "相对位置偏置表", "≈FLOPs/样本(224)", "前向耗时"],
                rows, title=None)
    print("  解读（与实测一致）：")
    print("  - 主效应：注意力点积 MACs ∝ M²，窗口越大注意力越贵（4:7:14 -> 16:49:196）；")
    print("  - 副作用：window_size 与各 stage 分辨率的整除性。224 配置下 stage 分辨率为")
    print("    56/28/14/7：window=4 在 14x14、7x7 上 pad 到 16/8（投影 +30%），")
    print("    window=14 在 7x7 上 pad 到 14（投影 +300%），pad 开销部分抵消注意力节省；")
    print("  - 实测总 FLOPs：window 7 (最省) < window 4 < window 14——56/28/14/7 恰能被 7")
    print("    整除，pad 全为零。这正是 Swin 选 window_size=7 的工程原因之一。")


def train_part(args):
    print("\n" + "=" * 78)
    print(f"实验 4-B：端到端精度 —— 不同 window_size（{args.dataset} @ {args.img_size}）")
    print("=" * 78)
    seed_all(args.seed)
    train_loader, val_loader, num_classes = build_cifar(
        args.dataset, args.img_size, args.batch, args.num_workers, args.data_dir)
    results = []
    for w in args.window_sizes:
        print(f"\n--- 训练 window_size={w} ---")
        model = swin_tiny(num_classes=num_classes, window_size=w)
        stats = model_stats(model, torch.randn(1, 3, args.img_size, args.img_size, device=args.device),
                            label=f"window={w}", device=args.device)
        print(f"  参数量 {fmt(stats['params'])} | ≈FLOPs {stats['gflops']:.2f} G | "
              f"前向 {stats['ms']:.2f} ms")
        res = train_model(model, train_loader, val_loader, args.epochs, lr=args.lr,
                          device=args.device, label=f"window={w}")
        results.append((w, stats, res))
    print("\n[4-B 结论表]")
    print_table(["window_size", "≈FLOPs/样本", "前向耗时", "最佳 val acc", "最终 val acc"], [
        [str(w), f"{s['gflops']:.2f} G", f"{s['ms']:.2f} ms",
         f"{r['best_acc']:.3f}", f"{r['final_acc']:.3f}"]
        for w, s, r in results
    ])


def main():
    ap = argparse.ArgumentParser(description="实验 4：window_size 消融")
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"],
                    help="训练数据集（本地已有则直接使用，不下载）")
    ap.add_argument("--window-sizes", type=int, nargs="+", default=[4, 7, 14])
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
    print("\n实验 4 完成。预期结论：窗口越大注意力点积越贵（∝ M²），但过小的窗口会在"
          "非整除分辨率上触发 pad（投影膨胀）。window=7 在 224 输入下与全部 stage 分辨率整除、"
          "注意力成本适中——精度与成本的工程平衡点，这也是 Swin 系列默认 window_size=7 的原因。")


if __name__ == "__main__":
    main()
