# -*- coding: utf-8 -*-
"""
模块 01 / 学习顺序第 1 步：实验 "Global vs Window Attention"
================================================================
本实验用最朴素的 CPU 计时，直观对比"全局多头自注意力（MSA）"与
"窗口多头自注意力（W-MSA）"的前向代价，并与公式推导的 MACs 对照。

实验设定（与 Swin-T 第一阶段一致）：
    特征图 H = W = 56  ->  hw = 56*56 = 3136 个 token
    通道 C = 96,  头数 num_heads = 3  ->  head_dim = 32
    batch B = 2,  窗口 M = 7  ->  每窗口 N = M^2 = 49,  nW = (56/7)^2 = 64

关键结论（请和运行输出对照）：
    全局 MSA  : ~2.0 G MACs，注意力矩阵 ~236 MB
    窗口 W-MSA : ~145 M MACs，注意力矩阵 ~3.7 MB
    总比值约 13.8x，其中"注意力部分"比值 = hw / M^2 = 64x

用法：
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe experiment.py            # CPU（默认）
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe experiment.py --device cuda   # 可选 GPU
"""

import argparse
import sys
import time

import torch

from window_attention import WindowAttention, msa_macs

# Windows 控制台默认 GBK 编码；强制 UTF-8 保证中文输出不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def bench(fn, warmup: int, repeats: int):
    """热身 warmup 次 + 计时 repeats 次，返回平均耗时（秒）。"""
    for _ in range(warmup):
        fn()
    # 同步点：GPU 上需要等 kernel 真正跑完
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def _heat_char(v: float) -> str:
    """把 [0,1] 的注意力权重映射成由浅到深的 ASCII 字符。"""
    ramp = " .:-=+*#%@"
    idx = min(int(v * (len(ramp) - 1) + 0.5), len(ramp) - 1)
    return ramp[idx]


def ascii_attention_map(attn_head: torch.Tensor) -> str:
    """把一个 head 的注意力矩阵（N×N，每行已归一化）渲染成字符热力图。

    attn_head : (N, N)，每行和为 1。行 = query token，列 = key token。
    """
    lines = []
    N = attn_head.shape[0]
    lines.append(f"  attention map (N={N}, 行=query, 列=key, 每行和=1)")
    lines.append("  " + " ".join(f"{j:2d}" for j in range(N)))
    for i in range(N):
        row = "".join(_heat_char(float(v)) for v in attn_head[i])
        lines.append(f"  {i:2d} {row}")
    return "\n".join(lines)


def run_benchmark(device: str) -> None:
    H = W = 56
    M = 7
    C = 96
    heads = 3
    B = 2
    hw = H * W
    win_tokens = M * M
    nW = (H // M) * (W // M)

    print("=" * 78)
    print("实验 1：Global MSA vs Window MSA —— 前向耗时与 MACs 对照")
    print("=" * 78)
    print(f"设定: H=W={H}, hw={hw}, C={C}, heads={heads}, head_dim={C//heads}, "
          f"B={B}, M={M}, 每窗口 N={win_tokens}, nW={nW}, device={device}")
    print()

    # 两个算子共用同一组权重，保证对比"只看结构、不看初始化"。
    global_msa = WindowAttention(dim=C, window_size=H, num_heads=heads).to(device).eval()
    window_msa = WindowAttention(dim=C, window_size=M, num_heads=heads).to(device).eval()
    window_msa.load_state_dict(global_msa.state_dict())  # 权重完全一致

    # 全局：整张图是唯一窗口，B_ = B, N = hw
    x_global = torch.randn(B, hw, C, device=device)
    # 窗口：所有窗口按 batch 拼接，B_ = B*nW, N = M^2
    x_window = torch.randn(B * nW, win_tokens, C, device=device)

    warmup, repeats = 2, 3
    t_global = bench(lambda: global_msa(x_global), warmup, repeats)
    t_window = bench(lambda: window_msa(x_window), warmup, repeats)

    macs_global = msa_macs(hw, C, hw)            # win_tokens = hw（全局）
    macs_window = msa_macs(hw, C, win_tokens)    # win_tokens = M^2（窗口）

    print("--- MACs（公式）对照 ---")
    print(f"  全局 MSA   MACs = {macs_global/1e6:10.1f} M   ({macs_global/1e9:.3f} G)")
    print(f"  窗口 W-MSA MACs = {macs_window/1e6:10.1f} M   ({macs_window/1e9:.3f} G)")
    print(f"  总比值          = {macs_global/macs_window:8.2f} x")
    print(f"  其中注意力部分比值 = hw / M^2 = {hw}/{win_tokens} = {hw/win_tokens:.1f} x")
    print()
    print("--- 前向耗时（实测，warmup=%d, repeats=%d）---" % (warmup, repeats))
    print(f"  全局 MSA   : {t_global*1e3:9.1f} ms / forward")
    print(f"  窗口 W-MSA : {t_window*1e3:9.1f} ms / forward")
    if t_window > 0:
        print(f"  实测加速比 : {t_global/t_window:8.2f} x")
    print()
    print("--- 注意力矩阵显存（float32，B·h·N²·4 字节）---")
    mem_global = B * heads * hw * hw * 4
    mem_window = (B * nW) * heads * win_tokens * win_tokens * 4
    print(f"  全局 MSA   : {mem_global/2**20:8.1f} MB")
    print(f"  窗口 W-MSA : {mem_window/2**20:8.1f} MB   （nW 个窗口各自 49x49）")
    print()
    print("  [说明] 实测耗时是『前向墙钟时间』，不严格等于 MACs 比值，因为 CPU 上的线性层、")
    print("         softmax、内存带宽等开销并非都与 MACs 成正比；但数量级的差距是真实的。")

    # ---------- hw 变化时的 MACs 对比表 ----------
    print()
    print("--- hw 变化时的 MACs 对比表（C=96, M=7, 单位 M MACs）---")
    print(f"  {'H=W':>5} {'hw':>7} {'全局 MSA':>11} {'窗口 MSA':>11} {'总比值':>8}")
    for size in (14, 28, 56, 112, 224):
        hh = size * size
        g = msa_macs(hh, C, hh)
        w = msa_macs(hh, C, win_tokens)
        print(f"  {size:5d} {hh:7d} {g/1e6:11.1f} {w/1e6:11.1f} {g/w:8.2f}x")
    print("  [读法] 窗口 MSA 随 hw 线性增长，全局 MSA 随 hw 二次增长 → 比值随 hw 线性变大。")

    # ---------- 小尺寸 attention map 的 ASCII 可视化 ----------
    print()
    print("=" * 78)
    print("小尺寸 attention map 可视化（8×8 图，window_size=4 → 每窗口 16 token）")
    print("=" * 78)
    small = WindowAttention(dim=8, window_size=4, num_heads=2).eval()
    x_small = torch.randn(1, 4 * 4, 8)          # 一个 4×4 窗口，16 个 token
    with torch.no_grad():
        B_, N_, C_ = x_small.shape
        qkv = small.qkv(x_small).reshape(B_, N_, 3, 2, 4).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        a = (q * small.scale) @ k.transpose(-2, -1)
        a = a.softmax(dim=-1)
    # 打印第 0 个 batch、第 0 个 head 的注意力矩阵
    print(ascii_attention_map(a[0, 0]))
    print()
    print("  [读法] 若输入是 4×4 的 8×8 大图切出的『第一个窗口』，每一行是该 token 对同窗口")
    print("         16 个 token 的注意力权重，行和 = 1。颜色越深（字符越靠右）权重越大。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模块01 实验：全局 vs 窗口注意力")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="运行设备，默认 cpu（快速可复现）")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 cpu")
        args.device = "cpu"
    run_benchmark(args.device)
