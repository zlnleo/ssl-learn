"""消融实验公共工具：FLOPs(MACs) 统计、显存测量、速度基准、CIFAR-10 数据、训练循环。

所有实验脚本（exp1-exp4）从本文件导入。依赖：torch / torchvision，无 thop/timm 等第三方。
"""
import contextlib
import os
import random
import sys
import time

# Windows 控制台默认 GBK，强制 UTF-8 输出避免特殊字符（²、Ω、×）报错
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 项目根目录加入 sys.path，使 swin 包可被导入
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from swin.utils import Mlp, DropPath                  # noqa: E402
from swin.window import window_partition, window_reverse, build_attn_mask  # noqa: E402

# CIFAR 归一化常数（各数据集常用值）
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
_CIFAR_NORM = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
}

__all__ = ["seed_all", "get_device", "print_table", "fmt",
           "count_macs", "benchmark_speed", "peak_gpu_memory", "attn_map_bytes",
           "build_cifar", "build_cifar10", "resolve_data_dir", "train_model", "evaluate",
           "AttentionBlock", "TokenClassifier", "analyze_connectivity",
           "model_stats"]


# ---------------------------------------------------------------- 基础工具
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device=None) -> str:
    """解析设备：None/'auto' -> cuda 若可用，否则 cpu。"""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def fmt(x: float) -> str:
    """人类可读数字：1.23K / 4.56M / 7.89G。"""
    for unit in ("", "K", "M", "G", "T"):
        if abs(x) < 1000 or unit == "T":
            return f"{x:.2f}{unit}" if unit else f"{x:.2f}"
        x /= 1000
    return f"{x:.2f}T"


def print_table(headers, rows, title=None):
    """对齐打印 markdown 风格表格。rows: list[list[str]]"""
    if title:
        print(f"\n=== {title} ===")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


# ---------------------------------------------------------------- FLOPs（MACs）统计
def _prod(seq):
    p = 1
    for s in seq:
        p *= s
    return p


def _matmul_macs(a, b) -> int:
    """一次 matmul 的 MACs = 输出元素数 × 归约维长度。"""
    sa, sb = list(a.shape), list(b.shape)
    if len(sa) == 1:
        sa = [1, sa[0]]
    if len(sb) == 1:
        sb = [sb[0], 1]
    m, k = sa[-2], sa[-1]
    n = sb[-1]
    batch = max(_prod(sa[:-2]) if len(sa) > 2 else 1,
                _prod(sb[:-2]) if len(sb) > 2 else 1)
    return batch * m * n * k


@contextlib.contextmanager
def macs_counter(model: nn.Module = None):
    """统计 matmul（@ 运算符与 torch.matmul 直呼）与 Linear/Conv2d 的 MACs。

    注意：`a @ b` 走 Tensor.__matmul__（Python 层），直呼 torch.matmul 走 dispatcher，
    因此两者都要 patch；@ 的补丁内部直接调用"补丁前的原始 dispatcher"，避免二次计数。

    with macs_counter(model) as st:
        model(x)
    st["macs"]  # 总 MACs；1 MAC ≈ 2 FLOPs
    """
    state = {"macs": 0}
    orig_matmul = torch.matmul
    orig_bmm = torch.bmm
    orig_tensor_matmul = torch.Tensor.__matmul__

    def patched_matmul(a, b, *args, **kw):
        state["macs"] += _matmul_macs(a, b)
        return orig_matmul(a, b, *args, **kw)

    def patched_tensor_matmul(self, other):
        state["macs"] += _matmul_macs(self, other)
        return orig_matmul(self, other)

    def patched_bmm(a, b, *args, **kw):
        state["macs"] += _matmul_macs(a, b)
        return orig_bmm(a, b, *args, **kw)

    def linear_hook(m, inp, out):
        state["macs"] += _prod(inp[0].shape[:-1]) * m.weight.numel()

    def conv_hook(m, inp, out):
        x = inp[0]
        k = m.kernel_size[0] * m.kernel_size[1]
        state["macs"] += x.shape[0] * m.out_channels * m.in_channels * k * out.shape[2] * out.shape[3]

    torch.matmul = patched_matmul
    torch.bmm = patched_bmm
    try:
        torch.Tensor.__matmul__ = patched_tensor_matmul
    except (AttributeError, TypeError):  # 极少数构建下不可赋值：退化为只统计 Linear/Conv
        pass
    hooks = []
    if model is not None:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))
            elif isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv_hook))
    try:
        yield state
    finally:
        torch.matmul, torch.bmm = orig_matmul, orig_bmm
        try:
            torch.Tensor.__matmul__ = orig_tensor_matmul
        except (AttributeError, TypeError):
            pass
        for h in hooks:
            h.remove()


def count_macs(model: nn.Module, *args, **kwargs) -> float:
    """前向一次的总 MACs。"""
    with macs_counter(model) as st:
        model(*args, **kwargs)
    return st["macs"]


# ---------------------------------------------------------------- 速度 / 显存
def benchmark_speed(fn, iters: int = 30, warmup: int = 5, device: str = "cpu") -> float:
    """单次 fn() 的平均耗时（ms）。fn 需自包含（前向或前向+反向）。"""
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def peak_gpu_memory(fn, backward: bool = False) -> float:
    """运行 fn 并返回 CUDA 峰值显存（MB）。仅 CUDA 可用时使用。"""
    assert torch.cuda.is_available(), "显存测量需要 CUDA"
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    if backward:
        if torch.is_tensor(out):
            out = out.sum()
        out.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def attn_map_bytes(B: int, heads: int, N: int, n_windows: int = 1) -> float:
    """注意力矩阵显存（字节）：B × nW × h × N² × 4。全局注意力 n_windows=1, N=hw。"""
    return B * n_windows * heads * N * N * 4


# ---------------------------------------------------------------- 数据
DATA_ROOT_CANDIDATES = [r"D:\project\self_supervised_learning\data", "./data"]


def resolve_data_dir(data_dir=None) -> str:
    """优先使用本地已有数据集目录（避免重复下载），否则回落到 ./data。"""
    if data_dir:
        return data_dir
    for p in DATA_ROOT_CANDIDATES:
        if os.path.isdir(p):
            return p
    return "./data"


def _make_cifar(dataset: str, data_dir: str, train: bool, transform):
    """已有数据时绝不触发下载（download=False），数据缺失时才允许下载。"""
    ds_cls = {"cifar10": torchvision.datasets.CIFAR10,
              "cifar100": torchvision.datasets.CIFAR100}[dataset]
    try:
        return ds_cls(data_dir, train=train, download=False, transform=transform)
    except RuntimeError:  # 本地数据不存在：允许联网下载
        return ds_cls(data_dir, train=train, download=True, transform=transform)


def build_cifar(dataset: str = "cifar10", img_size: int = 64, batch_size: int = 128,
                num_workers: int = 2, data_dir: str = None):
    """CIFAR-10/100（resize 到 img_size）。返回 (train_loader, val_loader, num_classes)。"""
    assert dataset in ("cifar10", "cifar100")
    data_dir = resolve_data_dir(data_dir)
    num_classes = 10 if dataset == "cifar10" else 100
    mean, std = _CIFAR_NORM[dataset]
    norm = T.Normalize(mean, std)
    train_tf = T.Compose([T.Resize(img_size), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    val_tf = T.Compose([T.Resize(img_size), T.ToTensor(), norm])
    train_ds = _make_cifar(dataset, data_dir, True, train_tf)
    val_ds = _make_cifar(dataset, data_dir, False, val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes


def build_cifar10(img_size: int, batch_size: int, num_workers: int = 2,
                  data_dir: str = None):
    """兼容别名：CIFAR-10 版。"""
    return build_cifar("cifar10", img_size, batch_size, num_workers, data_dir)


# ---------------------------------------------------------------- 训练循环
def evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train_model(model, train_loader, val_loader, epochs: int, lr: float = 3e-4,
                weight_decay: float = 0.05, device: str = "cuda", label: str = "model",
                warmup_epochs: int = 0):
    """AdamW + 余弦退火。返回 dict(best_acc, final_acc, history)。"""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1))
    if warmup_epochs > 0:  # 线性 warmup
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, sched],
                                                      milestones=[warmup_epochs])
    criterion = nn.CrossEntropyLoss()
    history = []
    best = 0.0
    for ep in range(epochs):
        model.train()
        correct = total = 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        sched.step()
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        history.append((train_acc, val_acc))
        best = max(best, val_acc)
        print(f"  [{label}] epoch {ep + 1}/{epochs}  train_acc={train_acc:.3f} "
              f"val_acc={val_acc:.3f}  ({dt:.1f}s)")
    return {"best_acc": best, "final_acc": history[-1][1], "history": history}


# ---------------------------------------------------------------- 实验 1/2 用模型
class AttentionBlock(nn.Module):
    """实验用注意力块：pre-norm + 注意力 + 残差 + pre-norm + MLP + 残差。

    mode: 'global'（全局 MSA）| 'window'（W-MSA）| 'shifted'（SW-MSA，roll + mask）。
    """

    def __init__(self, dim, num_heads, mode="window", window_size=4, mlp_ratio=4., drop=0.0):
        super().__init__()
        assert mode in ("global", "window", "shifted")
        self.mode = mode
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self._mask_cache = None
        self._mask_key = None

    def _attn(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)                       # (B_, h, N, N)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C))

    def _get_mask(self, H, W, device):
        if self.mode != "shifted":
            return None
        key = (H, W, str(device))
        if key != self._mask_key:
            self._mask_cache = build_attn_mask(H, W, self.window_size, self.shift_size, device)
            self._mask_key = key
        return self._mask_cache

    def forward(self, x, H, W):
        B, L, C = x.shape
        y = self.norm1(x)
        if self.mode == "global":
            # 全局注意力：直接在 (B, L, C) 上计算，无窗口/pad/roll
            y = self._attn(y.view(B, L, C))
        else:
            y = y.view(B, H, W, C)
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            y = F.pad(y, (0, 0, 0, pad_r, 0, pad_b))
            Hp, Wp = H + pad_b, W + pad_r
            if self.mode == "shifted":
                y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            y = window_partition(y, self.window_size).view(-1, self.window_size ** 2, C)
            y = self._attn(y, mask=self._get_mask(Hp, Wp, y.device))
            y = y.view(-1, self.window_size, self.window_size, C)
            y = window_reverse(y, self.window_size, Hp, Wp)
            if self.mode == "shifted":
                y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = y[:, :H, :W, :].contiguous().view(B, L, C)
        x = x + y
        return x + self.mlp(self.norm2(x))


class TokenClassifier(nn.Module):
    """实验 1/2 用小分类器：PatchEmbed -> n 个 AttentionBlock -> 平均池化 -> 分类头。

    modes: 与 depths 等长的 mode 列表（如 ("window","shifted")）。
    pos_embed: 可学习绝对位置编码（仅全局注意力变体开启，与 ViT 惯例一致）。
    """

    def __init__(self, img_size=64, patch_size=4, embed_dim=96, num_heads=3, num_classes=10,
                 modes=("window", "window"), window_size=4, mlp_ratio=4., drop=0.0,
                 pos_embed=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, mode=m, window_size=window_size,
                           mlp_ratio=mlp_ratio, drop=drop)
            for m in modes
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim)) \
            if pos_embed else None

    def forward(self, x):
        B = x.shape[0]
        H = W = self.img_size // self.patch_size
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, L, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, H, W)
        return self.head(x.mean(1))


# ---------------------------------------------------------------- 实验 2 连接性分析
def analyze_connectivity(H: int, W: int, window_size: int, shift_second: int):
    """两层注意力下每个 token 的"可达 token 数"（结构分析，与权重无关）。

    第一层恒为 shift=0（W-MSA）；第二层 shift = shift_second（0 表示 W-W，window//2 表示 W-SW）。
    返回 (reach_list, cover_map)：reach_list[t] 为 token t 的可达数；cover_map (H,W) 为中心 token 的覆盖 ASCII 图。
    """
    ids = torch.arange(H * W).view(1, H, W, 1).long()
    win1 = window_partition(ids, window_size).view(-1, window_size * window_size)  # (nW, M^2)
    nW = win1.shape[0]
    rolled = torch.roll(ids, shifts=(-shift_second, -shift_second), dims=(1, 2))
    win2 = window_partition(rolled, window_size).view(-1, window_size * window_size)
    tok_win1 = torch.zeros(H * W, dtype=torch.long)
    tok_win2 = torch.zeros(H * W, dtype=torch.long)
    for wi in range(nW):
        tok_win1[win1[wi]] = wi
        tok_win2[win2[wi]] = wi
    reach = []
    for t in range(H * W):
        first = set(win1[tok_win1[t]].tolist())
        second = set()
        for j in first:
            second |= set(win2[tok_win2[j]].tolist())
        reach.append(len(second))
    # 中心 token 覆盖图
    center = (H // 2) * W + (W // 2)
    first = set(win1[tok_win1[center]].tolist())
    second = set()
    for j in first:
        second |= set(win2[tok_win2[j]].tolist())
    cover = [["." for _ in range(W)] for _ in range(H)]
    for t in second:
        cover[t // W][t % W] = "#"
    cover[center // W][center % W] = "O"
    return reach, "\n".join("".join(row) for row in cover)


# ---------------------------------------------------------------- 模型统计
def model_stats(model, *args, label: str = "model", device: str = "cpu",
                backward: bool = False) -> dict:
    """一次性统计：参数量 / MACs / 前向速度 / 显存（CUDA 时）。args 为模型输入。"""
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        macs = count_macs(model, *args)
        speed = benchmark_speed(lambda: model(*args), iters=20, warmup=5, device=device)
    mem = None
    if device == "cuda":
        def fn():
            out = model(*args)
            if backward:
                out.sum().backward()
            return out
        mem = peak_gpu_memory(fn, backward=False) if not backward else peak_gpu_memory(
            lambda: model(*args).sum().backward(), backward=False)
    return {"label": label, "params": n_params, "macs": macs,
            "gflops": macs * 2 / 1e9, "ms": speed, "mem_mb": mem}
