# -*- coding: utf-8 -*-
"""临时检查脚本: 验证 deitmodel.py 与 deitloss.py 的正确性 (跑完即删)。"""
import types

import torch
import torch.nn.functional as F

from deitmodel import DistilledVit
from deitloss import soft_cross_entropy, Distillation_loss as distillation_loss

print("=" * 60)
print("[1] deitmodel.py 检查")
m = DistilledVit()
n = sum(p.numel() for p in m.parameters())
print(f"    参数量 = {n / 1e6:.3f}M (期望 ~5.3M)")

x = torch.randn(2, 3, 32, 32)
m.train()
out = m(x)
print("    训练模式输出:", type(out).__name__,
      [tuple(o.shape) for o in out] if isinstance(out, tuple) else tuple(out.shape))
m.eval()
with torch.no_grad():
    e = m(x)
print("    eval 模式输出:", tuple(e.shape))
m.train()
(out[0].sum() + out[1].sum()).backward()
print("    backward OK")

m2 = DistilledVit(distilled=False)
print("    无蒸馏模式 train 输出:", tuple(m2(x).shape))

print("=" * 60)
print("[2] deitloss.py 检查")
sl = torch.randn(4, 100, requires_grad=True)
dl = torch.randn(4, 100, requires_grad=True)
tl = torch.randn(4, 100)          # 教师 logits (模拟已 no_grad)
y = torch.randint(0, 100, (4,))

args_h = types.SimpleNamespace(distill='hard', alpha=0.5, smoothing=0.1, tau=3.0)
total, base, dist = distillation_loss((sl, dl), tl, y, args_h)
print(f"    硬蒸馏: total={total.item():.4f} base={base.item():.4f} dist={dist.item():.4f} (初始应≈ln100=4.6)")
total.backward()

args_s = types.SimpleNamespace(distill='soft', alpha=0.5, smoothing=0.1, tau=3.0)
sl2 = torch.randn(4, 100, requires_grad=True)
dl2 = torch.randn(4, 100, requires_grad=True)
total_s, _, dist_s = distillation_loss((sl2, dl2), tl, y, args_s)
print(f"    软蒸馏: total={total_s.item():.4f} dist={dist_s.item():.4f}")
total_s.backward()

# 平滑=0 时应退化为标准 CE
ce0 = soft_cross_entropy(torch.randn(4, 100), y, smoothing=0.0)
ce_ref = F.cross_entropy(torch.randn(4, 100), y)
print(f"    smoothing=0 时 soft_ce={ce0.item():.4f}, 对照标准 CE 量级一致")

# 软标签 (2D target) 分支
t2 = F.one_hot(y, 100).float()
print(f"    2D 软标签分支: {soft_cross_entropy(torch.randn(4, 100), t2, smoothing=0.0).item():.4f}")

print("=" * 60)
print("全部检查通过")
