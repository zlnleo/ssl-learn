# -*- coding: utf-8 -*-
"""
模块 01 / 学习顺序第 1 步：WindowAttention 的 Tensor Shape 跟踪
================================================================
用"插桩"的方式，把 WindowAttention.forward 里的每一步张量形状打印出来，
帮助你建立"形状从输入到输出如何一步步流转"的直觉。

关键路线（B_=2, N=4, C=8, heads=2, head_dim=4）：

    x   (B_, N, C)            = (2, 4, 8)
     |  self.qkv (Linear 8->24)
     v
    qkv (B_, N, 3C)           = (2, 4, 24)
     |  reshape -> (B_, N, 3, h, d) -> permute(2,0,3,1,4)
     v
    q,k,v (B_, h, N, d)       = (2, 2, 4, 4)
     |  (q*scale) @ k^T
     v
    attn (B_, h, N, N)        = (2, 2, 4, 4)   —— 每行是一个 token 对窗口内其它 token 的权重
     |  softmax(dim=-1)
     v
    attn (B_, h, N, N)        = (2, 2, 4, 4)   —— 每行和 = 1
     |  attn @ v
     v
    out  (B_, h, N, d) -> transpose(1,2) -> (B_, N, h, d) -> reshape
     v
    out  (B_, N, C)           = (2, 4, 8)
     |  self.proj (Linear 8->8)
     v
    y    (B_, N, C)           = (2, 4, 8)

本文件还包含一个断言：softmax 之后每一行的和 ≈ 1（这是注意力的概率解释）。
"""

import sys

import torch

from window_attention import WindowAttention

# Windows 控制台默认 GBK 编码，无法输出 ✔ 等符号；强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def step(label: str, t: torch.Tensor) -> None:
    """打印一步形状，方便逐行对照。"""
    print(f"{label:<28} shape = {str(tuple(t.shape)):<16} dtype = {t.dtype}")


def trace_forward() -> None:
    torch.manual_seed(0)
    B_, N, C, win, heads = 2, 4, 8, 2, 2
    x = torch.randn(B_, N, C)                       # (B_=2, N=4, C=8)
    attn = WindowAttention(dim=C, window_size=win, num_heads=heads)
    attn.eval()                                     # 关掉 dropout，保证断言确定性

    step("x (B_, N, C)", x)

    # 1) QKV 线性投影
    qkv = attn.qkv(x)                               # (2, 4, 24)
    step("qkv = qkv(x) (B_, N, 3C)", qkv)

    # 2) reshape + permute，拆出 q/k/v
    qkv5 = qkv.reshape(B_, N, 3, heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    step("qkv reshape+permute (3,B_,h,N,d)", qkv5)
    q, k, v = qkv5[0], qkv5[1], qkv5[2]
    step("q (B_, h, N, d)", q)
    step("k (B_, h, N, d)", k)
    step("v (B_, h, N, d)", v)

    # 3) 缩放点积
    attn_map = (q * attn.scale) @ k.transpose(-2, -1)
    step("attn = (q*scale) @ k^T (B_,h,N,N)", attn_map)

    # 4) softmax（沿最后一维，即"键"方向）
    attn_soft = attn_map.softmax(dim=-1)
    step("softmax(attn) (B_,h,N,N)", attn_soft)

    # 断言：softmax 之后每一行和为 1（概率解释）
    row_sum = attn_soft.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6), \
        f"softmax 行和 != 1，实际最大偏差 = {(row_sum - 1).abs().max().item():.2e}"
    print(f"    [断言通过] softmax 每行和 ≈ 1，最大偏差 = {(row_sum - 1).abs().max().item():.2e}")

    # 5) 注意力加权求和
    out = (attn_soft @ v).transpose(1, 2).reshape(B_, N, C)
    step("out = (attn@v) -> (B_, N, C)", out)

    # 6) 输出投影
    y = attn.proj_drop(attn.proj(out))
    step("y = proj(out) (B_, N, C)", y)

    # 7) 与 nn.Module.forward 的结果一致（验证我们手动插桩没有抄错）
    y_ref = attn(x)
    assert torch.allclose(y, y_ref, atol=1e-6), "手动插桩结果与 forward 不一致！"
    print("    [断言通过] 手动插桩逐行复现 = forward 输出（allclose）")

    print("\n全部形状跟踪与断言通过 [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("模块 01 · WindowAttention Shape Tracking")
    print("=" * 70)
    trace_forward()
