# -*- coding: utf-8 -*-
"""
模块 01 / 学习顺序第 1 步：WindowAttention 单元测试
================================================================
覆盖点：
    1. 输出形状正确；
    2. 不同 num_heads 下等价维度（head_dim = dim // num_heads）；
    3. softmax 行和为 1；
    4. window_size=1 退化为逐 token 变换（token 之间不交互，等价于逐 token 的
       qkv->proj 变换，与 head 无关）；
    5. 梯度可回传；
    6. qkv_bias=False 可运行；
    7. 数值验证"全局 = 一个覆盖全图的大窗口"：同一权重下
       WindowAttention(window_size=H) 与普通全局 MSA 结果一致。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe test_window_attention.py
"""

import sys
import unittest

import torch

from window_attention import WindowAttention

# Windows 控制台默认 GBK 编码；强制 UTF-8 保证中文输出不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_global_msa(dim: int, num_heads: int) -> torch.nn.Module:
    """手写一个"教科书式"全局 MSA（(B,N,C) 输入），用于与 WindowAttention 对拍。"""

    class GlobalMSA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(dim, dim * 3)
            self.proj = torch.nn.Linear(dim, dim)
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.num_heads = num_heads

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    return GlobalMSA()


class TestWindowAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.win, self.heads = 32, 4, 4
        self.B_, self.N = 6, self.win * self.win  # 6 个窗口，每窗口 16 token

    def test_output_shape(self):
        x = torch.randn(self.B_, self.N, self.dim)
        net = WindowAttention(self.dim, self.win, self.heads).eval()
        y = net(x)
        self.assertEqual(tuple(y.shape), (self.B_, self.N, self.dim))

    def test_head_dim_equivalence(self):
        """不同 num_heads 下，只要 dim 相同，head_dim = dim // num_heads 一致成立。"""
        for h in (1, 2, 4, 8):
            net = WindowAttention(self.dim, self.win, h)
            self.assertEqual(net.head_dim, self.dim // h)
            self.assertEqual(net.head_dim * h, self.dim)

    def test_softmax_rows_sum_to_one(self):
        net = WindowAttention(self.dim, self.win, self.heads).eval()
        x = torch.randn(self.B_, self.N, self.dim)
        B_, N = x.shape[0], x.shape[1]
        # 手工重算 forward 里的 softmax 输出并检查行和
        with torch.no_grad():
            qkv = net.qkv(x).reshape(B_, N, 3, self.heads, net.head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = ((q * net.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        row_sum = attn.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6))

    def test_window_size_one_is_per_token(self):
        """window_size=1 时 N=1，注意力退化为每个 token 自身的线性变换。"""
        net = WindowAttention(self.dim, window_size=1, num_heads=2).eval()
        x = torch.randn(5, 1, self.dim)   # 5 个"窗口"，各 1 个 token
        y = net(x)
        # 此时 attn 是 1x1 的 softmax=1，等价于 qkv 后取 q 的 V 位置... 直接用等价式对拍：
        # out = proj( attn(=1) @ v ) = proj(v)；v 来自 qkv 投影的第 3 段，再按 head 重组。
        with torch.no_grad():
            qkv = net.qkv(x).reshape(5, 1, 3, 2, net.head_dim).permute(2, 0, 3, 1, 4)
            v = qkv[2].transpose(1, 2).reshape(5, 1, self.dim)   # (5,1,dim)
            expected = net.proj(v)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_grad_flows(self):
        net = WindowAttention(self.dim, self.win, self.heads)
        x = torch.randn(self.B_, self.N, self.dim, requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(net.qkv.weight.grad)
        self.assertTrue(torch.isfinite(net.qkv.weight.grad).all())

    def test_qkv_bias_false(self):
        net = WindowAttention(self.dim, self.win, self.heads, qkv_bias=False).eval()
        x = torch.randn(self.B_, self.N, self.dim)
        y = net(x)
        self.assertEqual(tuple(y.shape), (self.B_, self.N, self.dim))
        self.assertIsNone(net.qkv.bias)

    def test_global_equals_single_big_window(self):
        """同一权重下，WindowAttention(window_size=H) 应等于手写全局 MSA。"""
        H = W = 6
        C, heads = 24, 3
        x = torch.randn(2, H * W, C)
        win = WindowAttention(C, window_size=H, num_heads=heads).eval()
        ref = make_global_msa(C, heads).eval()
        # 复制权重，确保两边完全一致
        ref.qkv.weight.data.copy_(win.qkv.weight.data)
        ref.qkv.bias.data.copy_(win.qkv.bias.data)
        ref.proj.weight.data.copy_(win.proj.weight.data)
        ref.proj.bias.data.copy_(win.proj.bias.data)
        with torch.no_grad():
            y_win = win(x)
            y_ref = ref(x)
        self.assertTrue(torch.allclose(y_win, y_ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
