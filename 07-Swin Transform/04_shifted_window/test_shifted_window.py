# -*- coding: utf-8 -*-
"""
模块编号: 04
学习顺序: 04 移位窗口 (本文件是单元测试)

test_shifted_window.py —— unittest 测试套件。

运行方式(项目根目录下):
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe "D:\\project\\self_supervised_learning\\07.Swin Transform\\04_shifted_window\\test_shifted_window.py"

覆盖点:
- window_partition 形状正确 (多 batch / 多窗口)
- window_reverse 是 window_partition 的逆 (round-trip)
- window_partition 的窗口排序 (行优先, 与手工切片一致)
- cyclic_shift 的 wrap-around 语义 (逐元素比对)
- cyclic_unshift 是 cyclic_shift 的逆
- shift_size=0 时恒等
- roll 不改变内容多重集 (只重排不丢失)
"""

import sys
import unittest

import torch

from shifted_window import (
    window_partition,
    window_reverse,
    cyclic_shift,
    cyclic_unshift,
)

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


class TestWindowPartition(unittest.TestCase):
    def test_partition_shape(self):
        for B in (1, 2):
            x = torch.randn(B, 8, 8, 3)
            windows = window_partition(x, 4)
            nW = (8 // 4) * (8 // 4)           # 4
            self.assertEqual(windows.shape, (B * nW, 4, 4, 3))

    def test_partition_shape_non_square(self):
        x = torch.randn(1, 8, 12, 3)           # H=8, W=12, 非正方形
        windows = window_partition(x, 4)
        nW = (8 // 4) * (12 // 4)              # 2 * 3 = 6
        self.assertEqual(windows.shape, (6, 4, 4, 3))

    def test_partition_ordering_matches_manual_slice(self):
        """窗口按行优先编号: 窗口 (r,c) 对应原图 [rM:(r+1)M, cM:(c+1)M]。"""
        H = W = 8
        M = 4
        ids = torch.arange(H * W).view(1, H, W, 1)
        windows = window_partition(ids, M)      # (4, 4, 4, 1)
        for r in range(2):
            for c in range(2):
                widx = r * 2 + c
                manual = ids[0, r*M:(r+1)*M, c*M:(c+1)*M, 0]   # (4,4)
                self.assertTrue(torch.equal(windows[widx, :, :, 0], manual),
                                f"窗口({r},{c}) 排序错误")

    def test_partition_reverse_roundtrip(self):
        for B in (1, 2):
            x = torch.randn(B, 8, 8, 4)
            windows = window_partition(x, 4)
            out = window_reverse(windows, 4, 8, 8)
            self.assertTrue(torch.equal(out, x), "partition/reverse 往返不一致")

    def test_reverse_infers_batch(self):
        """window_reverse 通过窗口数反推 batch。"""
        x = torch.randn(3, 8, 8, 2)
        windows = window_partition(x, 4)
        out = window_reverse(windows, 4, 8, 8)
        self.assertEqual(out.shape, (3, 8, 8, 2))


class TestCyclicShift(unittest.TestCase):
    def test_shift_semantics_wrap_around(self):
        """out[i,j] = x[(i+shift)%H, (j+shift)%W] 逐元素验证。"""
        H = W = 4
        x = torch.arange(H * W).reshape(1, H, W, 1)
        shift = 2
        out = cyclic_shift(x, shift)
        for i in range(H):
            for j in range(W):
                expected = x[0, (i + shift) % H, (j + shift) % W, 0]
                self.assertEqual(out[0, i, j, 0].item(), expected.item())

    def test_unshift_inverse(self):
        for shift in (1, 2, 3):
            x = torch.randn(1, 8, 8, 4)
            self.assertTrue(torch.equal(cyclic_unshift(cyclic_shift(x, shift), shift), x))

    def test_shift_zero_identity(self):
        x = torch.randn(1, 8, 8, 4)
        self.assertTrue(torch.equal(cyclic_shift(x, 0), x))
        self.assertTrue(torch.equal(cyclic_unshift(x, 0), x))

    def test_shift_preserves_multiset(self):
        """roll 只重排, 不丢内容: 排序后的多重集一致。"""
        x = torch.randn(1, 8, 8, 4)
        shifted = cyclic_shift(x, 3)
        self.assertTrue(torch.equal(torch.sort(x.flatten())[0],
                                    torch.sort(shifted.flatten())[0]))

    def test_shift_keeps_shape(self):
        x = torch.randn(2, 8, 8, 4)
        self.assertEqual(cyclic_shift(x, 2).shape, x.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
