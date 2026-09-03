# -*- coding: utf-8 -*-
"""
模块 02 / 学习顺序第 2 步：window_partition / window_reverse 单元测试
================================================================
覆盖点：
    1. 输出形状正确；
    2. 互逆性（多个随机尺寸）reverse(partition(x)) == x；
    3. 窗口编号顺序与手算一致（用位置 id 张量对拍）；
    4. 非整除尺寸应报错（view 无法整除 → RuntimeError）；
    5. dtype 保持。

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe test_window_partition.py
"""

import sys
import unittest

import torch

from window_partition import window_partition, window_reverse

# Windows 控制台默认 GBK 编码；强制 UTF-8 保证中文输出不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


class TestWindowPartition(unittest.TestCase):
    def test_output_shape(self):
        B, H, W, C, M = 2, 8, 8, 3, 2
        x = torch.randn(B, H, W, C)
        win = window_partition(x, M)
        nW = (H // M) * (W // M)
        self.assertEqual(tuple(win.shape), (B * nW, M, M, C))
        back = window_reverse(win, M, H, W)
        self.assertEqual(tuple(back.shape), (B, H, W, C))

    def test_inverse_multiple_sizes(self):
        torch.manual_seed(0)
        cases = [
            (1, 4, 4, 1, 2),
            (2, 8, 8, 3, 2),
            (3, 16, 8, 4, 4),
            (2, 12, 12, 6, 3),
            (1, 24, 16, 1, 4),
        ]
        for (B, H, W, C, M) in cases:
            x = torch.randn(B, H, W, C)
            back = window_reverse(window_partition(x, M), M, H, W)
            self.assertTrue(torch.equal(back, x),
                            f"互逆失败 @ (B={B},H={H},W={W},C={C},M={M})")

    def test_window_order_matches_hand_computation(self):
        """窗口编号顺序 = 行优先。用位置 id 张量与手算公式对拍。"""
        B, H, W, C, M = 2, 6, 8, 1, 2
        id_map = torch.arange(H * W).reshape(1, H, W, 1).expand(B, H, W, C).clone()
        win = window_partition(id_map, M)   # (B*nW, M, M, C)
        nW = win.shape[0] // B
        nh, nw = H // M, W // M
        for b in range(B):
            for k in range(nW):
                i, j = k // nw, k % nw          # 窗口行、列编号
                got = win[b * nW + k, :, :, 0].long().flatten().tolist()
                expect = []
                for m in range(M):
                    for n in range(M):
                        hh, ww = i * M + m, j * M + n
                        expect.append(hh * W + ww)   # 全局 id = h*W+w
                self.assertEqual(got, expect,
                                 f"窗口编号顺序不符 @ batch={b}, window={k}")

    def test_nondivisible_raises(self):
        """H 或 W 不能被 window_size 整除时，view 无法匹配元素总数 → 报错。"""
        x = torch.randn(1, 7, 8, 3)   # H=7 不能被 2 整除
        with self.assertRaises(RuntimeError):
            window_partition(x, 2)

    def test_dtype_preserved(self):
        for dtype in (torch.float32, torch.float64, torch.int64):
            x = torch.arange(2 * 8 * 8 * 3, dtype=dtype).reshape(2, 8, 8, 3)
            win = window_partition(x, 2)
            self.assertEqual(win.dtype, dtype)
            back = window_reverse(win, 2, 8, 8)
            self.assertEqual(back.dtype, dtype)
            self.assertTrue(torch.equal(back, x))


if __name__ == "__main__":
    unittest.main()
