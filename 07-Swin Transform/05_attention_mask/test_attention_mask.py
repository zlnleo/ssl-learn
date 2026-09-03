# -*- coding: utf-8 -*-
"""
模块编号: 05
学习顺序: 05 注意力掩码 (本文件是单元测试)

test_attention_mask.py —— unittest 测试套件。

运行方式(项目根目录下):
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe "D:\\project\\self_supervised_learning\\07.Swin Transform\\05_attention_mask\\test_attention_mask.py"

覆盖点:
- mask 形状正确 (nW, M^2, M^2)
- 值域只含 {0, -100}
- 对角线全 0 (自己永远可见)
- 对称性 attn_mask[i,j] == attn_mask[j,i]
- shift_size=0 时全 0 (等价无 mask)
- H=W=2*window_size、shift=window//2 手工构造小例子逐元素比对
"""

import sys
import unittest

import torch

from attention_mask import build_attn_mask, window_partition

# Windows 控制台默认 GBK 编码, 无法输出 ✔ 等符号; 强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _expected_mask_from_grid(grid):
    """由 4x4 区域号网格构造期望的 (16,16) mask: 区域号不同 -> -100, 相同 -> 0。"""
    g = torch.tensor(grid).view(-1)          # (16,)
    diff = g.unsqueeze(0) - g.unsqueeze(1)   # (16,16)
    out = torch.zeros_like(diff, dtype=torch.float)
    out = out.masked_fill(diff != 0, -100.0)
    return out


class TestBuildAttnMask(unittest.TestCase):
    H = W = 8
    M = 4
    S = 2

    def test_mask_shape(self):
        mask = build_attn_mask(self.H, self.W, self.M, self.S)
        nW = (self.H // self.M) * (self.W // self.M)   # 4
        N = self.M * self.M                            # 16
        self.assertEqual(mask.shape, (nW, N, N))

    def test_value_range(self):
        mask = build_attn_mask(self.H, self.W, self.M, self.S)
        vals = set(mask.flatten().tolist())
        self.assertEqual(vals, {0.0, -100.0})

    def test_diagonal_all_zero(self):
        """自己永远可见: 对角线全 0。"""
        mask = build_attn_mask(self.H, self.W, self.M, self.S)
        diag = mask.diagonal(dim1=1, dim2=2)
        self.assertTrue(bool((diag == 0).all()))

    def test_symmetry(self):
        """mask[i,j] == mask[j,i]。"""
        mask = build_attn_mask(self.H, self.W, self.M, self.S)
        self.assertTrue(bool((mask == mask.transpose(1, 2)).all()))

    def test_shift_zero_all_zero(self):
        """shift_size=0 时全 0, 等价无 mask。"""
        mask = build_attn_mask(self.H, self.W, self.M, 0)
        self.assertTrue(bool((mask == 0).all()))

    def test_handcrafted_windows(self):
        """H=W=2*window_size、shift=window//2: 手工构造 9 宫格区域号, 逐元素比对。"""
        mask = build_attn_mask(self.H, self.W, self.M, self.S)   # (4, 16, 16)

        # 手工给出的 4 个窗口的 4x4 区域号网格 (详见 README/math 的 9 宫格推导)
        grids = {
            0: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            1: [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
            2: [[3, 3, 3, 3], [3, 3, 3, 3], [6, 6, 6, 6], [6, 6, 6, 6]],
            3: [[4, 4, 5, 5], [4, 4, 5, 5], [7, 7, 8, 8], [7, 7, 8, 8]],
        }
        for widx, grid in grids.items():
            expected = _expected_mask_from_grid(grid)
            self.assertTrue(torch.equal(mask[widx], expected),
                            f"窗口 {widx} 与手工构造的 mask 不一致")

    def test_window_partition_consistency(self):
        """window_partition 与模块 04 语义一致 (顺带回归)。"""
        x = torch.randn(1, 8, 8, 3)
        windows = window_partition(x, 4)
        self.assertEqual(windows.shape, (4, 4, 4, 3))

    def test_count_masked_total(self):
        """8x8 / window4 / shift2 时总屏蔽数应为 448。"""
        mask = build_attn_mask(self.H, self.W, self.M, self.S)
        self.assertEqual(int((mask == -100).sum().item()), 448)


if __name__ == "__main__":
    unittest.main(verbosity=2)
