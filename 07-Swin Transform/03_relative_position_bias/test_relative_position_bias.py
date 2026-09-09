# -*- coding: utf-8 -*-
"""
模块编号: 03
学习顺序: 03 相对位置偏置 (本文件是单元测试)

test_relative_position_bias.py —— unittest 测试套件。

运行方式(项目根目录下):
    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe "D:\\project\\self_supervised_learning\\07.Swin Transform\\03_relative_position_bias\\test_relative_position_bias.py"

覆盖点:
- 索引表形状 / dtype / 取值范围
- 偏置表形状 / 参数总量 / requires_grad / 不进入 state_dict
- forward 输出形状
- (dh, dw) <-> 行号 的双射
- 转置对称性: index[a,b] 与 index[b,a] 对应相反位移
- 对角线(自己对自己)统一映射到 (0,0) 位移
"""

import sys
import unittest

import torch

# Windows 控制台默认 GBK 编码，无法输出 ✔ 等符号；强制 UTF-8 保证中文与符号不乱码、不报错。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from relative_position_bias import (
    build_relative_position_index,
    decode_relative_position_row,
    RelativePositionBias,
)


class TestBuildRelativePositionIndex(unittest.TestCase):
    def test_shape(self):
        for M in (2, 3, 4, 7):
            idx = build_relative_position_index(M)
            self.assertEqual(idx.shape, (M * M, M * M), f"M={M} 形状错误")

    def test_dtype_long(self):
        idx = build_relative_position_index(3)
        self.assertEqual(idx.dtype, torch.long, "索引表必须是 long 类型")

    def test_value_range(self):
        for M in (2, 3, 4, 7):
            span = 2 * M - 1
            idx = build_relative_position_index(M)
            self.assertGreaterEqual(idx.min().item(), 0)
            self.assertLess(idx.max().item(), span * span)

    def test_bijection_full_coverage(self):
        """每种相对位移(共 (2M-1)^2 种)在索引表中都至少出现一次。"""
        for M in (2, 3, 4):
            span = 2 * M - 1
            idx = build_relative_position_index(M)
            uniq = set(idx.flatten().tolist())
            self.assertEqual(uniq, set(range(span * span)), f"M={M} 未覆盖全部位移")

    def test_diagonal_is_center(self):
        """对角线元素(自己对自己)统一映射到位移 (0,0) 对应的行号。"""
        for M in (2, 3, 4, 7):
            span = 2 * M - 1
            idx = build_relative_position_index(M)
            center = (M - 1) * span + (M - 1)
            self.assertTrue(bool((idx.diagonal() == center).all()))

    def test_transpose_symmetry_opposite_displacement(self):
        """index[a, b] 与 index[b, a] 对应相反位移 (dh, dw) <-> (-dh, -dw)。"""
        for M in (2, 3, 4):
            idx = build_relative_position_index(M)
            dec = decode_relative_position_row(idx, M)          # (M^2, M^2, 2)
            d_ab = dec                                  # index[a,b] -> 位移
            d_ba = dec.transpose(0, 1)                  # index[b,a] -> 位移
            self.assertTrue(bool((d_ab == -d_ba).all()),
                            f"M={M} 转置不满足相反位移对称")


class TestRelativePositionBias(unittest.TestCase):
    def test_parameter_shape_and_count(self):
        M, H = 7, 3
        rpb = RelativePositionBias(M, H)
        self.assertEqual(rpb.relative_position_bias_table.shape, ((2 * M - 1) ** 2, H))
        self.assertEqual(rpb.relative_position_bias_table.numel(), (2 * M - 1) ** 2 * H)
        self.assertEqual(rpb.relative_position_bias_table.numel(), 507)

    def test_parameter_learnable(self):
        rpb = RelativePositionBias(3, 4)
        self.assertTrue(rpb.relative_position_bias_table.requires_grad)
        self.assertIsInstance(rpb.relative_position_bias_table, torch.nn.Parameter)

    def test_index_buffer_not_persistent(self):
        rpb = RelativePositionBias(3, 4)
        state = rpb.state_dict()
        self.assertNotIn("relative_position_index", state)
        # buffer 仍然可以通过名字访问
        self.assertTrue(hasattr(rpb, "relative_position_index"))

    def test_forward_shape(self):
        for M, H in ((2, 4), (3, 3), (4, 8)):
            rpb = RelativePositionBias(M, H)
            out = rpb.forward()
            self.assertEqual(out.shape, (M * M, M * M, H))

    def test_forward_is_gather_from_table(self):
        """forward 必须等价于 table[index], 即 gather 语义。"""
        M, H = 3, 4
        rpb = RelativePositionBias(M, H)
        out = rpb.forward()
        expected = rpb.relative_position_bias_table[rpb.relative_position_index]
        self.assertTrue(torch.equal(out, expected))

    def test_grad_flows_to_table(self):
        M, H = 3, 4
        rpb = RelativePositionBias(M, H)
        out = rpb.forward()                 # (M^2, M^2, H)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(rpb.relative_position_bias_table.grad)
        self.assertFalse(bool(torch.all(rpb.relative_position_bias_table.grad == 0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
