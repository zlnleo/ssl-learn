# -*- coding: utf-8 -*-
"""
模块 06：Patch Merging 单元测试
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe test_patch_merging.py
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_merging import PatchMerging


class TestPatchMerging(unittest.TestCase):

    def test_output_shape_standard(self):
        """标准形状：(2, 16*16, 96) 即 H=W=16 -> (2, 64, 192)。"""
        B, H, W, C = 2, 16, 16, 96
        x = torch.randn(B, H * W, C)
        y = PatchMerging(dim=C)(x, H, W)
        self.assertEqual(tuple(y.shape), (B, (H // 2) * (W // 2), 2 * C))

    def test_grouping_content_matches_manual(self):
        """2x2 分组内容与手算一致（逐元素比对）。

        用 C=1、值 0..15 的小例子，把 norm 与 reduction 换成恒等，
        让模块直接输出 4 路拼接结果 (1, 2, 2, 4)，再与手算比对。
        """
        H, W, C = 4, 4, 1
        vals = torch.arange(H * W, dtype=torch.float32).view(1, H, W, C)
        x = vals.view(1, H * W, C)

        model = PatchMerging(dim=C)
        model.norm = nn.Identity()
        model.reduction = nn.Identity()
        y = model(x, H, W)  # (1, (H/2)*(W/2), 4C) = (1, 4, 4)
        y = y.view(1, H // 2, W // 2, 4 * C)

        for i in range(H // 2):
            for j in range(W // 2):
                r0, c0 = 2 * i, 2 * j
                expected = [
                    vals[0, r0, c0, 0].item(),       # x0 左上
                    vals[0, r0 + 1, c0, 0].item(),   # x1 左下
                    vals[0, r0, c0 + 1, 0].item(),   # x2 右上
                    vals[0, r0 + 1, c0 + 1, 0].item(),  # x3 右下
                ]
                got = y[0, i, j].tolist()
                self.assertEqual(got, expected,
                                 f"新位置 ({i},{j}) 分组内容不符：{got} != {expected}")

    def test_odd_size(self):
        """奇数尺寸 H=5, W=5 输出 (H+1)//2 的分辨率。"""
        B, H, W, C = 1, 5, 5, 8
        x = torch.randn(B, H * W, C)
        y = PatchMerging(dim=C)(x, H, W)
        h_out, w_out = (H + 1) // 2, (W + 1) // 2
        self.assertEqual(tuple(y.shape), (B, h_out * w_out, 2 * C))

    def test_linear_equivalence(self):
        """Linear 等价性：手动构造权重复算 LN + Linear，与模块输出一致。"""
        B, H, W, C = 2, 4, 4, 2
        torch.manual_seed(0)
        x = torch.randn(B, H * W, C)

        model = PatchMerging(dim=C)
        # 手动设定已知权重，便于复算
        with torch.no_grad():
            model.norm.weight.fill_(1.0)
            model.norm.bias.zero_()
            model.reduction.weight.copy_(torch.randn(2 * C, 4 * C))

        # 手算：4 路切分 + 拼接 + LN + Linear
        x2d = x.view(B, H, W, C)
        x0 = x2d[:, 0::2, 0::2, :]
        x1 = x2d[:, 1::2, 0::2, :]
        x2 = x2d[:, 0::2, 1::2, :]
        x3 = x2d[:, 1::2, 1::2, :]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        x_norm = F.layer_norm(x_cat, (4 * C,), model.norm.weight, model.norm.bias,
                              model.norm.eps)
        manual_out = F.linear(x_norm, model.reduction.weight, None)

        model_out = model(x, H, W)
        self.assertTrue(torch.allclose(model_out, manual_out, atol=1e-6),
                        "模块输出与手算复现不一致")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
