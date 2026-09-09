# -*- coding: utf-8 -*-
"""
模块 08：BasicLayer 单元测试
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe test_basic_layer.py
"""

import unittest

import torch

from basic_layer import BasicLayer, PatchMerging, DropPath


class TestBasicLayer(unittest.TestCase):

    def test_two_stage_output_shape(self):
        """两 stage 输出形状正确：56x56 -> 28x28 -> 14x14，通道 96->192->384。"""
        torch.manual_seed(0)
        B, H, W = 1, 56, 56
        x = torch.randn(B, H * W, 96)
        stage1 = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7,
                            downsample=PatchMerging(96))
        stage2 = BasicLayer(dim=192, depth=2, num_heads=6, window_size=7,
                            downsample=PatchMerging(192))
        x, H, W = stage1(x, H, W)
        self.assertEqual((H, W, x.shape[-1]), (28, 28, 192))
        x, H, W = stage2(x, H, W)
        self.assertEqual((H, W, x.shape[-1]), (14, 14, 384))
        self.assertEqual(tuple(x.shape), (B, 14 * 14, 384))

    def test_downsample_none_preserves_shape(self):
        """downsample=None 时形状不变。"""
        torch.manual_seed(1)
        B, H, W, C = 1, 16, 16, 96
        x = torch.randn(B, H * W, C)
        layer = BasicLayer(dim=C, depth=2, num_heads=3, window_size=7, downsample=None)
        y, h, w = layer(x, H, W)
        self.assertEqual(tuple(y.shape), (B, H * W, C))
        self.assertEqual((h, w), (H, W))

    def test_shift_alternation(self):
        """blocks 数量正确且 shift 交替：第 0 块 shift=0、第 1 块 shift=window//2。"""
        layer = BasicLayer(dim=96, depth=4, num_heads=3, window_size=7)
        self.assertEqual(len(layer.blocks), 4)
        self.assertEqual(layer.blocks[0].shift_size, 0)
        self.assertEqual(layer.blocks[1].shift_size, 7 // 2)
        self.assertEqual(layer.blocks[2].shift_size, 0)
        self.assertEqual(layer.blocks[3].shift_size, 7 // 2)

    def test_drop_path_scalar_and_list(self):
        """drop_path 标量与列表两种传法都能构造。"""
        # 标量：所有块共享同一 drop_path
        layer_scalar = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7,
                                  drop_path=0.1)
        for blk in layer_scalar.blocks:
            self.assertIsInstance(blk.drop_path, DropPath)
            self.assertAlmostEqual(blk.drop_path.drop_prob, 0.1)

        # 列表：逐块取值（用非零值验证逐块赋值）
        layer_list = BasicLayer(dim=96, depth=2, num_heads=3, window_size=7,
                                drop_path=[0.05, 0.2])
        self.assertAlmostEqual(layer_list.blocks[0].drop_path.drop_prob, 0.05)
        self.assertAlmostEqual(layer_list.blocks[1].drop_path.drop_prob, 0.2)

        # drop_path=0 时是恒等（Identity），非 DropPath
        layer_zero = BasicLayer(dim=96, depth=1, num_heads=3, window_size=7, drop_path=0.0)
        self.assertNotIsInstance(layer_zero.blocks[0].drop_path, DropPath)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
