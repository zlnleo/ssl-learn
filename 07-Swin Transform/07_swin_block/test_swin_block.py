# -*- coding: utf-8 -*-
"""
模块 07：Swin Block 单元测试
学习顺序：06_patch_merging -> 07_swin_block -> 08_basic_layer

运行：D:\\env\\anaconda\\envs\\ssl_cv\\python.exe test_swin_block.py
"""

import unittest

import torch

import swin_block as sb
from swin_block import (SwinBlock, WindowAttention, DropPath,
                        window_partition, window_reverse)


class TestSwinBlock(unittest.TestCase):

    def _make(self, dim=96, num_heads=3, window_size=4, shift_size=0):
        return SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                         shift_size=shift_size)

    def test_output_shape_preserved(self):
        """输出形状 (B, L, C) 不变。"""
        torch.manual_seed(0)
        B, H, W, C = 2, 8, 8, 96
        x = torch.randn(B, H * W, C)
        for shift in (0, 2):
            blk = self._make(window_size=4, shift_size=shift)
            y = blk(x, H, W)
            self.assertEqual(tuple(y.shape), (B, H * W, C),
                             f"shift={shift} 输出形状错误")

    def test_shift0_equals_manual_partition_attention_reverse(self):
        """shift=0 的 block 与“手工 partition + WindowAttention + reverse”结果一致。"""
        torch.manual_seed(1)
        B, H, W, C, M = 2, 8, 8, 96, 4
        x = torch.randn(B, H * W, C)
        blk = self._make(window_size=M, shift_size=0)
        blk.eval()

        y = blk(x, H, W)

        # 手工复现 forward（shift=0，无 pad 无 roll）
        x2d = blk.norm1(x).view(B, H, W, C)
        win = window_partition(x2d, M).view(-1, M * M, C)
        attn_out = blk.attn(win, mask=None).view(-1, M, M, C)
        rev = window_reverse(attn_out, M, H, W).view(B, H * W, C)
        manual = x + rev                               # 残差 1（drop_path=0）
        manual = manual + blk.mlp(blk.norm2(manual))   # 残差 2

        self.assertTrue(torch.allclose(y, manual, atol=1e-5),
                        "shift=0 模块输出与手工 partition+attention+reverse 不一致")

    def test_two_blocks_stacked(self):
        """两个 block（W + SW）堆叠可运行。"""
        torch.manual_seed(2)
        B, H, W, C = 2, 8, 8, 96
        x = torch.randn(B, H * W, C)
        blk_w = self._make(window_size=4, shift_size=0)
        blk_sw = self._make(window_size=4, shift_size=2)
        y = blk_w(x, H, W)
        y = blk_sw(y, H, W)
        self.assertEqual(tuple(y.shape), (B, H * W, C))

    def test_mask_cache_hit(self):
        """mask 缓存：第二次前向命中，不重复构造 build_attn_mask。"""
        torch.manual_seed(3)
        calls = {"n": 0}
        orig = sb.build_attn_mask

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        sb.build_attn_mask = counting
        try:
            blk = self._make(window_size=4, shift_size=2)
            x = torch.randn(2, 8 * 8, 96)
            blk(x, 8, 8)
            blk(x, 8, 8)
            self.assertEqual(calls["n"], 1, "同尺寸第二次前向应命中缓存，不重复构造 mask")
            # 换尺寸才重新构造
            x12 = torch.randn(2, 12 * 12, 96)
            blk(x12, 12, 12)
            self.assertEqual(calls["n"], 2, "换尺寸应触发一次新构造")
        finally:
            sb.build_attn_mask = orig

    def test_drop_path_train_eval(self):
        """DropPath 在 train/eval 模式下的行为。"""
        torch.manual_seed(4)
        dp = DropPath(0.5)
        x = torch.randn(32, 4)

        dp.eval()
        self.assertTrue(torch.equal(dp(x), x), "eval 模式 DropPath 应为恒等")

        dp.train()
        out = dp(x)
        # 每一行要么整行置零，要么等于 x/keep_prob
        keep = 0.5
        row_kept = (out.abs().sum(dim=1) > 0)
        self.assertTrue(torch.all(out[row_kept] == x[row_kept] / keep),
                        "保留行应除以 keep_prob 保持期望")
        self.assertTrue(torch.all(out[~row_kept] == 0), "丢弃行应整行置零")
        n_kept = int(row_kept.sum().item())
        self.assertTrue(0 < n_kept < 32, f"drop_prob=0.5 时不应全保留或全丢弃，实际保留 {n_kept}")

    def test_non_divisible_hw(self):
        """H、W 不整除 window_size（10x10，window 4）也能运行且输出形状不变。"""
        torch.manual_seed(5)
        B, H, W, C = 2, 10, 10, 96
        x = torch.randn(B, H * W, C)
        for shift in (0, 2):
            blk = self._make(window_size=4, shift_size=shift)
            y = blk(x, H, W)
            self.assertEqual(tuple(y.shape), (B, H * W, C),
                             f"非整除 H=W=10, shift={shift} 输出形状错误")


class TestWindowHelpers(unittest.TestCase):
    def test_partition_reverse_roundtrip(self):
        """window_partition 与 window_reverse 互为逆运算。"""
        torch.manual_seed(6)
        B, H, W, C, M = 2, 8, 8, 96, 4
        x = torch.randn(B, H, W, C)
        win = window_partition(x, M)
        self.assertEqual(tuple(win.shape), (B * 2 * 2, M, M, C))
        back = window_reverse(win, M, H, W)
        self.assertTrue(torch.equal(back, x))

    def test_relative_position_index(self):
        """相对位置索引形状与取值范围。"""
        M = 4
        idx = sb.build_relative_position_index(M)
        self.assertEqual(tuple(idx.shape), (M * M, M * M))
        self.assertEqual(idx.min().item(), 0)
        self.assertEqual(idx.max().item(), (2 * M - 1) ** 2 - 1)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
