# -*- coding: utf-8 -*-
"""
模块 09 · test_swin_transformer.py
==================================
Swin-Tiny 总装模块的单元测试（unittest），直接运行：

    D:\\env\\anaconda\\envs\\ssl_cv\\python.exe 09_swin_transformer\\test_swin_transformer.py

覆盖点：前向形状（默认/自定义类别数/特征模式）、非标准分辨率 H/W 演化、
参数量区间、梯度回传、eval 确定性、depths/num_heads 长度不匹配报错。
"""

import sys
import unittest

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

from swin_transformer import SwinTransformer, swin_tiny


def trace_hw(model, x):
    """手动走一遍 forward_features 路径，返回 [(H, W), ...] 的演化列表。"""
    H, W = x.shape[2] // model.patch_size, x.shape[3] // model.patch_size
    hw = [(H, W)]
    xx = model.patch_embed(x)
    for layer in model.layers:
        xx, H, W = layer(xx, H, W)
        hw.append((H, W))
    return hw


class TestSwinTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_default_classes(self):
        """swin_tiny 默认 num_classes=1000：(2,3,224,224) -> (2,1000)。"""
        model = swin_tiny(num_classes=1000).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 1000))

    def test_forward_custom_classes(self):
        """num_classes=10 时输出 (2, 10)。"""
        model = swin_tiny(num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 10))

    def test_forward_feature_mode(self):
        """num_classes=0（特征/backbone 模式）时输出 (2, 768)。"""
        model = swin_tiny(num_classes=0).eval()
        with torch.no_grad():
            feat = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(feat.shape), (2, 768))

    def test_nonstandard_resolution_128(self):
        """非标准输入 128×128 也能跑，且各 stage H/W 演化正确。"""
        model = swin_tiny(num_classes=1000).eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, 1000))
        # 128/4=32 -> 16 -> 8 -> 4 -> 4
        hw = trace_hw(model, x)
        self.assertEqual(hw, [(32, 32), (16, 16), (8, 8), (4, 4), (4, 4)])

    def test_param_count_in_range(self):
        """参数量应落在 [27.5M, 29.5M] 区间（理论 ≈28.3M）。"""
        model = swin_tiny(num_classes=1000)
        total = sum(p.numel() for p in model.parameters())
        self.assertGreaterEqual(total, 27.5e6)
        self.assertLessEqual(total, 29.5e6)

    def test_gradient_to_patch_embed(self):
        """梯度能一路回传到 patch_embed（网络可训练）。"""
        model = swin_tiny(num_classes=10, drop_path_rate=0.0).train()
        x = torch.randn(2, 3, 224, 224)
        loss = model(x).sum()
        loss.backward()
        g = model.patch_embed.proj.weight.grad
        self.assertIsNotNone(g, "patch_embed.proj.weight 应存在梯度")
        self.assertGreater(g.abs().sum().item(), 0.0, "梯度不应全为 0")

    def test_eval_deterministic_with_zero_drop_path(self):
        """drop_path_rate=0 时 eval 模式两次前向结果完全一致。"""
        model = swin_tiny(num_classes=1000, drop_path_rate=0.0).eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            a = model(x)
            b = model(x)
        self.assertTrue(torch.equal(a, b), "eval 模式下两次前向应完全一致")

    def test_depths_heads_length_mismatch(self):
        """depths 与 num_heads 长度不匹配时应报错（num_heads 更短 -> IndexError）。"""
        with self.assertRaises(IndexError):
            SwinTransformer(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
