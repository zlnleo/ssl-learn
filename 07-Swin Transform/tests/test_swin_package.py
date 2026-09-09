"""工程包 swin/ 的集成测试（与 09 模块的测试互补，重点在包接口与工程开关）。"""
import os
import sys
import unittest

# Windows 控制台默认 GBK，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from swin import (SwinTransformer, swin_tiny, swin_small, swin_base, swin_large,
                  build_swin, SWIN_CONFIGS, window_partition, window_reverse,
                  build_relative_position_index, build_attn_mask)  # noqa: E402


class TestSwinTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = swin_tiny(num_classes=1000)
        cls.model.eval()

    def test_forward_shape_224(self):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (2, 1000))

    def test_param_count_tiny(self):
        n = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n, 27.5e6)
        self.assertLess(n, 29.5e6)

    def test_backbone_mode(self):
        m = swin_tiny(num_classes=0)
        m.eval()
        with torch.no_grad():
            feat = m(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(feat.shape), (1, 768))

    def test_stage_resolutions(self):
        x = torch.randn(1, 3, 224, 224)
        H = W = 56
        self.model.eval()
        with torch.no_grad():
            y = self.model.patch_embed(x)
            self.assertEqual(tuple(y.shape), (1, 3136, 96))
            for expect, layer in zip(((784, 192), (196, 384), (49, 768), (49, 768)),
                                     self.model.layers):
                y, H, W = layer(y, H, W)
                self.assertEqual((y.shape[1], y.shape[2]), expect)

    def test_determinism_eval(self):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            a = self.model(x)
            b = self.model(x)
        self.assertTrue(torch.equal(a, b))

    def test_gradient_flow(self):
        m = swin_tiny(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        loss = m(x).sum()
        loss.backward()
        g = m.patch_embed.proj.weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0)


class TestAblationSwitches(unittest.TestCase):
    def test_patch_merging_off(self):
        m = swin_tiny(num_classes=10, patch_merging=False)
        self.assertTrue(all(layer.downsample is None for layer in m.layers))
        with torch.no_grad():
            out = m(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 10))
        H = W = 56
        y = m.patch_embed(torch.randn(1, 3, 224, 224))
        for layer in m.layers:
            y, H, W = layer(y, H, W)
        self.assertEqual((H, W), (56, 56))  # 分辨率全程不变

    def test_window_size_override(self):
        for w in (4, 14):
            m = swin_tiny(num_classes=10, window_size=w)
            m.eval()
            with torch.no_grad():
                out = m(torch.randn(1, 3, 224, 224))
            self.assertEqual(tuple(out.shape), (1, 10))

    def test_non_square_input(self):
        m = swin_tiny(num_classes=10)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 128, 256))
        self.assertEqual(tuple(out.shape), (1, 10))


class TestFactoryAndConfigs(unittest.TestCase):
    def test_build_swin_names(self):
        for name in ("tiny", "small", "base", "large"):
            m = build_swin(name, num_classes=10)
            cfg = SWIN_CONFIGS[name]
            self.assertEqual(m.embed_dim, cfg["embed_dim"])
            self.assertEqual(m.num_layers, len(cfg["depths"]))

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            build_swin("nope")

    def test_depth_head_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            SwinTransformer(embed_dim=96, depths=(2, 2), num_heads=(3, 6, 12, 24))


class TestWindowUtils(unittest.TestCase):
    def test_partition_reverse_roundtrip(self):
        x = torch.randn(2, 56, 56, 96)
        w = window_partition(x, 7)
        self.assertEqual(tuple(w.shape), (2 * 64, 7, 7, 96))
        y = window_reverse(w, 7, 56, 56)
        self.assertTrue(torch.equal(x, y))

    def test_rel_index_range(self):
        idx = build_relative_position_index(7)
        self.assertEqual(tuple(idx.shape), (49, 49))
        self.assertTrue(idx.min() >= 0 and idx.max() < (2 * 7 - 1) ** 2)

    def test_attn_mask_values(self):
        mask = build_attn_mask(56, 56, 7, 3)
        self.assertEqual(tuple(mask.shape), (64, 49, 49))
        self.assertTrue(((mask == 0) | (mask == -100)).all())
        self.assertTrue((mask.diagonal(dim1=-2, dim2=-1) == 0).all())


if __name__ == "__main__":
    unittest.main()
