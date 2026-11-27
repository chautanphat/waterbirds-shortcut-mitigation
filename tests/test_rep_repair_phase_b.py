"""Unit tests for Direction 2 · Phase B stage extraction, controls, and counterfactuals."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from src.models.resnet import ResNetClassifier
from src.methods.rep_repair import controls, counterfactuals


class PhaseBTests(unittest.TestCase):
    def test_extract_stage_maps_matches_extract_features_and_preserves_keys(self):
        torch.manual_seed(0)
        model = ResNetClassifier(weights=None)
        model.eval()
        keys = list(model.state_dict().keys())
        self.assertTrue(all(k.startswith(("0.", "2.")) for k in keys))

        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            f_old = model.extract_features(x)
            stages = model.extract_stage_maps(x)
        self.assertEqual(tuple(stages["layer1"].shape), (2, 256, 56, 56))
        self.assertEqual(tuple(stages["layer2"].shape), (2, 512, 28, 28))
        self.assertEqual(tuple(stages["layer3"].shape), (2, 1024, 14, 14))
        self.assertEqual(tuple(stages["layer4_map"].shape), (2, 2048, 7, 7))
        self.assertEqual(tuple(stages["pooled"].shape), (2, 2048))
        self.assertLess(float((f_old - stages["pooled"]).abs().max()), 1e-6)

    def test_dar_weights_sum_to_one_and_solves_localized_foreground(self):
        rng = np.random.RandomState(0)
        n = 160
        c, h, w = 16, 7, 7
        fmap = rng.normal(0.0, 0.2, size=(n, c, h, w)).astype(np.float32)
        y = np.repeat([0, 1], n // 2)
        a = np.tile([0, 1], n // 2)
        g = np.array([f"{yi}{ai}" for yi, ai in zip(y, a)])
        # Strong spurious background across 48/49 cells; true core signal ONLY in center cell (3,3).
        bg_sign = (2 * a - 1).astype(np.float32)[:, None, None]
        fg_sign = (2 * y - 1).astype(np.float32)
        fmap[:, 0, :, :] += 2.0 * bg_sign
        fmap[:, 1, 3, 3] += 6.0 * fg_sign

        probs, attns = controls._fit_dar_single(
            fmap[:120], y[:120], g[:120], fmap[120:],
            attn_dim=32, weight_decay=1e-2, dropout=0.0, lr=1e-2, epochs=150,
            seeds=(0,), device="cpu",
        )
        np.testing.assert_allclose(attns.sum(axis=(1, 2)), 1.0, atol=1e-5)
        acc = float((probs.argmax(-1) == y[120:]).mean())
        self.assertGreater(acc, 0.90)

    def test_oracle_mask_pool_selects_foreground_cells_only(self):
        fmap = np.zeros((2, 4, 7, 7), dtype=np.float32)
        fmap[0, :, 3, 3] = [1.0, 2.0, 3.0, 4.0]
        fmap[0, :, 0, 0] = [99.0, 99.0, 99.0, 99.0]
        mask = np.zeros((2, 7, 7), dtype=np.float32)
        mask[0, 3, 3] = 1.0
        pooled = controls.pool_with_spatial_masks(fmap, mask)
        np.testing.assert_allclose(pooled[0], [1.0, 2.0, 3.0, 4.0], atol=1e-5)
        # Empty mask on item 1 falls back cleanly to finite uniform GAP.
        self.assertTrue(np.isfinite(pooled[1]).all())

    def test_affine_dfr_freezes_all_conv_weights_bitwise(self):
        model = ResNetClassifier(weights=None)
        params = controls.configure_adaptation_params(model, "affine_dfr")
        self.assertGreater(len(params), 0)
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                self.assertFalse(mod.weight.requires_grad, f"Conv weight should stay frozen: {name}")
            elif isinstance(mod, nn.BatchNorm2d):
                self.assertTrue(mod.weight.requires_grad, f"BN weight should be trainable: {name}")
                self.assertTrue(mod.bias.requires_grad, f"BN bias should be trainable: {name}")
        for p in model[2].parameters():
            self.assertTrue(p.requires_grad)

    def test_val_image_dataset_rejects_test_rows(self):
        with self.assertRaisesRegex(ValueError, "split == 1"):
            controls.ValImageDataset(
                image_dir=".",
                rows=[{"id": "foo.jpg", "y": 0, "group": "00", "split": 2}],
            )

    def test_counterfactual_compositor_preserves_foreground_and_swaps_background(self):
        fg = np.full((224, 224, 3), 0.8, dtype=np.float32)
        bg = np.full((224, 224, 3), 0.2, dtype=np.float32)
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[80:140, 80:140] = 1.0
        comp = counterfactuals.composite_foreground(fg, mask, bg)
        np.testing.assert_allclose(comp[80:140, 80:140], 0.8, atol=1e-6)
        np.testing.assert_allclose(comp[0:50, 0:50], 0.2, atol=1e-6)

        inpainted = counterfactuals.inpaint_small_mask(bg, mask, iterations=4)
        self.assertEqual(inpainted.shape, (224, 224, 3))
        np.testing.assert_allclose(inpainted, 0.2, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
