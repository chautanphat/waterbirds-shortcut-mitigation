"""Unit tests for Direction 2 · Phase C · Step 5 causal patching & BN-stage ablations."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from torch import nn

from src.models.resnet import ResNetClassifier
from src.methods.rep_repair import patching


class PhaseCPatchingTests(unittest.TestCase):
    def test_full_patch_reproduces_clean_forward_at_every_block(self):
        torch.manual_seed(0)
        model = ResNetClassifier(weights=None)
        model.eval()
        x_corrupt = torch.randn(1, 3, 224, 224)
        x_clean = torch.randn(1, 3, 224, 224)
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[64:160, 64:160] = 1.0

        clean_acts = patching.capture_block_outputs(model, x_clean)
        for bname in ("layer1", "layer3.5", "layer4.0", "layer4.2"):
            patched = patching.forward_with_spatial_patch(
                model, x_corrupt, clean_acts[bname], bname, mask, "full"
            )
            diff = float((patched["pooled"] - clean_acts["pooled"]).abs().max())
            self.assertLess(diff, 1e-5, f"Full patch mismatch at {bname}: {diff}")

    def test_fg_and_bg_patch_boundary_identities(self):
        torch.manual_seed(1)
        model = ResNetClassifier(weights=None)
        model.eval()
        x_corrupt = torch.randn(1, 3, 224, 224)
        x_clean = torch.randn(1, 3, 224, 224)
        clean_acts = patching.capture_block_outputs(model, x_clean)
        corrupt_pooled = model.extract_features(x_corrupt)

        ones = np.ones((224, 224), dtype=np.float32)
        zeros = np.zeros((224, 224), dtype=np.float32)

        fg_all = patching.forward_with_spatial_patch(model, x_corrupt, clean_acts["layer4.0"], "layer4.0", ones, "fg_only")
        fg_none = patching.forward_with_spatial_patch(model, x_corrupt, clean_acts["layer4.0"], "layer4.0", zeros, "fg_only")
        bg_none = patching.forward_with_spatial_patch(model, x_corrupt, clean_acts["layer4.0"], "layer4.0", ones, "bg_only")

        self.assertLess(float((fg_all["pooled"] - clean_acts["pooled"]).abs().max()), 1e-5)
        self.assertLess(float((fg_none["pooled"] - corrupt_pooled).abs().max()), 1e-5)
        self.assertLess(float((bg_none["pooled"] - corrupt_pooled).abs().max()), 1e-5)

    def test_bottleneck_branch_decomposition_zero_for_identical_inputs(self):
        torch.manual_seed(2)
        model = ResNetClassifier(weights=None)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[70:150, 70:150] = 1.0

        same = patching.bottleneck_branch_decomposition(model, x, x, mask, ("layer4.0", "layer4.2"))
        for row in same:
            self.assertAlmostEqual(row["skip_fg_drift"], 0.0, places=6)
            self.assertAlmostEqual(row["residual_conv_fg_drift"], 0.0, places=6)
            self.assertAlmostEqual(row["post_relu_fg_drift"], 0.0, places=6)
            self.assertAlmostEqual(row["fg_relu_flip_rate"], 0.0, places=6)

        diff = patching.bottleneck_branch_decomposition(model, x, x2, mask, ("layer4.0", "layer4.2"))
        for row in diff:
            self.assertGreater(row["post_relu_fg_drift"], 0.0)
            self.assertTrue(0.0 <= row["residual_delta_share"] <= 1.0)
            self.assertTrue(0.0 <= row["fg_relu_flip_rate"] <= 1.0)

    def test_configure_bn_stage_params_and_shift_summary(self):
        model = ResNetClassifier(weights=None)
        patching.configure_bn_stage_params(model, "bn_l4")

        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                self.assertFalse(mod.weight.requires_grad)
        for mod in model[0][4].modules():  # layer1 BN must stay frozen
            if isinstance(mod, nn.BatchNorm2d):
                self.assertFalse(mod.weight.requires_grad)
        for mod in model[0][7].modules():  # layer4 BN must be trainable
            if isinstance(mod, nn.BatchNorm2d):
                self.assertTrue(mod.weight.requires_grad)
                self.assertTrue(mod.bias.requires_grad)

        adapted = copy.deepcopy(model)
        with torch.no_grad():
            for mod in adapted[0][7].modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.weight.add_(0.25)
        summary = {r["stage"]: r for r in patching.summarize_bn_affine_shifts(model, adapted)}
        self.assertAlmostEqual(summary["layer1"]["mean_abs_dgamma"], 0.0, places=5)
        self.assertAlmostEqual(summary["layer4"]["mean_abs_dgamma"], 0.25, places=4)


if __name__ == "__main__":
    unittest.main()
