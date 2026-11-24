"""Unit tests for Direction 2 · Phase C · Step 6 Pre-Layer4 mechanism-derived repair."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.resnet import ResNetClassifier
from src.methods.rep_repair.repair import (
    CosineClassifier,
    LoRAConv2d,
    PreLayer4RepairModel,
    PreLayer4SpatialGate,
    context_invariance_loss,
)


class PhaseCStep6Tests(unittest.TestCase):
    def test_gate_near_identity_init_and_bounds(self):
        torch.manual_seed(0)
        gate = PreLayer4SpatialGate(1024, hidden=64, dropout=0.0, init_bias=2.5)
        gate.eval()
        x = torch.randn(2, 1024, 14, 14)
        with torch.inference_mode():
            g = gate(x)
        self.assertEqual(tuple(g.shape), (2, 1, 14, 14))
        self.assertTrue(bool(((g > 0.85) & (g < 0.99)).all()))

    def test_cosine_classifier_scale_invariance_to_positive_feature_scaling(self):
        torch.manual_seed(1)
        head = CosineClassifier(2048, 2, init_scale=10.0)
        x = torch.randn(4, 2048) + 3.0
        # Positive affine transform per row should leave row-standardized L2-unit cosine logits identical.
        out1 = head(x)
        out2 = head(5.0 * x - 7.0)
        self.assertLess(float((out1 - out2).abs().max()), 1e-5)

    def test_all_conv_weights_stay_bitwise_frozen_and_gradients_flow_to_gate_and_bn(self):
        torch.manual_seed(2)
        base = ResNetClassifier(weights=None)
        model = PreLayer4RepairModel(base, use_gate=True, head_type="cosine", bn_scope="bn_l3_l4")
        model.set_repair_train_mode()

        # Verify every single Conv2d weight is frozen, layer1/2 BN frozen, layer3/4 BN trainable.
        for mod in model.backbone.modules():
            if isinstance(mod, nn.Conv2d):
                self.assertFalse(mod.weight.requires_grad)
        for mod in model.backbone[0][4].modules():
            if isinstance(mod, nn.BatchNorm2d):
                self.assertFalse(mod.weight.requires_grad)
        for mod in model.backbone[0][7].modules():
            if isinstance(mod, nn.BatchNorm2d):
                self.assertTrue(mod.weight.requires_grad)

        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        a = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        out = model.forward_with_context_swap(x, a)
        ce = nn.CrossEntropyLoss()(out["logits"], y)
        inv = context_invariance_loss(
            out["logits"], out["logits_swap"],
            out["layer4_map"], out["layer4_map_swap"], out["gate_map"],
        )
        total = ce + inv["total"]
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertGreaterEqual(float(inv["drift"]), 0.0)
        self.assertGreaterEqual(float(inv["cons"]), -1e-6)
        total.backward()

        # Gate, layer4 BN, and cosine head must all receive finite non-zero gradients.
        self.assertIsNotNone(model.gate.fc2.bias.grad)
        self.assertGreater(float(model.gate.fc2.bias.grad.abs().max()), 0.0)
        self.assertIsNotNone(model.head.weight.grad)
        self.assertGreater(float(model.head.weight.grad.abs().max()), 0.0)
        l4_bn_grads = [
            float(m.weight.grad.abs().max())
            for m in model.backbone[0][7].modules()
            if isinstance(m, nn.BatchNorm2d) and m.weight.grad is not None
        ]
        self.assertGreater(max(l4_bn_grads), 0.0)

    def test_extract_repaired_stage_maps_matches_forward(self):
        torch.manual_seed(3)
        base = ResNetClassifier(weights=None)
        model = PreLayer4RepairModel(base, use_gate=True, head_type="cosine", bn_scope="bn_all")
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            logits_fwd = model(x)
            maps = model.extract_repaired_stage_maps(x)
        self.assertLess(float((logits_fwd - maps["logits"]).abs().max()), 1e-5)
        self.assertEqual(tuple(maps["gate_map"].shape), (2, 1, 14, 14))
        self.assertEqual(tuple(maps["layer4_map"].shape), (2, 2048, 7, 7))

    def test_peak_neutral_gate_preserves_fg_peak_and_uses_neutral_baseline(self):
        torch.manual_seed(4)
        base = ResNetClassifier(weights=None)
        model = PreLayer4RepairModel(
            base, use_gate=True, gate_mode="peak_neutral", head_type="linear", bn_scope="bn_all"
        )
        mu = torch.full((1, 1024, 1, 1), 0.75)
        model.set_neutral_c3(mu)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            logits_fwd = model(x)
            maps = model.extract_repaired_stage_maps(x)
        # Peak of effective gate must be exactly 1.0 per image, and forward must match stage extraction.
        peaks = maps["gate_map"].amax(dim=(2, 3))
        self.assertLess(float((peaks - 1.0).abs().max()), 1e-6)
        self.assertLess(float((logits_fwd - maps["logits"]).abs().max()), 1e-5)
        # Verify neutral fill when gate approaches 0 on a synthetic tensor.
        c3 = torch.full((1, 1024, 14, 14), 5.0)
        g_zero = torch.zeros((1, 1, 14, 14))
        g_zero[0, 0, 7, 7] = 1.0
        gated = model._apply_gate(c3, g_zero)
        self.assertAlmostEqual(float(gated[0, 0, 7, 7]), 5.0, places=5)
        self.assertAlmostEqual(float(gated[0, 0, 0, 0]), 0.75, places=5)

        # Verify finite gradients flow into gate, BN, and linear head under gentle context-swap loss.
        model.set_repair_train_mode()
        y = torch.tensor([0, 1], dtype=torch.long)
        a = torch.tensor([0, 1], dtype=torch.long)
        out = model.forward_with_context_swap(x, a)
        loss = nn.CrossEntropyLoss()(out["logits"], y) + context_invariance_loss(
            out["logits"], out["logits_swap"],
            out["layer4_map"], out["layer4_map_swap"], out["gate_map"],
            drift_weight=0.05, cons_weight=0.10, gate_sparsity=0.005,
        )["total"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(model.gate.fc2.weight.grad)
        self.assertGreater(float(model.gate.fc2.weight.grad.abs().max()), 0.0)
        self.assertGreater(float(model.head.weight.grad.abs().max()), 0.0)

    def test_layer4_conv2_lora_identity_init_and_base_weights_frozen(self):
        torch.manual_seed(5)
        base = ResNetClassifier(weights=None)
        plain = PreLayer4RepairModel(base, use_gate=False, head_type="linear", bn_scope="bn_all", use_l4_conv2_lora=False)
        lora = PreLayer4RepairModel(
            base, use_gate=False, head_type="linear", bn_scope="bn_all", use_l4_conv2_lora=True, lora_rank=8
        )
        plain.eval()
        lora.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            diff = float((plain(x) - lora(x)).abs().max())
        # Zero-initialized lora_b guarantees exact identity at epoch 0.
        self.assertLess(diff, 1e-6)

        # Exactly the 3 layer4.{0,1,2}.conv2 modules must be wrapped in LoRAConv2d with base weights frozen.
        wrapped = [m for m in lora.backbone.modules() if isinstance(m, LoRAConv2d)]
        self.assertEqual(len(wrapped), 3)
        base_snapshots = [m.base.weight.detach().clone() for m in wrapped]
        for m in wrapped:
            self.assertFalse(m.base.weight.requires_grad)
            self.assertTrue(m.lora_a.weight.requires_grad)
            self.assertTrue(m.lora_b.weight.requires_grad)

        # All non-LoRA Conv2d weights across the entire backbone remain frozen.
        lora_conv_ids = {id(m.lora_a) for m in wrapped} | {id(m.lora_b) for m in wrapped}
        for mod in lora.backbone.modules():
            if isinstance(mod, nn.Conv2d) and id(mod) not in lora_conv_ids:
                self.assertFalse(mod.weight.requires_grad)

        # Run 1 optimizer step with parameter_groups and verify base weights stay bitwise identical while LoRA updates.
        lora.set_repair_train_mode()
        groups = lora.parameter_groups(lr=3e-3, lora_lr=1e-3)
        self.assertEqual(len(groups), 2)
        opt = torch.optim.AdamW(groups, weight_decay=1e-2)
        loss = nn.CrossEntropyLoss()(lora(x), torch.tensor([0, 1], dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        opt.step()
        for m, snap in zip(wrapped, base_snapshots):
            self.assertTrue(torch.equal(m.base.weight.detach(), snap))
            self.assertGreater(float(m.lora_b.weight.detach().abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
