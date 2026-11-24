"""Unit tests for Direction 2 · Phase C · Step 7c ViT transfer, 4-class repair, and OOD test loaders."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.datasets.spawrious_o2o import (
    CLASSES,
    FILLER_BG,
    TEST_SHIFTED_BG,
    TRAIN_SPURIOUS_BG,
    load_o2o_hard_ood_test_rows,
    load_o2o_hard_train_val_rows,
)
from src.methods.rep_repair.phase_b_assets import recover_kaggle_kernel_files
from src.methods.rep_repair.repair import PreLayer4RepairModel, context_invariance_loss
from src.models.resnet import ResNetClassifier
from src.models.vit import PreBlockViTRepairModel, ViTClassifier


class Step7cTests(unittest.TestCase):
    def test_pre_block_vit_repair_peak_neutral_and_frozen_attention_mlp(self):
        torch.manual_seed(0)
        base_vit = ViTClassifier(weights=None, num_classes=2)
        model = PreBlockViTRepairModel(base_vit, gate_before_block=6, use_gate=True, ln_scope="ln_late")
        model.set_neutral_tokens(torch.full((1, 1, 768), 0.25))
        model.set_repair_train_mode()

        # Verify all Attention/MLP Linear weights, conv_proj, class_token, pos_embedding, and early LNs stay frozen.
        self.assertFalse(model.backbone.vit.conv_proj.weight.requires_grad)
        self.assertFalse(model.backbone.vit.class_token.requires_grad)
        self.assertFalse(model.backbone.vit.encoder.pos_embedding.requires_grad)
        for mod in model.backbone.vit.encoder.layers[0].modules():
            if isinstance(mod, nn.LayerNorm):
                self.assertFalse(mod.weight.requires_grad)
        for mod in model.backbone.vit.encoder.layers[11].modules():
            if isinstance(mod, nn.Linear):
                self.assertFalse(mod.weight.requires_grad)
            elif isinstance(mod, nn.LayerNorm):
                self.assertTrue(mod.weight.requires_grad)
        self.assertTrue(model.backbone.vit.heads.head.weight.requires_grad)

        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        a = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        out = model.forward_with_context_swap(x, a)
        self.assertEqual(tuple(out["gate_map"].shape), (4, 1, 14, 14))
        self.assertLess(float((out["gate_map"].amax(dim=(2, 3)) - 1.0).abs().max()), 1e-6)

        inv = context_invariance_loss(
            out["logits"], out["logits_swap"],
            out["patch_map"], out["patch_map_swap"], out["gate_map"],
            drift_weight=0.05, cons_weight=0.10, gate_sparsity=0.005,
        )
        loss = nn.CrossEntropyLoss()(out["logits"], y) + inv["total"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(model.gate.fc2.weight.grad)
        self.assertGreater(float(model.gate.fc2.weight.grad.abs().max()), 0.0)
        self.assertGreater(float(model.backbone.vit.heads.head.weight.grad.abs().max()), 0.0)

    def test_pre_layer4_repair_supports_4_class_spawrious_context_swap(self):
        torch.manual_seed(1)
        base4 = ResNetClassifier(weights=None, num_classes=4)
        model = PreLayer4RepairModel(
            base4, use_gate=True, gate_mode="peak_neutral", head_type="linear", bn_scope="bn_all"
        )
        model.set_neutral_c3(torch.zeros(1, 1024, 1, 1))
        model.set_repair_train_mode()

        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        bg_attrs = torch.tensor([0, 1, 2, 4], dtype=torch.long)
        out = model.forward_with_context_swap(x, bg_attrs)
        self.assertEqual(tuple(out["logits"].shape), (4, 4))
        inv = context_invariance_loss(
            out["logits"], out["logits_swap"],
            out["layer4_map"], out["layer4_map_swap"], out["gate_map"],
            drift_weight=0.05, cons_weight=0.10, gate_sparsity=0.005,
        )
        loss = nn.CrossEntropyLoss()(out["logits"], y) + inv["total"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(float(model.gate.fc2.weight.grad.abs().max()), 0.0)
        self.assertGreater(float(model.head.weight.grad.abs().max()), 0.0)

    def test_load_o2o_hard_ood_test_rows_loads_only_shifted_combinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "spawrious224"
            for domain in (0, 1):
                for cls_name in CLASSES:
                    for bg in (TRAIN_SPURIOUS_BG[cls_name], FILLER_BG, TEST_SHIFTED_BG[cls_name]):
                        folder = root / str(domain) / bg / cls_name
                        folder.mkdir(parents=True, exist_ok=True)
                        for k in range(3):
                            Image.new("RGB", (16, 16), color=(k * 20, 40, 60)).save(folder / f"img_{k}.png")

            train_rows, val_rows = load_o2o_hard_train_val_rows(root, val_fraction=0.2, split_seed=0)
            ood_rows = load_o2o_hard_ood_test_rows(root)
            self.assertEqual(len(ood_rows), 2 * 4 * 3)
            for r in ood_rows:
                cls_name = CLASSES[r["y"]]
                self.assertEqual(r["bg"], TEST_SHIFTED_BG[cls_name])
                self.assertEqual(r["split"], "test_ood")
            self.assertEqual(len({r["id"] for r in train_rows + val_rows} & {r["id"] for r in ood_rows}), 0)

    def test_recover_kaggle_kernel_files_prefers_local_or_input_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "kaggle_input" / "subdir"
            input_root.mkdir(parents=True, exist_ok=True)
            f1 = input_root / "vit_b16_s0_ckpt_last.pt"
            f1.write_bytes(b"ckpt0")

            dest_dir = tmp_path / "dest"
            dest_dir.mkdir(parents=True, exist_ok=True)
            f2 = dest_dir / "vit_b16_s1_ckpt_last.pt"
            f2.write_bytes(b"ckpt1")

            resolved = recover_kaggle_kernel_files(
                "unused-slug",
                ["vit_b16_backbones/vit_b16_s0_ckpt_last.pt", "vit_b16_backbones/vit_b16_s1_ckpt_last.pt"],
                dest_dir,
                input_search_root=tmp_path / "kaggle_input",
            )
            self.assertEqual(resolved["vit_b16_backbones/vit_b16_s0_ckpt_last.pt"], f1.resolve())
            self.assertEqual(resolved["vit_b16_backbones/vit_b16_s1_ckpt_last.pt"], f2.resolve())

    def test_in_memory_preload_matches_eval_transform_and_single_pass_ood_eval(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from src.methods.rep_repair.phase_b_assets import eval_image_transform
        from scripts.analysis.rep_repair_step7c_waterbirds_test import (
            InMemoryNormalizedDataset,
            _fit_dar_val_and_test,
            preload_images_uint8,
        )
        from scripts.analysis.rep_repair_step7c_spawrious_test import (
            ARM_ORDER,
            _fit_dar_5fold,
            evaluate_all_arms_single_ood_pass,
        )

        # 1) Verify preload_images_uint8 + InMemoryNormalizedDataset matches eval_image_transform() to <1e-5.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bird.png"
            rng = np.random.RandomState(42)
            raw = rng.randint(0, 256, size=(300, 320, 3), dtype=np.uint8)
            Image.fromarray(raw).save(p)
            rows = [{"id": "bird.png", "y": 1, "group": "10"}]
            u8 = preload_images_uint8(Path(tmp), rows)
            ds = InMemoryNormalizedDataset(u8, rows)
            _, x_mem, y_mem, g_mem = ds[0]
            with Image.open(p) as im:
                x_ref = eval_image_transform()(im.convert("RGB"))
            self.assertEqual(y_mem, 1)
            self.assertEqual(g_mem, "10")
            self.assertLess(float((x_mem - x_ref).abs().max()), 1e-5)

        # 2) Verify single-pass OOD evaluation across ALL 6 arms (ERM, DFR, DAR, 3 neural arms).
        torch.manual_seed(7)
        base4 = ResNetClassifier(weights=None, num_classes=4)
        base4.eval()
        F_val = np.random.randn(16, 2048, 7, 7).astype(np.float32)
        y_val = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 2, dtype=int)
        g_val = np.array(["0_spur", "0_beach", "1_spur", "1_beach", "2_spur", "2_beach", "3_spur", "3_beach"] * 2)
        folds = np.array([0] * 8 + [1] * 8, dtype=int)
        val_dar, dar_heads, n_dar = _fit_dar_5fold(F_val, y_val, g_val, folds, seed=0, device="cpu")
        self.assertEqual(val_dar.shape, (16, 4))
        self.assertGreater(n_dar, 0)

        sc = StandardScaler().fit(np.random.randn(16, 2048).astype(np.float32))
        clf = LogisticRegression(max_iter=50).fit(sc.transform(np.random.randn(16, 2048).astype(np.float32)), y_val)
        rep = PreLayer4RepairModel(base4, use_gate=True, gate_mode="peak_neutral", head_type="linear", bn_scope="bn_all")
        neural_arms = {"affine_dfr_bn_l4": [base4], "affine_dfr_bn_all": [base4], "pre_l4_repair_w005": [rep]}
        ood_loader = [(torch.arange(4), torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long), ["0_spur"] * 4)]
        ood_out = evaluate_all_arms_single_ood_pass(base4, [(sc, clf)], dar_heads, neural_arms, ood_loader, "cpu")
        self.assertEqual(set(ood_out.keys()), set(ARM_ORDER))
        for arm, arr in ood_out.items():
            self.assertEqual(arr.shape, (4, 4))
            self.assertTrue(np.isfinite(arr).all(), f"Non-finite output for {arm}")
            self.assertTrue(np.allclose(arr.sum(axis=1), 1.0, atol=1e-4))

        # 3) Verify Waterbirds streaming DAR helper (2 classes).
        base2 = ResNetClassifier(weights=None, num_classes=2)
        y2 = np.array([0, 0, 1, 1] * 2, dtype=int)
        g2 = np.array(["00", "01", "10", "11"] * 2)
        f2 = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        val_p2, test_p2, _ = _fit_dar_val_and_test(base2, F_val[:8], y2, g2, ood_loader, f2, seed=0, device="cpu")
        self.assertEqual(val_p2.shape, (8, 2))
        self.assertEqual(test_p2.shape, (4, 2))
        self.assertTrue(np.allclose(test_p2.sum(axis=1), 1.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
