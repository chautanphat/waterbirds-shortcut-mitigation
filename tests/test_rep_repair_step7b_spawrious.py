"""Unit tests for Direction 2 · Phase C · Step 7b Spawrious o2o_hard dataset and 4-class models."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.spawrious_o2o import (
    CLASSES,
    FILLER_BG,
    GROUPS_8,
    TEST_SHIFTED_BG,
    TRAIN_SPURIOUS_BG,
    assert_not_o2o_hard_test,
    load_o2o_hard_train_val_rows,
    spawrious_balanced_weights,
    spawrious_group_metrics,
    stratified_8group_folds,
)
from src.methods.rep_repair.controls import DARHead
from src.models.resnet import ResNetClassifier


class Step7bSpawriousTests(unittest.TestCase):
    def test_resnet_and_dar_support_4_classes_without_breaking_binary_default(self):
        torch.manual_seed(0)
        res2 = ResNetClassifier(weights=None)
        res4 = ResNetClassifier(weights=None, num_classes=4)
        dar2 = DARHead()
        dar4 = DARHead(num_classes=4)

        x = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            self.assertEqual(tuple(res2(x).shape), (2, 2))
            self.assertEqual(tuple(res4(x).shape), (2, 4))
            maps = res4.extract_stage_maps(x)
            self.assertEqual(tuple(maps["layer4_map"].shape), (2, 2048, 7, 7))
            logits2, w2 = dar2(maps["layer4_map"])
            logits4, w4 = dar4(maps["layer4_map"])
            self.assertEqual(tuple(logits2.shape), (2, 2))
            self.assertEqual(tuple(logits4.shape), (2, 4))
            self.assertEqual(tuple(w2.shape), (2, 7, 7))
            self.assertEqual(tuple(w4.shape), (2, 7, 7))

    def test_spawrious_rejects_all_test_combinations_and_splits_8_groups_cleanly(self):
        for cls_name, bad_bg in TEST_SHIFTED_BG.items():
            with self.assertRaises(ValueError):
                assert_not_o2o_hard_test(bad_bg, cls_name)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "spawrious224"
            # Populate synthetic train-domain folders AND decoy OOD test folders.
            for domain in (0, 1):
                for cls_name in CLASSES:
                    for bg in (TRAIN_SPURIOUS_BG[cls_name], FILLER_BG, TEST_SHIFTED_BG[cls_name]):
                        folder = root / str(domain) / bg / cls_name
                        folder.mkdir(parents=True, exist_ok=True)
                        n_imgs = 4 if bg == TEST_SHIFTED_BG[cls_name] else 10
                        for k in range(n_imgs):
                            Image.new("RGB", (32, 32), color=(k * 10, 50, 100)).save(folder / f"img_{k}.png")

            train_rows, val_rows = load_o2o_hard_train_val_rows(root, val_fraction=0.2, split_seed=0)
            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(val_rows), 0)

            train_ids = {r["id"] for r in train_rows}
            val_ids = {r["id"] for r in val_rows}
            self.assertEqual(len(train_ids & val_ids), 0)

            # Verify all 8 groups exist in both splits and NO shifted test folder ever appears.
            self.assertEqual({r["group"] for r in train_rows}, set(GROUPS_8))
            self.assertEqual({r["group"] for r in val_rows}, set(GROUPS_8))
            for r in train_rows + val_rows:
                cls_name = CLASSES[r["y"]]
                self.assertNotEqual(r["bg"], TEST_SHIFTED_BG[cls_name])

            folds = stratified_8group_folds(val_rows, n_splits=4, seed=0)
            self.assertEqual(len(folds), len(val_rows))
            self.assertEqual(set(folds.tolist()), {0, 1, 2, 3})

    def test_spawrious_group_metrics_and_balanced_weights(self):
        groups = np.array(["0_spur", "0_spur", "0_beach", "1_spur", "1_beach", "2_spur", "2_beach", "3_spur", "3_beach"])
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
        # Miss one of two in 0_spur (50%) and miss 1_beach completely (0%); all others 100%.
        y_pred = np.array([0, 1, 0, 1, 0, 2, 2, 3, 3])

        m = spawrious_group_metrics(y_true, y_pred, groups)
        self.assertAlmostEqual(m["wga"], 0.0, places=5)
        self.assertAlmostEqual(m["minority_acc"], 75.0, places=5)
        self.assertAlmostEqual(m["majority_acc"], 87.5, places=5)
        self.assertAlmostEqual(m["overall"], 100.0 * 7.0 / 9.0, places=4)

        w = spawrious_balanced_weights(groups)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        group_totals = [float(w[groups == g].sum()) for g in GROUPS_8]
        for tot in group_totals[1:]:
            self.assertAlmostEqual(tot, group_totals[0], places=5)


if __name__ == "__main__":
    unittest.main()
