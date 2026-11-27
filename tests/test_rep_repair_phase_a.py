"""Unit tests for Direction 2 · Phase A validation-only failure audit and readout ladder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analysis import rep_repair_frozen_readouts as readouts
from src.methods.rep_repair import phase_a_data as data


class PhaseATests(unittest.TestCase):
    def test_safe_val_file_blocks_test_paths_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "val_raw.npz").touch()
            (root / "test_raw.npz").touch()
            (root / "test.npz").touch()
            outside = root.parent / "val_raw.npz"
            outside.touch()
            try:
                self.assertEqual(
                    data.safe_val_file(root, "val_raw.npz", ("val_raw.npz",)),
                    root / "val_raw.npz",
                )
                with self.assertRaisesRegex(ValueError, "Blocked"):
                    data.safe_val_file(root, "test_raw.npz", ("val_raw.npz", "test_raw.npz"))
                with self.assertRaisesRegex(ValueError, "Blocked"):
                    data.safe_val_file(root, "test.npz", ("val_raw.npz",))
                with self.assertRaisesRegex(ValueError, "escapes"):
                    data.safe_val_file(root, "../val_raw.npz", ("val_raw.npz",))
            finally:
                outside.unlink(missing_ok=True)

    def test_tern_relabel_and_drop_match_canonical_counts(self):
        rows = [
            {"id": "021.Eastern_Towhee/a.jpg", "y": 1, "a": 0, "group": "10"},
            {"id": "088.Western_Meadowlark/b.jpg", "y": 1, "a": 1, "group": "11"},
            {"id": "102.Western_Wood_Pewee/c.jpg", "y": 1, "a": 0, "group": "10"},
            {"id": "001.Black_footed_Albatross/d.jpg", "y": 1, "a": 0, "group": "10"},
            {"id": "009.Brewer_Blackbird/e.jpg", "y": 0, "a": 1, "group": "01"},
        ]
        keep_rel, rel = data.apply_label_condition(rows, "relabel")
        self.assertEqual(keep_rel.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual([r["y"] for r in rel], [0, 0, 0, 1, 0])
        self.assertEqual([r["group"] for r in rel], ["00", "01", "00", "10", "01"])

        keep_drop, dropped = data.apply_label_condition(rows, "drop")
        self.assertEqual(keep_drop.tolist(), [3, 4])
        self.assertEqual([r["group"] for r in dropped], ["10", "01"])

        # Also check exact counts on official Waterbirds validation CSV if present.
        meta_path = Path("E:/APCS/SC203/waterbirds-shortcut-mitigation/data/waterbirds/waterbird_complete95_forest2water2/metadata.csv")
        if meta_path.is_file():
            val_rows = data.load_val_metadata(meta_path)
            mask = data.tern_mask([r["id"] for r in val_rows])
            self.assertEqual(int(mask.sum()), 18)
            self.assertEqual(sum(m and r["group"] == "10" for r, m in zip(val_rows, mask)), 11)
            self.assertEqual(sum(m and r["group"] == "11" for r, m in zip(val_rows, mask)), 7)
            keep_d, d_rows = data.apply_label_condition(val_rows, "drop")
            self.assertEqual(len(keep_d), 1181)
            self.assertEqual(len(d_rows), 1181)

    def test_stratified_group_folds_disjoint_and_balanced(self):
        rows = []
        counts = {"00": 45, "01": 45, "10": 15, "11": 15}
        idx = 0
        for g, n in counts.items():
            for _ in range(n):
                rows.append({"id": f"img_{idx:04d}.jpg", "y": int(g[0]), "group": g})
                idx += 1
        folds = data.stratified_group_folds(rows, n_splits=5, seed=0)
        self.assertEqual(set(folds.tolist()), {0, 1, 2, 3, 4})
        for g, total in counts.items():
            g_mask = np.array([r["group"] == g for r in rows])
            per_fold = [int(np.sum(folds[g_mask] == f)) for f in range(5)]
            self.assertEqual(sum(per_fold), total)
            self.assertLessEqual(max(per_fold) - min(per_fold), 1)

    def test_nonlinear_readouts_solve_synthetic_xor_where_linear_fails(self):
        rng = np.random.RandomState(42)
        n_per_group = 60
        Xs, ys, gs = [], [], []
        # Class y = XOR(sign(x0), sign(x1)); spurious background a = sign(x0) > 0.
        # Linear classifier cannot separate XOR, while MLP & RBF-SVM should exceed 95% WGA.
        for g_str, (s0, s1, y_val) in [
            ("00", (-1.0, -1.0, 0)),
            ("01", (1.0, 1.0, 0)),
            ("10", (-1.0, 1.0, 1)),
            ("11", (1.0, -1.0, 1)),
        ]:
            pts = rng.normal(0.0, 0.25, size=(n_per_group, 8))
            pts[:, 0] += s0 * 1.5
            pts[:, 1] += s1 * 1.5
            Xs.append(pts)
            ys.append(np.full(n_per_group, y_val, dtype=int))
            gs.append(np.full(n_per_group, g_str))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        g = np.concatenate(gs)
        rows = [{"id": f"s_{i:04d}", "y": int(y[i]), "group": str(g[i])} for i in range(len(y))]
        folds = data.stratified_group_folds(rows, n_splits=4, seed=0)

        lin_probs, _ = readouts.nested_cv_oof(
            X, y, g, folds, inner_splits=3, seed=0,
            candidates=[{"penalty": "l2", "c_val": 1.0}],
            fit_fn=readouts.fit_predict_logistic,
        )
        svm_probs, _ = readouts.nested_cv_oof(
            X, y, g, folds, inner_splits=3, seed=0,
            candidates=[{"c_val": 3.0, "gamma": "scale", "norm": "col_zscore"}],
            fit_fn=readouts.fit_predict_rbf_svm,
        )
        mlp_probs, _ = readouts.nested_cv_oof(
            X, y, g, folds, inner_splits=3, seed=0,
            candidates=[{"hidden": 32, "dropout": 0.1, "weight_decay": 1e-2, "lr": 1e-2, "epochs": 120, "seeds": [0, 1]}],
            fit_fn=readouts.fit_predict_mlp,
        )
        lin_wga = data.group_metrics(y, lin_probs.argmax(-1), g)["wga"]
        svm_wga = data.group_metrics(y, svm_probs.argmax(-1), g)["wga"]
        mlp_wga = data.group_metrics(y, mlp_probs.argmax(-1), g)["wga"]

        self.assertLess(lin_wga, 65.0)
        self.assertGreater(svm_wga, 95.0)
        self.assertGreater(mlp_wga, 95.0)


if __name__ == "__main__":
    unittest.main()
