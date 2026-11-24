"""Unit tests for Direction 2 · Phase C · Step 7d zero-GPU statistical & threshold audit."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.analysis.rep_repair_step7d_posthoc_audit import (
    ROOT,
    SPAWRIOUS_SPUR_DISTRACTOR,
    fast_binary_auc_pct,
    paired_stratified_bootstrap,
    spawrious_confusion_metrics,
    sweep_wga_curve,
)
from src.methods.rep_repair.phase_a_data import group_metrics


class Step7dPosthocAuditTests(unittest.TestCase):
    def test_sweep_wga_curve_matches_group_metrics_exactly(self):
        rng = np.random.default_rng(7)
        y = np.array([0] * 40 + [1] * 30, dtype=int)
        g = np.array(["00"] * 20 + ["01"] * 20 + ["10"] * 15 + ["11"] * 15, dtype=str)
        p1 = np.clip(y * 0.6 + 0.2 + rng.normal(0.0, 0.2, size=len(y)), 0.0, 1.0)
        # Include exact threshold ties to exercise side='left' boundary behavior.
        p1[0] = 0.35
        p1[41] = 0.35
        grid = np.linspace(0.05, 0.95, 37)
        vec_curve = sweep_wga_curve(p1, y, g, grid)
        ref_curve = np.array([group_metrics(y, (p1 >= t).astype(int), g)["wga"] for t in grid])
        self.assertLess(float(np.max(np.abs(vec_curve - ref_curve))), 1e-9)

    def test_fast_binary_auc_pct_matches_expected_with_ties(self):
        y = np.array([0, 0, 1, 1, 1], dtype=int)
        s = np.array([0.1, 0.4, 0.4, 0.8, 0.9], dtype=float)
        # Pairs (pos, neg): (0.4,0.1)=1, (0.4,0.4)=0.5, (0.8,0.1)=1, (0.8,0.4)=1, (0.9,0.1)=1, (0.9,0.4)=1 -> 5.5/6
        self.assertAlmostEqual(fast_binary_auc_pct(y, s), (5.5 / 6.0) * 100.0, places=9)

    def test_spawrious_confusion_metrics_tracks_distractors(self):
        # 4 samples per class (16 total)
        y = np.repeat(np.arange(4), 4)
        probs = np.full((16, 4), 0.01, dtype=float)
        for i, c in enumerate(y):
            probs[i, c] = 0.97
        # Inject one spurious distractor error for dachshund (c=2 -> labrador=3) and one non-spurious error (c=2 -> corgi=1)
        probs[8] = [0.01, 0.01, 0.01, 0.97]  # distractor for dachshund@snow
        probs[9] = [0.01, 0.97, 0.01, 0.01]  # non-spurious error for dachshund@snow
        self.assertEqual(SPAWRIOUS_SPUR_DISTRACTOR[2], 3)

        res = spawrious_confusion_metrics(probs, y)
        self.assertAlmostEqual(res["ood_worst_class_acc"], 50.0, places=6)
        self.assertAlmostEqual(res["ood_mean_acc"], 87.5, places=6)
        self.assertAlmostEqual(res["spur_rates"][2], 25.0, places=6)
        self.assertAlmostEqual(res["ood_spur_confusion_mean"], 6.25, places=6)
        self.assertAlmostEqual(res["ood_non_spur_err_mean"], 6.25, places=6)

    def test_paired_stratified_bootstrap_determinism_and_dominance(self):
        strata = [np.arange(0, 50), np.arange(50, 100)]
        a = np.concatenate([np.full(50, 92.0), np.full(50, 94.0)])
        b = np.concatenate([np.full(50, 87.0), np.full(50, 89.0)])

        fn_a = lambda idx: np.array([a[idx].mean(), a[idx].mean() + 1.0])
        fn_b = lambda idx: np.array([b[idx].mean(), b[idx].mean()])
        r1 = paired_stratified_bootstrap(fn_a, fn_b, strata, n_boot=200, seed=42)
        r2 = paired_stratified_bootstrap(fn_a, fn_b, strata, n_boot=200, seed=42)
        self.assertEqual(r1["ci_low"], r2["ci_low"])
        self.assertEqual(r1["ci_high"], r2["ci_high"])
        self.assertAlmostEqual(r1["delta_mean"], 5.5, places=6)
        self.assertGreater(r1["ci_low"], 0.0)
        self.assertEqual(r1["seed_wins"], "2/2")

    def test_tracked_step7d_summary_matches_alignment_and_arms(self):
        summary_path = ROOT / "results/rep_repair/step7d_posthoc_audit_20261124/step7d_summary.json"
        if not summary_path.is_file():
            self.skipTest("Step 7d summary not generated yet")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        align = data["alignment_check"]
        self.assertLess(align["max_wb_resnet_wga_diff"], 1e-3)
        self.assertLess(align["max_wb_vit_wga_diff"], 1e-3)
        self.assertLess(align["max_spawrious_acc_diff"], 1e-3)
        self.assertEqual(len(data["spawrious_confusion_rates"]), 6)


if __name__ == "__main__":
    unittest.main()
