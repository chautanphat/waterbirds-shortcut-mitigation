"""Focused checks for the XRM baseline adaptations and report aggregation."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "baselines" / "xrm"
sys.path.insert(0, str(BASELINE))

from algorithms import ERM, GroupDRO  # noqa: E402
from scripts.summarize_xrm_baselines import read_run  # noqa: E402
from utils import get_metrics, load_checkpoint, save_checkpoint  # noqa: E402


class XrmBaselineTests(unittest.TestCase):
    def setUp(self):
        self.hparams = {"device": "cpu", "num_y": 2, "num_m": 2, "eta": 0.7}
        self.net = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer.lr_scheduler = None

    def test_erm_objective_and_one_update(self):
        method = ERM(self.hparams, self.net, self.optimizer)
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        self.assertTrue(torch.allclose(method.get_loss(logits, labels), F.cross_entropy(logits, labels)))
        x = torch.randn(4, 3)
        before = self.net.weight.detach().clone()
        method.update((torch.arange(4), x, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])))
        self.assertFalse(torch.equal(before, self.net.weight.detach()))

    def test_groupdro_objective_tracks_four_group_losses(self):
        method = GroupDRO(self.hparams, self.net, self.optimizer)
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        attributes = torch.tensor([0, 0, 1, 1])
        losses = F.cross_entropy(logits, labels, reduction="none")
        groups = 2 * labels + attributes
        expected_q = torch.zeros(4)
        expected_q[groups] = torch.exp(self.hparams["eta"] * losses.detach())
        expected_q /= expected_q.sum()
        group_losses = torch.zeros(4)
        group_losses[groups] = losses.detach()
        expected_loss = (expected_q * group_losses).sum()
        actual = method.get_loss(logits, labels, attributes)
        self.assertTrue(torch.allclose(actual, expected_loss))
        self.assertTrue(torch.allclose(method.q, expected_q))
        self.assertFalse(method.q.requires_grad)

    def test_metric_uses_all_groups(self):
        y = torch.tensor([0, 0, 1, 1])
        attrs = torch.tensor([0, 1, 0, 1])
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0]])
        metrics = get_metrics({"num_m": 2}, y, logits, attrs)
        self.assertEqual(metrics["avg_acc"], 0.75)
        self.assertEqual(metrics["wga"], 0.0)

    def test_groupdro_checkpoint_restores_q_optimizer_and_torch_rng(self):
        method = GroupDRO(self.hparams, self.net, self.optimizer)
        method.q.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        torch.manual_seed(52)
        with tempfile.TemporaryDirectory() as temp:
            hparams = {"ckpt_file": str(Path(temp) / "state.pt"), "device": "cpu"}
            save_checkpoint(hparams, method, 17)
            expected_rng = torch.get_rng_state().clone()
            method.q.fill_(0.25)
            with torch.no_grad():
                method.net.weight.zero_()
            torch.manual_seed(999)
            self.assertEqual(load_checkpoint(hparams, method), 18)
            self.assertTrue(torch.allclose(method.q, torch.tensor([0.1, 0.2, 0.3, 0.4])))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertGreater(float(method.net.weight.detach().abs().sum()), 0.0)

    def test_windows_checkpoint_pattern_and_incomplete_log_aggregation(self):
        from utils import get_star_value_from_file_name
        self.assertEqual(get_star_value_from_file_name(r"C:\runs\ckpt_hpcomb_*.pt", r"C:\runs\ckpt_hpcomb_8.pt"), 8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results_hpcomb_4_seed0.json"
            rows = [
                {"algorithm_name": "ERM", "seed": 0, "hparams_comb": 4, "num_step": 3},
                {"step": 1, "va_wga": 0.5, "te_wga": 0.4, "te_avg_acc": 0.7},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = read_run(path, "ERM", 0, 4, 3)
            self.assertEqual(result["status"], "incomplete")
            rows.append({"step": 2, "va_wga": 0.6, "te_wga": 0.55, "te_avg_acc": 0.75})
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = read_run(path, "ERM", 0, 4, 3)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["selected_step"], 2)


if __name__ == "__main__":
    unittest.main()
