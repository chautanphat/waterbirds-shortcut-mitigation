import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.analysis import dpe_head_diversity as audit
from src.methods.dpe_head_reweighting import ensemble


class HeadDiversityTests(unittest.TestCase):
    def test_same_decision_direction_with_orthogonal_prototypes(self):
        # e and v are orthogonal and zero-mean, as are the supported features.
        e = np.array([1., -1., 0., 0.]) / np.sqrt(2)
        v = np.array([0., 0., 1., -1.]) / np.sqrt(2)
        prototypes = np.array([[-e + v, e + v], [-e - v, e - v]])
        normalized, _, effective = audit.decision_directions(prototypes)
        np.testing.assert_allclose(np.sum(normalized[0] * normalized[1], axis=-1), 0, atol=1e-12)
        normals = audit.unit_rows(effective)
        np.testing.assert_allclose(normals[0], normals[1])
        raw = np.array([e, -e, e + 3 * v, -e + 3 * v], dtype=np.float32)
        state = dict(
            version=1, prototypes=torch.tensor(prototypes, dtype=torch.float32),
            distance_scales=torch.ones(2), temperatures=torch.ones(2), head_ids=["a", "b"],
            selected_epochs=[1, 1], classes=[0, 1], groups=list(audit.GROUPS),
            preprocessing=ensemble.PREPROCESSING, provenance={"synthetic": True},
        )
        scores = ensemble.predict(state, raw, device="cpu")
        np.testing.assert_array_equal(scores.argmax(axis=-1), [[1, 0, 1, 0], [1, 0, 1, 0]])
        p, scaled, checks = audit.verify_replay(state, raw, scores, 1e-6)
        self.assertEqual(checks["analytic_near_tie_disagreements"], 0)
        self.assertEqual(checks["rescale_max_abs_error"], 0)
        np.testing.assert_allclose(scaled / np.array([.5, 2.])[None, :, None], p)
        bad_scores = scores.copy()
        bad_scores[0, 0] = bad_scores[0, 0, ::-1]
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            audit.verify_replay(state, raw, bad_scores, 1e-6)

    def test_projection_removes_common_feature_coordinate(self):
        prototypes = np.array([[[1., 0., 0.], [0., 1., 1.]]])
        _, nominal, effective = audit.decision_directions(prototypes)
        self.assertNotAlmostEqual(float(nominal.sum()), 0.)
        np.testing.assert_allclose(effective.sum(axis=-1), 0., atol=1e-12)
        np.testing.assert_allclose(effective @ [1., -1., 0.], nominal @ [1., -1., 0.])

    def test_signed_loss_matches_training_formula_and_cancels(self):
        prototypes = np.array([[[1., 0.], [1., 0.]], [[1., 0.], [-1., 0.]]])
        signed, termwise, fraction = audit.stage_penalties(prototypes)[1]
        saved = torch.from_numpy(prototypes).transpose(0, 1)
        newest_first = torch.cat([saved[:, 1:], saved[:, :1]], dim=1)
        covariance = torch.einsum("ijk,ilk->ijl", newest_first, newest_first)
        expected = covariance[:, 0, 1:].sum(1).div(newest_first.shape[1]).mean().abs()
        self.assertEqual(signed, expected.item())
        self.assertEqual((signed, termwise, fraction), (0.0, 0.5, 1.0))
        scaled = audit.stage_penalties(prototypes * np.array([.5, 2.])[None, :, None])
        self.assertEqual(scaled[0], (0.0, 0.0, None))
        self.assertGreater(scaled[1][0], 0.)

    def test_counts_whether_both_heads_can_rescue_errors(self):
        a = np.array([True, True, False, False, True, True])
        b = np.array([True, False, True, False, True, False])
        self.assertEqual(audit.pair_behavior(a, b), {
            "count": 6, "both_correct": 2, "only_a_correct": 2, "only_b_correct": 1,
            "both_wrong": 1, "disagreement": 3, "oracle_extra_correct": 1,
        })
        self.assertEqual(audit.pair_behavior(a, a)["oracle_extra_correct"], 0)

    def test_gate_requires_counts_and_a_group_level_coverage_gap(self):
        limits = dict(primary=dict(max_prototype_similarity=.5, min_boundary_similarity=.95,
                                   max_control_boundary_similarity=.8))
        config = dict(min_pairs_per_comparison=3, min_shared_error_rows=3, min_oracle_gain_gap_rows=2)
        affected = dict(group="10", mean_abs_prototype_cos=.5, effective_direction_cos=.95,
                        both_wrong=3, oracle_extra_correct=1)
        control = dict(group="10", mean_abs_prototype_cos=.5, effective_direction_cos=.8,
                       both_wrong=3, oracle_extra_correct=3)
        rows = [affected] * 3 + [control] * 3
        result = audit.gate_for_seed(rows, limits, config)["primary"]["groups"]
        self.assertTrue(result["10"]["qualifies"])
        self.assertFalse(result["11"]["qualifies"])
        self.assertFalse(audit.gate_for_seed(rows[:-1], limits, config)["primary"]["groups"]["10"]["qualifies"])

    def test_source_allowlist_rejects_test_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            (directory / "test.npz").touch()
            with self.assertRaisesRegex(ValueError, "Unexpected source file"):
                audit.source_file(directory, "test.npz", "val.npz")


if __name__ == "__main__":
    unittest.main()
