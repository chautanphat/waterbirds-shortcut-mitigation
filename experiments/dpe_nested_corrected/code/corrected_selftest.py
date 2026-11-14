"""Regression tests; --smoke also fits real heads on synthetic data only."""
import argparse
import ast
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import corrected_nested as c


def isolated_folds():
    path = c.HERE / "src/methods/dpe_independent/sources.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "folds")
    ns = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["folds"]


def synthetic_source(n=100, dim=8, recipe="normal_erm", directory=None):
    rng = np.random.RandomState(12)
    g = np.repeat(c.old.GROUPS, n // 4)
    y = np.array([int(value[0]) for value in g])
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X[:, 0] += 2 * y - 1
    tg = np.repeat(c.old.GROUPS, 10)
    ty = np.array([int(value[0]) for value in tg])
    Xt = rng.normal(size=(len(ty), dim)).astype(np.float32)
    Xt[:, 0] += 2 * ty - 1
    source = SimpleNamespace(rows=[dict(id=f"image-{i:04d}", y=int(y[i]), group=str(g[i]), split=1)
                                   for i in range(n)], _val=X, _test=Xt, val_y=y, val_g=g,
                             test_y=ty, test_g=tg, spec=dict(id="synthetic-" + recipe, recipe=recipe, seed=0))
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, features in (("val", X), ("test", Xt)):
            np.savez_compressed(directory / (name + ".npz"), features=features)
        (directory / "metadata.json").write_text(json.dumps(source.rows))
        source.spec.update(features={k: str(directory / (k + ".npz")) for k in ("val", "test")},
                           metadata=str(directory / "metadata.json"))
    return source


def fake_models():
    def fit(x, y, g, kind, setting, cfg, seed):
        return dict(kind=kind, a=1.2 if kind == "affine" else 1., b=float(np.mean(y) - .5))
    def predict(model, x, cfg):
        return 1 / (1 + np.exp(-(model["a"] * c.old.logit_clip(x[:, 0]) + model["b"])))
    return fit, predict


class CorrectedTests(unittest.TestCase):
    def test_exact_search_recovers_narrow_improvement(self):
        g = np.repeat(c.old.GROUPS, 2)
        y = np.repeat([0, 0, 1, 1], 2)
        z = np.array([.001, -.020, .001, -.020, .010, .030, .010, .030])
        p = 1 / (1 + np.exp(-z))
        model = c.fit_threshold_exact(p, y, g)
        self.assertEqual(model["legacy_grid_fit_wga"], 50.)
        self.assertEqual(model["fit_wga"], 100.)
        self.assertTrue(np.array_equal(c.apply_threshold(model, p), y))

    def test_exact_search_matches_exhaustive_classifier_partitions(self):
        rng = np.random.RandomState(4)
        for _ in range(30):
            g = np.repeat(c.old.GROUPS, 7)
            y = np.array([int(v[0]) for v in g])
            p = rng.choice([0., .001, .49, .5, .501, .51, .999, 1.], len(y))
            model = c.fit_threshold_exact(p, y, g)
            candidates = np.r_[-1., np.unique(p), 2.]
            oracle = max(min(c.old.group_accuracy(p > t, y, g).values()) * 100 for t in candidates)
            self.assertAlmostEqual(model["fit_wga"], oracle)
            self.assertAlmostEqual(min(c.old.group_accuracy(c.apply_threshold(model, p), y, g).values()) * 100, oracle)

    def test_equal_scores_move_together_and_default_wins_ties(self):
        g = np.array(c.old.GROUPS)
        y = np.array([0, 0, 1, 1])
        model = c.fit_threshold_exact(np.repeat(.5, 4), y, g)
        self.assertEqual(model["threshold"], .5)
        self.assertTrue(np.array_equal(c.apply_threshold(model, np.repeat(.5, 4)), np.zeros(4, bool)))
        with self.assertRaises(ValueError):
            c.fit_threshold_exact([.3, .8], [0, 1], ["00", "11"])

    def test_postmean_control_is_one_threshold_on_the_mean(self):
        fit, predict = fake_models()
        folds = np.array([[.99, .489], [.365, .489], [.365, .489], [.365, .489], [.365, .489]])
        affine, bias = dict(a=1.5, b=.05), dict(a=1., b=.05)
        with patch.object(c.old, "predict_model", side_effect=predict):
            out = c.calibration_predictions(affine, bias, folds, np.array([.49, .489]), {})
        expected = predict(bias, folds.mean(0)[:, None], {})
        np.testing.assert_array_equal(out["crossfit_postmean_bias"], expected)
        self.assertTrue(np.all(np.diff(out["crossfit_postmean_bias"]) < 0))
        self.assertLess(out["crossfit_bias_foldwise"][0], .5)
        self.assertGreater(out["crossfit_bias_foldwise"][1], .5)

    def test_every_inner_head_excludes_the_entire_evaluation_fold(self):
        source = synthetic_source()
        folds = isolated_folds()
        captured = []
        fit, predict = fake_models()
        def train(src, idx, seed, cfg, scratch):
            captured.append(set(idx))
            return dict(offset=float(src.val_y[idx].mean() - .5))
        def probabilities(heads, raw):
            p = 1 / (1 + np.exp(-(raw[:, 0] + heads["offset"])))
            return np.tile(np.stack([1 - p, p], axis=-1)[None, :, :], (15, 1, 1))
        with tempfile.TemporaryDirectory() as td, \
             patch.object(c.old, "adapters", return_value={"sources": SimpleNamespace(folds=folds)}), \
             patch.object(c.old, "train_ensemble", side_effect=train), \
             patch.object(c.old, "head_probabilities", side_effect=probabilities), \
             patch.object(c.old, "fit_model", side_effect=fit), \
             patch.object(c.old, "predict_model", side_effect=predict):
            scores, _ = c.strict_inner_scores(source, 0, {}, 0, 0, td, float("inf"))
            assign = folds(source.rows, 5, 10000)
            self.assertEqual(len(captured), 30)
            for f in range(5):
                forbidden = set(np.flatnonzero(assign == f))
                for training in captured[f * 6:(f + 1) * 6]:
                    self.assertFalse(training & forbidden)
            # Replaying a completed selection uses checkpoints, not new fits.
            with patch.object(c.old, "train_ensemble", side_effect=AssertionError("unexpected retraining")):
                again, _ = c.strict_inner_scores(source, 0, {}, 0, 0, td, float("inf"))
            self.assertEqual(scores, again)

    def test_heldout_label_perturbation_cannot_change_inner_context(self):
        source = synthetic_source()
        changed = copy.deepcopy(source)
        folds = isolated_folds()
        assign = folds(source.rows, 5, 10000)
        train, held = np.flatnonzero(assign != 0), np.flatnonzero(assign == 0)
        changed.val_y[held] = 1 - changed.val_y[held]
        def learn(src, idx, *args):
            return dict(offset=float(src.val_y[idx].mean()))
        def probabilities(model, raw):
            p = 1 / (1 + np.exp(-(raw[:, 0] + model["offset"])))
            return np.tile(np.stack([1 - p, p], axis=-1)[None, :, :], (15, 1, 1))
        with tempfile.TemporaryDirectory() as td, \
             patch.object(c.old, "adapters", return_value={"sources": SimpleNamespace(folds=folds)}), \
             patch.object(c.old, "train_ensemble", side_effect=learn), \
             patch.object(c.old, "head_probabilities", side_effect=probabilities):
            a = c.build_context(source, train, held, False, 0, 0, {}, Path(td)/"a", float("inf"), held)
            b = c.build_context(changed, train, held, False, 0, 0, {}, Path(td)/"b", float("inf"), held)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_cache_identity_and_overlap_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "identity.json"
            c.bind_manifest(path, dict(version=2))
            c.bind_manifest(path, dict(version=2))
            with self.assertRaises(ValueError):
                c.bind_manifest(path, dict(version=1))
            with self.assertRaises(ValueError):
                c.cached_scores(synthetic_source(), [0, 1], [1, 2], False, 0, {}, td, float("inf"))
            with self.assertRaises(ValueError):
                c.cached_scores(synthetic_source(), [0, 1], [2], False, 0, {}, td, float("inf"), [1])

    def test_budget_stops_before_a_new_fit(self):
        with tempfile.TemporaryDirectory() as td, patch.object(c.old, "train_ensemble") as trainer:
            with self.assertRaises(c.BudgetReached):
                c.cached_scores(synthetic_source(), [0, 1], [2], False, 0, {}, td, 0)
            trainer.assert_not_called()

    def test_corrupt_prediction_cache_is_rejected(self):
        source = synthetic_source()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            c.bind_manifest(path / "indices.json", dict(train_idx=[0, 1], val_eval_idx=[2],
                            forbidden_idx=[], test_eval=False, seed=0))
            c.atomic_npz(path / "scores.npz", val=np.array([.5]), test=np.empty(0))
            c.old.write_json(path / "complete.json", dict(sha256="incorrect"))
            with self.assertRaises(ValueError):
                c.cached_scores(source, [0, 1], [2], False, 0, {}, path, float("inf"))

    def test_empty_and_partial_reports_label_declared_missing_cells(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            c.old.write_json(out / "sources.json", dict(sources=[dict(id="test-source")]))
            with patch("builtins.print"):
                c.cmd_summarize(SimpleNamespace(out=out))
            report = (out / "REPORT.md").read_text()
            self.assertIn("Completed: 0/10", report)
            self.assertIn("PARTIAL RUN", report)
            arms = {name: dict(wga=80., overall=85., groups={g: 80. for g in c.old.GROUPS})
                    for name in ("fullval_uniform", "bag75_uniform", "crossfit_uniform", "pool_uniform", "dfr_lite")}
            grid, inner = {}, {}
            for family in (*c.FAMILIES, "pool_affine", "pool_linear"):
                grid[family] = {}
                inner[family] = {}
                for setting in c.SETTINGS:
                    grid[family][c.old.name(setting)] = dict(wga=81.)
                    inner[family][c.old.name(setting)] = dict(wga=81., nll=.5)
                for suffix in ("fixed", "sel_nll", "sel_wga"):
                    arms[family + "_" + suffix] = dict(wga=81., overall=86., groups={g: 81. for g in c.old.GROUPS},
                                                       selected=c.old.name(c.FIXED))
            for name in ("crossfit_postmean_thrwga", "fullval_postmean_thrwga"):
                arms[name] = dict(wga=82., overall=87., groups={g: 82. for g in c.old.GROUPS},
                                  threshold=.501, fit_wga=90., legacy_grid_fit_wga=88.)
            c.old.write_json(out / "results/test-source/split00.json", dict(protocol=c.PROTOCOL,
                            fingerprint="synthetic", source="test-source", recipe="normal_erm", split=0,
                            arms=arms, grid=grid, inner=inner))
            with patch("builtins.print"):
                c.cmd_summarize(SimpleNamespace(out=out))
            report = (out / "REPORT.md").read_text()
            self.assertIn("Completed: 1/10", report)
            self.assertIn("crossfit_postmean_thrwga", report)
            self.assertNotIn("crossfit_threshold", report)
            import csv
            with (out / "results_long.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 28)


def real_smoke():
    """Exercise both real head trainers and complete cell/report paths.

    One-epoch heads and eight optimizer steps on synthetic 32-D features;
    these are software checks, not Waterbirds experiments or performance claims.
    """
    import torch
    torch.set_num_threads(1)
    cfg = copy.deepcopy(c.old.adapters()["grid"].load_config())
    cfg["head_training"]["epochs"] = 1
    cfg["steps"] = 8
    results = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        specs = []
        for recipe in ("normal_erm", "ermstar"):
            source = synthetic_source(1200, 32, recipe, root / recipe)
            specs.append(source.spec)
            with patch.object(c.old, "get_source", return_value=source):
                task = (source.spec, 0, str(root / "out"), cfg, "synthetic-smoke-only", time.time() + 600)
                result = c.run_task(task)
                if result[2] != "done":
                    errors = list((root / "out/errors").glob("*.txt"))
                    raise AssertionError("Smoke failed: " + str(result) + "\n" + "\n".join(p.read_text() for p in errors))
                cell = c.old.read_json(root / "out/results" / source.spec["id"] / "split00.json")
                assert len(cell["inner"]) == 7
                assert len(cell["arms"]) == 28
                assert c.run_task(task)[2] == "cached"
                # A changed protocol cannot reuse the completed cell/checkpoints.
                mismatch = list(task)
                mismatch[4] = "changed-protocol"
                assert c.run_task(tuple(mismatch))[2] == "ERROR"
                assert c.run_task(task)[2] == "cached"  # clear the deliberately injected guard error
                results.append(dict(recipe=recipe, status="passed", arms=len(cell["arms"]),
                                    head_ensembles=41, epochs=1, calibration_steps=8))
        c.old.write_json(root / "out/sources.json", dict(sources=specs, blockers=[]))
        c.cmd_summarize(SimpleNamespace(out=root / "out"))
        assert "PARTIAL RUN" in (root / "out/REPORT.md").read_text()
    print("SYNTHETIC SMOKE PASSED:", json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorrectedTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.smoke:
        real_smoke()
