"""Mechanism, solver and leakage checks. Synthetic scores are not research results."""
import numpy as np
import pytest
import torch
from scipy.special import softmax

from src.methods.dpe_head_reweighting import ensemble
from src.methods.dpe_head_reweighting.aggregation_source import partition as old_partition
from src.methods.dpe_independent import fitting, geometry, replay, runner
from src.methods.dpe_independent.grid import candidates, crossfit_candidates, load_config
from src.methods.dpe_independent.sources import folds, partition
from src.methods.dpe_independent.storage import (Budget, atomic_json, atomic_npz, frozen_json,
                                               read_json)


def problem():
    rng = np.random.RandomState(4)
    groups = np.repeat(["00", "01", "10", "11"], 12)
    y = np.array([int(g[0]) for g in groups])
    q = rng.uniform(.2, .8, (15, len(y)))
    q[0] = .1+.8*y
    return np.stack([1-q, q], axis=-1), y, groups


def config():
    result = load_config()
    result["steps"] = 8
    return result


def test_subset_endpoints_exposure_and_permutation():
    z = np.random.RandomState(1).normal(size=(15, 24, 2))
    for k in (1, 2, 3, 5, 15):
        for membership in ("cyclic", "permuted"):
            m = geometry.memberships(k, membership)
            np.testing.assert_array_equal(np.bincount(m.ravel(), minlength=15), np.full(15, k))
            p = geometry.subset_heads(z, k, .1, membership)
            np.testing.assert_allclose(p.sum(-1), 1)
            if k in (1, 15):
                np.testing.assert_allclose(p.mean(0), geometry.subset_heads(z, k, .1).mean(0))
    np.testing.assert_allclose(geometry.subset_heads(z, 1), softmax(z, axis=-1))
    full = geometry.subset_heads(z, 15)
    np.testing.assert_allclose(full[0], geometry.global_pool(z))
    np.testing.assert_allclose(full, np.broadcast_to(full[0], full.shape))
    np.testing.assert_allclose(geometry.local_topk(z, 15), full[0])
    np.testing.assert_allclose(geometry.global_pool(z, alpha=np.full((15, 2), 1/15)), full[0])


def test_local_topk_is_classwise_and_dynamic():
    z = np.zeros((15, 2, 2)); z[0, 0, 0] = 10; z[7, 0, 1] = 11
    z[9, 1, 0] = 11; z[3, 1, 1] = 10
    expected = softmax([[10., 11.], [11., 10.]], axis=-1)
    np.testing.assert_allclose(geometry.local_topk(z, 1), expected)
    counts = geometry.usage(z, 1, 1., np.array(["00", "11"]))["all"]
    assert counts[0][0] == counts[0][9] == counts[1][7] == counts[1][3] == 1


@pytest.mark.parametrize("temperature", [1e-8, .05, 1., 1e8])
def test_equal_scores_and_extremes(temperature):
    z = np.full((15, 8, 2), -1e6)
    np.testing.assert_allclose(geometry.subset_heads(z, 3, temperature), .5)
    np.testing.assert_allclose(geometry.local_topk(z, 3, temperature), .5)
    z[:, :, 1] = -2e6
    assert np.isfinite(geometry.global_pool(z, temperature)).all()


def test_original_normalization_scale_and_distance_reconstruction():
    rng = np.random.RandomState(4)
    raw = rng.normal(size=(8, 4)).astype("float32")
    heads = dict(version=1, prototypes=torch.tensor(rng.normal(size=(15, 2, 4)), dtype=torch.float32),
                 distance_scales=torch.linspace(-2, 2, 15), temperatures=torch.ones(15),
                 head_ids=[str(i) for i in range(15)], selected_epochs=[1]*15, classes=[0, 1],
                 groups=["00", "01", "10", "11"], preprocessing=ensemble.PREPROCESSING, provenance={"fixture": True})
    d = geometry.distances(heads, raw)
    reconstructed = softmax(geometry.scores(d, heads["distance_scales"].numpy()), axis=-1)
    # Legacy float32 accumulation is checked at the declared historical replay
    # tolerance; an independent float64 distance reference checks the new path.
    np.testing.assert_allclose(reconstructed, ensemble.predict(heads, raw), atol=2e-7, rtol=0)
    from scipy.spatial.distance import cdist
    normalized = ensemble.standardize(raw).numpy().astype(float)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    expected = []
    for prototype in heads["prototypes"].numpy().astype(float):
        prototype /= np.linalg.norm(prototype, axis=1, keepdims=True)
        expected.append(cdist(normalized, prototype))
    np.testing.assert_allclose(d, expected, atol=1e-12, rtol=0)
    shared = geometry.scores(d, heads["distance_scales"].numpy(), "shared")
    np.testing.assert_allclose(shared, -d*np.median(np.abs(heads["distance_scales"].numpy())))
    with pytest.raises(ValueError):
        geometry.global_pool(shared, alpha=np.ones((15, 2)))


@pytest.mark.parametrize("kind", ["importance", "temperature", "bias", "temperature_bias"])
@pytest.mark.parametrize("loss", ["average", "maximum"])
def test_prototype_fitting_and_independent_replay(kind, loss):
    p, y, g = problem(); z = np.log(p)-np.arange(15)[:, None, None]/20
    spec = dict(family="prototype", kind=kind, temperature=.1, loss=loss, penalty=.01)
    model = fitting.PrototypeModel(kind, .1, config())
    with torch.inference_mode():
        original = model.logits(torch.tensor(z)).sigmoid().numpy()
    np.testing.assert_allclose(original, geometry.global_pool(z, .1)[:, 1])
    record = fitting.fit_adam(z, y, g, spec, config(), 123)
    np.testing.assert_allclose(fitting.predict_adam(record, z, config()), replay.adam_prediction(record, z), atol=1e-12)
    assert record["history"][-1]["step"] == 8
    if "log_temperature" in record["state"]:
        assert .01 <= np.exp(record["state"]["log_temperature"]) <= 10
    if "bias" in record["state"]:
        assert -5 <= record["state"]["bias"] <= 5


@pytest.mark.parametrize("kind", ["affine", "linear", "nonlinear"])
def test_residual_zero_initialization_and_replay(kind):
    p, y, g = problem(); q = p[:, :, 1].T
    with torch.random.fork_rng():
        model = fitting.ResidualModel(kind)
    with torch.inference_mode():
        np.testing.assert_allclose(model.logits(torch.tensor(q)).sigmoid(), q.mean(1))
    spec = dict(family="residual", kind=kind, loss="maximum", penalty=.1)
    record = fitting.fit_adam(q, y, g, spec, config(), 123)
    np.testing.assert_allclose(fitting.predict_adam(record, q, config()), replay.adam_prediction(record, q), atol=1e-12)


@pytest.mark.parametrize("loss", ["average", "maximum"])
def test_brier_simplex_certificate_and_useful_expert(loss):
    p, y, g = problem()
    fit = fitting.fit_brier(p, y, g, .01, loss, config()["solver"])
    w = np.asarray(fit["weights"])
    assert np.all(w >= 0) and abs(w.sum()-1) < 1e-12
    assert fit["objective_gap_bound"] <= 1e-7
    assert w[0] > .5
    assert fit["objective"] < np.square(p.mean(0)[:, 1]-y).mean()


def test_failed_brier_solver_cannot_become_uniform(monkeypatch):
    class Failure:
        x = np.full(16, np.nan)
        success = False
        message = "injected failure"
        nit = 0
    monkeypatch.setattr(fitting, "minimize", lambda *args, **kwargs: Failure())
    p, y, g = problem()
    with pytest.raises(ValueError, match="failed feasibility"):
        fitting.fit_brier(p, y, g, .01, "maximum", config()["solver"])


@pytest.mark.parametrize("contrast", [1e-4, 1e-2])
def test_brier_certificate_with_weak_nearly_identical_heads(contrast):
    rng = np.random.RandomState(27)
    g = np.repeat(["00", "01", "10", "11"], [187, 186, 53, 53])
    y = np.array([int(value[0]) for value in g])
    q = .5+contrast*(rng.normal(size=(15, len(y)))+.2*(2*y-1))
    p = np.stack([1-q, q], axis=-1)
    fit = fitting.fit_brier(p, y, g, .1, "maximum", config()["solver"])
    assert fit["objective_gap_bound"] <= 1e-7


def test_grid_independent_all_settings_and_aliases():
    c = load_config(); full, split, cf = candidates(c, False), candidates(c), crossfit_candidates(c)
    assert (len(full), len(split), len(cf)) == (85, 240, 35)
    assert len({s["id"] for s in split}) == len(split)
    assert sum(s["family"] == "prototype" for s in split) == 64
    assert sum(s["stage"] == "combination" for s in split) == 66
    assert all(s.get("input", {}).get("k") != 15 for s in split if s["stage"] == "combination")
    assert not any("selected" in s for s in split)


def test_partitions_reproduce_historical_primary_and_folds_cover():
    rows = [dict(id=f"species/{i}.jpg", y=i%2, group=f"{i%2}{i%3 == 0:d}", split=1) for i in range(1199)]
    assert partition(rows, 0) == old_partition(rows)
    f = folds(rows)
    assert f.shape == (1199,) and set(f) == set(range(5))
    assert np.bincount(f).max()-np.bincount(f).min() <= 1
    for group in ("00", "01", "10", "11"):
        counts = np.bincount(f[np.array([r["group"] == group for r in rows])], minlength=5)
        assert counts.max()-counts.min() <= 1


def test_frozen_state_refuses_input_change(tmp_path):
    p = tmp_path/"inputs.json"
    frozen_json(p, dict(seed=0)); frozen_json(p, dict(seed=0))
    with pytest.raises(ValueError, match="Frozen inputs changed"):
        frozen_json(p, dict(seed=1))


def test_cache_detects_tampering(tmp_path, monkeypatch):
    head = tmp_path/"head.pt"; head.write_bytes(b"test-fixture")
    feature_path = tmp_path/"features.npz"; feature_path.write_bytes(b"feature-fixture")
    class Source:
        spec = dict(features={"val": str(feature_path)})
        def features(self, split):
            return np.arange(12, dtype="float32").reshape(3, 4)
    monkeypatch.setattr(geometry, "distances", lambda *a, **k: np.zeros((15, 3, 2), dtype="float32"))
    monkeypatch.setattr(ensemble, "predict", lambda *a, **k: np.full((15, 3, 2), .5, dtype="float32"))
    runner.distance_cache(Source(), {"distance_scales": torch.ones(15)}, head, "val", np.arange(3), tmp_path, "cpu")
    atomic_npz(tmp_path/"val-distances.npz", distances=np.ones((15, 3, 2)), scales=np.ones(15), probabilities=np.full((15, 3, 2), .5))
    with pytest.raises(ValueError, match="cache bytes changed"):
        runner.distance_cache(Source(), {"distance_scales": torch.ones(15)}, head, "val", np.arange(3), tmp_path, "cpu")


def test_duplicated_head_simplex_degeneracy():
    p, _, _ = problem()
    duplicate = np.repeat(p[:1], 15, axis=0)
    weights = np.arange(1, 16, dtype=float); weights /= weights.sum()
    np.testing.assert_allclose(np.einsum("h,hnc->nc", weights, duplicate), p[0])


def arm_fixture(tmp_path, monkeypatch):
    from pathlib import Path
    feature = tmp_path/"raw.npz"; feature.write_bytes(b"synthetic-feature-identity")
    head = tmp_path/"heads.pt"; head.write_bytes(b"synthetic-head-identity")
    metadata = tmp_path/"metadata.csv"; metadata.write_text("synthetic identity only")
    provenance = tmp_path/"provenance.json"; provenance.write_text("{}")
    test_rows = [dict(image_path=f"species/image_{i}.jpg", y=i%2, a=(i//2)%2, split="test") for i in range(5794)]
    rows = [dict(id=f"species/validation_{i}.jpg", y=i%2, group=f"{i%2}{(i//2)%2}", split=1) for i in range(1199)]
    class Fixture:
        spec = dict(id="synthetic-seed0", seed=0, recipe="normal_erm", metadata=str(metadata),
                    features=dict(val=str(feature), test=str(feature)), provenance=str(provenance), backbone_sha256="a"*64)
        def heads(self, *args):
            return {}, head
        def head_indices(self, arm):
            return np.arange(720), np.arange(720, 1199)
    source = Fixture(); source.rows = rows; source.test_rows = test_rows
    specs = [dict(id="uniform", family="uniform", stage="geometry"),
             dict(id="affine", family="residual", kind="affine", loss="average", penalty=.01, stage="h2")]
    monkeypatch.setattr(runner, "candidates", lambda *a, **k: specs)
    events = []
    def cache(source, heads, head_path, split, indices, output, device):
        events.append(split)
        if split == "test":
            assert (output.parent/"fit_complete.json").exists()
        targets = np.array([r["y"] for r in (rows if split == "val" else test_rows)])[indices]
        q = np.broadcast_to(.2+.6*targets, (15, len(indices)))
        return dict(probabilities=np.stack([1-q, q], axis=-1), distances=np.zeros((15, len(indices), 2)), scales=np.ones(15))
    monkeypatch.setattr(runner, "distance_cache", cache)
    impl = dict(snapshot="synthetic-fixture", sha256="a"*64, files={})
    return source, events, impl


def test_all_models_freeze_before_test_and_resume_preserves_exports(tmp_path, monkeypatch):
    source, events, impl = arm_fixture(tmp_path, monkeypatch)
    out = tmp_path/"arm"
    result = runner.run_arm(source, "partition0", out, config(), impl, Budget(1))
    assert events == ["val", "test"]
    assert all(r["status"] == "completed" for r in result.values()), result
    exports = {str(p): p.read_bytes() for p in out.rglob("predictions.csv")}
    assert len(exports) == 2
    monkeypatch.setattr(runner, "fit_one", lambda *a, **k: pytest.fail("Resume refitted a frozen model"))
    runner.run_arm(source, "partition0", out, config(), impl, Budget(1))
    assert exports == {str(p): p.read_bytes() for p in out.rglob("predictions.csv")}


def test_interrupted_fit_never_opens_test_and_resumes(tmp_path, monkeypatch):
    source, events, impl = arm_fixture(tmp_path, monkeypatch)
    class Interrupted:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("injected interruption")
    out = tmp_path/"arm"
    with pytest.raises(TimeoutError, match="injected"):
        runner.run_arm(source, "partition0", out, config(), impl, Interrupted())
    assert events == ["val"]
    assert not (out/"fit_complete.json").exists()
    result = runner.run_arm(source, "partition0", out, config(), impl, Budget(1))
    assert all(r["status"] == "completed" for r in result.values()), result


def test_crossfit_oof_coverage_and_test_freeze(tmp_path, monkeypatch):
    source, _, impl = arm_fixture(tmp_path, monkeypatch)
    assignments = folds(source.rows)
    trained, observed = [], []
    def train(indices, path, config, device):
        fold = len(trained); trained.append(indices)
        assert not set(indices) & set(np.flatnonzero(assignments == fold))
        assert 959 <= len(indices) <= 960
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f"fold-{fold}".encode())
        return {}, path
    source.train_heads = train
    def cache(source, heads, path, split, indices, output, device):
        if split == "test":
            assert (output.parent.parent/"fit_complete.json").exists()
            records = source.test_rows
        else:
            observed.extend(indices.tolist()); records = source.rows
        y = np.array([r["y"] for r in records])[indices]
        q = np.broadcast_to(.2+.6*y, (15, len(indices)))
        return dict(probabilities=np.stack([1-q, q], axis=-1), distances=np.zeros((15, len(indices), 2)), scales=np.ones(15))
    specs = [dict(id="crossfit-original-uniform", family="uniform", input=dict(family="original", scale="native"), stage="crossfit"),
             dict(id="crossfit-original-affine", family="residual", kind="affine", input=dict(family="original", scale="native"),
                  loss="average", penalty=.01, stage="crossfit")]
    monkeypatch.setattr(runner, "crossfit_candidates", lambda c: specs)
    monkeypatch.setattr(runner, "combination_inputs", lambda c: [])
    monkeypatch.setattr(runner, "distance_cache", cache)
    result = runner.run_crossfit(source, tmp_path/"crossfit", config(), impl, Budget(1))
    assert sorted(observed) == list(range(1199))
    assert all(r["status"] == "completed" for r in result.values()), result


def test_missing_backbone_never_constructs_replacement(tmp_path):
    from src.methods.dpe_independent.extraction import extract_saved
    with pytest.raises(ValueError, match="replacement training is forbidden"):
        extract_saved(tmp_path/"missing.pt", "a"*64, tmp_path, tmp_path/"features")


def test_kaggle_notebook_compiles_and_uses_normal_imports():
    from pathlib import Path
    notebook = read_json(Path(__file__).resolve().parents[3]/"notebooks/dpe_independent_methods_kaggle.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            compile(code, "kaggle-cell", "exec")
            assert "exec(" not in code and "base64" not in code
