"""Targeted, CPU-only audit. Executes selected pure functions from verified source.

No backbone/head fitting and no test-dependent hyperparameter selection.
Historical held-out caches illustrate grid resolution; they are not the missing
OOF predictions from the 70 nested-run cells.
"""
import ast
import io
import zipfile
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE / "audit"
CODE = PACKAGE / "code"
ORIGINAL_RUN = PACKAGE.parents[1] / "results/dpe/nested_selection_20261115"


def functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


ns = {"np": np, "GROUPS": ("00", "01", "10", "11"),
      "SETTINGS": [("average", .1), ("maximum", .1), ("average", .01), ("maximum", .01)]}
functions(ORIGINAL_RUN / "code/nested_selection.py", ["logit_clip", "fit_threshold_wga", "group_accuracy",
          "balanced_nll", "inner_score", "choose", "name"], ns)
functions(CODE / "src/methods/dpe_independent/sources.py", ["folds"], ns)
logit = ns["logit_clip"]


def exact_threshold(p, y, groups):
    z = logit(p)
    boundaries = -np.unique(z)
    candidates = np.unique(np.r_[0., boundaries, np.nextafter(boundaries, -np.inf),
                                 np.nextafter(boundaries, np.inf)])
    best = None
    for b in candidates:
        pred = z + b > 0
        wga = min(ns["group_accuracy"](pred, y, groups).values())
        key = (-round(wga, 9), abs(b))
        if best is None or key < best[0]:
            best = key, float(b), float(wga)
    return {"b": best[1], "wga_pct": 100 * best[2]}


def compare_thresholds(p, y, g):
    coarse = ns["fit_threshold_wga"](p[:, None], y, g)
    coarse_wga = min(ns["group_accuracy"](logit(p) + coarse["b"] > 0, y, g).values()) * 100
    exact = exact_threshold(p, y, g)
    assert exact["wga_pct"] >= coarse_wga - 1e-10
    return {"coarse_b": coarse["b"], "coarse_wga_pct": coarse_wga,
            "exact_b": exact["b"], "exact_wga_pct": exact["wga_pct"],
            "missed_validation_wga_pp": exact["wga_pct"] - coarse_wga,
            "logit_quantiles": np.quantile(logit(p), [0, .1, .5, .9, 1]).tolist()}


out = {"source_hashes": {}}
for local, expected in {'nested_selection.py': 'bf17f74598733fd16b67ef41af65e6ccc39393b03f50e25a895e4300961f97a8', 'src/methods/dpe_independent/fitting.py': '82ebfc48925d70493528cddb2fa89dc75df29f5b998f905865ab0f2b2df58c4e', 'src/methods/dpe_independent/sources.py': '1c9bab8a32ffd4122d7c7c098646f63b0a1f9c3edcca41c8c5be3de5761906e5'}.items():
    path = ORIGINAL_RUN / "code/nested_selection.py" if local == "nested_selection.py" else CODE / local
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (local, actual, expected)
    out["source_hashes"][local] = actual

# A constructive counterexample using the actual notebook function.
g = np.repeat(["00", "01", "10", "11"], 2)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
z = np.array([.001, -.020, .001, -.020, .010, .030, .010, .030])
p = 1 / (1 + np.exp(-z))
out["threshold_counterexample"] = compare_thresholds(p, y, g)
assert out["threshold_counterexample"]["coarse_wga_pct"] == 50
assert out["threshold_counterexample"]["exact_wga_pct"] == 100

# The implemented foldwise bias is not a scalar threshold of the uniform score.
high_mean = np.array([.99, .365, .365, .365, .365])
low_mean = np.repeat(.489, 5)
shift = lambda p: float(np.mean(1 / (1 + np.exp(-(logit(p) + .05)))))
assert high_mean.mean() > low_mean.mean()
assert shift(high_mean) < .5 < shift(low_mean)
out["foldwise_bias_ranking_counterexample"] = {
    "b": .05, "higher_uniform_mean": float(high_mean.mean()),
    "higher_uniform_mean_after_foldwise_bias": shift(high_mean),
    "lower_uniform_mean": float(low_mean.mean()),
    "lower_uniform_mean_after_foldwise_bias": shift(low_mean)}

# Real historical held-out predictions, not the nested run's OOF scores.
out["historical_heldout_threshold_checks"] = {}
with zipfile.ZipFile(ROOT / "historical_heldout_caches.zip") as caches:
    for name in sorted(caches.namelist()):
        with np.load(io.BytesIO(caches.read(name)), allow_pickle=False) as data:
            p = data["probabilities"].astype(np.float64).mean(0)[:, 1]
            out["historical_heldout_threshold_checks"][name] = compare_thresholds(p, data["labels"], data["groups"])

cells = [json.loads(p.read_text()) for p in sorted((ORIGINAL_RUN / "results").glob("*/split*.json"))]
assert len(cells) == 70
assert {c["split"] for c in cells} == set(range(10))
out["cells"] = len(cells)
out["thrwga_b_values"] = sorted({c["arms"]["crossfit_thrwga"]["b"] for c in cells})
out["threshold_control_metrics_identical_cells"] = {}
for arm, control in [("crossfit_thrwga", "crossfit_uniform"), ("fullval_thrwga_oof", "fullval_uniform")]:
    out["threshold_control_metrics_identical_cells"][arm] = sum(
        all(c["arms"][arm][k] == c["arms"][control][k] for k in ("wga", "overall", "groups"))
        for c in cells)

out["selection_checks"] = {}
out["contrasts"] = {}
for recipe in ["normal_erm", "ermstar"]:
    subset = [c for c in cells if c["recipe"] == recipe]
    assert len(subset) == (50 if recipe == "normal_erm" else 20)
    info = {}
    for family in ["crossfit_affine", "crossfit_threshold", "pool_affine", "pool_linear"]:
        for c in subset:
            for rule in ["nll", "wga"]:
                assert ns["choose"](c["inner"][family], rule) == c["arms"][family + "_sel_" + rule]["selected"]
        info[family] = {
            "selected_nll": dict(Counter(c["arms"][family + "_sel_nll"]["selected"] for c in subset)),
            "four_way_inner_wga_ties": sum(len({round(v["wga"], 6) for v in c["inner"][family].values()}) == 1 for c in subset),
            "mean_inner_nll_l001_minus_l01": float(np.mean([
                c["inner"][family]["average-l0.01"]["nll"] - c["inner"][family]["average-l0.1"]["nll"] for c in subset])),
            "mean_inner_wga_l001_minus_l01_pp": float(np.mean([
                c["inner"][family]["average-l0.01"]["wga"] - c["inner"][family]["average-l0.1"]["wga"] for c in subset]))}
    out["selection_checks"][recipe] = info
    contrasts = {}
    for arm, control in [("crossfit_affine_sel_nll", "fullval_uniform"),
                         ("crossfit_affine_fixed", "fullval_uniform"),
                         ("crossfit_affine_fixed", "crossfit_uniform"),
                         ("crossfit_affine_sel_nll", "crossfit_affine_fixed"),
                         ("crossfit_affine_sel_nll", "crossfit_threshold_sel_nll"),
                         ("crossfit_affine_sel_nll", "fullval_threshold_oof_sel_nll"),
                         ("fullval_threshold_oof_sel_nll", "fullval_uniform")]:
        contrasts[arm + " minus " + control] = {
            metric: float(np.mean([c["arms"][arm][metric] - c["arms"][control][metric] for c in subset]))
            for metric in ["wga", "overall"]}
    out["contrasts"][recipe] = contrasts

# Trace label ancestry using the actual stratified folds and official metadata.
with (CODE / "data/metadata.csv").open(newline="") as f:
    rows = [{"id": r["img_filename"], "y": int(r["y"]), "group": r["y"] + r["place"]}
            for r in csv.DictReader(f) if r["split"] == "1"]
assert len(rows) == 1199
out["nesting_ancestry"] = []
for split in range(10):
    outer = ns["folds"](rows, 5, split)
    inner = ns["folds"](rows, 5, 10000 + split)
    for fold in range(5):
        held = inner == fold
        counts = np.array([np.sum(held & (outer != outer[i])) for i in np.flatnonzero(~held)])
        out["nesting_ancestry"].append({"split": split, "inner_fold": fold,
            "held_labels": int(held.sum()), "calibrator_training_rows": int((~held).sum()),
            "training_scores_depending_on_held_labels": int((counts > 0).sum()),
            "min_held_labels_in_each_training_score_base_fit": int(counts.min()),
            "max_held_labels_in_each_training_score_base_fit": int(counts.max())})
assert all(r["calibrator_training_rows"] == r["training_scores_depending_on_held_labels"]
           for r in out["nesting_ancestry"])

(ROOT / "audit_checks.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "nesting_ancestry"}, indent=2))
print("Ancestry example:", out["nesting_ancestry"][0])
print("PASS: source hashes, 70 cells, every saved selection, threshold counterexample, ancestry checks")
