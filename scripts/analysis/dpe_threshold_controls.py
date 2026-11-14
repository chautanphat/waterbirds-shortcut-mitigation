"""Run threshold/offset controls for the five-fold cross-fit DPE ensemble on recovered sources.

Reuses the independent study's fold split, head training, distance caches and affine fit, so the
uncalibrated and affine rows should reproduce results/dpe/independent_methods_20261113/per_seed.csv.
Fixed settings only; nothing is selected on test data.

    python scripts/analysis/dpe_threshold_controls.py --sources outputs/dpe_recovered/sources.json \
        --output outputs/dpe_controls
    python scripts/analysis/dpe_threshold_controls.py --summarize --output outputs/dpe_controls \
        --results results/dpe/threshold_controls_20261115
"""
import argparse
import csv
import json
import statistics
from itertools import product
from pathlib import Path

import numpy as np

T95 = {4: 2.776445105}  # Student t, two-sided 95%, four degrees of freedom
SETTINGS = list(product(("average", "maximum"), (0.01, 0.1)))
REFERENCE = Path("results/dpe/independent_methods_20261113/per_seed.csv")


def run(sources_path, output, only=None):
    from src.methods.dpe_head_reweighting.reweighting import metrics
    from src.methods.dpe_independent import controls, fitting
    from src.methods.dpe_independent.grid import DEFAULT, load_config
    from src.methods.dpe_independent.runner import distance_cache
    from src.methods.dpe_independent.sources import Source, folds

    config = load_config(DEFAULT)
    specs = json.loads(Path(sources_path).read_text())["sources"]
    for spec in specs:
        if only and spec["id"] not in only:
            continue
        target = Path(output)/spec["id"]/"controls.json"
        if target.exists():
            print("skip", spec["id"], flush=True)
            continue
        print("source", spec["id"], flush=True)
        source = Source(spec)
        work = Path(output)/spec["id"]
        y = np.array([r["y"] for r in source.rows]); g = np.array([r["group"] for r in source.rows])
        yt = np.array([r["y"] for r in source.test_rows]); gt = np.array([f"{r['y']}{r['a']}" for r in source.test_rows])
        seed = spec["seed"] + config["seed_offset"]

        assignments = folds(source.rows, config["crossfit"]["folds"], config["crossfit"]["split_seed"])
        oof = np.full(1199, np.nan); fold_test = []
        for k in range(5):
            train_idx, heldout = np.flatnonzero(assignments != k), np.flatnonzero(assignments == k)
            heads, path = source.train_heads(train_idx, work/"heads"/f"fold{k}"/"ensemble.pt", config, "cpu")
            val = distance_cache(source, heads, path, "val", heldout, work/"cache"/f"fold{k}", "cpu")
            oof[heldout] = val["probabilities"].astype(np.float64).mean(0)[:, 1]
            test = distance_cache(source, heads, path, "test", np.arange(5794), work/"cache"/f"fold{k}", "cpu")
            fold_test.append(test["probabilities"].astype(np.float64).mean(0)[:, 1])
            print(f"  fold {k} done", flush=True)
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete out-of-fold coverage")
        heads, path = source.heads("full_validation", work/"full", config, "cpu")
        full = distance_cache(source, heads, path, "test", np.arange(5794), work/"cache"/"full", "cpu")
        full_p = full["probabilities"].astype(np.float64).mean(0)[:, 1]

        s_oof = controls.score(oof)
        s_fold = [controls.score(p) for p in fold_test]
        s_full = controls.score(full_p)
        rows, params = {}, {}

        def add(name, p1):
            p1 = np.asarray(p1, dtype=np.float64)
            rows[name] = metrics(np.stack([1-p1, p1], -1), yt, gt)

        add("full_validation", full_p)
        add("fold_uniform", np.mean(fold_test, axis=0))
        for loss, lam in SETTINGS:
            tag = f"{loss}-l{lam:g}"
            spec_fit = dict(family="residual", kind="affine", loss=loss, penalty=lam)
            model = fitting.fit_adam(oof[:, None], y, g, spec_fit, config, seed)
            p = np.mean([fitting.predict_adam(model, f[:, None], config)[:, 1] for f in fold_test], axis=0)
            add(f"fold_affine_{tag}", p)
            params[f"affine_{tag}"] = model["state"]
            b = controls.fit_offset(s_oof, y, g, loss, lam, config, seed)
            params[f"offset_{tag}"] = b
            add(f"fold_offset_{tag}", controls.offset_average(s_fold, b))
            add(f"full_offset_{tag}", (s_full + b > 0).astype(float))
        t, val_wga = controls.wga_threshold(s_oof, y, g)
        params["wga_threshold"] = dict(t=t, oof_wga=val_wga)
        add("fold_threshold_wga", (controls.score(np.mean(fold_test, axis=0)) > t).astype(float))
        add("full_threshold_wga", (s_full > t).astype(float))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(source=spec["id"], recipe=spec["recipe"], seed=spec["seed"],
                                          metrics=rows, parameters=params), indent=1))
        print({k: round(v["wga"]*100, 2) for k, v in rows.items()}, flush=True)


def summarize(output, results):
    records = [json.loads(p.read_text()) for p in sorted(Path(output).glob("*/controls.json"))]
    if not records:
        raise SystemExit("No controls.json found")
    reference = {}
    with REFERENCE.open() as f:
        for r in csv.DictReader(f):
            if r["status"] == "completed":
                reference[(r["source_id"], r["arm"], r["configuration"])] = float(r["wga"])
    checks = []
    for rec in records:
        sid = rec["source"]
        pairs = [("full_validation", ("full_validation", "uniform")),
                 ("fold_uniform", ("crossfit", "crossfit-original-uniform"))]
        pairs += [(f"fold_affine_{l}-l{x:g}", ("crossfit", f"crossfit-original-affine-{l}-l{x:g}")) for l, x in SETTINGS]
        for name, key in pairs:
            ref = reference.get((sid, *key))
            checks.append(dict(source=sid, method=name, reproduced=rec["metrics"][name]["wga"], historical=ref,
                               match=ref is not None and abs(ref - rec["metrics"][name]["wga"]) < 1e-9))
    methods = list(records[0]["metrics"])
    long_rows = [dict(source=r["source"], recipe=r["recipe"], method=m, wga=r["metrics"][m]["wga"]*100,
                      overall=r["metrics"][m]["overall"]*100,
                      **{f"group_{k}": v*100 for k, v in r["metrics"][m]["groups"].items()})
                 for r in records for m in methods]
    contrasts = [("fold_affine_average-l0.1", "full_validation"),
                 ("fold_affine_average-l0.1", "fold_offset_average-l0.1"),
                 ("fold_affine_average-l0.1", "full_offset_average-l0.1"),
                 ("fold_affine_average-l0.1", "fold_threshold_wga"),
                 ("fold_affine_average-l0.1", "full_threshold_wga"),
                 ("fold_offset_average-l0.1", "full_validation"),
                 ("full_offset_average-l0.1", "full_validation"),
                 ("full_threshold_wga", "full_validation"),
                 ("fold_threshold_wga", "full_validation"),
                 ("fold_uniform", "full_validation")]
    contrasts += [(f"fold_affine_{l}-l{x:g}", f"fold_offset_{l}-l{x:g}") for l, x in SETTINGS]
    table = []
    for recipe in ("normal_erm", "ermstar"):
        recs = [r for r in records if r["recipe"] == recipe]
        if not recs:
            continue
        for m in methods:
            w = [r["metrics"][m]["wga"]*100 for r in recs]
            table.append(dict(recipe=recipe, kind="mean", a=m, b="", n=len(w), mean=statistics.mean(w),
                              per_source=w, low=None, high=None))
        for a, b in contrasts:
            d = [(r["metrics"][a]["wga"]-r["metrics"][b]["wga"])*100 for r in recs]
            low = high = None
            if len(d)-1 in T95:
                half = T95[len(d)-1]*statistics.stdev(d)/len(d)**.5
                low, high = statistics.mean(d)-half, statistics.mean(d)+half
            table.append(dict(recipe=recipe, kind="paired", a=a, b=b, n=len(d), mean=statistics.mean(d),
                              per_source=d, low=low, high=high))
    results = Path(results); results.mkdir(parents=True, exist_ok=True)
    with (results/"per_source.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0])); w.writeheader(); w.writerows(long_rows)
    (results/"summary.json").write_text(json.dumps(dict(table=table, reproduction=checks,
        parameters={r["source"]: r["parameters"] for r in records}), indent=1))
    fmt = lambda v: "" if v is None else f"{v:+.2f}"
    lines = ["# Threshold controls for cross-fitted DPE", "",
             f"Sources: {', '.join(r['source'] for r in records)}.",
             f"Reproduction of historical WGA: {sum(c['match'] for c in checks)}/{len(checks)} rows exact.", ""]
    for recipe in ("normal_erm", "ermstar"):
        rows = [t for t in table if t["recipe"] == recipe]
        if not rows:
            continue
        lines += [f"## {recipe}", "", "| Method | Mean WGA | Per source |", "| --- | ---: | --- |"]
        lines += [f"| {t['a']} | {t['mean']:.2f} | {', '.join(f'{v:.2f}' for v in t['per_source'])} |"
                  for t in rows if t["kind"] == "mean"]
        lines += ["", "| Contrast (pp) | Mean | 95% t interval | Per source |", "| --- | ---: | --- | --- |"]
        lines += [f"| {t['a']} minus {t['b']} | {t['mean']:+.2f} | "
                  f"{'' if t['low'] is None else f'[{fmt(t['low'])}, {fmt(t['high'])}]'} | "
                  f"{', '.join(f'{v:+.2f}' for v in t['per_source'])} |" for t in rows if t["kind"] == "paired"]
        lines.append("")
    (results/"REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", nargs="+")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("results/dpe/threshold_controls_20261115"))
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output, args.results)
    else:
        run(args.sources, args.output, args.source_id)
