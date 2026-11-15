"""Effect of the Waterbirds species-label error on validation-label methods (DPE, DFR-lite).

Three CUB species (021 Eastern Towhee, 088 Western Meadowlark, 102 Western Wood-Pewee) are
labelled waterbird in metadata.csv only because the generator matches the keyword "Tern" inside
"Eastern"/"Western". This script refits full-validation DPE and DFR-lite on the validation split
under three label conditions and scores every model on official and corrected test labels:

* official : metadata labels (what published results use);
* relabel  : the 18 affected validation images relabelled landbird (group follows);
* drop     : the 18 affected validation images removed (1,181 images).

Head seeds follow the nested-selection run (source seed + 1000*s), so official/s=0 reproduces
the saved full-validation ensembles. Nothing is selected on test data.

    PYTHONPATH=. python scripts/analysis/dpe_label_correction.py run --sources outputs/dpe_recovered/sources.json \
        --output outputs/label_correction --seeds 0 1 2 3 4 --workers 6
    PYTHONPATH=. python scripts/analysis/dpe_label_correction.py summarize --output outputs/label_correction \
        --results results/dpe/label_correction_20261115
"""
import argparse
import contextlib
import csv
import json
import os
import statistics
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SPECIES = ("021.Eastern_Towhee", "088.Western_Meadowlark", "102.Western_Wood_Pewee")
GROUPS = ("00", "01", "10", "11")
CONDITIONS = ("official", "relabel", "drop")
DFR_C = (0.01, 0.03, 0.1, 0.3, 1.0)
T95 = {4: 2.776445105}


def affected(ids):
    return np.array([str(i).split("/")[0] in SPECIES for i in ids])


def corrected(y, g, mask):
    y = np.asarray(y).copy(); g = np.asarray(g).astype(str).copy()
    y[mask] = 0
    g[mask] = np.array(["0" + x[1] for x in g[mask]])
    return y, g


def scores(pred, y, g):
    pred, y, g = np.asarray(pred), np.asarray(y), np.asarray(g)
    acc = {k: float((pred[g == k] == y[g == k]).mean()) * 100 for k in GROUPS}
    return dict(wga=min(acc.values()), overall=float((pred == y).mean()) * 100, groups=acc)


def condition_rows(rows, cond):
    mask = affected([r["id"] for r in rows])
    if cond == "official":
        return np.arange(len(rows)), [dict(r) for r in rows]
    if cond == "drop":
        keep = np.flatnonzero(~mask)
        return keep, [dict(rows[i]) for i in keep]
    if cond.startswith("randdrop"):
        # Control: drop as many correctly labelled images from each official group as the
        # misannotated images occupy there (11 waterbird-on-land, 7 waterbird-on-water).
        rng = np.random.RandomState(500 + int(cond.removeprefix("randdrop")))
        drop = []
        for group in GROUPS:
            need = int(sum(m and r["group"] == group for r, m in zip(rows, mask)))
            pool = sorted([i for i, (r, m) in enumerate(zip(rows, mask)) if r["group"] == group and not m],
                          key=lambda i: rows[i]["id"])
            drop += [pool[i] for i in rng.choice(len(pool), need, replace=False)] if need else []
        keep = np.setdiff1d(np.arange(len(rows)), drop)
        return keep, [dict(rows[i]) for i in keep]
    out = []
    for r, m in zip(rows, mask):
        r = dict(r)
        if m:
            r["y"], r["group"] = 0, "0" + r["group"][1]
        out.append(r)
    return np.arange(len(rows)), out


def evaluate(pred, test_rows):
    y = np.array([r["y"] for r in test_rows]); g = np.array([f"{r['y']}{r['a']}" for r in test_rows])
    mask = affected([r["image_path"] for r in test_rows])
    yc, gc = corrected(y, g, mask)
    return dict(official=scores(pred, y, g), corrected=scores(pred, yc, gc),
                removed=scores(pred[~mask], y[~mask], g[~mask]),
                affected_errors_official=int((pred[mask] != y[mask]).sum()))


def dpe_task(args):
    spec, cond, s, out = args
    path = Path(out)/spec["id"]/f"dpe-{cond}-s{s}.json"
    if path.exists():
        return str(path), "cached"
    import torch
    torch.set_num_threads(1)
    from scipy.special import softmax
    from src.methods.dpe_independent import geometry
    from src.methods.dpe_independent.grid import load_config
    from src.methods.dpe_independent.sources import Source
    from src.methods.dpe_head_reweighting import ensemble
    cfg = load_config(); settings = cfg["head_training"]
    source = Source(spec)
    val, test = source.features("val"), source.features("test")
    idx, rows = condition_rows(source.rows, cond)
    seed = spec["seed"] + 1000 * s
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        if spec["recipe"] == "ermstar":
            import shutil, tempfile
            from src.methods.dpe_ermstar.run import train_heads
            tmp = Path(tempfile.mkdtemp())
            try:
                heads = train_heads(val[idx], rows, tmp/"head_progress.pt", seed, settings, "cpu")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            from src.methods.dpe_head_reweighting.train_heads import train
            heads = train(val[idx], np.array([r["y"] for r in rows]), np.array([r["group"] for r in rows]),
                          settings, seed, dict(source=spec["id"], seed=seed, n=len(rows), condition=cond), "cpu",
                          progress=False)
    ensemble.validate(heads, count=15)
    z = geometry.scores(geometry.distances(heads, test), heads["distance_scales"].numpy())
    p = softmax(z, axis=-1).astype(np.float32).astype(np.float64).mean(0)
    pred = p.argmax(-1)
    result = dict(source=spec["id"], recipe=spec["recipe"], method="dpe", condition=cond, seed_index=s,
                  n_fit=len(rows), **evaluate(pred, source.test_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), pred.astype(np.int8))
    path.write_text(json.dumps(result, indent=1))
    return str(path), "done"


def dfr_task(args):
    spec, cond, _, out = args
    path = Path(out)/spec["id"]/f"dfr-{cond}.json"
    if path.exists():
        return str(path), "cached"
    from sklearn.linear_model import LogisticRegression
    from src.methods.dpe_independent.sources import Source, folds
    source = Source(spec)
    idx, rows = condition_rows(source.rows, cond)
    X = source.features("val")[idx].astype(np.float64)
    mean, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    X = (X-mean)/sd
    Xt = (source.features("test").astype(np.float64)-mean)/sd
    y = np.array([r["y"] for r in rows]); g = np.array([r["group"] for r in rows])

    def weights(gg):
        w = np.zeros(len(gg))
        for k in GROUPS:
            w[gg == k] = 1/(gg == k).sum()
        return w*len(gg)/w.sum()

    def fit(Xa, ya, ga, C):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=2000, tol=1e-3).fit(
                Xa, ya, sample_weight=weights(ga))

    assign = folds(rows, 5, 20_000)
    inner = {}
    for C in DFR_C:
        oof = np.zeros(len(y))
        for f in range(5):
            tr, te = assign != f, assign == f
            oof[te] = fit(X[tr], y[tr], g[tr], C).predict_proba(X[te])[:, 1]
        inner[C] = scores((oof > .5).astype(int), y, g)["wga"]
    best = max(DFR_C, key=lambda C: (inner[C], -C))
    pred = (fit(X, y, g, best).predict_proba(Xt)[:, 1] > .5).astype(int)
    result = dict(source=spec["id"], recipe=spec["recipe"], method="dfr_lite", condition=cond, seed_index=0,
                  n_fit=len(rows), C=best, **evaluate(pred, source.test_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return str(path), "done"


def run(sources, out, seeds, workers, conditions=CONDITIONS, dfr=True, only=None):
    specs = json.loads(Path(sources).read_text())["sources"]
    if only:
        specs = [s for s in specs if s["id"] in only]
    tasks = [(dpe_task, (spec, c, s, str(out))) for spec in specs for c in conditions for s in seeds]
    if dfr:
        tasks += [(dfr_task, (spec, c, 0, str(out))) for spec in specs for c in conditions]
    # ERM* heads are slower; start them first.
    tasks.sort(key=lambda t: (t[1][0]["recipe"] != "ermstar", t[0] is dfr_task))
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, a) for fn, a in tasks]
        for i, f in enumerate(futures):
            print(f"[{i+1}/{len(futures)}]", *f.result(), flush=True)


def summarize(out, results):
    recs = [json.loads(p.read_text()) for p in sorted(Path(out).glob("*/*.json"))]
    rows = []
    for r in recs:
        for label in ("official", "corrected", "removed"):
            rows.append(dict(source=r["source"], recipe=r["recipe"], method=r["method"], condition=r["condition"],
                             seed_index=r["seed_index"], test_labels=label, wga=r[label]["wga"],
                             overall=r[label]["overall"], **{f"g{k}": v for k, v in r[label]["groups"].items()},
                             affected_errors_official=r["affected_errors_official"]))
    results = Path(results); results.mkdir(parents=True, exist_ok=True)
    with (results/"per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def per_source(method, recipe, cond, label, key="wga"):
        out_ = {}
        for r in rows:
            if (r["method"], r["recipe"], r["condition"], r["test_labels"]) == (method, recipe, cond, label):
                out_.setdefault(r["source"], []).append(r[key])
        return {k: statistics.mean(v) for k, v in sorted(out_.items())}

    lines = ["# Label-correction study", "",
             "WGA in %, averaged over head seeds within each source, then over sources. "
             "Paired differences (pp) against the official-label fit of the same method, same test labels.", ""]
    for recipe in ("normal_erm", "ermstar"):
        lines += [f"## {recipe}", "", "| Method | Fit labels | Test: official | Test: corrected | Test: 90 removed | Errors on 90 (official) |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for method in ("dpe", "dfr_lite"):
            for cond in CONDITIONS:
                vals = [per_source(method, recipe, cond, lab) for lab in ("official", "corrected", "removed")]
                err = per_source(method, recipe, cond, "official", "affected_errors_official")
                if not vals[0]:
                    continue
                lines.append(f"| {method} | {cond} | " + " | ".join(f"{statistics.mean(v.values()):.2f}" for v in vals)
                             + f" | {statistics.mean(err.values()):.1f} |")
        lines += ["", "| Contrast | Test labels | Mean (pp) | 95% t interval | Per source |", "| --- | --- | ---: | --- | --- |"]
        for method in ("dpe", "dfr_lite"):
            for cond in ("relabel", "drop"):
                for lab in ("official", "corrected"):
                    a, b = per_source(method, recipe, cond, lab), per_source(method, recipe, "official", lab)
                    if not a:
                        continue
                    d = [a[k]-b[k] for k in a]
                    ci = ""
                    if len(d)-1 in T95:
                        h = T95[len(d)-1]*statistics.stdev(d)/len(d)**.5
                        ci = f"[{statistics.mean(d)-h:+.2f}, {statistics.mean(d)+h:+.2f}]"
                    lines.append(f"| {method}: {cond} minus official | {lab} | {statistics.mean(d):+.2f} | {ci} | "
                                 + ", ".join(f"{x:+.2f}" for x in d) + " |")
        lines.append("")
    (results/"REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def rescore(sources, controls, results):
    """Official vs corrected test labels for the cross-fit variants (official-label fits, split seed 0)."""
    from src.evaluation.prediction_export import read_metadata
    specs = json.loads(Path(sources).read_text())["sources"]
    lg = lambda p: np.log(np.clip(p, 1e-12, 1-1e-12)) - np.log1p(-np.clip(p, 1e-12, 1-1e-12))
    rows = []
    for sp in specs:
        test_rows = read_metadata(sp["metadata"], "test")
        c = Path(controls)/sp["id"]
        par = json.loads((c/"controls.json").read_text())["parameters"]
        load = lambda d: np.load(c/"cache"/d/"test-distances.npz")["probabilities"].astype(float).mean(0)[:, 1]
        full, folds_ = load("full"), [load(f"fold{k}") for k in range(5)]
        a, b = par["affine_average-l0.1"]["a"], par["affine_average-l0.1"]["b"]
        preds = dict(full_validation=full > .5, fold_uniform=np.mean(folds_, 0) > .5,
                     fold_affine_average_l01=np.mean([1/(1+np.exp(-(a*lg(f)+b))) for f in folds_], 0) > .5)
        for name, p in preds.items():
            e = evaluate(p.astype(int), test_rows)
            rows.append(dict(source=sp["id"], recipe=sp["recipe"], method=name,
                             official=e["official"]["wga"], corrected=e["corrected"]["wga"]))
    results = Path(results); results.mkdir(parents=True, exist_ok=True)
    with (results/"rescored_variants.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for recipe in ("normal_erm", "ermstar"):
        base = {r["source"]: r for r in rows if r["recipe"] == recipe and r["method"] == "full_validation"}
        for name in ("full_validation", "fold_uniform", "fold_affine_average_l01"):
            sel = [r for r in rows if r["recipe"] == recipe and r["method"] == name]
            do = [r["official"]-base[r["source"]]["official"] for r in sel]
            dc = [r["corrected"]-base[r["source"]]["corrected"] for r in sel]
            print(f"{recipe} {name}: official {statistics.mean(r['official'] for r in sel):.2f} "
                  f"corrected {statistics.mean(r['corrected'] for r in sel):.2f} | vs full: official {statistics.mean(do):+.2f} "
                  f"corrected {statistics.mean(dc):+.2f} per source {', '.join(f'{x:+.2f}' for x in dc)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--sources", required=True); r.add_argument("--output", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); r.add_argument("--workers", type=int, default=6)
    r.add_argument("--conditions", nargs="+", default=list(CONDITIONS)); r.add_argument("--no-dfr", action="store_true")
    r.add_argument("--only", nargs="*")
    s = sub.add_parser("summarize"); s.add_argument("--output", required=True)
    s.add_argument("--results", default="results/dpe/label_correction_20261115")
    q = sub.add_parser("rescore"); q.add_argument("--sources", required=True)
    q.add_argument("--controls", default="outputs/dpe_controls")
    q.add_argument("--results", default="results/dpe/label_correction_20261115")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.sources, a.output, a.seeds, a.workers, a.conditions, not a.no_dfr, a.only)
    elif a.cmd == "rescore":
        rescore(a.sources, a.controls, a.results)
    else:
        summarize(a.output, a.results)
