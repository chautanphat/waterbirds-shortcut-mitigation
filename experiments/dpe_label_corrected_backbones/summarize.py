"""Score every finished job on official and corrected test labels and write the declared contrasts.

Each job directory holds test_predictions.npz (probabilities, y, groups, filenames) written by the
runner, plus job.json describing backbone seed, backbone label set, validation label set and head
seed. The test labels in the npz follow the job's validation label set; both label sets are
rebuilt here from the three species names, so every job is scored on both.

Declared before the run (do not edit after results are read):
  primary   : (backbone official, val corrected) - (backbone official, val official), corrected test,
              mean over head seeds within a backbone, then over backbones.
  secondary : (backbone corrected, val corrected) - (backbone official, val official), corrected test;
              (backbone corrected, val official) - (backbone official, val official), corrected test;
              all contrasts also on official test labels.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

SPECIES = ("021.Eastern_Towhee", "088.Western_Meadowlark", "102.Western_Wood_Pewee")
GROUPS = ("00", "01", "10", "11")
REGIMES = (("official", "official"), ("official", "corrected"), ("corrected", "official"), ("corrected", "corrected"))
T95 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105, 5: 2.570581836}
CONTRASTS = [(("official", "corrected"), ("official", "official"), "primary: fix validation labels"),
             (("corrected", "corrected"), ("official", "official"), "fix training and validation labels"),
             (("corrected", "official"), ("official", "official"), "fix training labels only"),
             (("corrected", "corrected"), ("official", "corrected"), "add training fix to validation fix")]


def affected(filenames):
    return np.array([any(s in str(f) for s in SPECIES) for f in filenames])


def score(pred, y, a):
    g = np.array([f"{yy}{aa}" for yy, aa in zip(y, a)])
    acc = {k: float((pred[g == k] == y[g == k]).mean()) * 100 for k in GROUPS}
    return dict(wga=min(acc.values()), overall=float((pred == y).mean()) * 100, groups=acc)


def score_job(job_dir):
    job = json.loads((job_dir/"job.json").read_text())
    with np.load(job_dir/"test_predictions.npz", allow_pickle=False) as z:
        p, y, groups, files = z["probabilities"], z["y"].astype(int), z["groups"].astype(str), z["filenames"]
    if len(y) != 5794 or p.shape != (5794, 2):
        raise ValueError(f"{job_dir}: expected 5,794 test predictions")
    mask = affected(files)
    if int(mask.sum()) != 90:
        raise ValueError(f"{job_dir}: expected 90 affected test images, found {int(mask.sum())}")
    a = np.array([int(g[1]) for g in groups])
    pred = p.argmax(1)
    y_off, y_cor = y.copy(), y.copy()
    y_off[mask], y_cor[mask] = 1, 0
    return dict(job, official=score(pred, y_off, a), corrected=score(pred, y_cor, a),
                affected_predicted_landbird=int((pred[mask] == 0).sum()))


def summarize(results, out):
    results, out = Path(results), Path(out)
    rows, failures = [], []
    for d in sorted(p for p in results.iterdir() if (p/"job.json").exists()):
        if not (d/"test_predictions.npz").exists():
            failures.append(d.name)
            continue
        try:
            rows.append(score_job(d))
        except Exception as e:  # keep going; report the failure
            failures.append(f"{d.name}: {e!r}")
    out.mkdir(parents=True, exist_ok=True)
    flat = [dict(job=r["name"], backbone_seed=r["backbone_seed"], backbone_labels=r["backbone_labels"],
                 val_labels=r["val_labels"], head_seed=r["head_seed"], test_labels=t, wga=r[t]["wga"],
                 overall=r[t]["overall"], **{f"g{k}": v for k, v in r[t]["groups"].items()},
                 affected_predicted_landbird=r["affected_predicted_landbird"])
            for r in rows for t in ("official", "corrected")]
    if flat:
        with (out/"summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)

    def cell(seed, regime, test):
        v = [r[test]["wga"] for r in rows if r["backbone_seed"] == seed and (r["backbone_labels"], r["val_labels"]) == regime]
        return statistics.mean(v) if v else None, len(v)

    seeds = sorted({r["backbone_seed"] for r in rows})
    lines = ["# Label-corrected backbones: results", "",
             f"Jobs scored: {len(rows)}. Missing or failed: {len(failures)}.", ""]
    if failures:
        lines += ["Failures:", *[f"- {x}" for x in failures], ""]
    for test in ("corrected", "official"):
        lines += [f"## Test labels: {test}", "", "WGA (%) per backbone seed, mean over head seeds (n head seeds in brackets).", "",
                  "| Backbone labels / val labels | " + " | ".join(f"seed {s}" for s in seeds) + " |",
                  "| --- |" + " ---: |" * len(seeds)]
        for regime in REGIMES:
            vals = [cell(s, regime, test) for s in seeds]
            lines.append(f"| {regime[0]} / {regime[1]} | " + " | ".join("" if m is None else f"{m:.2f} ({n})" for m, n in vals) + " |")
        lines += ["", "| Contrast | Mean (pp) | 95% t interval | Backbones | Per backbone |", "| --- | ---: | --- | ---: | --- |"]
        for a_, b_, label in CONTRASTS:
            d = []
            for s in seeds:
                x, y = cell(s, a_, test)[0], cell(s, b_, test)[0]
                if x is not None and y is not None:
                    d.append((s, x-y))
            if not d:
                lines.append(f"| {label} | | | 0 | |"); continue
            vals = [v for _, v in d]
            m = statistics.mean(vals)
            ci = ""
            if len(vals) >= 2:
                h = T95[len(vals)-1]*statistics.stdev(vals)/len(vals)**.5
                ci = f"[{m-h:+.2f}, {m+h:+.2f}]"
            lines.append(f"| {label} | {m:+.2f} | {ci} | {len(vals)} | " + ", ".join(f"s{s}: {v:+.2f}" for s, v in d) + " |")
        lines.append("")
    (out/"REPORT.md").write_text("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(sys.argv[1], sys.argv[2]))
