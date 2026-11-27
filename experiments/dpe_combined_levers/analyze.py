"""Declared contrasts of PLAN.md for the combined-levers run (A) and the random-drop control (B).

python experiments/dpe_combined_levers/analyze.py --run outputs/combined_levers/run \
    --randdrop outputs/combined_levers/randdrop --out results/dpe/combined_levers_20261115
"""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
STD = [f"normal-erm-seed{i}" for i in range(5)]
OFFICIAL_RESCORE = Path("results/dpe/corrected_test_rescore_20261115/per_arm.csv")
LABEL_RUNS = Path("results/dpe/label_correction_20261115/per_run.csv")
VIEW_KEY = dict(official="wga_official", corrected="wga_corrected", removed="wga_removed")


def interval(values):
    m = st.mean(values)
    if len(values) < 2:
        return m, float("nan"), float("nan")
    h = T975[len(values) - 1] * st.stdev(values) / len(values) ** .5
    return m, m - h, m + h


def fmt(stats, per):
    m, lo, hi = stats
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}], {sum(x > 0 for x in per)}/{len(per)}"


def load_run(run, mode):
    cells = {}
    for p in sorted((Path(run) / "results" / mode).glob("*/split*.json")):
        c = json.loads(p.read_text())
        cells[(c["source"], c["split"])] = c
    return cells


def load_official():
    out = defaultdict(dict)
    with OFFICIAL_RESCORE.open() as f:
        for r in csv.DictReader(f):
            out[(r["source"], int(r["split"]))][r["arm"]] = {v: float(r[k]) for v, k in VIEW_KEY.items()}
    return out


def contrast(get, cells, arm, control, view, splits):
    per = []
    for s in STD:
        per.append(st.mean(get(cells, s, sp, arm, view) - get(cells, s, sp, control, view) for sp in splits))
    return interval(per), per


def get_new(cells, s, sp, arm, view):
    return cells[(s, sp)]["arms"][arm][view]["wga"]


def get_old(cells, s, sp, arm, view):
    return cells[(s, sp)][arm][view]


def analyze_a(run, lines, summary):
    new = load_run(run, "relabel")
    old = load_official()
    splits = sorted(sp for sp in range(10) if all((s, sp) in new for s in STD))
    lines += ["## Run A: nested post-hoc procedures on corrected validation labels", "",
              f"Splits complete for all five standard-ERM backbones: {splits} ({len(splits)} of 10).", ""]
    summary["splits"] = splits
    if not splits:
        return
    contrasts = [("C1 (primary)", "crossfit_affine_sel_nll", "fullval_uniform"),
                 ("C2", "fullval_affine_oof_sel_nll", "fullval_uniform"),
                 ("C3", "pool_linear_sel_nll", "fullval_uniform"),
                 ("C4", "pool_linear_sel_nll", "pool_affine_sel_nll"),
                 ("T1", "crossfit_postmean_thrwga", "crossfit_uniform"),
                 ("--", "crossfit_affine_sel_wga", "fullval_uniform"),
                 ("--", "crossfit_uniform", "fullval_uniform"),
                 ("--", "pool_uniform", "fullval_uniform"),
                 ("--", "fullval_postmean_thrwga", "fullval_uniform")]
    rows = ["| ID | Contrast | Clean, corrected val labels | Clean, official val labels (same splits) | Interaction (corrected − official) | Corrected test labels, corrected val |",
            "| --- | --- | --- | --- | --- | --- |"]
    for cid, arm, control in contrasts:
        (cn, pn) = contrast(get_new, new, arm, control, "removed", splits)
        (co, po) = contrast(get_old, old, arm, control, "removed", splits)
        inter = [a - b for a, b in zip(pn, po)]
        (cc, pc) = contrast(get_new, new, arm, control, "corrected", splits)
        rows.append(f"| {cid} | `{arm}` − `{control}` | {fmt(cn, pn)} | {fmt(co, po)} | {fmt(interval(inter), inter)} | {fmt(cc, pc)} |")
        summary.setdefault("contrasts", {})[f"{arm}-{control}"] = dict(
            id=cid, corrected_val=dict(mean=cn[0], lo=cn[1], hi=cn[2], per_source=pn),
            official_val=dict(mean=co[0], lo=co[1], hi=co[2], per_source=po),
            interaction=dict(zip(("mean", "lo", "hi"), interval(inter)), per_source=inter),
            corrected_test=dict(mean=cc[0], lo=cc[1], hi=cc[2], per_source=pc))
    lines += rows + [""]
    # Levels and per-group accuracies (clean subset)
    lines += ["### Clean-subset WGA and per-group accuracy (mean over splits, then backbones)", "",
              "| Val labels | Arm | WGA | Land/land | Land/water | Water/land | Water/water | worst group counts |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm in ("fullval_uniform", "crossfit_affine_sel_nll", "fullval_affine_oof_sel_nll", "pool_linear_sel_nll",
                "pool_affine_sel_nll", "crossfit_postmean_thrwga"):
        recs = [new[(s, sp)]["arms"][arm]["removed"] for s in STD for sp in splits]
        wga = st.mean(st.mean(new[(s, sp)]["arms"][arm]["removed"]["wga"] for sp in splits) for s in STD)
        groups = [st.mean(r["groups"][k] for r in recs) for k in ("00", "01", "10", "11")]
        worst = defaultdict(int)
        for r in recs:
            worst[min(r["groups"], key=r["groups"].get)] += 1
        lines.append(f"| corrected | `{arm}` | {wga:.2f} | " + " | ".join(f"{g:.2f}" for g in groups) +
                     f" | {dict(sorted(worst.items()))} |")
        summary.setdefault("levels", {})[arm] = dict(wga=wga, groups=groups, worst=dict(worst))
    for arm in ("fullval_uniform", "crossfit_affine_sel_nll"):
        wga = st.mean(st.mean(old[(s, sp)][arm]["removed"] for sp in splits) for s in STD)
        lines.append(f"| official | `{arm}` | {wga:.2f} | | | | | |")
    # Combined lever against the official-label DPE reference
    tot = [st.mean(get_new(new, s, sp, "crossfit_affine_sel_nll", "removed") - get_old(old, s, sp, "fullval_uniform", "removed")
                   for sp in splits) for s in STD]
    lab = [st.mean(get_new(new, s, sp, "fullval_uniform", "removed") - get_old(old, s, sp, "fullval_uniform", "removed")
                   for sp in splits) for s in STD]
    lines += ["", f"Label lever alone (full-val DPE, corrected − official val labels, same seeds): {fmt(interval(lab), lab)}",
              f"Both levers (cross-fit recalibration on corrected labels − full-val DPE on official labels): {fmt(interval(tot), tot)}", ""]
    summary["label_lever_nested_seeds"] = dict(zip(("mean", "lo", "hi"), interval(lab)), per_source=lab)
    summary["both_levers"] = dict(zip(("mean", "lo", "hi"), interval(tot)), per_source=tot)
    sel = defaultdict(int)
    for s in STD:
        for sp in splits:
            sel[new[(s, sp)]["arms"]["crossfit_affine_sel_nll"]["removed"].get("selected", "?")] += 1
    lines += [f"NLL-selected setting for cross-fit recalibration (corrected labels): {dict(sel)}", ""]


def analyze_b(randdrop, lines, summary):
    recs = [json.loads(p.read_text()) for p in sorted(Path(randdrop).glob("*/dpe-*.json"))]
    if not recs:
        return
    by = defaultdict(dict)
    for r in recs:
        by[(r["source"], r["seed_index"])][r["condition"]] = r["removed"]["wga"]
    ref = {}
    with LABEL_RUNS.open() as f:
        for r in csv.DictReader(f):
            if r["method"] == "dpe" and r["test_labels"] == "removed":
                ref[(r["source"], int(r["seed_index"]), r["condition"])] = float(r["wga"])
    seeds = sorted({k[1] for k in by})
    draws = sorted({c for v in by.values() for c in v if c.startswith("randdrop")})
    complete = all(len(by[(s, i)]) == 1 + len(draws) for s in STD for i in seeds)
    repro = max(abs(by[(s, i)]["drop"] - ref[(s, i, "drop")]) for s in STD for i in seeds if "drop" in by[(s, i)])
    lines += ["## Run B: matched random-drop control (full-validation DPE, clean subset)", "",
              f"Head seeds {seeds}; random draws {draws}; complete: {complete}. "
              f"`drop` reproduces label_correction_20261115 to within {repro:.4f} pp.", ""]
    def per_source(f):
        return [st.mean(f(s, i) for i in seeds) for s in STD]
    rnd = lambda s, i: st.mean(by[(s, i)][d] for d in draws)
    off = lambda s, i: ref[(s, i, "official")]
    rows = [("drop misannotated − official", per_source(lambda s, i: by[(s, i)]["drop"] - off(s, i))),
            ("drop random (mean of draws) − official", per_source(lambda s, i: rnd(s, i) - off(s, i))),
            ("**C5:** drop misannotated − drop random", per_source(lambda s, i: by[(s, i)]["drop"] - rnd(s, i)))]
    lines += ["| Contrast | Mean [95% interval], positive backbones |", "| --- | --- |"]
    for name, per in rows:
        lines.append(f"| {name} | {fmt(interval(per), per)} |")
        summary.setdefault("randdrop", {})[name] = dict(zip(("mean", "lo", "hi"), interval(per)), per_source=per)
    per_draw = {d: st.mean(st.mean(by[(s, i)][d] - off(s, i) for i in seeds) for s in STD) for d in draws}
    lines += ["", "Per draw, random drop − official: " + ", ".join(f"{d} {v:+.2f}" for d, v in per_draw.items()), ""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--randdrop", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lines, summary = ["# Combined levers and random-drop control: declared contrasts", "",
                      "Paired per backbone (five standard-ERM backbones), averaged over split/head seeds, "
                      "pointwise 95% t interval over backbones; differences in pp of test WGA. "
                      "Clean = 5,704 correctly labelled test images.", ""], {}
    analyze_a(a.run, lines, summary)
    analyze_b(a.randdrop, lines, summary)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
