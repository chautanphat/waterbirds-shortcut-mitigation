"""Declared analysis for the cross-dataset test (PLAN.md Sect. 5-7). Do not edit after the freeze.

    python summarize.py OUT [OUT ...] --report REPORT_DIR

Each OUT is one `xdpe.nested run` output folder (one dataset, or one dataset with the
validation set subsampled). For every OUT: per backbone, average the paired test-WGA
difference over split seeds (pairs share a split seed); then mean, pointwise 95% t interval
over backbones and the number of backbones with a positive difference. Official test labels.

Decision rule (PLAN Sect. 5): supported if mean > 0 and the interval excludes 0; otherwise
not supported. Fewer than 3 backbones: no interval, "not assessable".
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REFERENCE = "fullval_uniform"
CONTRASTS = [  # (id, arm, control, role)
    ("H1", "crossfit_affine_sel_nll", REFERENCE, "primary on CelebA: cross-fitted recalibration"),
    ("H2", "fullval_affine_oof_sel_nll", REFERENCE, "out-of-fold map applied to full-validation DPE"),
    ("H3", "pool_linear_sel_nll", REFERENCE, "linear residual on head disagreements"),
    ("H4", "pool_linear_sel_nll", "pool_affine_sel_nll", "head information beyond scalar recalibration"),
]
GROUP_ARMS = [REFERENCE, "crossfit_affine_sel_nll", "fullval_affine_oof_sel_nll", "pool_linear_sel_nll"]
EXPECTED_SPLITS = list(range(10))
PRIMARY = ("CelebA", "H1")


def t_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    if n < 3:
        return mean, float("nan"), float("nan")
    from scipy import stats
    half = stats.t.ppf(.975, n - 1) * values.std(ddof=1) / math.sqrt(n)
    return mean, mean - half, mean + half


def decision(mean, lo, hi):
    if math.isnan(lo):
        return "not assessable (fewer than 3 backbones)"
    return "supported" if mean > 0 and lo > 0 else "not supported"


def fmt(v, digits=2, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def load(out):
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    sources = json.loads((out / "sources.json").read_text())
    cells = [json.loads(p.read_text()) for p in sorted((out / "results").glob("*/split*.json"))]
    if any(c["protocol"] != "xdpe-strict-nested-postmean-v2" for c in cells):
        raise ValueError("Unexpected protocol in results")
    if protocol.get("smoke"):
        raise ValueError(f"{out} is a smoke run; it has no test scores")
    return protocol, sources, cells


def paired(cells, arm, control):
    """{source: mean over shared splits of (arm - control) WGA}."""
    by = {}
    for c in cells:
        if arm in c["arms"] and control in c["arms"]:
            by.setdefault(c["source"], []).append(c["arms"][arm]["wga"] - c["arms"][control]["wga"])
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def per_source_mean(cells, arm, key="wga"):
    by = {}
    for c in cells:
        if arm in c["arms"]:
            by.setdefault(c["source"], []).append(c["arms"][arm][key])
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def erm_reference(sources):
    """ERM backbone test WGA from each backbone manifest, if present."""
    values = {}
    for spec in sources["sources"]:
        try:
            m = json.loads(Path(spec["manifest"]).read_text())
            values[spec["id"]] = m["erm_metrics"]["test"]["wga"] * 100
        except (KeyError, OSError, ValueError):
            pass
    return values


def analyse(out):
    protocol, sources, cells = load(out)
    dataset = protocol["dataset"]
    subsample = sources["sources"][0].get("val_subsample") if sources["sources"] else None
    label = dataset + (f" (validation subsampled to {subsample})" if subsample else "")
    planned = [s["id"] for s in sources["sources"]]
    present = {(c["source"], c["split"]) for c in cells}
    missing = [(s, k) for s in planned for k in EXPECTED_SPLITS if (s, k) not in present]
    result = dict(out=str(out), dataset=dataset, label=label, backbones=planned, cells=len(cells),
                  missing=missing, contrasts=[], arms={}, groups={}, selection={})
    for cid, arm, control, role in CONTRASTS:
        d = paired(cells, arm, control)
        mean, lo, hi = t_interval(list(d.values()))
        result["contrasts"].append(dict(id=cid, arm=arm, control=control, role=role, mean=mean, lo=lo, hi=hi,
                                        positive=sum(v > 0 for v in d.values()), n=len(d), per_source=d,
                                        decision=decision(mean, lo, hi),
                                        primary=(dataset, cid) == PRIMARY and not subsample))
    arms = sorted({a for c in cells for a in c["arms"]})
    for arm in arms:
        wga = per_source_mean(cells, arm)
        oa = per_source_mean(cells, arm, "overall")
        d = paired(cells, arm, REFERENCE)
        mean, lo, hi = t_interval(list(d.values()))
        result["arms"][arm] = dict(wga=float(np.mean(list(wga.values()))), overall=float(np.mean(list(oa.values()))),
                                   delta=mean, lo=lo, hi=hi, positive=sum(v > 0 for v in d.values()), n=len(d))
    for arm in GROUP_ARMS:
        rows = [c["arms"][arm] for c in cells if arm in c["arms"]]
        if rows:
            names = sorted(rows[0]["groups"])
            acc = {g: float(np.mean([r["groups"][g] for r in rows])) for g in names}
            counts = rows[0].get("counts", {})
            result["groups"][arm] = dict(accuracy=acc, counts=counts, worst=min(acc, key=acc.get))
    for arm in arms:
        picks = [c["arms"][arm]["selected"] for c in cells if "selected" in c["arms"][arm] and "_sel_" in arm]
        if picks:
            result["selection"][arm] = {k: picks.count(k) for k in sorted(set(picks))}
    sds = []
    for s in planned:
        v = [c["arms"][REFERENCE]["wga"] for c in cells if c["source"] == s and REFERENCE in c["arms"]]
        if len(v) > 1:
            sds.append(float(np.std(v, ddof=1)))
    result["reference_within_backbone_sd"] = float(np.mean(sds)) if sds else float("nan")
    result["erm_wga"] = erm_reference(sources)
    return result


def render(results, script_sha):
    lines = ["# Cross-dataset test of DPE post-hoc procedures", "",
             f"Analysis script SHA-256: `{script_sha}`. Official test labels. Differences in percentage points (pp) "
             "against full-validation DPE retrained with the same seeds; mean over split seeds within a backbone, then "
             "mean and pointwise 95% t interval over backbones. Decision: supported if mean > 0 and the interval excludes 0.", ""]
    for r in results:
        lines += [f"## {r['label']}", "",
                  f"Backbones: {len(r['backbones'])}; cells: {r['cells']}/{len(r['backbones']) * len(EXPECTED_SPLITS)}.",
                  ("**PARTIAL RUN: missing cells " + str(r["missing"]) + ". Do not report these numbers as the declared experiment.**")
                  if r["missing"] else "All declared cells complete.", ""]
        rows = [[c["id"] + (" (primary)" if c["primary"] else ""), f"`{c['arm']}` - `{c['control']}`", fmt(c["mean"], 2, True),
                 f"[{fmt(c['lo'], 2, True)}, {fmt(c['hi'], 2, True)}]", f"{c['positive']}/{c['n']}", c["decision"],
                 ", ".join(fmt(v, 2, True) for v in c["per_source"].values())] for c in r["contrasts"]]
        lines += [table(["contrast", "arms", "mean", "95% interval", "backbones > 0", "decision", "per backbone"], rows), ""]
        ref = r["arms"].get(REFERENCE, {})
        erm = list(r["erm_wga"].values())
        lines += [f"Full-validation DPE: WGA {fmt(ref.get('wga'))}%, overall {fmt(ref.get('overall'))}%; "
                  f"within-backbone SD over split seeds {fmt(r['reference_within_backbone_sd'])} pp; "
                  f"ERM backbone WGA {fmt(float(np.mean(erm))) if erm else 'n/a'}%.", ""]
        if r["groups"]:
            names = sorted(next(iter(r["groups"].values()))["accuracy"])
            counts = next(iter(r["groups"].values()))["counts"]
            rows = [[f"`{arm}`", *[fmt(v["accuracy"][g]) for g in names], v["worst"]] for arm, v in r["groups"].items()]
            lines += ["Per-group test accuracy (%), mean over backbones and split seeds:", "",
                      table(["arm", *[f"{g} (n={counts.get(g, '?')})" for g in names], "worst group"], rows), ""]
        rows = [[f"`{a}`", fmt(v["wga"]), fmt(v["overall"]), fmt(v["delta"], 2, True),
                 f"[{fmt(v['lo'], 2, True)}, {fmt(v['hi'], 2, True)}]", f"{v['positive']}/{v['n']}"]
                for a, v in r["arms"].items()]
        lines += ["All arms (supplementary):", "",
                  table(["arm", "WGA", "overall", "vs full-val DPE", "95% interval", "backbones > 0"], rows), ""]
        if r["selection"]:
            rows = [[f"`{a}`", ", ".join(f"{k}: {v}" for k, v in s.items())] for a, s in r["selection"].items()]
            lines += ["Settings chosen on validation data (cells):", "", table(["arm", "picks"], rows), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outs", nargs="+", type=Path)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = [analyse(o) for o in a.outs]
    a.report.mkdir(parents=True, exist_ok=True)
    text = render(results, script_sha)
    (a.report / "REPORT.md").write_text(text, encoding="utf-8")
    (a.report / "summary.json").write_text(json.dumps(dict(script_sha256=script_sha, results=results), indent=1,
                                                      default=float), encoding="utf-8")
    with (a.report / "contrasts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label", "contrast", "arm", "control", "mean", "lo", "hi", "positive", "n", "decision"])
        for r in results:
            for c in r["contrasts"]:
                w.writerow([r["dataset"], r["label"], c["id"], c["arm"], c["control"], c["mean"], c["lo"], c["hi"],
                            c["positive"], c["n"], c["decision"]])
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
