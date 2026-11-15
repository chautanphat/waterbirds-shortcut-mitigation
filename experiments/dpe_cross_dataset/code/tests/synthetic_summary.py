"""summarize.py on synthetic results with known answers (PLAN Sect. 8.2).

    python tests/synthetic_summary.py WORK_DIR
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
GROUPS = ["00", "01", "10", "11"]


def arm(wga, **extra):
    return dict(wga=wga, overall=wga + 5, groups={g: wga + i for i, g in enumerate(GROUPS)},
                counts={g: 100 for g in GROUPS}, **extra)


def build(out, missing_one=False):
    out.mkdir(parents=True)
    (out / "protocol.json").write_text(json.dumps(dict(protocol="xdpe-strict-nested-postmean-v2", dataset="CelebA", smoke=False)))
    sources = []
    for s in range(5):
        manifest = out / f"bb{s}" / "backbone.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps(dict(erm_metrics=dict(test=dict(wga=0.40 + 0.01 * s)))))
        sources.append(dict(id=f"CelebA-seed{s}", seed=s, manifest=str(manifest)))
    (out / "sources.json").write_text(json.dumps(dict(dataset="CelebA", sources=sources)))
    rng = np.random.RandomState(0)
    h3_offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]  # mean 0 -> not supported
    for s in range(5):
        for k in range(10):
            if missing_one and (s, k) == (4, 9):
                continue
            base = 85 + rng.normal()
            arms = dict(fullval_uniform=arm(base),
                        crossfit_affine_sel_nll=arm(base + 1.0, selected="average-l0.01"),  # +1 everywhere -> supported
                        fullval_affine_oof_sel_nll=arm(base - 0.2, selected="average-l0.1"),  # -0.2 -> not supported
                        pool_linear_sel_nll=arm(base + h3_offsets[s], selected="maximum-l0.1"),
                        pool_affine_sel_nll=arm(base + h3_offsets[s] - 0.3, selected="average-l0.1"))  # H4 = +0.3
            path = out / "results" / f"CelebA-seed{s}" / f"split{k:02d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(protocol="xdpe-strict-nested-postmean-v2", source=f"CelebA-seed{s}", split=k, arms=arms)))


def main():
    work = Path(sys.argv[1]).resolve()
    if work.exists():
        raise SystemExit("use a fresh directory")
    full, partial = work / "full", work / "partial"
    build(full)
    build(partial, missing_one=True)
    subprocess.run([sys.executable, str(CODE / "summarize.py"), str(full), str(partial), "--report", str(work / "report")],
                   check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((work / "report/summary.json").read_text())
    full_r, partial_r = summary["results"]
    c = {x["id"]: x for x in full_r["contrasts"]}
    checks = [
        ("H1 mean +1", abs(c["H1"]["mean"] - 1.0) < 1e-9),
        ("H1 supported, primary", c["H1"]["decision"] == "supported" and c["H1"]["primary"]),
        ("H1 5/5 positive", c["H1"]["positive"] == 5),
        ("H2 mean -0.2, not supported", abs(c["H2"]["mean"] + 0.2) < 1e-9 and c["H2"]["decision"] == "not supported"),
        ("H3 mean 0, not supported", abs(c["H3"]["mean"]) < 1e-9 and c["H3"]["decision"] == "not supported"),
        ("H3 interval symmetric", abs(c["H3"]["lo"] + c["H3"]["hi"]) < 1e-9 and c["H3"]["hi"] > 0),
        ("H3 2/5 positive", c["H3"]["positive"] == 2),
        ("H4 mean +0.3 supported", abs(c["H4"]["mean"] - 0.3) < 1e-9 and c["H4"]["decision"] == "supported"),
        ("ERM WGA read", abs(np.mean(list(full_r["erm_wga"].values())) - 42.0) < 1e-9),
        ("full run complete", not full_r["missing"]),
        ("partial run flagged", partial_r["missing"] == [["CelebA-seed4", 9]] or partial_r["missing"] == [("CelebA-seed4", 9)]),
        ("report warns on partial", "PARTIAL RUN" in (work / "report/REPORT.md").read_text()),
    ]
    for label, ok in checks:
        print("ok  " if ok else "FAIL", label)
    if not all(ok for _, ok in checks):
        return 1
    print("summarize.py: all synthetic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
