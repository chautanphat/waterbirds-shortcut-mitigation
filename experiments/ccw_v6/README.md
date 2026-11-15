# CCW v6 — B-only NO-GO; no production integration

Completed feasible work on2026-11-15. The frozen primary CCW_FULL fails the
prespecified group-blind B selector against matched G1:

| Method | Mean worst-class NLL on B OOF (lower is better) |
|---|---:|
| G1_frozen | **0.62063432** |
| CCW_FULL | 0.62518272 |
| CCW_A | 0.62520816 |
| G2_z | 0.63253482 |
| U0 | 0.63601003 |

CCW_FULL meets the strict AA floor on all seeds (delta versus U0:
−0.326264/−0.161551/0.000000 pp), but loses the fixed robustness proxy to G1.
No novel method was locked. Under the preregistered stop rule, no full-B
deployment refit or new CCW C replay was run. CCW WGA improvement is **not
measured or confirmed**. The old C CSV contains only reproduced U0/G1 controls.

```powershell
.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase all --resume
.venv/Scripts/python.exe -m experiments.ccw_v6.run --status
.venv/Scripts/python.exe -m experiments.ccw_v6.run --verify
.venv/Scripts/python.exe -m scripts.finalize_ccw_v6
```

`--phase0|1|2` selects input verification, B fitting/locking, or final diagnostic
and confirmation disposition (use a space, e.g. `--phase 1`). `--max-tasks N`
stops at a durable boundary. Source/input/config fingerprints reject incompatible
reuse; successful receipts validate every artifact's SHA/size. Model/table fits
are deterministic closed-form statistics; no DPE training epoch is required.

Nine matching B caches (240 images each) were verified against exact v5 head
order/ABC manifests and exported without group fields. Four inner folds fit all
normalizers, quantiles, q20, class/boundary weights, relative-advantage tables and
sigma_A on B-inner-train only. Thirty-six fits and180 validation caches cover
the five fixed variants. First two real recovery fits were reused. Seed pooling
averages repeated B IDs before AA/uncertainty:613/619/615 unique images rather
than720 independent rows. The three head seeds share one ERM* backbone.

Nine tests passed. Synthetic and actual-B workers were killed mid-inner-fold
after a completed fold; restart matched uninterrupted models/predictions and
preserved the completed fold. Full separate-process resume recomputed zero
experiments. Independent audit verified all36 fitted normalizers/target tables,
180 prediction caches, train/validation separation and unchanged v5 production
source hashes. All496 task receipts were subsequently verified.

Source r2 fixes only missing-fresh-data report serialization; r1 manifests,
sources, completed scientific results and failure logs remain intact. Atomic
fsync/replace retries handle transient file locks. OS locking rejects concurrent
writers. Original v1-v5 results and Uniform/G1 inference remain unchanged.

Current report: `reports/FINAL_REPORT_VERIFIED.md` (earlier reports are retained).
Lightweight export: `results/dpe/ccw_v6_20261115/FINAL_REPORT.md`.
Result package: `outputs/ccw_v6/DPE_CCW_v6_results.zip`.
All fitted tables and detailed arrays remain under `checkpoints/` and `results/`.

**CONFIRMATION BLOCKED — NO FRESH DATA.** Read `INPUT_REQUEST.md` for exact fresh
cohort/head/cache/annotation provenance requirements. A fresh file cannot itself
prove freshness or override the B NO-GO. A modified proposal requires a separate
preregistered study; never rescue this result by tuning on old C/test labels.

## Separately requested exploratory C diagnostic

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

The user subsequently authorized a separate measurement of the five frozen
methods on all nine existing C caches despite the B NO_GO. The original
workflow and its decision remain unchanged. `--phase all` does not opt in.

```powershell
.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase exploratory_c --resume
.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase exploratory_c --resume --verify
```

Outputs and independent recovery state live in `exploratory_c/`. Read
`exploratory_c/EXPLORATORY_C_REPORT.md`; the five requested deliverables and
`DPE_CCW_v6_Exploratory_C.zip` are in that directory. `--max-tasks N` stops at
a durable boundary. Source revision r3 adds only this explicit CLI route;
the frozen model mathematics, original protocol, selection and scientific
artifacts are unchanged. Historical reports above describe the original study.

| Method | Mean AA (%) | Mean WGA (%) |
|---|---:|---:|
| Uniform | 95.746455 | 92.481203 |
| G1 | 95.857659 | 92.982456 |
| G2 | 95.829858 | 92.982456 |
| CCW-A | 95.802057 | 92.982456 |
| CCW-Full | 95.829858 | 92.982456 |

CCW-Full improves observed WGA over Uniform by 0.501253 pp and ties G1 in
every seed. Both per-seed AA constraints pass. The gain over Uniform includes
one net correction in worst group 2 in each of seeds 1 and 2; seed 0 has one
correction and one breakage. There are no extra worst-group corrections versus
G1. The advantage term changes weights (mean total variation 0.00688089), but
changes only one classification versus CCW-A and adds no WGA gain. Paired
image-bootstrap uncertainty spans zero for the WGA differences. All three head
seeds reuse 1,199 images and one frozen ERM* backbone; this is not an ERM-only
result or independent confirmation.

Twelve tests passed. Real full-B forced-kill recovery and full zero-task resume
passed; all 94 diagnostic/delivery receipts and 1,586 protected original files
were verified. No retraining, parameter search, promotion or production
inference changes were performed.
