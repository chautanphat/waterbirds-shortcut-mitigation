# Score-mechanism audit and signed piecewise lookup (v4)

Completed 2026-11-15. Primary R4, alpha=.25, beta=2, kappa=50: matched nested
mean WGA **−0.501253 pp**, AA **+0.166806 pp**. WGA improves in zero of three
seeds; every AA floor passes. **Promotion FAIL, no production integration.**

```powershell
.venv/Scripts/python.exe -m experiments.score_audit_v4.run --phase all --resume
.venv/Scripts/python.exe -m experiments.score_audit_v4.run --status
.venv/Scripts/python.exe -m scripts.report_score_audit_v4
```

The first command supports `--phase 0|1|2`, `--max-tasks N`, source/input/config
hash guards, atomic task commits, signal-safe boundaries, and automatic reuse.
Separate-process forced-kill recovery and full no-recompute resume passed.
Do not change guarded Python sources inside a completed run without preserving
the old manifest/source and recording an explicit revision. Missing/corrupt
committed files or mismatched manifests fail transparently.

Full report: `results/dpe/score_audit_v4_20261115/FINAL_REPORT.md`.
Checkpoints/state: `outputs/score_audit_v4/run1/`.
Compact deliverable: `outputs/score_audit_v4/DPE_WGA_v4_results.zip`.
The package includes fitted lookup/calibrator tables and per-image prediction
caches, with references/hashes for all 135 original nested head checkpoints.

Six preregistered maps were screened; R4/R3 advanced to the bounded grid.
Exactly one recipe was locked using class-only worst-class log loss and AA
floor before WGA evaluation. Phase 0 historical test rows are diagnostic-only.
Phase 1 and nested folds reuse previously inspected development data and share
one verified **ERM*** backbone, so these are not untouched confirmatory results
or standard-ERM results. All 135 A-only heads were verified and reused; all
temperatures, bins and lookup tables fit B only. Old LCMSW weights are unused.

The independent reporting script audits arithmetic, head identity, A/B/C IDs,
epoch checkpoints, subgroup flips and separates calibration from weighting.
Earlier reports, failure evidence and source/manifest revisions remain intact.
