# Strictly nested rerun of validation-only selection (Kaggle, 2026-11-15)

Run: public Kaggle notebook `noerula/dpe-test` (CPU, 3 h 01 min, version 1), built from
[`experiments/dpe_nested_corrected`](../../../experiments/dpe_nested_corrected/README.md)
(protocol `strict-nested-postmean-v2`, embedded bundle SHA-256 `d5f0f59e...f3bd`). Files here
are the notebook's `corrected_nested_results.zip`, unmodified. It fixes the issues found in the
audit of [`../nested_selection_20261115`](../nested_selection_20261115/README.md)
([audit](../../../docs/dpe_nested_audit_and_rerun.md)): every head and calibrator is rebuilt
inside each inner training fold, the inner CV scores the full prediction procedure used on
test, and WGA thresholds are searched exactly after averaging. Same 7 sources, 10 split seeds,
4 calibration settings and selection rules (primary: balanced NLL; secondary: WGA).

Checks: 10/10 self-tests passed; saved-ensemble regression check passed for all 7 sources
(`check.json`); 70/70 cells completed (`completion.json`). Full tables: [`REPORT.md`](REPORT.md),
[`CORRECTED_DIAGNOSTICS.md`](CORRECTED_DIAGNOSTICS.md); every arm: `results_long.csv`,
`results/<source>/splitNN.json`.

## Main results (standard ERM, n=5; pp against same-seed full-validation DPE, 89.88%)

| Arm | Mean | 95% interval | Sources > 0 |
| --- | ---: | --- | ---: |
| Cross-fit affine, selected by NLL (primary) | -0.17 | [-0.33, -0.02] | 0/5 |
| Cross-fit affine, selected by WGA | -0.08 | [-0.23, +0.07] | 1/5 |
| Cross-fit uniform | +0.13 | [-0.28, +0.53] | 4/5 |
| Cross-fit + exact WGA threshold after averaging | -0.50 | [-1.11, +0.11] | 1/5 |
| Full-validation DPE + exact WGA threshold | -0.31 | [-0.59, -0.03] | 0/5 |
| 720/479 affine, selected by NLL | -0.05 | [-0.17, +0.06] | 2/5 |
| 720/479 linear residual, selected by NLL | -0.86 | [-1.32, -0.39] | 0/5 |
| DFR-lite (standardiser fitted inside CV) | -1.73 | [-2.27, -1.19] | 0/5 |

Reading:

* The primary contrast is identical to the leaky original run (-0.172 pp): the NLL rule picks
  average/0.01 in all 50 standard-ERM cells under both protocols, so the deployed predictors
  are the same. The audit's nesting issue did not change the conclusion.
* Exact WGA thresholds raise fitted-validation WGA (by 0.63 pp on average over the coarse
  grid) but lower test WGA: -0.63 pp [-0.98, -0.28] against cross-fit uniform.
* DFR-lite moves from -1.51 to -1.73 pp once its preprocessing stays inside the folds.

Interval and table numbers above were recomputed from `results/*/split*.json` by
`report/ACIIDS_2027/build_levers.py` and an inline check. The test set and backbones were
inspected before this run, so it remains exploratory.
