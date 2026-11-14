# Nested validation-only selection for cross-fitted DPE (Kaggle, 2026-11-15)

Run: public Kaggle notebook `noerula/dpe-nested-code`, CPU, saved features and heads only
(no backbone training). Code as executed: [`code/nested_selection.py`](code/nested_selection.py)
(its docstring declares the protocol before the run). Full report: [`REPORT.md`](REPORT.md);
every arm, source and split: [`results_long.csv`](results_long.csv); per-cell records:
`results/<source>/splitNN.json`; saved-ensemble regression check: [`check.json`](check.json) (ok).

## Design

* 7 sources (5 standard ERM, 2 ERM*) x 10 split seeds = 70 cells, all completed.
* Split seed `s` sets the cross-fit fold split, the 720/479 partition and the head-training
  seeds (`seed + 1000*s`). Full-validation DPE is retrained with the same seeds in every cell,
  so comparisons are paired per cell. Split seed 0 reproduces the earlier saved cross-fit run
  to 0.00 pp in all 35 compared rows.
* Calibration settings: (average|maximum group loss) x (lambda 0.1|0.01). Selection rule
  (primary): inner 5-fold CV **on the out-of-fold scores** (heads are not refitted inside the
  inner CV), lowest group-balanced NLL; secondary: highest inner WGA.
* Controls: cross-fit uniform; 75-head bag (five full-validation ensembles); offset-only
  (`crossfit_threshold_*`) and transferred offset/affine on full-validation DPE
  (`fullval_*_oof_*`); pooled 720/479 uniform, affine and linear residual; DFR-lite
  (group-weighted L1 logistic regression on standardised raw features, C chosen by
  validation CV).
* Intervals: pointwise 95% t over sources of the per-source mean over split seeds (n=5);
  ERM* (n=2) is descriptive.

## Main results (standard ERM, n=5; differences in pp against same-seed full-validation DPE)

| Comparison | Mean | 95% interval | Sources > 0 |
| --- | ---: | --- | ---: |
| Cross-fit affine, selected by NLL (primary) | -0.17 | [-0.33, -0.02] | 0/5 |
| Cross-fit affine, selected by WGA | -0.08 | [-0.17, +0.01] | 1/5 |
| Cross-fit affine, fixed average/0.1 | +0.17 | [-0.01, +0.34] | 4/5 |
| Cross-fit uniform | +0.13 | [-0.28, +0.53] | 4/5 |
| Transferred offset on full-validation DPE (NLL-selected) | -0.57 | [-0.76, -0.39] | 0/5 |
| Pooled linear residual, NLL-selected | -0.86 | [-1.32, -0.39] | 0/5 |
| DFR-lite | -1.51 | [-1.83, -1.20] | 0/5 |

Mean full-validation DPE over split seeds: 89.88% (standard ERM), 93.88% (ERM*). ERM*
differences are all within about +/-0.3 pp except DFR-lite (-1.28 pp).

Reading:

* Under validation-only selection no cross-fit or recalibration variant beats
  full-validation DPE. The earlier test-highlighted +0.47 pp (split seed 0) shrinks to
  +0.17 pp averaged over ten split seeds; split seed 0 had a below-average full-validation
  draw (89.60 vs 89.88).
* NLL selection chose average/0.01 in all 70 cells; that setting is weaker on test.
* Cross-fit affine and cross-fit offset-only give the same mean WGA at every setting: the
  affine map acts as a threshold shift.
* Cross-fit uniform is level with a 75-head bag (+0.07 pp, [-0.32, +0.46]).
* Head-seed noise: SD of full-validation DPE over split seeds is about 0.4 pp per source.

## Caveats

* The direct WGA-threshold arms (`crossfit_thrwga`, `fullval_thrwga_oof`) search offsets on a
  0.05 grid. DPE scores lie within about +/-0.1 logit of the boundary, so these arms always
  chose 0 and equal the uncalibrated arms. See
  [`../threshold_controls_20261115`](../threshold_controls_20261115/README.md) for an exact
  threshold search.
* The test set and the seven backbones were inspected before this run.
