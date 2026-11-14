# Threshold and offset controls for cross-fitted DPE (local CPU, 2026-11-15)

Code: [`src/methods/dpe_independent/controls.py`](../../../src/methods/dpe_independent/controls.py)
(unit tests in `tests/methods/dpe_independent/test_controls.py`) and
[`scripts/analysis/dpe_threshold_controls.py`](../../../scripts/analysis/dpe_threshold_controls.py).
Inputs: features and heads recovered with `python -m src.methods.dpe_independent recover`
from the public notebooks `noerula/dpe-exploration` and `noerula/dpe-mlp-dropout-verification`
(kept under the gitignored `outputs/`).

Same fold split (seed 0), fold ensembles and four (loss, lambda) settings as
[`../independent_methods_20261113`](../independent_methods_20261113/README.md). All 42 historical
WGA values (full-validation DPE, uncalibrated fold ensemble, four affine settings, 7 sources)
were reproduced exactly. Fixed settings; nothing selected on test.

Controls:

* `fold_offset_*`: a = 1, b fitted with the affine objective (same group loss, penalty
  lambda*b^2, 400 Adam steps), applied to each fold ensemble before averaging.
* `fold_threshold_wga`: threshold on the fold-average score maximising out-of-fold WGA
  (candidates: 0 and midpoints between consecutive scores; ties to smallest |t|).
* `full_offset_*`, `full_threshold_wga`: the same offset or threshold transferred to
  full-validation DPE.

Standard ERM (n=5), average loss, lambda=0.1, differences against full-validation DPE:

| Procedure | WGA | Diff (pp) | 95% interval |
| --- | ---: | ---: | --- |
| Full-validation DPE | 89.60 | | |
| Fold ensemble, uniform | 90.14 | +0.54 | [-0.37, +1.46] |
| Fold ensemble, affine | 90.06 | +0.47 | [+0.16, +0.77] |
| Fold ensemble, offset only | 90.06 | +0.47 | [+0.16, +0.77] |
| Fold ensemble, WGA threshold | 89.06 | -0.53 | [-1.47, +0.41] |
| Full DPE, transferred offset | 89.16 | -0.44 | [-0.65, -0.22] |
| Full DPE, transferred WGA threshold | 89.34 | -0.25 | [-1.16, +0.65] |

Affine and offset-only have identical overall and per-group accuracy on all seven sources at
this setting. Fitted slopes are 1.20-1.31 and offsets |b| <= 0.011, while the median absolute
test logit of full-validation DPE is 0.10-0.16, so the map shifts the boundary by a few
thousandths. Full tables (both recipes, all settings): [`REPORT.md`](REPORT.md),
[`per_source.csv`](per_source.csv), [`summary.json`](summary.json).

This is split seed 0 only; the ten-seed picture is in
[`../nested_selection_20261115`](../nested_selection_20261115/README.md).
