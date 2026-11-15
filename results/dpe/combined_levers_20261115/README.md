# Combined levers and random-drop control (local CPU, 2026-11-15)

Plan, declared before scoring: [`experiments/dpe_combined_levers/PLAN.md`](../../../experiments/dpe_combined_levers/PLAN.md)
(commit `c8bf571`). Code: [`combined_levers.py`](../../../experiments/dpe_combined_levers/combined_levers.py)
(wrapper around the frozen strict nested pipeline, protocol `strict-nested-postmean-v2`),
[`check_modes.py`](../../../experiments/dpe_combined_levers/check_modes.py),
[`analyze.py`](../../../experiments/dpe_combined_levers/analyze.py), and the `randdropK` conditions of
[`scripts/analysis/dpe_label_correction.py`](../../../scripts/analysis/dpe_label_correction.py).
Full tables: [`REPORT.md`](REPORT.md); numbers: [`summary.json`](summary.json).

**Status: exploratory.** The test set and backbones were inspected in earlier stages.

## What was run

- **Run A.** The strictly nested pipeline (all heads, calibrators and selections) refitted with the
  18 misannotated validation images relabelled landbird. Five standard-ERM backbones × 10 split
  seeds = 50 cells, all completed (`cells/relabel/`). Counterpart on official validation labels:
  `../strict_nested_20261115`, re-scored in `../corrected_test_rescore_20261115/per_arm.csv`.
- **Run B.** Full-validation DPE (5 head seeds) after dropping the 18 misannotated images (`drop`)
  or 18 random correctly labelled images from the same official groups (11 waterbird-on-land,
  7 waterbird-on-water; five draws, `randdrop0..4`). 150 fits (`randdrop/`).
- Environment: Windows, 16-core CPU, Python 3.11.1, torch 2.5.1+cpu, numpy 2.2.1, scikit-learn
  1.6.1. Features recovered from the public Kaggle outputs with
  `python -m src.methods.dpe_independent recover` (no backbone retraining).

## Checks

- `nested_selection.py check`: all 7 saved ensembles reproduce their reference test WGA (delta 0.000).
- `check_modes.py`: official mode leaves the source unchanged; relabel changes exactly 18 validation
  labels; drop removes exactly those 18 rows with features aligned (standard ERM and ERM*).
- The wrapper in official mode reproduces the saved strict nested cell `normal-erm-seed0` split 0
  exactly: all 28 arms, official and clean views, and selected settings (`cells/official_regression/`).
- `drop` in run B reproduces `../label_correction_20261115` exactly.

## Results (clean subset, 5,704 images; pp; 95% interval over five backbones)

| Contrast | Official val labels | Corrected val labels |
| --- | --- | --- |
| **C1** cross-fit recalibration (NLL) − full-val DPE | +1.11 [+0.67, +1.56], 5/5 | **−1.10 [−1.48, −0.71], 0/5** |
| C2 full-val DPE + out-of-fold map − full-val DPE | +1.39 [+1.13, +1.65], 5/5 | −0.63 [−0.98, −0.28], 0/5 |
| C3 linear residual (720/479) − full-val DPE | +1.74 [+1.45, +2.03], 5/5 | −0.77 [−1.02, −0.53], 0/5 |
| C4 linear residual − affine map (same heads) | +1.16 [+0.79, +1.54], 5/5 | +0.83 [+0.65, +1.01], 5/5 |
| T1 cross-fit exact WGA threshold − cross-fit uniform | −0.79 [−1.13, −0.45], 0/5 | +0.96 [+0.50, +1.43], 5/5 |
| Full-val DPE exact WGA threshold − full-val DPE | −0.73 [−1.08, −0.37], 0/5 | +0.82 [+0.44, +1.20], 5/5 |

- Label lever alone, same seeds (full-val DPE, corrected − official val labels): +2.14 [+1.62, +2.67], 5/5.
- Both levers (C1 arm on corrected labels − full-val DPE on official labels): +1.04 [+0.51, +1.58], 5/5,
  i.e. less than the label correction alone.
- Clean-subset WGA on corrected labels: full-val DPE 92.70, cross-fit recalibration 91.60,
  out-of-fold map 92.07, linear residual 91.93, cross-fit exact threshold 93.24. The worst group stays
  landbird-on-water in 47–50 of 50 cells for every arm except the linear residual (40/50).
- **Run B, C5:** drop misannotated − drop random = +2.28 [+1.85, +2.72], 5/5. Random drop − official
  = −0.01 [−0.58, +0.56] (per draw −0.07 to +0.11).

## Reading

- Declared rule: C1–C3 are **not supported** on corrected validation labels; all three lower WGA on
  every backbone. The declared prediction (the levers overlap, I1–I3 < 0) holds, more strongly than
  predicted: the post-hoc gains do not just shrink, they reverse.
- So the post-hoc gains reported in the paper exist only while the validation labels contain the
  error. One reading: on official validation labels the recalibration partly undoes the bias
  that the 18 wrong labels put into the heads; once the labels are corrected there is no such bias
  and the likelihood-fitted shift moves the boundary away from the worst group. Not tested here.
- Head disagreements still carry information beyond scalar recalibration (C4 positive, 5/5).
- The exact WGA threshold, which hurts on official validation labels, helps on corrected ones
  (T1 +0.96). This supports the paper's explanation that the threshold search tunes to the wrong
  worst group when the validation labels contain the error.
- Removing the 18 misannotated images helps because they are wrong, not because the validation set
  is smaller (C5).
- ERM* was not run.
