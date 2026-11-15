# Strictly nested run re-scored on corrected and clean test labels (local CPU, 2026-11-15)

Code: [`scripts/analysis/dpe_rescore_corrected_test.py`](../../../scripts/analysis/dpe_rescore_corrected_test.py)
(every arm, three test views) and [`scripts/analysis/dpe_rescore_groups.py`](../../../scripts/analysis/dpe_rescore_groups.py)
(per-group accuracies of key arms). Input: `corrected_nested_resume.zip` from Kaggle `noerula/dpe-test`
(the checkpoints of [`../strict_nested_20261115`](../strict_nested_20261115/README.md); 116 MB, not tracked).

## What was done

Each arm's test predictions were rebuilt from the run's own checkpoints: the test probabilities of every
cross-fit fold, full-refit, bag and 720/479 head ensemble, and the calibration maps, chosen settings and
thresholds recorded in `fit_complete.json` before test scoring. The 720/479 combiners were refitted
deterministically from the saved head scores. Nothing was reselected. Every rebuilt arm reproduces its
recorded official-label WGA exactly: 0 mismatches over 70 cells x 27 arms (DFR-lite, which needs raw
features, is excluded). Validation labels and fits stay official; only the test labels change.

Three views of the 5,794 test images:

* `official`: released labels;
* `corrected`: the 90 images of 021 Eastern Towhee, 088 Western Meadowlark and 102 Western Wood-Pewee
  relabelled landbird (group follows);
* `removed` (the "clean subset" in the paper): those 90 images dropped, 5,704 left, no label changed.

## Main results (standard ERM, n=5, pp against same-seed full-validation DPE)

| Arm | Official | Corrected | Clean | 95% interval (clean) |
| --- | ---: | ---: | ---: | --- |
| Cross-fit + affine, selected (NLL) | -0.17 | +1.21 | +1.11 | [+0.67, +1.56], 5/5 |
| Full DPE + out-of-fold map, selected (NLL) | -0.58 | +1.44 | +1.39 | [+1.13, +1.65], 5/5 |
| 720/479 linear residual, selected (NLL) | -0.86 | +1.75 | +1.74 | [+1.45, +2.03], 5/5 |
| 720/479 uniform | -0.37 | -0.87 | -0.90 | [-1.49, -0.32], 0/5 |
| Cross-fit, exact WGA threshold | -0.50 | -1.10 | -1.14 | [-1.81, -0.46], 0/5 |

Full-validation DPE: 89.88 (official), 90.49 (corrected), 90.56 (clean). On the clean subset its worst
group is landbird-on-water (90.56%); waterbird-on-land is 95.54% (90.00% on official labels). The post-hoc
procedures raise landbird-on-water by 1.1-2.2 pp (`per_group.csv`). Linear residual minus affine on the
same heads, clean subset: +1.16 [+0.79, +1.54] (selected), +0.50 [+0.27, +0.73] (fixed).
Cross-fit affine minus offset-only, selected, clean: -0.07 [-0.16, +0.02].

All arms, both recipes and all three views: `summary.json`, `per_arm.csv`. Fig. 1 of the ACIIDS draft is
built from these files by [`scripts/analysis/build_aciids_levers.py`](../../../scripts/analysis/build_aciids_levers.py)
(audit: `levers_audit.json`).

## Caveats

This re-scoring was decided after the nested run had been scored on official labels; it changes no fit or
selection. The candidate procedures came from exploratory work in which the test set was inspected.
ERM* gains on the clean subset are smaller (+0.88, +0.73, +0.08 pp for the three selected arms).
