# Waterbirds species-label error and validation-label methods (local CPU, 2026-11-15)

Code: [`scripts/analysis/dpe_label_correction.py`](../../../scripts/analysis/dpe_label_correction.py)
(`run`, `summarize`, `rescore`). Inputs: the recovered features of the 7 sources (see
[`../threshold_controls_20261115`](../threshold_controls_20261115/README.md)).

## The label error and prior reports

`metadata.csv` labels every image of 021 Eastern Towhee, 088 Western Meadowlark and 102 Western
Wood-Pewee as waterbird. The generator (`group_DRO/dataset_scripts/generate_waterbirds.py`)
marks a species as waterbird if its lower-cased name contains a keyword from a seabird/waterfowl
list; `tern` occurs inside `eastern` and `western`. A pass over all 46 waterbird-labelled
species finds no other case (Western Grebe and Western Gull also contain `tern` but are
waterbirds through `grebe`/`gull`). Affected images: 72 train, 18 validation (11 of 133
waterbird-on-land, 7 of 133 waterbird-on-water), 90 test (43 of 642 waterbird-on-land, 47 of 642
waterbird-on-water).

The error is already known: the group_DRO README lists the three species ("doesn't affect the
validity of the dataset"); MaskTune (Asgari Taghanaki et al., NeurIPS 2022) corrected them,
attributing the error to similar species names, together with a multi-bird background fix,
and reported ERM WGA rising from 60 to 80.8 after both fixes; a 2025 paper (arXiv:2510.02818)
evaluates Group DRO on corrected labels. What this folder adds is the effect on methods that
fit on the *validation* group labels (DPE, DFR), which those works do not measure.

## Design

Full-validation DPE (5 head seeds per source, seeds `source + 1000*s`, s = 0..4) and DFR-lite
(group-weighted L1 logistic regression, C by 5-fold validation WGA) are refitted under three
validation label sets: `official`; `relabel` (the 18 images relabelled landbird, group follows);
`drop` (the 18 images removed, 1,181 left). Every model is scored on official test labels,
corrected test labels (the 90 images relabelled landbird) and with the 90 images removed.
`official`, s=0 reproduces the saved full-validation DPE exactly (normal-erm-seed0: 89.56).
126 runs, all completed. Full table: [`REPORT.md`](REPORT.md); every run: [`per_run.csv`](per_run.csv).

## Results (standard ERM, n=5; paired differences against the official-label fit)

| Method | Fit labels | Test official | Test corrected | Diff on corrected test (pp) |
| --- | --- | ---: | ---: | --- |
| DPE | official | 89.89 | 90.34 | |
| DPE | relabel | 89.07 | 92.81 | +2.48 [+1.80, +3.15], 5/5 positive |
| DPE | drop | 89.07 | 92.79 | +2.46 [+1.90, +3.01], 5/5 positive |
| DFR-lite | official | 88.62 | 88.97 | |
| DFR-lite | relabel | 86.70 | 90.53 | +1.56 [+1.19, +1.94], 5/5 positive |

ERM* (n=2, descriptive): DPE +2.12 (relabel) and +2.08 (drop) pp on corrected test; DFR-lite
-0.04 and -0.02 pp.

Reading:

* Fixing 18 of 1,199 validation labels improves DPE's corrected-test WGA by about 2.5 pp on
  every standard-ERM backbone, more than any aggregation effect in this project (including
  test-selected settings) and about six times the head-seed SD (about 0.4 pp).
* On the official test labels the same fix *lowers* WGA (DPE -0.83 pp, [-1.34, -0.31]): the
  corrected model calls more of the 90 mislabelled test images landbird (errors on them rise
  from 78 to 86 of 90 on average), which the official labels count as mistakes. A benchmark
  that keeps the error penalises the correct fix.

## Method rankings change under corrected test labels

`rescore` (official-label fits of split seed 0, from the threshold-controls caches), standard
ERM, difference against full-validation DPE: [`rescored_variants.csv`](rescored_variants.csv).

| Variant | Official test | Corrected test | Per source, corrected |
| --- | ---: | ---: | --- |
| Fold ensemble, uniform | +0.54 | -0.18 | +0.91, -0.78, +0.43, -2.43, +0.96 |
| Fold ensemble, affine avg/0.1 | +0.47 | +0.64 | +1.87, +0.30, +0.74, -1.52, +1.82 |

The uniform fold ensemble's lead over full-validation DPE disappears on corrected labels:
differences of this size between methods depend on whether the label error is fixed.

## Caveats

* The test set and backbones were inspected before; the label error was first noticed in test
  errors, but its cause is in the generator code and its correction does not use any model.
* Backbones were trained on the official training labels (72 affected images); only the
  validation and test labels are corrected here.
* "Corrected" changes only these three species; other possible label issues (e.g. images
  with several birds, raised by MaskTune) are not addressed.
