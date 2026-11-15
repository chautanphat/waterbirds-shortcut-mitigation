# DPE with corrected labels on fresh backbones (Kaggle, 2026-11-15)

Run: Kaggle notebook `noerula/dpe-corrected-label`, script version 369638136, GPU T4 x2,
built from [`experiments/dpe_label_corrected_backbones`](../../../experiments/dpe_label_corrected_backbones/README.md).
Files here are the notebook's `label-corrected-evidence.zip` without the feature caches
(`feats_*.npy`, about 55 MB per backbone), the per-head `subsets.json` files and the smoke test.
Per-image test predictions (`results/*/test_predictions.npz`), manifests, commands and logs are kept.

## Design (declared in `summarize.py` before the run)

Backbone training labels {official, corrected} x DPE validation labels {official, corrected} x
head seeds {s, s+1000, s+2000}, backbone seeds s = 5..8, SubpopBench ERM (default hyper-parameters)
and the authors' unmodified DPE `main.py`. Every model is scored on official and corrected test
labels. Primary contrast: official backbone, corrected minus official validation labels, on
corrected test labels. The official backbones of seeds 5 and 6 were reused from
`noerula/dpe-reproduction`; their checkpoint SHA-256 matches
[`../upstream_reproduction_20261115`](../upstream_reproduction_20261115/README.md), and their
head-seed-5/6 fits reproduce its DPE WGA exactly (89.41, 86.47).

48/48 jobs completed (6 backbone trainings, each also fitting one head seed, and 42 head-only
jobs); no failures. The notebook's report is [`report/REPORT.md`](report/REPORT.md).

## Results (WGA %, mean over 3 head seeds, then over 4 backbones)

| Backbone labels / val labels | Official test | Corrected test |
| --- | ---: | ---: |
| official / official | 88.11 | 88.40 |
| official / corrected | 87.21 | 91.44 |
| corrected / official | 88.94 | 89.45 |
| corrected / corrected | 86.94 | 91.72 |

| Contrast (corrected test labels) | Mean (pp) | 95% t interval (n=4) | Per backbone s5..s8 |
| --- | ---: | --- | --- |
| **Primary: fix validation labels** | **+3.04** | [+0.59, +5.48] | +1.75, +5.17, +2.10, +3.13 |
| Fix training and validation labels | +3.32 | [+0.78, +5.85] | +1.86, +5.59, +2.90, +2.92 |
| Fix training labels only | +1.05 | [-1.57, +3.66] | -0.78, +2.53, +2.33, +0.10 |
| Add training fix to validation fix | +0.28 | [-0.40, +0.96] | +0.11, +0.42, +0.80, -0.20 |

On official test labels the primary contrast is -0.89 pp ([-3.19, +1.40]).

Reading:

* The validation-label effect replicates on four fresh backbones with the authors' code:
  positive on all four, mean +3.04 pp, similar to +2.48 pp on our seven saved backbones.
* Correcting the 72 training labels adds little once the validation labels are fixed
  (+0.28 pp); for DPE the validation labels carry most of the effect.
* ERM backbones alone (raw features, own label set): official 68.07-69.31%, corrected
  74.29-77.63%. These are scored on different label sets and are not a paired comparison.

## Verification

All 48 prediction files were re-scored independently from the official `metadata.csv`
(labels rebuilt from the three species names, not taken from the run's own fields); every cell
and contrast above matches the notebook's report. The message "Error(s) in loading state_dict"
in every `stage1.log` is the upstream DPE code loading a linear-head ERM checkpoint into the
prototype model, followed by "Load partial state dict"; it is expected.

Caveats: 4 backbones, so intervals are wide; the test set is the same one inspected in earlier
work; only the three "tern"-substring species are corrected.
