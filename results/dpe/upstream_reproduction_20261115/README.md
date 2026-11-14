# Upstream DPE reproduction on fresh ERM backbones (Kaggle, 2026-11-15)

Run: public Kaggle notebook `noerula/dpe-reproduction` (GPU T4 x2). Official SubpopBench ERM
training (`resnet_sup_in1k`, `train_attr=no`, hparams seed 0, fixed final checkpoint), then
the official DPE CLI on the frozen backbone. Two new seeds (5, 6), not the backbones used in
the rest of this study. Checkpoints, features and predictions were not copied into Git;
hashes are in each `seed*/manifest.json`.

| Seed | ERM backbone alone, WGA | DPE, WGA | DPE, overall acc. |
| ---: | ---: | ---: | ---: |
| 5 | 69.31 | 89.41 | 92.46 |
| 6 | 69.31 | 86.47 | 91.54 |
| Paper | 69.1 | 91.0 | |

Per-group accuracies are in [`summary.csv`](summary.csv) and `seed*/metrics.json`;
backbone-only metrics on raw features in `seed*/erm_metrics_raw_features.json`.

Reading: the official pipeline matches the paper's ERM backbone (69.3 vs 69.1) but, like our
own implementation (89.60 saved heads, 89.88 averaged over head seeds), stays below the
reported 91.0 for DPE. With two seeds this is supporting evidence that the gap is not a bug in
our reimplementation, not an estimate of the true mean. The paper's selected ERM artefact and
hyper-parameters were not recovered.
