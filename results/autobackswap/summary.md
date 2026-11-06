# AutoBackSwap on Waterbirds: results summary

Test split of `waterbird_complete95_forest2water2`, 5794 images. Group definition: `group = y*2 + place + 1` (y: 0 landbird, 1 waterbird; place: 0 land, 1 water). Accuracies are recomputed from per-example predictions of the best-validation checkpoint and equal the `Overall Accuracy` and `Worst Group Accuracy` in each run's `test_accuracies.txt`. The upstream text file swaps the labels of the two middle groups; the names below follow the dataset metadata.

Status: **partial reproduction** (see `docs/methods.md`). Paper values are mean ± std over 5 seeds; this table has 1 to 3 seeds per setting.

## Per run

| Run | Seed | Landbird on Land (n=2255) | Landbird on Water (n=2255) | Waterbird on Land (n=642) | Waterbird on Water (n=642) | Worst-group acc | Overall acc | Errors |
|---|---|---|---|---|---|---|---|---|
| ERM, without minority | 0 | 99.601 | 34.900 | 60.125 | 97.040 | 34.900 | 69.762 | 1752 |
| ERM, without minority | 1 | 99.690 | 31.885 | 52.025 | 97.508 | 31.885 | 67.777 | 1867 |
| ERM, without minority | 2 | 99.601 | 33.348 | 53.738 | 97.508 | 33.348 | 68.502 | 1825 |
| ERM, with minority | 0 | 99.734 | 92.683 | 77.414 | 94.081 | 77.414 | 93.890 | 354 |
| ERM, with minority | 1 | 99.468 | 91.175 | 80.997 | 94.860 | 80.997 | 93.683 | 366 |
| ERM, with minority | 2 | 99.512 | 91.885 | 76.947 | 93.925 | 76.947 | 93.424 | 381 |
| AutoBackSwap, without minority | 0 | 96.674 | 92.195 | 92.368 | 92.835 | 92.195 | 94.028 | 346 |
| AutoBackSwap, without minority | 1 | 97.384 | 92.284 | 93.769 | 94.860 | 92.284 | 94.719 | 306 |
| AutoBackSwap, without minority | 2 | 97.472 | 92.506 | 92.835 | 94.704 | 92.506 | 94.719 | 306 |
| AutoBackSwap, with minority | 0 | 98.049 | 94.457 | 92.835 | 92.368 | 92.368 | 95.444 | 264 |

## Mean ± std over seeds vs. paper

Std is the sample standard deviation (n-1) over the seeds we ran; with one seed no std is shown. "Within 2σ" compares our mean with the paper mean using the paper std.

| Setting | Seeds | WGA (ours) | WGA (paper) | Acc (ours) | Acc (paper) | WGA within 2σ | Acc within 2σ |
|---|---|---|---|---|---|---|---|
| AutoBackSwap, without minority | 3 | 92.33 ± 0.16 | 92.2 ± 0.4 | 94.49 ± 0.40 | 95.1 ± 0.3 | yes (0.32σ) | no (2.04σ) |
| AutoBackSwap, with minority | 1 | 92.37 | 93.0 ± 0.9 | 95.44 | 95.0 ± 0.5 | yes (0.70σ) | yes (0.89σ) |
| ERM, without minority | 3 | 33.38 ± 1.51 | 33.4 ± 4.4 | 68.68 ± 1.00 | 68.4 ± 1.9 | yes (0.01σ) | yes (0.15σ) |
| ERM, with minority | 3 | 78.45 ± 2.22 | 74.9 ± 3.3 | 93.67 ± 0.23 | 93.6 ± 0.4 | yes (1.08σ) | yes (0.16σ) |

Notes:
- AutoBackSwap with minority has seed 0 only; it reused the detector and generated images of the without-minority seed 0 run (classifier retrained). Single-seed values against a 5-seed paper mean say little about run-to-run variation.
- AutoBackSwap without minority: overall accuracy is below the paper mean for all three seeds (94.03, 94.72, 94.72 vs 95.1 ± 0.3). Seed 0 is 3.6σ below, seeds 1 and 2 are 1.27σ below. The cause was not isolated. Untested differences from the paper setup: fp32 instead of bf16 + `torch.compile`, chunked generation (needed to fit a 15 GB T4), `detector_n_samples` 200, the WILDS copy of the data, how the paper defines "Acc".
- ERM with minority, seed 1: worst-group accuracy 80.997 is 1.85σ above the paper mean, close to the 2σ edge.
- Spawrious and Spurious Vehicles were not run.
