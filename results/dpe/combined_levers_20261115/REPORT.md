# Combined levers and random-drop control: declared contrasts

Paired per backbone (five standard-ERM backbones), averaged over split/head seeds, pointwise 95% t interval over backbones; differences in pp of test WGA. Clean = 5,704 correctly labelled test images.

## Run A: nested post-hoc procedures on corrected validation labels

Splits complete for all five standard-ERM backbones: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (10 of 10).

| ID | Contrast | Clean, corrected val labels | Clean, official val labels (same splits) | Interaction (corrected − official) | Corrected test labels, corrected val |
| --- | --- | --- | --- | --- | --- |
| C1 (primary) | `crossfit_affine_sel_nll` − `fullval_uniform` | -1.10 [-1.48, -0.71], 0/5 | +1.11 [+0.67, +1.56], 5/5 | -2.21 [-2.68, -1.74], 0/5 | -1.08 [-1.44, -0.71], 0/5 |
| C2 | `fullval_affine_oof_sel_nll` − `fullval_uniform` | -0.63 [-0.98, -0.28], 0/5 | +1.39 [+1.13, +1.65], 5/5 | -2.02 [-2.17, -1.86], 0/5 | -0.63 [-0.98, -0.28], 0/5 |
| C3 | `pool_linear_sel_nll` − `fullval_uniform` | -0.77 [-1.02, -0.53], 0/5 | +1.74 [+1.45, +2.03], 5/5 | -2.51 [-2.83, -2.20], 0/5 | -0.82 [-1.05, -0.58], 0/5 |
| C4 | `pool_linear_sel_nll` − `pool_affine_sel_nll` | +0.83 [+0.65, +1.01], 5/5 | +1.16 [+0.79, +1.54], 5/5 | -0.33 [-0.62, -0.04], 1/5 | +0.77 [+0.60, +0.95], 5/5 |
| T1 | `crossfit_postmean_thrwga` − `crossfit_uniform` | +0.96 [+0.50, +1.43], 5/5 | -0.79 [-1.13, -0.45], 0/5 | +1.75 [+1.45, +2.05], 5/5 | +0.95 [+0.48, +1.42], 5/5 |
| -- | `crossfit_affine_sel_wga` − `fullval_uniform` | -0.59 [-0.99, -0.19], 0/5 | +1.02 [+0.65, +1.39], 5/5 | -1.61 [-2.09, -1.13], 0/5 | -0.58 [-0.96, -0.19], 0/5 |
| -- | `crossfit_uniform` − `fullval_uniform` | -0.42 [-0.79, -0.05], 0/5 | -0.35 [-0.90, +0.20], 1/5 | -0.07 [-0.71, +0.57], 3/5 | -0.41 [-0.78, -0.04], 0/5 |
| -- | `pool_uniform` − `fullval_uniform` | -1.03 [-1.34, -0.72], 0/5 | -0.90 [-1.49, -0.32], 0/5 | -0.13 [-0.55, +0.29], 2/5 | -1.01 [-1.32, -0.70], 0/5 |
| -- | `fullval_postmean_thrwga` − `fullval_uniform` | +0.82 [+0.44, +1.20], 5/5 | -0.73 [-1.08, -0.37], 0/5 | +1.55 [+1.34, +1.75], 5/5 | +0.79 [+0.41, +1.17], 5/5 |

### Clean-subset WGA and per-group accuracy (mean over splits, then backbones)

| Val labels | Arm | WGA | Land/land | Land/water | Water/land | Water/water | worst group counts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| corrected | `fullval_uniform` | 92.70 | 93.97 | 92.71 | 94.98 | 97.21 | {'00': 2, '01': 48} |
| corrected | `crossfit_affine_sel_nll` | 91.60 | 93.53 | 91.60 | 95.33 | 97.67 | {'01': 50} |
| corrected | `fullval_affine_oof_sel_nll` | 92.07 | 93.51 | 92.08 | 95.25 | 97.45 | {'00': 2, '01': 48} |
| corrected | `pool_linear_sel_nll` | 91.93 | 93.55 | 92.03 | 95.06 | 97.33 | {'00': 10, '01': 40} |
| corrected | `pool_affine_sel_nll` | 91.09 | 93.49 | 91.09 | 95.12 | 97.73 | {'01': 50} |
| corrected | `crossfit_postmean_thrwga` | 93.24 | 94.70 | 93.25 | 94.64 | 96.97 | {'01': 47, '10': 3} |
| official | `fullval_uniform` | 90.56 | | | | | |
| official | `crossfit_affine_sel_nll` | 91.67 | | | | | |

Label lever alone (full-val DPE, corrected − official val labels, same seeds): +2.14 [+1.62, +2.67], 5/5
Both levers (cross-fit recalibration on corrected labels − full-val DPE on official labels): +1.04 [+0.51, +1.58], 5/5

NLL-selected setting for cross-fit recalibration (corrected labels): {'average-l0.01': 50}

## Run B: matched random-drop control (full-validation DPE, clean subset)

Head seeds [0, 1, 2, 3, 4]; random draws ['randdrop0', 'randdrop1', 'randdrop2', 'randdrop3', 'randdrop4']; complete: True. `drop` reproduces label_correction_20261115 to within 0.0000 pp.

| Contrast | Mean [95% interval], positive backbones |
| --- | --- |
| drop misannotated − official | +2.27 [+1.72, +2.82], 5/5 |
| drop random (mean of draws) − official | -0.01 [-0.58, +0.56], 3/5 |
| **C5:** drop misannotated − drop random | +2.28 [+1.85, +2.72], 5/5 |

Per draw, random drop − official: randdrop0 -0.07, randdrop1 +0.11, randdrop2 +0.01, randdrop3 -0.05, randdrop4 -0.05
