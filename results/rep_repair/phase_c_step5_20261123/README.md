# Direction 2 · Phase C · Step 5 Causal Localization & Affine-DFR Audit (Validation Only)

## A. Spatial Causal Patching on 26 Persistent Failures (`orig` patched from `neutral_gray`, Std ERM 5-seed mean)

| Block | `bg_only` (DFR-L2) | `fg_only` (DFR-L2) | `bg_only` (Oracle-Mask) | `fg_only` (Oracle-Mask) | `full` (DFR-L2) |
|---|---:|---:|---:|---:|---:|
| `layer1` | 70.0% | 8.5% | 77.7% | 46.1% | 76.2% |
| `layer2` | 72.3% | 28.5% | 68.5% | 60.0% | 76.2% |
| `layer3.0` | 70.8% | 34.6% | 70.0% | 60.0% | 76.2% |
| `layer3.2` | 70.0% | 30.8% | 80.0% | 63.9% | 76.2% |
| `layer3.5` | 70.0% | 40.8% | 73.1% | 69.2% | 76.2% |
| `layer4.0` | 66.2% | 43.1% | 66.2% | 72.3% | 76.2% |
| `layer4.1` | 60.8% | 51.5% | 63.1% | 73.8% | 76.2% |
| `layer4.2` | 54.6% | 60.0% | 56.1% | 76.2% | 76.2% |

## B. Bottleneck Branch Drift Attribution (Skip vs. 3x3 Residual Conv, `orig -> neutral_gray`)

| Cohort | Block | Skip FG Drift | 3x3 Residual FG Drift | Post-ReLU FG Drift | Residual Delta Share | FG ReLU Flip Rate |
|---|---|---:|---:|---:|---:|---:|
| standard_erm | `layer3.0` | 0.2665 | 0.3483 | 0.2287 | 61.2% | 15.6% |
| standard_erm | `layer3.5` | 0.3758 | 0.2430 | 0.3726 | 53.8% | 18.1% |
| standard_erm | `layer4.0` | 0.3262 | 0.4150 | 0.6279 | 57.5% | 13.6% |
| standard_erm | `layer4.1` | 0.6279 | 0.4445 | 0.7700 | 73.8% | 16.2% |
| standard_erm | `layer4.2` | 0.7700 | 0.5996 | 0.8512 | 75.5% | 20.9% |
| erm_star | `layer3.0` | 0.2255 | 0.3131 | 0.2055 | 62.7% | 14.4% |
| erm_star | `layer3.5` | 0.3131 | 0.2123 | 0.3129 | 55.1% | 15.0% |
| erm_star | `layer4.0` | 0.2801 | 0.3300 | 0.5434 | 53.9% | 11.1% |
| erm_star | `layer4.1` | 0.5434 | 0.3902 | 0.6708 | 76.4% | 13.3% |
| erm_star | `layer4.2` | 0.6708 | 0.5236 | 0.6862 | 78.0% | 18.1% |

## C1. Stage-Scoped BN-Affine Ablation (5 Std ERM Seeds, 5-Fold Cleaned Val OOF)

| BN Scope | Raw WGA (%) | OOF-Thr WGA (%) | OOF-Thr Acc (%) | Minority-Grp AUROC (%) | All AUROC (%) |
|---|---:|---:|---:|---:|---:|
| `bn_l1_l2` | 85.41 | 91.02 ± 1.09 | 93.28 | 97.38 | 98.60 |
| `bn_l3` | 88.52 | 91.63 ± 1.21 | 94.35 | 97.93 | 98.88 |
| `bn_l4` | 86.39 | 91.29 ± 0.93 | 93.19 | 97.17 | 98.30 |
| `bn_l3_l4` | 88.03 | 92.35 ± 0.81 | 94.43 | 97.82 | 98.81 |
| `bn_all` | 88.69 | 92.69 ± 0.64 | 94.81 | 98.02 | 98.87 |

## C2. Affine-DFR Layer-4 Foreground Drift (`orig -> neutral_gray` / `matched_bg`)

| Model | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|---|---:|---:|---:|---:|
| `affine_dfr_bn_all` | 0.0644 | 0.1507 | 0.3557 | 0.7101 |
| `affine_dfr_bn_l3_l4` | 0.0643 | 0.1501 | 0.3671 | 0.7367 |

## D. Channel-Level Bleed Identifiability (Spearman Correlation with Mask-Free Observational Group Shift `G_c`)

| Cohort | Seed | Spearman(`S_c`, `G_c` raw) | Spearman(`S_c`, `G_c` std) | Spearman(`S_c`, Core `C_c`) | Top-10% Channels Bleed Share |
|---|---:|---:|---:|---:|---:|
| standard_erm | 0 | 0.130 | -0.089 | 0.125 | 23.7% |
| standard_erm | 1 | 0.138 | -0.087 | 0.121 | 24.7% |
| standard_erm | 2 | 0.150 | -0.080 | 0.093 | 24.6% |
| standard_erm | 3 | 0.116 | -0.099 | 0.126 | 24.5% |
| standard_erm | 4 | 0.136 | -0.084 | 0.098 | 24.7% |
| erm_star | 0 | 0.093 | -0.128 | 0.158 | 24.2% |
| erm_star | 1 | 0.105 | -0.114 | 0.184 | 24.7% |
