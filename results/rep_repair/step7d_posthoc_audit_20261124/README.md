# Direction 2 · Phase C · Step 7d Zero-GPU Statistical & Threshold-Sensitivity Audit

Generated 100% locally on CPU from the saved Step 7c held-out test probability caches (`step7c_waterbirds_test_probs.npz` and `step7c_spawrious_ood_probs.npz`) with zero test-set tuning or checkpoint changes.

> **Protocol note:** Pre-registered validation-calibrated operating points (`off_val_thr_wga`, `rel_val_thr_wga`, `rel_raw_wga` at `p = 0.5`) remain the primary benchmark claims. Post-hoc threshold sweeps (`rel_diag_ceil_wga = max_t WGA_test(t)`) and `rel_val_bal_wga` are explicitly labelled diagnostic upper-bound / sensitivity checks separating representation quality from small-validation threshold variance.

## 1. Waterbirds ResNet-50 Threshold Sensitivity & Representation Ceiling (Cleaned Held-Out Test)

| Cohort | Arm | Off ValThr WGA (%) | Raw `p=0.5` WGA (%) | Pre-Reg ValThr WGA (%) | Smooth ValBal WGA (%) | Diag Ceiling `max_t WGA(t)` (%) | `>=93%` Span | Min-Grp AUC (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet_std` | `erm_raw` | 72.74 ± 0.79 | 71.09 ± 1.50 | **73.73 ± 1.58** | 67.48 | **75.82 ± 0.75** | 0.000 | **80.57** |
| `resnet_std` | `dfr_l2` | 88.50 ± 0.59 | 87.95 ± 0.82 | **91.49 ± 0.30** | 90.14 | **93.02 ± 0.24** | 0.007 | **98.18** |
| `resnet_std` | `dar` | 87.26 ± 0.74 | 85.04 ± 0.48 | **90.30 ± 0.69** | 88.38 | **91.87 ± 0.20** | 0.000 | **97.62** |
| `resnet_std` | `pre_l4_gate_only` | 88.38 ± 0.45 | 90.95 ± 0.74 | **92.82 ± 0.84** | 90.56 | **93.76 ± 0.38** | 0.069 | **98.36** |
| `resnet_std` | `affine_dfr_linear` | 89.10 ± 0.76 | 90.08 ± 1.25 | **93.75 ± 0.72** | 92.97 | **94.56 ± 0.27** | 0.180 | **98.77** |
| `resnet_std` | `pre_l4_repair_w005` | 89.91 ± 1.81 | 91.22 ± 0.44 | **92.52 ± 1.06** | 91.23 | **95.24 ± 0.33** | 0.220 | **98.95** |
| `resnet_star` | `erm_star_raw` | 79.83 ± 0.55 | 79.22 ± 1.53 | **82.91 ± 0.15** | 82.33 | **83.52 ± 0.17** | 0.000 | **90.15** |
| `resnet_star` | `erm_star_dfr_l2` | 90.89 ± 0.99 | 91.15 ± 0.00 | **95.00 ± 0.31** | 92.15 | **95.22 ± 0.25** | 0.234 | **99.00** |

## 2. Paired Group-Stratified Bootstrap 95% CIs (2,000 Replicates)

| Cohort | Comparison (`A − B`) | Metric | Δ Mean (pp) | 95% Bootstrap CI | 2-Sided `p` | Seed Wins |
|---|---|---|---:|---:|---:|---:|
| `resnet_std` | `pre_l4_repair_w005` vs `affine_dfr_linear` | `rel_min_auc` | +0.18 | [+0.06, +0.29] | 0.0060 | 5/5 |
| `resnet_std` | `pre_l4_gate_only` vs `dar` | `rel_min_auc` | +0.74 | [+0.34, +1.10] | 0.0010 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dfr_l2` | `rel_min_auc` | +0.77 | [+0.50, +1.07] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dar` | `rel_min_auc` | +1.33 | [+0.96, +1.71] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `affine_dfr_linear` | `rel_all_auc` | +0.10 | [+0.04, +0.16] | 0.0020 | 5/5 |
| `resnet_std` | `pre_l4_gate_only` vs `dar` | `rel_all_auc` | +0.31 | [+0.10, +0.50] | 0.0040 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dfr_l2` | `rel_all_auc` | +0.58 | [+0.41, +0.76] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dar` | `rel_all_auc` | +0.65 | [+0.48, +0.85] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `affine_dfr_linear` | `rel_raw_wga` | +1.14 | [+0.33, +2.00] | 0.0040 | 4/5 |
| `resnet_std` | `pre_l4_gate_only` vs `dar` | `rel_raw_wga` | +5.91 | [+4.37, +7.55] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dfr_l2` | `rel_raw_wga` | +3.27 | [+1.40, +5.11] | 0.0010 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dar` | `rel_raw_wga` | +6.18 | [+4.44, +8.01] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `affine_dfr_linear` | `off_val_thr_wga` | +0.81 | [-0.15, +1.34] | 0.0930 | 4/5 |
| `resnet_std` | `pre_l4_gate_only` vs `dar` | `off_val_thr_wga` | +1.12 | [+0.24, +2.13] | 0.0140 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dfr_l2` | `off_val_thr_wga` | +1.40 | [-0.14, +2.53] | 0.0780 | 4/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dar` | `off_val_thr_wga` | +2.65 | [+1.14, +3.69] | 0.0040 | 4/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `affine_dfr_linear` | `rel_val_thr_wga` | -1.23 | [-1.53, -0.58] | 0.0000 | 1/5 |
| `resnet_std` | `pre_l4_gate_only` vs `dar` | `rel_val_thr_wga` | +2.53 | [+1.78, +3.14] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dfr_l2` | `rel_val_thr_wga` | +1.04 | [+0.15, +1.91] | 0.0180 | 4/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dar` | `rel_val_thr_wga` | +2.23 | [+1.48, +3.09] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `affine_dfr_linear` | `rel_diag_ceil_wga` | +0.67 | [+0.34, +1.06] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_gate_only` vs `dar` | `rel_diag_ceil_wga` | +1.88 | [+1.18, +2.48] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dfr_l2` | `rel_diag_ceil_wga` | +2.21 | [+1.35, +3.07] | 0.0000 | 5/5 |
| `resnet_std` | `pre_l4_repair_w005` vs `dar` | `rel_diag_ceil_wga` | +3.37 | [+2.54, +4.12] | 0.0000 | 5/5 |
| `vit` | `vit_ln_dfr_late` vs `vit_dfr_l2` | `rel_min_auc` | +0.12 | [-0.01, +0.26] | 0.0740 | 1/3 |
| `vit` | `vit_pre_b6_repair_w005` vs `vit_ln_dfr_late` | `rel_min_auc` | -0.20 | [-0.31, -0.09] | 0.0000 | 0/3 |
| `vit` | `vit_ln_dfr_late` vs `vit_dfr_l2` | `rel_raw_wga` | +1.17 | [+0.17, +2.17] | 0.0240 | 2/3 |
| `vit` | `vit_pre_b6_repair_w005` vs `vit_ln_dfr_late` | `rel_raw_wga` | -0.45 | [-1.22, +0.33] | 0.2840 | 1/3 |
| `vit` | `vit_ln_dfr_late` vs `vit_dfr_l2` | `rel_val_thr_wga` | +0.25 | [-0.42, +0.84] | 0.5200 | 1/3 |
| `vit` | `vit_pre_b6_repair_w005` vs `vit_ln_dfr_late` | `rel_val_thr_wga` | -3.36 | [-4.66, -2.12] | 0.0000 | 1/3 |
| `vit` | `vit_ln_dfr_late` vs `vit_dfr_l2` | `rel_diag_ceil_wga` | +0.39 | [-0.18, +0.84] | 0.2060 | 2/3 |
| `vit` | `vit_pre_b6_repair_w005` vs `vit_ln_dfr_late` | `rel_diag_ceil_wga` | -0.41 | [-0.91, +0.08] | 0.1040 | 0/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_all` | `ood_mean_acc` | +3.30 | [+3.11, +3.48] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_l4` | `ood_mean_acc` | +4.47 | [+4.25, +4.69] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dfr_l2` | `ood_mean_acc` | +1.07 | [+0.84, +1.27] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dar` | `ood_mean_acc` | +0.11 | [-0.14, +0.36] | 0.4030 | 2/3 |
| `spawrious` | `affine_dfr_bn_all` vs `dfr_l2` | `ood_mean_acc` | -2.23 | [-2.44, -2.02] | 0.0000 | 0/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_all` | `ood_worst_class_acc` | +5.58 | [+5.06, +6.08] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_l4` | `ood_worst_class_acc` | +11.34 | [+10.66, +12.04] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dfr_l2` | `ood_worst_class_acc` | -1.44 | [-2.06, -0.84] | 0.0000 | 1/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dar` | `ood_worst_class_acc` | +1.51 | [+0.77, +2.28] | 0.0000 | 2/3 |
| `spawrious` | `affine_dfr_bn_all` vs `dfr_l2` | `ood_worst_class_acc` | -7.02 | [-7.65, -6.40] | 0.0000 | 0/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_all` | `ood_macro_auc` | +1.67 | [+1.61, +1.74] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_l4` | `ood_macro_auc` | +1.97 | [+1.88, +2.05] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dfr_l2` | `ood_macro_auc` | +0.79 | [+0.71, +0.88] | 0.0000 | 3/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dar` | `ood_macro_auc` | -0.28 | [-0.36, -0.20] | 0.0000 | 0/3 |
| `spawrious` | `affine_dfr_bn_all` vs `dfr_l2` | `ood_macro_auc` | -0.88 | [-0.96, -0.80] | 0.0000 | 0/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_all` | `ood_spur_confusion_mean` | -2.75 | [-2.92, -2.57] | 0.0000 | 0/3 |
| `spawrious` | `pre_l4_repair_w005` vs `affine_dfr_bn_l4` | `ood_spur_confusion_mean` | -4.30 | [-4.52, -4.09] | 0.0000 | 0/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dfr_l2` | `ood_spur_confusion_mean` | -0.44 | [-0.63, -0.24] | 0.0000 | 1/3 |
| `spawrious` | `pre_l4_repair_w005` vs `dar` | `ood_spur_confusion_mean` | +0.74 | [+0.51, +0.95] | 0.0000 | 2/3 |
| `spawrious` | `affine_dfr_bn_all` vs `dfr_l2` | `ood_spur_confusion_mean` | +2.31 | [+2.12, +2.51] | 0.0000 | 3/3 |

## 3. Spawrious (`o2o_hard`) OOD Spurious-Background Confusion Attribution (25,344 OOD Images, 3 Seeds)

Each breed is paired on OOD test with a background that was 87–97% correlated with a **different** breed during training:
- `dachshund @ snow` → distractor `labrador` (trained on snow); `bulldog @ mountain` → distractor `dachshund`; `labrador @ desert` → distractor `corgi`; `corgi @ jungle` → distractor `bulldog`.

| Arm | OOD Mean Acc (%) | OOD Worst-Cls (%) | `dach@snow→lab` (%) | `bull@mtn→dach` (%) | `lab@des→cor` (%) | Macro Spur-BG Err (%) | Share of All Errors (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `erm_raw` | 83.43 ± 0.53 | 63.15 ± 1.96 | 36.75 | 11.34 | 3.62 | **13.07** | 78.9% |
| `dfr_l2` | 87.07 ± 0.16 | 72.36 ± 0.54 | 27.50 | 10.57 | 3.40 | **10.48** | 81.1% |
| `dar` | 88.03 ± 0.11 | 69.41 ± 0.83 | 30.33 | 5.46 | 1.09 | **9.31** | 77.7% |
| `affine_dfr_bn_l4` | 83.66 ± 1.19 | 59.58 ± 3.86 | 40.35 | 12.81 | 3.61 | **14.35** | 87.8% |
| `affine_dfr_bn_all` | 84.84 ± 0.69 | 65.34 ± 2.07 | 34.65 | 13.34 | 2.56 | **12.79** | 84.4% |
| `pre_l4_repair_w005` | 88.14 ± 0.62 | 70.92 ± 1.78 | 29.01 | 8.97 | 1.77 | **10.04** | 84.6% |
