# Direction 2 · Phase C · Step 7c-1 Waterbirds One-Shot Held-Out Test Confirmation

All checkpoints, neutral baselines, and decision thresholds calibrated **strictly on validation** (`split == 1`, `1,199` rows); `split == 2` (`5,794` rows) unsealed once.

## 1. ResNet-50 Held-Out Test (5 Standard-ERM Seeds + 2 ERM* Reference Seeds)

| Cohort | Arm | Params | Val OOF WGA | Official Test WGA | Cleaned Test WGA | Δ vs Affine (Wins) | Cleaned Acc | Min-Grp AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `standard_erm` | `erm_raw` | 23,512,130 | 71.70 | 72.74 ± 0.79 | **73.73 ± 1.58** | -20.03 (0/5) | 86.97 | **80.57** |
| `standard_erm` | `dfr_l2` | 4,098 | 90.54 | 88.50 ± 0.59 | **91.49 ± 0.30** | -2.27 (0/5) | 93.69 | **98.18** |
| `standard_erm` | `dar` | 270,595 | 90.33 | 87.26 ± 0.74 | **90.30 ± 0.69** | -3.46 (0/5) | 93.62 | **97.62** |
| `standard_erm` | `pre_l4_gate_only` | 71,811 | 90.71 | 88.38 ± 0.45 | **92.82 ± 0.84** | -0.93 (0/5) | 94.73 | **98.36** |
| `standard_erm` | `affine_dfr_linear` | 57,218 | 91.64 | 89.10 ± 0.76 | **93.75 ± 0.72** | +0.00 (0/5) | 95.41 | **98.77** |
| `standard_erm` | `pre_l4_repair_w005` | 124,931 | 91.88 | 89.91 ± 1.81 | **92.52 ± 1.06** | -1.23 (1/5) | 95.04 | **98.95** |
| `erm_star` | `erm_star_raw` | 23,512,130 | 85.20 | 79.83 ± 0.55 | **82.91 ± 0.15** | — | 91.48 | **90.15** |
| `erm_star` | `erm_star_dfr_l2` | 4,098 | 94.67 | 90.89 ± 0.99 | **95.00 ± 0.31** | — | 96.71 | **99.00** |

## 2. ViT-B/16 Cross-Architecture Transfer Held-Out Test (3 Standard-ERM Seeds, Pre-`block_6` Repair)

| Arm | Params | Val OOF WGA | Official Test WGA | Cleaned Test WGA | Δ vs LN-Late (Wins) | Cleaned Acc | Min-Grp AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vit_erm_raw` | 85,800,194 | 62.02 | 61.32 ± 2.07 | **65.55 ± 2.22** | -25.97 (0/3) | 89.89 | **81.03** |
| `vit_dfr_l2` | 1,538 | 91.26 | 87.38 ± 2.59 | **91.27 ± 0.83** | -0.25 (2/3) | 94.52 | **98.27** |
| `vit_ln_dfr_late` | 15,362 | 91.16 | 87.80 ± 1.51 | **91.52 ± 0.60** | +0.00 (0/3) | 94.60 | **98.39** |
| `vit_pre_b6_gate_only` | 52,355 | 89.89 | 87.00 ± 2.81 | **89.69 ± 1.10** | -1.83 (0/3) | 94.26 | **98.09** |
| `vit_pre_b6_repair_w005` | 66,179 | 90.98 | 83.18 ± 4.49 | **88.16 ± 3.31** | -3.36 (1/3) | 95.59 | **98.18** |
