# Direction 2 · Phase C · Step 7c-2 Spawrious (`o2o_hard`) Sealed OOD Test Confirmation

Validation adaptation set: **3,800** images | Shifted OOD test set (evaluated once): **25,344** images.

| Arm | Params | Val WGA | OOD Mean Acc (%) | OOD Worst-Cls (%) | Δ Mean vs Affine (Wins) | OOD Macro-AUC (%) |
|---|---:|---:|---:|---:|---:|---:|
| `erm_raw` | 23,516,228 | 87.72 | **83.43 ± 0.53** | **63.15 ± 1.96** | -1.41 (0/3) | **94.17** |
| `dfr_l2` | 8,196 | 93.42 | **87.07 ± 0.16** | **72.36 ± 0.54** | +2.23 (3/3) | **96.46** |
| `dar` | 274,693 | 87.72 | **88.03 ± 0.11** | **69.41 ± 0.83** | +3.19 (3/3) | **97.53** |
| `affine_dfr_bn_l4` | 30,724 | 92.98 | **83.66 ± 1.19** | **59.58 ± 3.86** | -1.18 (0/3) | **95.28** |
| `affine_dfr_bn_all` | 61,316 | 88.16 | **84.84 ± 0.69** | **65.34 ± 2.07** | +0.00 (0/3) | **95.57** |
| `pre_l4_repair_w005` | 129,029 | 87.72 | **88.14 ± 0.62** | **70.92 ± 1.78** | +3.30 (3/3) | **97.24** |
