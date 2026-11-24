# Direction 2 · Phase C · Step 6 Minimal Repair & 6-Arm Ablation (Validation Only)

## 1. Cleaned 5-Fold Validation OOF Results (5 Standard-ERM Seeds)

| Arm | Trainable Params | Raw WGA (%) | OOF-Thr WGA (%) | OOF-Thr Acc (%) | Min-Grp AUROC (%) | All AUROC (%) |
|---|---:|---:|---:|---:|---:|---:|
| `affine_dfr_linear` | 57,218 | 85.57 | **92.24 ± 0.85** | 94.11 | **97.93** | 98.76 |
| `affine_dfr_cosine` | 57,217 | 85.90 | **91.30 ± 0.94** | 93.88 | **97.57** | 98.66 |
| `pre_l4_gate_only` | 71,811 | 86.07 | **91.03 ± 1.08** | 93.48 | **97.57** | 98.56 |
| `pre_l4_gate_bn_ce` | 124,930 | 86.39 | **91.12 ± 1.26** | 93.43 | **97.49** | 98.54 |
| `pre_l4_repair_bn_l3l4` | 114,818 | 83.87 | **85.02 ± 5.64** | 87.74 | **94.57** | 96.23 |
| `pre_l4_repair_bn_all` | 124,930 | 86.39 | **88.62 ± 2.29** | 89.87 | **95.56** | 96.66 |

## 2. Counterfactual Robustness (26 Persistent Failures) & Layer-4 Foreground Drift

| Arm | `orig` (%) | `matched` (%) | `flipped` (%) | `neutral` (%) | `ood` (%) | L4 FG Drift (neu) | Gate FG / BG (`Δ`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `affine_dfr_linear` | 64.6 | 95.4 | 86.9 | 89.2 | 92.3 | 0.7048 | — |
| `affine_dfr_cosine` | 63.1 | 91.5 | 83.1 | 88.5 | 88.5 | 0.7068 | — |
| `pre_l4_gate_only` | 44.6 | 96.2 | 75.4 | 85.4 | 91.5 | 0.8160 | 0.88 / 0.54 (+0.34) |
| `pre_l4_gate_bn_ce` | 59.2 | 90.0 | 80.8 | 86.2 | 90.8 | 0.7125 | 0.85 / 0.60 (+0.26) |
| `pre_l4_repair_bn_l3l4` | 60.0 | 80.8 | 73.8 | 73.1 | 76.2 | 0.6315 | 0.48 / 0.10 (+0.37) |
| `pre_l4_repair_bn_all` | 60.0 | 85.4 | 74.6 | 73.8 | 81.5 | 0.6469 | 0.51 / 0.10 (+0.41) |
