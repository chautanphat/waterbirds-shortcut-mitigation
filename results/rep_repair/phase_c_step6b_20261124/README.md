# Direction 2 · Phase C · Step 6b Fixed Pre-Layer4 Repair & Layer4.conv2 LoRA Ablation (Validation Only)

## 1. Cleaned 5-Fold Validation OOF Results (5 Standard-ERM Seeds)

| Arm | Params | Raw WGA (%) | OOF-Thr WGA (%) | Δ WGA (Wins) | OOF-Thr Acc (%) | Min-Grp AUROC (%) | Δ AUC (Wins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `affine_dfr_linear` | 57,218 | 85.25 | **92.70 ± 0.59** | +0.00 (0/5) | 94.65 | **97.92** | +0.00 (0/5) |
| `pre_l4_gate_bn_ce` | 124,931 | 88.52 | **92.86 ± 0.37** | +0.16 (2/5) | 94.68 | **97.96** | +0.04 (2/5) |
| `pre_l4_repair_w005` | 124,931 | 86.56 | **93.28 ± 0.90** | +0.58 (4/5) | 94.71 | **97.94** | +0.02 (3/5) |
| `pre_l4_repair_w015` | 124,931 | 86.39 | **92.57 ± 0.69** | -0.13 (1/5) | 94.03 | **97.69** | -0.23 (1/5) |
| `l4_conv2_lora_bn` | 180,098 | 86.07 | **92.78 ± 1.40** | +0.09 (2/5) | 94.53 | **97.85** | -0.07 (1/5) |
| `l4_conv2_lora_repair_w005` | 247,811 | 86.56 | **92.30 ± 0.93** | -0.40 (1/5) | 94.88 | **98.19** | +0.27 (4/5) |

## 2. Counterfactual Robustness (26 Persistent Failures) & Layer-4 Foreground Drift

| Arm | `orig` (%) | `matched` (%) | `flipped` (%) | `neutral` (%) | `ood` (%) | L4 FG Drift (neu) | Gate FG / BG (`Δ`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `affine_dfr_linear` | 59.2 | 93.8 | 83.8 | 89.2 | 93.1 | 0.7078 | — |
| `pre_l4_gate_bn_ce` | 70.0 | 88.5 | 86.2 | 89.2 | 92.3 | 0.6743 | 0.91 / 0.59 (+0.32) |
| `pre_l4_repair_w005` | 70.0 | 90.0 | 90.8 | 88.5 | 95.4 | 0.6424 | 0.84 / 0.34 (+0.50) |
| `pre_l4_repair_w015` | 73.1 | 86.2 | 87.7 | 84.6 | 89.2 | 0.6334 | 0.77 / 0.30 (+0.47) |
| `l4_conv2_lora_bn` | 66.9 | 90.8 | 86.9 | 90.0 | 93.8 | 0.6848 | — |
| `l4_conv2_lora_repair_w005` | 69.2 | 89.2 | 85.4 | 90.8 | 92.3 | 0.6217 | 0.85 / 0.35 (+0.49) |
