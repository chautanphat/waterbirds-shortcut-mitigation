# Direction 2 · Phase C · Step 7a Waterbirds ViT-B/16 ERM & Mechanism Diagnostic (Validation Only)

Failure subset: **45** minority val images (`missed_by_>=2/3_dfr_l2_seeds`); matched passing controls: **26**.

## 1. Cleaned 5-Fold Validation OOF Control Ladder (3 Standard-ERM ViT-B/16 Seeds)

| Arm | Trainable Params | Raw WGA (%) | OOF-Thr WGA (%) | OOF-Thr Acc (%) | Min-Grp AUROC (%) | All AUROC (%) |
|---|---:|---:|---:|---:|---:|---:|
| `erm_raw` | 85,800,194 | 54.92 | **62.02 ± 4.80** | 89.16 | **80.99** | 94.32 |
| `dfr_l2` | 1,538 | 87.98 | **89.62 ± 1.25** | 93.36 | **97.11** | 97.62 |
| `oracle_fg_token_pool` | 1,538 | 91.53 | **92.48 ± 0.62** | 94.63 | **98.24** | 98.65 |
| `ln_dfr_late` | 15,362 | 89.34 | **91.16 ± 1.29** | 93.63 | **96.98** | 97.45 |
| `ln_dfr_all` | 39,938 | 86.34 | **89.66 ± 1.76** | 93.97 | **96.51** | 97.81 |

## 2. Counterfactual Robustness on ViT Persistent Failures

| Arm | `orig` (%) | `matched` (%) | `flipped` (%) | `neutral` (%) | `ood` (%) |
|---|---:|---:|---:|---:|---:|
| `dfr_l2` | 28.1 | 76.3 | 54.1 | 66.7 | 65.9 |
| `oracle_fg_token_pool` | 51.1 | 74.8 | 60.7 | 71.8 | 72.6 |
| `ln_dfr_all` | 38.5 | 80.7 | 58.5 | 77.8 | 75.6 |

## 3. Block-Wise Foreground Token Drift (`orig -> neutral_gray`)

| Site | Frozen Failures | Frozen Passing Ctrls | `ln_dfr_all` Failures |
|---|---:|---:|---:|
| `embed` | 0.2947 | 0.2413 | 0.2947 |
| `block_0` | 0.3821 | 0.3443 | 0.3659 |
| `block_1` | 0.6061 | 0.6070 | 0.5437 |
| `block_2` | 0.6405 | 0.6468 | 0.6013 |
| `block_3` | 0.6502 | 0.6380 | 0.6240 |
| `block_4` | 0.6375 | 0.6018 | 0.6226 |
| `block_5` | 0.6406 | 0.5847 | 0.6340 |
| `block_6` | 0.6869 | 0.5917 | 0.6918 |
| `block_7` | 0.6926 | 0.5702 | 0.7009 |
| `block_8` | 0.6957 | 0.5221 | 0.6688 |
| `block_9` | 0.6598 | 0.4411 | 0.6176 |
| `block_10` | 0.5973 | 0.3852 | 0.5418 |
| `block_11` | 0.6509 | 0.4429 | 0.6139 |
| `final_cls` | 1.2546 | 0.5083 | 1.0297 |

## 4. Spatial Token Patching (`orig <-> neutral_gray`)

| Site | BG-Only DFR (%) | FG-Only DFR (%) | BG-Only Oracle-FG (%) | FG-Only Oracle-FG (%) |
|---|---:|---:|---:|---:|
| `embed` | 63.0 | 30.4 | 68.9 | 49.6 |
| `block_0` | 64.4 | 28.9 | 69.6 | 51.1 |
| `block_2` | 66.7 | 34.8 | 69.6 | 54.8 |
| `block_4` | 65.2 | 41.5 | 69.6 | 60.7 |
| `block_6` | 56.3 | 46.7 | 64.4 | 65.2 |
| `block_8` | 46.7 | 58.5 | 53.3 | 74.8 |
| `block_9` | 49.6 | 60.0 | 56.3 | 71.1 |
| `block_10` | 40.0 | 57.8 | 60.0 | 68.9 |
| `block_11` | 28.1 | 28.1 | 60.0 | 69.6 |
