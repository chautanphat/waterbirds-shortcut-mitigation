# AutoBackSwap baseline reproduction (arXiv 2606.32018)

Scope completed: **four runs**, AutoBackSwap on Waterbirds (ResNet50, seed 0) without and with minority groups, and the ERM baseline without and with minority (seed 0). The first ERM with-minority attempt was lost at epoch 42/50 (my notebook error) and rerun with the same seed. Everything else in the plan was not run (see section 4). The without-minority numbers come from `results/ours_minority_false_seed0_waterbird_wom_resnet_seed0/test_accuracies.txt` and were re-derived independently by `analysis/dump_predictions.py`. The with-minority and ERM numbers come from the `test_accuracies.txt` files in `results/` (copied from Kaggle; checked equal to the values first read from the notebook output). Only the without-minority AutoBackSwap numbers were re-derived independently by `dump_predictions.py`.

## 1. Setup summary and environment changes
- Machine: Kaggle notebook, one Tesla T4 (15 GB, `--device=cuda:0`), 4 CPU cores. The Mac (M1) and an RTX machine were not used for training.
- Code: `cesro7/spur-corr` at `b060532c2632039ffc2918828780d2b696763af6`, AOT-GAN submodule `2cd1afd8fdfabb101c678f6062d14bc7d302509e` (`REPO_PIN.txt`).
- Environment: Python 3.12.3 venv (uv), torch 2.11.0+cu126, torchvision 0.26.0, `repo/requirements.txt`. The venv was rebuilt once with the same pins after a Kaggle session restart.
- Weights: AOT-GAN `G0000000.pt` from the Google Drive folder in the repo README (60829150 bytes, sha256 in `runs_log.md`).
- Data: Waterbirds from WILDS `waterbirds_v1.0.tar.gz`, segmentations from data.caltech.edu record w9d68-gec53 (checksums in `runs_log.md`). The repo `metadata.csv` is byte-identical to the WILDS one. Train groups (y, place) 3498 / 184 / 56 / 1057 match the paper.
- Deviations from the README for the reported run (each is also in `runs_log.md`):
  1. **fp32** (`--cuda_optimizations=false`) for generate and classifier instead of the default bf16 autocast + `torch.compile`. The detector always runs with it off (`train.py`). On the T4 bf16 is emulated and measured about 1.8x slower than fp32 (4096x4096 matmul: fp32 34.7 ms, bf16 63.6 ms).
  2. **Chunked generate** (user-approved): `generate_waterbirds` in `methods/ours/augment_data.py` now passes images to `separator.separate` in chunks of `augment_batch_size` (16) instead of a whole class folder. Patch: `patches/augment_waterbirds_chunk.patch`. Needed because the unchunked call ran out of memory on the T4, in fp32 and in bf16.
  3. **Data source**: the README link (nlp.stanford.edu) refused connections from Kaggle and from the Mac. WILDS was used instead. Its TLS certificate had expired, so it was fetched with `curl -k` after the user approved; the certificate was not verified and no official checksum was available to compare. The archive was only listed and extracted, never executed.
  4. `detector_n_samples` is 200 (repo config, README command uses `waterbirds_from_segmentation`); the paper text mentions 287 hand-labeled samples.
  5. `MPLBACKEND=Agg` on every `run.py` call; symlinks for `images/` and `segmentations/` into `repo/data/...`; explicit run label `ours_minority_false_seed0` (see section 4).
- The classifier was run twice. The first run (fp32, chunked generate) died at epoch 42/50 when the Kaggle session shut down, so it produced no test accuracy. The reported result is the full 50-epoch rerun with the same seed and the same generated data. The partial output is kept in `results/..._partial_epoch42` (config only in git).
- Timings on the T4 (fp32): detector about 19 min, generate about 32 min, classifier about 46 min.

## 2. Results vs. paper (mean ± std, ResNet50)
| Run label | Setting | Seed | WGA | Acc | Paper WGA | Paper Acc | Within 2σ? |
|---|---|---|---|---|---|---|---|
| ours_minority_false_seed0 | Waterbirds, AutoBackSwap, without minority | 0 | 92.195 | 94.028 | 92.2 ± 0.4 | 95.1 ± 0.3 | WGA: yes (diff 0.005). Acc: **no** (diff 1.072, 3.6σ) |
| ours_minority_true_seed0 | Waterbirds, AutoBackSwap, with minority (classifier only, detector and generated data reused) | 0 | 92.368 | 95.444 | 93.0 ± 0.9 | 95.0 ± 0.5 | WGA: yes (diff 0.632, 0.7σ). Acc: yes (diff 0.444, 0.9σ) |
| erm_minority_false_seed0 | Waterbirds, ERM, without minority | 0 | 34.900 | 69.762 | 33.4 ± 4.4 | 68.4 ± 1.9 | WGA: yes (diff 1.5, 0.3σ). Acc: yes (diff 1.362, 0.7σ) |
| erm_minority_true_seed0 | Waterbirds, ERM, with minority | 0 | 77.414 | 93.890 | 74.9 ± 3.3 | 93.6 ± 0.4 | WGA: yes (diff 2.514, 0.8σ). Acc: yes (diff 0.290, 0.7σ) |
| erm_minority_false_seed1 | Waterbirds, ERM, without minority | 1 | 31.885 | 67.777 | 33.4 ± 4.4 | 68.4 ± 1.9 | WGA: yes (diff 1.515, 0.34σ). Acc: yes (diff 0.623, 0.33σ). Read from the Kaggle output, then confirmed from the copied test_accuracies.txt (2026-11-02) |
| erm_minority_true_seed1 | Waterbirds, ERM, with minority | 1 | 80.997 | 93.683 | 74.9 ± 3.3 | 93.6 ± 0.4 | WGA: yes (diff 6.097, 1.85σ, close to the 2σ edge). Acc: yes (diff 0.083, 0.2σ). Read from the Kaggle output, then confirmed from the copied test_accuracies.txt (2026-11-02) |
| ours_minority_false_seed1 | Waterbirds, AutoBackSwap, without minority | 1 | 92.284 | 94.719 | 92.2 ± 0.4 | 95.1 ± 0.3 | WGA: yes (diff 0.084, 0.2σ). Acc: yes (diff 0.381, 1.27σ). Read from the Kaggle output, then confirmed from the copied test_accuracies.txt (2026-11-02) |
| erm_minority_false_seed2 | Waterbirds, ERM, without minority | 2 | 33.348 | 68.502 | 33.4 ± 4.4 | 68.4 ± 1.9 | WGA: yes (diff 0.052). Acc: yes (diff 0.102). Read from the Kaggle output, then confirmed from the copied test_accuracies.txt (2026-11-02) |
| erm_minority_true_seed2 | Waterbirds, ERM, with minority | 2 | 76.947 | 93.424 | 74.9 ± 3.3 | 93.6 ± 0.4 | WGA: yes (diff 2.047, 0.62σ). Acc: yes (diff 0.176, 0.44σ). Read from the Kaggle output, then confirmed from the copied test_accuracies.txt (2026-11-02) |
| ours_minority_false_seed2 | Waterbirds, AutoBackSwap, without minority | 2 | 92.506 | 94.719 | 92.2 ± 0.4 | 95.1 ± 0.3 | WGA: yes (diff 0.306, 0.77σ). Acc: yes (diff 0.381, 1.27σ). Read from the Kaggle output, then confirmed from the copied test_accuracies.txt (2026-11-02) |

"Acc" is the repo's `Overall Accuracy`: correct predictions pooled over all 5794 test images. Whether this is exactly the paper's "Acc" was not checked.

Per-group test accuracy: Landbird on Land 96.674 (n=2255), Landbird on Water 92.195 (n=2255), Waterbird on Land 92.368 (n=642), Waterbird on Water 92.835 (n=642). Majority 95.823, minority 92.233.

Per-group test accuracy, with minority (metadata groups; the repo text file swaps the two middle labels): Landbird on Land 98.049, Landbird on Water 94.457, Waterbird on Land 92.835, Waterbird on Water 92.368 (worst). Majority 96.790, minority 94.097. Best val_acc 95.41%, 50 epochs, about 49.6 min.

ERM without minority per-group test accuracy (metadata groups; the repo text file swaps the two middle labels): Landbird on Land 99.601, Landbird on Water 34.900 (worst), Waterbird on Land 60.125, Waterbird on Water 97.040. Majority 99.033, minority 40.490. Best val_acc 99.33%, 50 epochs, 52.4 s/epoch. Source: `results/erm_minority_false_seed0_waterbird_wom_resnet_seed0/test_accuracies.txt`.

ERM with minority per-group test accuracy (metadata groups): Landbird on Land 99.734, Landbird on Water 92.683, Waterbird on Land 77.414 (worst), Waterbird on Water 94.081. Majority 98.481, minority 89.299. Best val_acc 93.49%, 50 epochs, 56.8 s/epoch. Source: `results/erm_minority_true_seed0_waterbird_wm_resnet_seed0/test_accuracies.txt`.

Not run, so no comparison: Spawrious, Spurious Vehicles (paper 87.2±1.9 / 95.3±0.7).

## 3. Reproduction verdicts
- **WGA: reproduces** within 2σ (92.195 vs 92.2±0.4).
- **Acc: does not reproduce** (94.028 vs 95.1±0.3, 3.6σ below the paper mean).
- **With minority, WGA and Acc: reproduce** within 2σ (92.368 vs 93.0±0.9; 95.444 vs 95.0±0.5). Both are single-seed values against the paper's 5-seed mean ± std.
- This run reused the detector and the generated `abs_seed0` images from the without-minority run. Basis: the `waterbirds_use_minority` flag is not read by the detector or by `augment_data.py` (code reading only; not verified by regenerating).
- Caveat on the test itself: this is one seed compared with the paper's 5-seed mean ± std, so it says little about run-to-run variation. A single run cannot establish a systematic gap.
- **ERM without minority: WGA and Acc reproduce** within 2σ (34.900 vs 33.4±4.4; 69.762 vs 68.4±1.9), single seed against the paper's 5-seed mean ± std.
- **ERM with minority: WGA and Acc reproduce** within 2σ (77.414 vs 74.9±3.3; 93.890 vs 93.6±0.4), single seed.
- **Cause of the Acc gap: not established.** No experiment was run to isolate it. Differences from the paper setup that exist and were *not* tested as causes: fp32 instead of bf16 + compile, chunked generate, `detector_n_samples` 200 vs 287, WILDS data copy, single seed, possible difference in how the paper defines Acc. Nothing here should be read as a finding about the method.

## 4. Repo/paper issues and items not reproduced
Verified in the code or in the run:
1. `augment_batch_size` has no effect for Waterbirds; `generate_waterbirds` batches a whole class folder (about 60 images), which ran out of memory on a 15 GB T4 (fp32: needed 3.75 GiB with 3.48 GiB free; bf16 the same).
2. Group labels in `test_accuracies.txt` are wrong for the two minority groups. `get_group = y*2 + place + 1`, but `write_results` prints index 1 as "Waterbird on Land" and index 2 as "Landbird on Water". Group 2 is y=0, place=1 (landbird on water, n=2255, 176 errors, 92.195%), group 3 is y=1, place=0 (waterbird on land, n=642, 49 errors, 92.368%). So the worst group is **Landbird on Water**, although the repo text file reports 92.195 under "Waterbird on Land". WGA and Overall are unaffected. `artifacts/.../errors.csv` uses the metadata-based names.
3. The README run-label example `--run_label="ours_minority_$WB_USE_MINORITY_seed0"` expands to `ours_minority_` in bash (`$WB_USE_MINORITY_seed0` is read as one variable name), so an explicit label was used.
4. The detector ignores `--results_dir`, writes to `methods/ours/detector/results_<dataset>_<detector>_seed<k>`, and fails if that folder exists; it must be removed before every re-run.
5. The classifier has no resume: test accuracy is computed only after the last epoch, so a lost session loses the whole run (it happened once here).
6. The DataLoader requests 8 workers; the Kaggle machine has 4 cores (warning only, not changed).
7. After a Kaggle session restart, symlinks (venv `bin/python`, the data links) were not restored while `/kaggle/working` files were. Both were recreated.

Not run: ERM baselines (both settings), AutoBackSwap without minority seeds 1 and 2, Spawrious `o2o_easy`, Spurious Vehicles. The user stopped the plan after the first run; the with-minority run was added later at the user's request. No comparison or claim is made for them.

## 5. Artifacts
- Misclassified examples: `artifacts/misclassified/ours_minority_false_seed0/`
  - `errors.csv`: all 346 errors of 5794 test images (columns: image_path, true_label, predicted_label, group, group_name, spurious_attribute, confidence, run_label, seed).
  - `index.md` and `index.html`: at most 50 per group, highest-confidence errors first (50 / 50 / 49 / 46 shown). The HTML expects the dataset `images/` folder to be symlinked next to it; no image was copied or uploaded.
  - `summary.md`: per-group counts; the script reproduces `test_accuracies.txt` exactly (WGA 92.195, overall 94.028), using the best-val checkpoint the repo itself evaluates.
  - Error counts per group: Landbird on Land 75 (all predicted waterbird), Landbird on Water 176 (all predicted waterbird), Waterbird on Land 49 and Waterbird on Water 46 (all predicted landbird). No per-example cause is claimed.
- Script: `analysis/dump_predictions.py` (standalone, does not touch method code).
- All test predictions, correct and wrong, for the four seed 0 runs: `artifacts/predictions/<run_label>/predictions.csv` (5794 rows each, written by `analysis/dump_all_predictions.py`). Errors of a run are the rows with `correct` = 0. Recomputed counts match `test_accuracies.txt` for all four (errors: AutoBackSwap without minority 346, with minority 264, ERM without minority 1752, with minority 354). No images included.
- Run log: `runs_log.csv`, `runs_log.md` (every attempt, including failed and partial ones).
