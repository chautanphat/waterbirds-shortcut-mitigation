# Experiment Log

Record every meaningful run here. This is an index, not a replacement for configurations, method READMEs, or untracked output files.

## Entry template

```markdown
## <experiment-id> — YYYY-MM-DD

- **Method / version:**
- **Purpose:** smoke test / original reproduction / Waterbirds adaptation / final comparison
- **Branch or commit:**
- **Configuration:** `configs/...` or `experiments/...`
- **Dataset and split:**
- **Seed:**
- **Environment and hardware:**
- **Command:**
- **Output location:** local, untracked path
- **Results:** overall accuracy; per-group accuracy; worst-group accuracy
- **Paper comparison or expected behavior:**
- **Status:** exploratory / smoke test / partial reproduction / verified reproduction / final
- **Notes and follow-up:**
```

Use one entry per meaningful run or a short grouped entry for repeated seeds with the same configuration. Never present a result without its configuration, command, seed, and evaluation context.

Use repository-relative paths in every entry. Link implementation changes to `baselines/` or `src/methods/`, tracked settings to `configs/`, run metadata to `experiments/`, and lightweight summaries to `results/`; keep large outputs and checkpoints local and untracked.

---

## exp-shared-erm-smoke-001 — 2026-11-07

- **Method / version:** Project-owned shared ERM, `src/methods/erm/`; branch `feature/shared-erm`, initial implementation at `c3a64f6`.
- **Purpose:** CPU smoke test and compatibility validation, not a benchmark reproduction.
- **Configuration:** `configs/erm/waterbirds_standard.yaml`, with smoke overrides: random initialization (`model.weights: null`), one epoch, train/eval batch sizes 2/3, workers 0, CPU. SGD settings, class-balanced sampling, preprocessing and updated BatchNorm retained.
- **Dataset and split:** Generated 12-image RGB fixture, four examples per split and one per target/background group; no official Waterbirds data used. A separate training integration test omits test data entirely.
- **Seed:** 0.
- **Environment and hardware:** CPU; Python 3.12.14, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu; one compute thread.
- **Command:** `python scripts/train_erm.py --config <temporary-smoke-config.yaml>`; then `python scripts/evaluate_erm.py --checkpoint <run>/ckpt_best_wga.pt --data-dir <fixture> --device cpu --workers 0 --batch-size 3 --output <run>/test.json`; `python scripts/extract_erm_features.py --checkpoint <run>/ckpt_last.pt --data-dir <fixture> --device cpu --workers 0 --batch-size 3 --splits train val test`. The temporary config is generated from `ERMConfig` with the overrides above. `python -m unittest discover -s tests -v` regenerates fixtures and exercises the training/export contracts without downloads.
- **Output location:** Temporary, untracked fixture and run artifacts; no weights/features committed.
- **Results:** Two optimizer updates with the actual ResNet-50 completed. Validation on synthetic images: overall accuracy 50%, group accuracies [100%, 100%, 0%, 0%] with one image each, WGA 0%; these values carry no benchmark meaning. Ten CPU regression tests passed. Direct comparison against the preserved DPE modules confirmed identical preprocessing tensors, split ordering, labels and sampling weights, plus strict checkpoint loading with identical logits and features.
- **Paper comparison or expected behavior:** No accuracy comparison with papers is appropriate for this fixture. This initial smoke run used weight decay `1e-4` from the DPE YAML. Shared defaults were subsequently aligned with the effective DPE notebook command: weight decay 0, evaluation batch size 256 and last-checkpoint handoff. The smoke numbers above describe the initial run only.
- **Status:** smoke test.
- **Notes and follow-up:** Run official Waterbirds training on GPU for benchmark results and multi-seed validation. The test set is never loaded during shared training or checkpoint selection. Historical baseline results are unchanged.

## exp-dpe-waterbirds-ermstar-001 — 2026-11-02

- **Method / version:** Diverse Prototypical Ensembles (DPE) / commit `68b8114`
- **Purpose:** Original reproduction on Waterbirds (Enhanced $\text{ERM}^*$ Track with Photometric Distortion)
- **Branch or commit:** `main`
- **Configuration:** `notebooks/dpe_waterbirds_end_to_end.ipynb` (default data augmentation enabled)
- **Dataset and split:** Waterbirds (`waterbird_complete95_forest2water2`), official train/val/test splits (train=4,795, val=1,199, test=5,794)
- **Seed:** 0
- **Environment and hardware:** Kaggle Linux container, 2x Tesla T4 GPUs (15 GB VRAM each), PyTorch 2.x, CUDA 12.x
- **Command:** End-to-end execution of `notebooks/dpe_waterbirds_end_to_end.ipynb` (Stage 0 ResNet-50 ERM 100 epochs with `RandomPhotometricDistort(p=0.5)` + Stage 1+ 15-head DPE ensemble with IsoMax+ and covariance regularization)
- **Output location:** `checkpoints/waterbirds_stage0_erm/` (untracked)
- **Results:**
  - **Stage 0 ERM Backbone (100 ep):** Overall Acc: 93.63%; Per-group Acc: [99.03%, 77.10%, 77.30%, 99.75%]; Worst-Group Acc (WGA): 77.10%
  - **Stage 1 DPE (1 Head):** Worst-Group Acc (WGA): 92.50%
  - **Stage 15 Full DPE (Cov Reg + Random Subsets):** Overall Acc: 93.30%; Per-group Acc: [93.6%, 93.0%, 92.7%, 93.9%]; Worst-Group Acc (WGA): 92.70%
  - **Stage 15 Ablation 1 (Random Subsets Only, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.70%; Per-group Acc: [94.2%, 93.5%, 92.8%, 94.3%]; Worst-Group Acc (WGA): 92.80%
  - **Stage 15 Ablation 2 (Single Fixed Subset, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.70%; Per-group Acc: [94.3%, 93.7%, 92.7%, 94.1%]; Worst-Group Acc (WGA): 92.70%
- **Paper comparison or expected behavior:** Closely matches the paper's reported $\text{ERM}^*$ baseline ($77.9\% \pm 3.0\%$ WGA) and $\text{ERM}^*$ + DPE benchmark ($93.0\% \pm 0.4\%$ / $94.1\% \pm 0.2\%$ WGA). The photometric augmentations produced robust backbone representations, leading to relatively flat ablation metrics across the ensemble variants.
- **Status:** verified reproduction
- **Notes and follow-up:** Validates the paper's enhanced data-augmented track. The presence of `RandomPhotometricDistort` mitigated background bias during Stage 0 (+10% WGA over standard ERM).


## exp-dpe-waterbirds-standard-002 — 2026-11-02

- **Method / version:** Diverse Prototypical Ensembles (DPE) / commit `68b8114`
- **Purpose:** Original reproduction on Waterbirds (Standard SubpopBench ERM Track, `--no_augmentation`)
- **Branch or commit:** `main`
- **Configuration:** `configs/dpe4subpop/waterbirds_stage0_erm.yaml` (`no_augmentation: true`), `configs/dpe4subpop/waterbirds_stage1_dpe.yaml`
- **Dataset and split:** Waterbirds (`waterbird_complete95_forest2water2`), official train/val/test splits (train=4,795, val=1,199, test=5,794)
- **Seed:** 0
- **Environment and hardware:** Kaggle Linux container, 2x Tesla T4 GPUs (15 GB VRAM each), PyTorch 2.x, CUDA 13.0
- **Command:** End-to-end execution of `notebooks/dpe_waterbirds_end_to_end.ipynb` with `--no_augmentation` passed to Stage 0 training (deterministic `Resize(256) -> CenterCrop(224)`) + Stage 1+ 15-head DPE ensemble
- **Output location:** `checkpoints/waterbirds_stage0_standard_erm/` (untracked)
- **Results:**
  - **Stage 0 ERM Backbone (100 ep):** Overall Acc: 88.63%; Per-group Acc: [99.47%, 80.49%, 62.46%, 93.30%]; Worst-Group Acc (WGA): 62.46%
  - **Stage 1 DPE (1 Head):** Overall Acc: 92.70%; Per-group Acc: [94.8%, 92.5%, 88.9%, 90.0%]; Worst-Group Acc (WGA): 88.90%
  - **Stage 15 Full DPE (Cov Reg + Random Subsets):** Overall Acc: 91.40%; Per-group Acc: [93.8%, 89.0%, 89.9%, 92.7%]; Worst-Group Acc (WGA): 89.00%
  - **Stage 15 Ablation 1 (Random Subsets Only, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.50%; Per-group Acc: [96.3%, 93.5%, 88.2%, 89.6%]; Worst-Group Acc (WGA): 88.20%
  - **Stage 15 Ablation 2 (Single Fixed Subset, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.30%; Per-group Acc: [95.5%, 93.2%, 88.8%, 90.3%]; Worst-Group Acc (WGA): 88.80%
- **Paper comparison or expected behavior:** Matches the paper's canonical Standard ERM baseline ($69.1\% \pm 4.7\%$ WGA) and reproduces the Standard ERM + DPE benchmark ($91.0\% \pm 0.5\%$ WGA) within 2.0% on a single seed. On standard shortcut-susceptible representations, Full DPE outperformed the no-covariance-regularization ablation by +0.8% WGA (89.00% vs. 88.20%), confirming the empirical benefit of prototype covariance regularization.
- **Status:** verified reproduction
- **Notes and follow-up:** Confirms the primary paper claims across both Standard and Enhanced tracks. Next steps: multi-seed runs (seeds 1 & 2) and unannotated subsampling evaluation (`train_attr=no`, `subsample_type=class`).


## exp-poter-waterbirds-standard-seed1-001 — 2026-11-04

- **Method / version:** POTER / NeurIPS 2026
- **Purpose:** Seed-1 Waterbirds benchmark run and validation-based checkpoint selection
- **Branch or commit:** `phat/poter-windows-cmnist`
- **Configuration:** `configs/poter/waterbirds_standard.yaml`
- **Dataset and split:** Waterbirds official train/validation/test splits (4,795 / 1,199 / 5,794)
- **Seed:** 1
- **Environment and hardware:** Kaggle Linux container, 1× NVIDIA Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0
- **Command:** `scripts/kaggle_waterbirds.py --data-dir /kaggle/working/poter-data --cache /kaggle/working/poter-features/waterbirds_standard.pt --run-dir /kaggle/working/poter-results/CUB/Clean/POTER_CUB_seed1_kaggle --seed 1 --epochs 300 --batch-size 128 --max-hours 9.5`
- **Output location:** `poter-results/CUB/Clean/POTER_CUB_seed1_kaggle/` (untracked); train/validation aggregate in `results/poter/seed1_full_metrics.csv`
- **Results:**
  - Selected epoch: 258 (zero-based), selected only by validation WGA.
  - Validation WGA: 90.10%; validation average accuracy: 91.60%.
  - Validation group accuracies: [92.50%, 90.10%, 92.50%, 92.50%].
  - Held-out test WGA, overall accuracy, adjusted average, group accuracies and counts: **not present in tracked artifacts**.
- **Paper comparison or expected behavior:** The paper reports held-out test WGA 90.9 ± 0.3% over three seeds. The seed-1 validation result is not directly comparable and no reproduction verdict is drawn from the 0.8-point numerical difference.
- **Status:** partial reproduction
- **Notes and follow-up:** Recover `log.txt`, `val.csv`, and `test.csv` from the Kaggle backup and run the summarizer with `--seeds 1`. If `test.csv` is unavailable, evaluate `best_validation.pt` once on the official test split. Then run seeds 2 and 3 for the paper-level aggregate.

## exp-autobackswap-waterbirds-wom-003 — 2026-10-31 to 2026-11-01

- **Method / version:** AutoBackSwap, `baselines/autobackswap/` (upstream commit `b060532c2632039ffc2918828780d2b696763af6`), plus `baselines/autobackswap/patches/augment_waterbirds_chunk.patch`
- **Purpose:** partial reproduction
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_autobackswap_resnet50.yaml` (`use_minority: false`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0, 1, 2
- **Environment and hardware:** Kaggle, Tesla T4, Python 3.12, torch 2.11.0+cu126, fp32
- **Command:** see `baselines/autobackswap/README.md`; full per-run lines in `results/autobackswap/runs_log.csv`
- **Output location:** local, untracked (`results/<run_label>_waterbird_wom_resnet_seed<k>/`); copies of `test_accuracies.txt` in `results/autobackswap/test_accuracies/`
- **Results:** worst-group acc 92.195 / 92.284 / 92.506; overall acc 94.028 / 94.719 / 94.719 (per-group values and group sizes in `results/autobackswap/summary.md`)
- **Paper comparison or expected behavior:** paper 92.2 ± 0.4 / 95.1 ± 0.3. Worst-group within 2σ; overall accuracy mean 94.49 is 2.04σ below.
- **Status:** partial reproduction
- **Notes and follow-up:** cause of the overall-accuracy gap not isolated. Seed 0 first attempts failed or were interrupted (out of memory, session shutdown); they are in `results/autobackswap/runs_log.md`.

## exp-autobackswap-waterbirds-wm-004 — 2026-10-31

- **Method / version:** AutoBackSwap, same as above
- **Purpose:** partial reproduction
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_autobackswap_resnet50.yaml` (`use_minority: true`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** classifier step only (`--mode=ours --waterbirds_use_minority=true`), reusing the detector and generated images of `exp-autobackswap-waterbirds-wom-003` seed 0
- **Output location:** local, untracked
- **Results:** worst-group acc 92.368, overall acc 95.444
- **Paper comparison or expected behavior:** paper 93.0 ± 0.9 / 95.0 ± 0.5, both within 2σ (single seed)
- **Status:** partial reproduction
- **Notes and follow-up:** one seed only; reuse of generated data was inferred from code reading, not verified by regenerating.

## exp-erm-waterbirds-wom-005 — 2026-10-31 to 2026-11-01

- **Method / version:** ERM from the AutoBackSwap codebase (`--mode=erm`)
- **Purpose:** baseline for the AutoBackSwap comparison
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_erm_resnet50.yaml` (`use_minority: false`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0, 1, 2
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** see `baselines/autobackswap/README.md`
- **Output location:** local, untracked; `test_accuracies.txt` copies in `results/autobackswap/test_accuracies/`
- **Results:** worst-group acc 34.900 / 31.885 / 33.348; overall acc 69.762 / 67.777 / 68.502
- **Paper comparison or expected behavior:** paper 33.4 ± 4.4 / 68.4 ± 1.9, both within 2σ
- **Status:** partial reproduction
- **Notes and follow-up:** 3 seeds against the paper's 5.

## exp-erm-waterbirds-wm-006 — 2026-10-31 to 2026-11-01

- **Method / version:** ERM from the AutoBackSwap codebase
- **Purpose:** baseline for the AutoBackSwap comparison
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_erm_resnet50.yaml` (`use_minority: true`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0, 1, 2
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** see `baselines/autobackswap/README.md`
- **Output location:** local, untracked; `test_accuracies.txt` copies in `results/autobackswap/test_accuracies/`
- **Results:** worst-group acc 77.414 / 80.997 / 76.947; overall acc 93.890 / 93.683 / 93.424
- **Paper comparison or expected behavior:** paper 74.9 ± 3.3 / 93.6 ± 0.4, both within 2σ (seed 1 worst-group is 1.85σ above the paper mean)
- **Status:** partial reproduction
- **Notes and follow-up:** the first seed 0 attempt stopped at epoch 42 of 50 (most likely a notebook kernel restart, not verified) and was rerun with the same seed.
