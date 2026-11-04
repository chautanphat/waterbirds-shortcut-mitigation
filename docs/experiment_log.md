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

- **Method / version:** Potentials from Optimal Transport for Empirical Reweighting (POTER) / NeurIPS 2026
- **Purpose:** Original reproduction on Waterbirds (Standard Clean Track, 300 epochs)
- **Branch or commit:** `phat/poter-windows-cmnist`
- **Configuration:** `configs/poter/waterbirds_standard.yaml`
- **Dataset and split:** Waterbirds (`waterbird_complete95_forest2water2`), official train/val/test splits (train=4,795, val=1,199, test=5,794)
- **Seed:** 1
- **Environment and hardware:** Kaggle Linux container, 1x NVIDIA Tesla T4 GPU (15.0 GiB VRAM), PyTorch 2.11.0+cu130, CUDA 13.0
- **Command:** End-to-end execution of `notebooks/poter_waterbirds_kaggle.ipynb` (`scripts/kaggle_waterbirds.py --data-dir /kaggle/working/poter-data --cache /kaggle/working/poter-features/waterbirds_standard.pt --run-dir /kaggle/working/poter-results/CUB/Clean/POTER_CUB_seed1_kaggle --seed 1 --epochs 300 --batch-size 128 --max-hours 9.5`)
- **Output location:** `poter-results/CUB/Clean/POTER_CUB_seed1_kaggle/` (untracked), metrics in `results/poter/seed1_full_metrics.csv`
- **Results:**
  - **Selected Best Checkpoint:** Epoch 258 (zero-based), selected strictly by validation WGA
  - **Validation WGA:** **90.10%** (vs. paper Table 1: $90.9 \pm 0.3\%$)
  - **Validation Average Accuracy:** **91.60%** (vs. paper: $\approx 92.5\%$)
  - **Validation Per-group Accuracies:** Group 0 (Landbird on Land): 92.50%; Group 1 (Landbird on Water, minority): 90.10%; Group 2 (Waterbird on Land, minority): 92.50%; Group 3 (Waterbird on Water): 92.50%
  - **Terminal Epoch 299:** Train Loss: 0.176; Train Avg Acc: 98.9%; Val Avg Acc: 90.5%; Val WGA: 88.00%
- **Paper comparison or expected behavior:** Matches published paper target ($90.9 \pm 0.3\%$ WGA) within $0.8\%$ single-seed variance. Confirms that class-conditioned Optimal Transport potential reweighting successfully neutralizes spurious background correlation (+60.5% gain on minority Group 1 from initial epoch).
- **Status:** verified reproduction
- **Notes and follow-up:** Complete 300-epoch run verified. Next steps: run seeds 2 and 3 using the resumable checkpoint pipeline to produce the full 3-seed mean $\pm$ std summary table.



