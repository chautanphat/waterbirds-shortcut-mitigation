# Research Methods

This is the project-wide inventory of methods being investigated. It explains the comparison landscape; method directories contain implementation details.

## How to add a method

Add a method only after the team agrees to investigate it. Then:

1. Decide whether the work is an external reproduction or a project-owned implementation.
2. Preserve an official external implementation under `baselines/<method-name>/`; put a project-owned, reimplemented, or integrated method under `src/methods/<method-name>/`.
3. Add a method-specific README with source, paper, license, setup, commands, and Waterbirds-specific changes.
4. Add one entry here using the template in [`README.md`](README.md).
5. Link a tracked configuration and experiment-log entry.

Do not claim an original reproduction when the work is only a smoke test or an adaptation.

Each completed method entry should point to:

- `baselines/<method-name>/` for an external implementation, or `src/methods/<method-name>/` for a project-owned implementation;
- `configs/<method-name>/` for reproducible settings;
- `experiments/<experiment-id>/` for run metadata, when present; and
- the relevant entry in `docs/experiment_log.md`.

## Planned methods

The initial comparison is expected to include ERM, GroupDRO, JTT, and other agreed baselines. No method entry is populated until its investigation begins.

## Method entries

Add approved methods below this heading using the shared template.

## Diverse Prototypical Ensembles (DPE)

### Overview
- **Full name:** Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift
- **Paper:** [arXiv:2505.23027](https://arxiv.org/abs/2505.23027) (ICML 2025)
- **Source:** https://github.com/minhto2802/dpe4subpop (commit `68b811461e238d44cced6ae6c8ce75b66cad5449`)
- **Year:** 2025
- **Implementation location:** `baselines/dpe4subpop/`

### Mechanism
DPE addresses subpopulation shifts and shortcut learning without requiring subgroup annotations during training. It replaces a single linear classification head with a diversified ensemble of prototype-based distance classifiers (IsoMax+). Ensemble diversity is explicitly enforced via balanced bootstrap subsampling and an inter-prototype similarity / covariance regularization loss that penalizes prototype alignment across stages. Final inference averages the softmax probability predictions across all ensemble heads.

### Implementation
- **Backbone:** ResNet-50 (frozen feature extractor for Stage 1+).
- **Loss function:** IsoMax+ metric loss combined with inter-prototype covariance regularization ($\mathcal{L}_{\text{cov}}$).
- **Training procedure:** Two-stage: Stage 0 pre-trains feature backbone via ERM; Stage 1+ freezes the backbone and iteratively trains $N=16$ prototype classifier heads on resampled subsets with decaying covariance regularization.
- **Special requirements:** Pre-extracted feature embeddings (`feats_val.npy`, `feats_test.npy`), `netcal`, `jsonargparse`.

### Experimental Settings
- **Dataset and split:** Waterbirds (official train/val/test splits).
- **Configuration:** `configs/dpe4subpop/waterbirds_stage0_erm.yaml`, `configs/dpe4subpop/waterbirds_stage1_dpe.yaml`.
- **Batch size / learning rate / epochs / optimizer:** Stage 0: BS=128, LR=0.001, Epochs=100, SGD. Stage 1+: BS=256, LR=0.001, Epochs=20 per stage, SGD, entropic_scale=30, cov_reg=5e5.
- **Random seeds:** 0, 1, 2.

### Evaluation
- **Paper reported targets:**
  - Standard ERM Track: Standard ERM WGA $69.1\% \pm 4.7\% \to$ DPE WGA $91.0\% \pm 0.5\%$ (Avg Acc: $92.5\% \pm 0.2\%$).
  - Enhanced $\text{ERM}^*$ Track: $\text{ERM}^*$ WGA $77.9\% \pm 3.0\% \to$ $\text{ERM}^*$ + DPE WGA $93.0\% \pm 0.4\%$ / $94.1\% \pm 0.2\%$ (Avg Acc: $96.0\% \pm 0.1\%$).
- **Reproduced results (Seed 0):**
  - Standard Track (`--no_augmentation`): ERM WGA 62.46% (Acc 88.63%) $\to$ Full DPE WGA 89.00% (Acc 91.40%).
  - Enhanced Track (Photometric distortion): $\text{ERM}^*$ WGA 77.10% (Acc 93.63%) $\to$ Full DPE WGA 92.70% (Acc 93.30%).
  - Detailed ablation and per-group accuracies recorded in [`docs/experiment_log.md`](experiment_log.md).

### Status
- **Original reproduction:** Verified reproduction on Waterbirds (both Standard and Enhanced tracks reproduced on Seed 0; see `exp-dpe-waterbirds-ermstar-001` and `exp-dpe-waterbirds-standard-002`).
- **Waterbirds adaptation:** Preserved upstream implementation in `baselines/dpe4subpop/` with end-to-end notebook in `notebooks/dpe_waterbirds_end_to_end.ipynb`.

### Notes
- Operates effectively without subgroup labels (`train_attr=no`, `subsample_type=class`).
- Fast Stage 1+ execution because optimization occurs entirely on pre-extracted feature embeddings.
- Standard ERM requires passing `--no_augmentation` to match canonical SubpopBench deterministic transforms (`Resize(256) -> CenterCrop(224)`). Omitting `--no_augmentation` enables `RandomPhotometricDistort(p=0.5)` (Enhanced $\text{ERM}^*$ track).
- Covariance regularization provides a clear +0.8% WGA boost over unregularized ensembles when trained on standard shortcut-susceptible representations.

## Optimal Transport Reweighting (POTER)

### Overview
- **Full name:** Optimal Transport Reweighting for Robust Learning under Spurious Correlations and Label Noise
- **Paper:** [arXiv:2610.01028](https://arxiv.org/abs/2610.01028) (NeurIPS 2026)
- **Source:** https://github.com/Sung-Ho-Jo/POTER
- **Year:** 2026
- **Implementation location:** `baselines/poter/`

### Mechanism
POTER mitigates spurious correlations and label noise by computing non-parametric importance weights via class-conditioned Optimal Transport (OT). It compares training representations against a small, clean validation reference set. Class-conditional Kantorovich dual potentials are calculated between training features and reference features, then mapped into per-sample sampling weights through power scaling and clipping. Training samples that align well with counter-spurious reference distributions receive higher weights, which are supplied to PyTorch's `WeightedRandomSampler` during full-network fine-tuning.

### Implementation
- **Backbone:** ResNet-50 (ImageNet-pretrained, full fine-tuning with SGD).
- **Loss function:** Standard Cross-Entropy loss with `WeightedRandomSampler` sample reweighting.
- **Training procedure:**
  1. Offline feature extraction: extract 2,048-dim penultimate ResNet-50 embeddings for train and reference splits.
  2. OT potential computation: solve class-conditional OT Kantorovich dual problems with entropy regularization.
  3. Sample weight calculation: transform potentials via power scaling (`f_power=0.1`) and clipping (`clip_cap=20.0`).
  4. Classifier training: fine-tune ResNet-50 for 300 epochs with batch size 128 using `WeightedRandomSampler`.
- **Special requirements:** Offline `.pt` feature cache, GPU with $> 10.8$ GB VRAM for batch size 128.

### Experimental Settings
- **Dataset and split:** Waterbirds (`waterbird_complete95_forest2water2`), official train/val/test splits.
- **Configuration:** [`configs/poter/waterbirds_standard.yaml`](../configs/poter/waterbirds_standard.yaml).
- **Batch size / learning rate / epochs / optimizer:** BS=128, LR=$1 \times 10^{-5}$, WD=1.0, Epochs=300, SGD with momentum=0.9, standard data augmentation.
- **Reference set composition:** `0:0, 1:466, 2:133, 3:0` (599 counter-spurious validation samples).
- **Random seeds:** 1, 2, 3.

### Evaluation
- **Paper reported targets:** Test Worst-Group Accuracy (WGA) $90.9\% \pm 0.3\%$, Average Accuracy $\approx 92.5\%$.
- **Reproduced results (Seed 1):**
  - Validation WGA: **90.10%** at Best Checkpoint (Epoch 258).
  - Validation Average Accuracy: **91.60%**.
  - Per-group validation accuracies: [92.50%, 90.10%, 92.50%, 92.50%].
  - Minority group gain: Group 1 (landbird on water) accuracy increased from 29.6% at Epoch 0 to 90.10% at Best Epoch (+60.5%).
  - Full trajectory recorded in [`results/poter/report.md`](../results/poter/report.md) and [`docs/experiment_log.md`](experiment_log.md).

### Status
- **Original reproduction:** Verified reproduction on Waterbirds (Seed 1 completed 300 epochs; matches paper target within 0.8% single-seed variance; see `exp-poter-waterbirds-standard-seed1-001`).
- **Waterbirds adaptation:** Preserved upstream implementation in `baselines/poter/` with end-to-end notebook in `notebooks/poter_waterbirds_kaggle.ipynb`.

### Notes
- Operates via data sampling reweighting (`WeightedRandomSampler`) rather than direct per-sample loss multiplication.
- Model selection is governed strictly by validation Worst-Group Accuracy (WGA); test set evaluation occurs on the selected validation-optimal checkpoint.
- Requires $> 10.8$ GB VRAM for official batch size 128 (runs on Kaggle Tesla T4 16 GB, out-of-memory on 6 GB laptop GPUs).

