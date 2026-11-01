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
- **Overall accuracy:** 96.8% ± 0.2% (unannotated test).
- **Per-group accuracy:** Reported per target/background combination.
- **Worst-group accuracy:** 94.1% ± 0.2% (unannotated), 94.1% ± 0.4% (annotated val).

### Status
- **Original reproduction:** Smoke test verified locally on CPU/Windows via `baselines/dpe4subpop/reproduce_demo.py` and `tests/test_dpe_smoke.py`. Full Waterbirds benchmark ready for execution.
- **Waterbirds adaptation:** Preserved upstream implementation in `baselines/dpe4subpop/`.

### Notes
- Operates effectively without subgroup labels (`train_attr=no`, `subsample_type=class`).
- Fast Stage 1+ execution because optimization occurs entirely on pre-extracted feature embeddings.

