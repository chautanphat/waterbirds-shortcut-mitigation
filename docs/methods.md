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

## Shared empirical risk minimization (ERM)

- **Implementation:** [`src/methods/erm/`](../src/methods/erm/README.md), project-owned extraction of DPE's standard stage-0 training recipe; shared data/model/evaluation utilities are available to future methods.
- **Source:** [DPE](https://github.com/minhto2802/dpe4subpop) (MIT); deterministic Waterbirds preprocessing inherited from [SubpopBench](https://github.com/YyzHarry/SubpopBench) (MIT). Citations and source details are in the [method README](../src/methods/erm/README.md).
- **Mechanism:** Full-backbone ResNet-50 fine-tuning with mean cross entropy and class-balanced replacement sampling. Training group labels are unused.
- **Settings:** [`configs/erm/waterbirds_standard.yaml`](../configs/erm/waterbirds_standard.yaml): ImageNet V2 weights, updated BatchNorm statistics, 100 epochs, train/eval batch sizes 128/256, SGD LR 0.001, momentum 0.9, weight decay 0, no scheduler or augmentation. These match the effective stage-0 command in the existing DPE notebook, including CLI defaults.
- **Evaluation:** `ckpt_last.pt` is the default, matching DPE's last-epoch handoff. An additional `ckpt_best_wga.pt` tracks validation WGA for experiments using validation selection. Held-out test evaluation is a separate command. Reports include overall/per-group accuracy, WGA and counts.
- **Status:** One standard seed-0 shared-training run is recorded in the [H2 experiment](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md). Backbone validation metrics are available; that record does not establish an ERM-only test result or a multi-seed reproduction.
- **Reuse:** Project-level train/evaluate/export scripts operate independently of all preserved baselines. Checkpoints and raw ordered feature caches can be consumed by DPE; historical DPE runs remain unchanged.

## DPE head reweighting (H2)

### Overview

- **Full name:** Validation-derived reweighting of Diverse Prototypical Ensemble heads.
- **Paper/source:** Project-owned adaptation of [DPE](https://arxiv.org/abs/2505.23027), ICML 2025; the weighting rule is not a paper result.
- **Implementation location:** [src/methods/dpe_head_reweighting/](../src/methods/dpe_head_reweighting/README.md).

### Mechanism

Freeze one ERM representation and 15 A-trained prototype heads. Estimate each head's WGA on B, construct score-power weights with uniform shrinkage, select aggregation on C and compare with the same heads under uniform averaging on test. This experiment uses group-annotated validation data.

The [multi-rule notebook](../notebooks/dpe_weighting_rules.ipynb) now runs the full study from Waterbirds setup through per-seed backbones, heads, weighting comparisons and a combined ZIP. Its [default study plan](../configs/dpe_head_reweighting/study.yaml) covers seeds 0/1/2; matching attached original H2 bundles are reused, while missing seeds train automatically. New backbones support epoch-boundary resume, with completed stages reused after validation. A cached-only mode remains available. The [unchanged weighting grid](../configs/dpe_head_reweighting/weighting_rules.yaml) fits on B and selects on C before new test inference; each family includes uniform. See the [protocol and hosted-run instructions](../src/methods/dpe_head_reweighting/README.md#one-end-to-end-weighting-study). The new orchestration has synthetic integration coverage; the completed standard seed-0 comparison below remains the available real-data evidence, and multi-seed conclusions are pending.

### Implementation

- **Backbone:** Shared ImageNet-pretrained ResNet-50; final-epoch standard ERM checkpoint.
- **Loss:** IsoMax+ with training entropic scale 30, prototype penalty and sequential diversity regularization; inference temperature 1.
- **Training procedure:** Validation A/B/C roles are separated; A supplies head fitting and epoch selection. Test is held out until aggregation selection is persisted.
- **Special requirements:** Ordered raw features, group annotations and matching input provenance.

### Experimental Settings

- **Dataset and split:** Official Waterbirds; validation A/B/C=720/241/238, partition seed 0.
- **Configuration:** [Recorded H2 settings](../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/config.yaml), [backbone settings](../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/stage0_config.yaml).
- **Optimization:** Backbone SGD 100 epochs, batch 128; 15 heads × 20 epochs, batch 256; learning rate 0.001 and momentum 0.9.
- **Random seeds:** One completed standard-track run, seed 0; seeds 1 and 2 remain pending.

### Evaluation

Uniform and selected test accuracy are both 91.2323%, with WGA 89.7561%. Group accuracies/counts and full candidate results are in the [run report](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md). All 13 candidates predict identical C labels; selection retains uniform weights, yielding zero paired improvement.

The subsequent [20-candidate suite](../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/report.md) selects top-k10 overall after a one-image C improvement. Test accuracy/WGA are 91.1978/89.6231%, with paired changes of -0.0345/-0.1330 percentage points. It fixes 27 test errors and introduces 29. Power, softmax and joint families select uniform; no selected weighting procedure improves test WGA in this seed. This suite is a separately recorded exploratory result, reviewed for aggregate consistency; the uploaded suite archive does not contain prediction caches for independent re-inference.

### Status

- **Original reproduction:** Not claimed; H2 changes validation allocation and head-training data budget.
- **Waterbirds adaptation:** Completed exploratory seed-0 run; independently audited saved predictions and records. See [experiment index](experiment_log.md#exp-dpe-h2-waterbirds-standard-seed0-001--2026-11-08) and [provenance](../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/run.json).

### Notes

The audit cannot rerun image-to-feature inference from the supplied bundle because raw images, backbone and raw features are excluded. [Summary diagnostics](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md#additional-summary-diagnostics--2026-11-08) find unstable B-to-C WGA rankings (Spearman -0.0276) and mild concentration under the original power grid (minimum effective heads 14.3720). The additional rules have now been measured for seed 0; complete seeds 1/2 with the fixed rule grid before drawing multi-seed conclusions. A separately labeled study can assess score/partition stability. Uniform fallback and top-k selection outcomes are observations, not evidence of a general advantage or disadvantage of weighting.

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
- **Paper:** [arXiv:2610.01028](https://arxiv.org/abs/2610.01028) (NeurIPS 2026)
- **Upstream source:** https://github.com/Sung-Ho-Jo/POTER
- **Implementation:** `baselines/poter/`

POTER constructs class-conditioned optimal-transport weights from a limited validation reference set with group annotations, then supplies those weights to `WeightedRandomSampler` while fine-tuning an ImageNet-pretrained ResNet-50. It does not require training-set group labels, but it does use validation-set group annotations to construct the reference distribution.

### Waterbirds settings
- Official train/validation/test splits.
- Reference groups: `0:0, 1:466, 2:133, 3:0`.
- ResNet-50, SGD, batch size 128, learning rate `1e-5`, weight decay `1.0`, 300 epochs and standard augmentation.
- Checkpoint selection: maximum validation WGA, with the earliest epoch as tie-breaker.
- Paper target: held-out test WGA `90.9 ± 0.3%` over three seeds.

### Current evidence
Seed 1 has a tracked 300-epoch train/validation trajectory. Its unique validation-WGA maximum is 90.10% at epoch 258; validation average accuracy is 91.60%, and validation group accuracies are [92.50%, 90.10%, 92.50%, 92.50%].

These are validation results, not held-out test results. The PR does not contain the generated `test.csv`, raw `val.csv`, `log.txt`, or `best_validation.pt`, so seed-1 test WGA, test accuracy, per-group test accuracy and group counts cannot yet be verified.

### Status
- **Original reproduction:** Partial reproduction. Training and validation-based model selection for seed 1 are documented; held-out test evaluation is pending.
- **Required for verified reproduction:** Recover the original seed-1 `test.csv` (preferred) or evaluate `best_validation.pt` once on the official test split, then complete seeds 2 and 3 and report the three-seed test mean ± sample standard deviation.
- **Detailed audit:** [`results/poter/report.md`](../results/poter/report.md).

## AutoBackSwap

### Overview
- **Full name:** Automated Background Swapping for Robustness against Spurious Backgrounds
- **Paper:** [arXiv:2606.32018](https://arxiv.org/abs/2606.32018)
- **Source:** https://github.com/cesro7/spur-corr (commit `b060532c2632039ffc2918828780d2b696763af6`)
- **Year:** 2026
- **Implementation location:** `baselines/autobackswap/`

### Mechanism
Builds new training images that break the background shortcut: a small detector finds the foreground, AOT-GAN inpaints the removed foreground to get clean backgrounds, and different foregrounds are pasted on them. The classifier is then trained on the original plus generated images. It does not need a single minority-group image in the training data.

### Implementation
- Backbone: ResNet50 (ImageNet weights) classifier; EfficientNet-B0 detector.
- Loss function: weighted cross-entropy with label smoothing 0.1.
- Training procedure: three steps (`--mode=ours_detector`, `ours_generate`, `ours`), then the same trainer as ERM on the augmented data.
- Special requirements: AOT-GAN submodule and its `G0000000.pt` weights, foreground segmentation masks, torch 2.11.0 / Python 3.12.

### Experimental Settings
- Dataset and split: Waterbirds, official split, test 5794 images.
- Configuration: `configs/autobackswap/waterbirds_autobackswap_resnet50.yaml`, `configs/autobackswap/waterbirds_erm_resnet50.yaml`.
- Batch size / learning rate / epochs / optimizer: 32 / 0.001 (ERM 0.003) / 50 / SGD momentum 0.9.
- Random seeds: 0, 1, 2 (AutoBackSwap with minority: 0 only).

### Evaluation
Mean over seeds (paper: mean ± std over 5 seeds). Per-run and per-group numbers, with group sizes, are in `results/autobackswap/summary.md`.

| Setting | Worst-group acc (ours / paper) | Overall acc (ours / paper) |
|---|---|---|
| AutoBackSwap, without minority (3 seeds) | 92.33 / 92.2 ± 0.4 | 94.49 / 95.1 ± 0.3 |
| AutoBackSwap, with minority (1 seed) | 92.37 / 93.0 ± 0.9 | 95.44 / 95.0 ± 0.5 |
| ERM, without minority (3 seeds) | 33.38 / 33.4 ± 4.4 | 68.68 / 68.4 ± 1.9 |
| ERM, with minority (3 seeds) | 78.45 / 74.9 ± 3.3 | 93.67 / 93.6 ± 0.4 |

### Status
- Original reproduction: **partial reproduction.** Worst-group accuracy is within 2σ of the paper in all four settings. Overall accuracy of AutoBackSwap without minority is 2.04σ below the paper (all three seeds below the paper mean); the cause was not isolated. Spawrious and Spurious Vehicles were not run.
- Waterbirds adaptation: none needed (Waterbirds is a native dataset of the method). Only a memory patch for small GPUs, kept separate in `baselines/autobackswap/patches/`.

### Notes
- fp32 instead of the default bf16 + `torch.compile`; chunked generation to fit a 15 GB GPU; WILDS copy of the data. All deviations are listed in `baselines/autobackswap/README.md`.
- The upstream `test_accuracies.txt` swaps the labels of the two middle groups. Worst group is Landbird on Water without minority (ERM and AutoBackSwap) and Waterbird on Land for ERM with minority.
- The upstream repository has no licence file at the pinned commit.
