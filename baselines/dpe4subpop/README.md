# Diverse Prototypical Ensembles (DPE) — Baseline Reproduction Guide

> **Preserved external implementation for benchmark comparison.**  
> Based on the official code for **"Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift"** by **Minh Nguyen Nhat To, Paul F. R. Wilson, Viet Nguyen, Mohamed Harmanani, Michael Cooper, Fahimeh Fooladgar, Purang Abolmaesumi, Parvin Mousavi, and Rahul G. Krishnan** (ICML 2025).  
> **Original repository:** https://github.com/minhto2802/dpe4subpop  
> **Preserved upstream commit:** `68b811461e238d44cced6ae6c8ce75b66cad5449`  
> **Original paper:** https://arxiv.org/abs/2505.23027  
> **Upstream README:** [`docs/README.upstream.md`](docs/README.upstream.md)  
> **License:** MIT License ([`LICENSE`](LICENSE))

---

## Table of Contents

1. [Overview & Motivation](#1-overview--motivation)
2. [How DPE Works](#2-how-dpe-works)
3. [Environment Setup](#3-environment-setup)
4. [Step-by-Step Waterbirds Reproduction](#4-step-by-step-waterbirds-reproduction)
5. [Expected Benchmark Results](#5-expected-benchmark-results)
6. [Repository Structure](#6-repository-structure)
7. [Limitations & Practical Tips](#7-limitations--practical-tips)
8. [References](#8-references)

---

## 1. Overview & Motivation

### The Problem
Deep neural networks trained via Empirical Risk Minimization (ERM) on datasets with spurious correlations often learn "shortcuts" instead of core causal features. In the **Waterbirds** benchmark:
- **Target task ($y$):** Classify bird as *Waterbird* ($y=1$) or *Landbird* ($y=0$).
- **Spurious attribute ($a$):** Background is *Water* ($a=1$) or *Land* ($a=0$).
- In training, 95% of waterbirds appear on water and 95% of landbirds appear on land.
- A standard ERM classifier relies on the background shortcut and fails catastrophically on minority groups (waterbirds on land or landbirds on water), resulting in poor **Worst-Group Accuracy (WGA)** (~69%–78%).

### Why Single Linear Heads Fail
Prior methods (e.g. GroupDRO, DFR) typically retrain or constrain a single linear classification head ($W^T f(x) + b$). A single linear hyperplane inevitably forces a compromise that collapses into the dominant majority clusters. Furthermore, many methods require explicit group annotations during training.

### The DPE Solution
**Diverse Prototypical Ensembles (DPE)** eliminates the need for subgroup annotations during training. Instead of a single linear head, DPE freezes the feature extractor and trains an **ensemble of prototype-based distance classifiers**. By explicitly enforcing diversity across prototype heads through **covariance regularization** and **balanced data subsampling**, individual ensemble members specialize in different latent subpopulation combinations.

---

## 2. How DPE Works

```text
Input Features z = f(x) in R^d (from frozen ResNet-50)
              |
    +---------+---------+---------+ ... (Stage 1 to N)
    |                   |                   |
Prototype Head 1   Prototype Head 2   Prototype Head N
 (p^(1), |d_s1|)    (p^(2), |d_s2|)    (p^(N), |d_sN|)
    |                   |                   |
    +---- Covariance Regularization --------+ (Penalizes prototype alignment)
    |                   |                   |
Softmax Prob P_1   Softmax Prob P_2   Softmax Prob P_N
    \                   |                  /
     +------------------+-----------------+
                        |
            Probability Averaging:
       P_ens(y|x) = (1/N) * sum_j P_j(y|x)
```

1. **Distance-Based Classification (IsoMax+):**
   Features $z$ and class prototypes $p_k$ are normalized onto the unit hypersphere:
   $$D(z, p_k) = |d_s| \cdot \left\lVert \frac{z}{\|z\|_2} - \frac{p_k}{\|p_k\|_2} \right\rVert_2$$
   Logits are scaled negative distances $-\tau \cdot D(z, p_k)$. The entropic loss maximizes distance to non-target prototypes.

2. **Dual Diversification:**
   - **Data Diversification:** Each prototype stage is trained on a different resampled bootstrap subset. When group labels are unavailable, it resamples class-balanced subsets (`subsample_type='class'`).
   - **Inter-Prototype Similarity / Covariance Regularization:** For stage $j$, newly trained prototypes are penalized for having high inner product with frozen prototypes from earlier stages:
     $$\mathcal{L}_{\text{cov}} = \frac{1}{K} \sum_{k=1}^K \frac{1}{j-1} \sum_{m=1}^{j-1} \frac{\left| \langle p_k^{(j)}, p_k^{(m)} \rangle \right|}{d - 1}$$
   - **Covariance Scheduler:** $\lambda_{\text{cov}}(j) = \frac{\lambda_0}{1 + \alpha j}$ gradually loosens the penalty as more members are added.

3. **Inference:**
   Softmax output distributions across all $N$ heads are averaged to produce the final classification.

---

## 3. Environment Setup

### Dependencies
Install the required dependencies:
```bash
pip install -r baselines/dpe4subpop/requirements.txt
```

Key packages:
- `torch>=2.0`
- `torchvision>=0.15`
- `netcal>=1.3.5` (used for ECE metric computation)
- `jsonargparse>=4.37.0` (CLI argument parsing)
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `tqdm`
- `wandb` (optional, can be disabled with `--no_wandb`)

---

## 4. Step-by-Step Waterbirds Reproduction

Waterbirds reproduction follows a **2-stage pipeline**:

### Step 4.1: Dataset Structure
Ensure the Waterbirds dataset follows the standard SubpopBench directory format:
```text
data/waterbirds/
├── metadata_waterbirds.csv
└── waterbird_complete95_forest2water2/
    └── [image directories]
```

### Step 4.2: Stage-0 Backbone Training & Feature Pre-extraction
Train an ImageNet-pretrained ResNet-50 via ERM for 100 epochs, then extract feature representations (`feats_val.npy` and `feats_test.npy`).

There are two evaluation tracks:

#### Track A: Standard ERM Track (Canonical SubpopBench Benchmark)
Pass `--no_augmentation` to enforce deterministic evaluation preprocessing (`Resize(256) -> CenterCrop(224)`) without random photometric distortions:
```bash
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds \
  --data_dir data \
  --ckpt_dir checkpoints/waterbirds_stage0_standard_erm \
  --model_name resnet50 \
  --pretrained_imgnet \
  --train_mode full \
  --loss_name ce \
  --epochs 100 \
  --lr 0.001 \
  --batch_size 128 \
  --no_augmentation \
  --force_saving_feats \
  --no_wandb \
  --seed 0
```
*Expected Stage 0 WGA: ~62%–69%.*

#### Track B: Historical Augmented ERM Adaptation
Omit `--no_augmentation` to enable random resized crops, photometric distortion and horizontal flips during Stage 0 training. The upstream Waterbirds launcher uses 400 epochs for the enhanced recipe; the recorded project reproduction used a shorter 100-epoch augmented adaptation:
```bash
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds \
  --data_dir data \
  --ckpt_dir checkpoints/waterbirds_stage0_erm \
  --model_name resnet50 \
  --pretrained_imgnet \
  --train_mode full \
  --loss_name ce \
  --epochs 100 \
  --lr 0.001 \
  --batch_size 128 \
  --force_saving_feats \
  --no_wandb \
  --seed 0
```
*Expected Stage 0 WGA: ~77%–78%.*

### Step 4.3: Stage-1+ Diverse Prototypical Ensemble Training
Train 15 prototype heads (stages 1 through 15; `num_stages=16` is exclusive) on pre-extracted embeddings (runs in ~2 minutes):

**Class-balanced head fitting (`train_attr no`; WGA selection still uses group labels):**
```bash
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds \
  --data_dir data \
  --pretrained_path checkpoints/waterbirds_stage0_standard_erm/ckpt_last.pt \
  --ckpt_dir checkpoints/waterbirds_dpe \
  --loss_name isomax \
  --stage 1 \
  --num_stages 16 \
  --epochs 20 \
  --batch_size 256 \
  --lr 0.001 \
  --train_attr no \
  --subsample_type group \
  --cov_reg 5.e5 \
  --entropic_scale 30 \
  --train_mode freeze \
  --ensemble_criterion wga_val \
  -ncbt \
  -sit \
  --no_wandb \
  --seed 0
```

**Attribute-Aware Validation Setting (`train_attr yes`):**
```bash
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds \
  --data_dir data \
  --pretrained_path checkpoints/waterbirds_stage0_standard_erm/ckpt_last.pt \
  --ckpt_dir checkpoints/waterbirds_dpe \
  --loss_name isomax \
  --stage 1 \
  --num_stages 16 \
  --epochs 20 \
  --batch_size 256 \
  --lr 0.001 \
  --train_attr yes \
  --subsample_type group \
  --cov_reg 5.e5 \
  --entropic_scale 30 \
  --train_mode freeze \
  --ensemble_criterion wga_val \
  -ncbt \
  -sit \
  --no_wandb \
  --seed 0
```

---

## 5. Expected & Reproduced Benchmark Results

### Benchmark Comparison (Waterbirds)

| Method / Track | Group Info (Train / Val) | Paper Worst-Group Acc | Paper Overall Acc | Reproduced WGA (Seed 0) | Reproduced Overall Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard ERM** (`--no_augmentation`) | None / None | 69.1% ± 4.7% | 84.1% ± 1.7% | **62.5%** | 88.6% |
| **Standard ERM + DPE (annotated)** | None / Val | 91.0% ± 0.4% | 92.5% ± 0.2% | **89.0%** | 91.4% |
| **Enhanced $\text{ERM}^*$** (with Photometric Distort) | None / None | 77.9% ± 3.0% | 92.1% ± 0.2% | **77.1%** | 93.6% |
| **Enhanced $\text{ERM}^*$ + DPE (annotated)** | None / Val | 94.1% ± 0.4% | 96.0% ± 0.1% | **92.7%** | 93.3% |
| **GroupDRO** (reference) | Train & Val | 89.9% ± 0.9% | 93.5% ± 0.3% | — | — |
| **DFR** (reference) | Val only | 92.9% ± 0.4% | 94.0% ± 0.3% | — | — |

The DPE paper values above use its annotated-validation Table 2. Table 1 reports unannotated WGA 91.0 ± 0.5 for ERM + DPE and 94.1 ± 0.2 for ERM* + DPE. ERM/ERM* describe backbone recipes, independently of annotation availability. Historical reproduced ERM values above require re-evaluation: the original project notebook fed standardized DPE features to the raw-feature ERM head. The DPE head path already used its intended standardized inputs. Preserve those historical numbers as records rather than relabeling them as corrected measurements.

*Note: All runs, ablation studies (Covariance Regularization vs. Random Subsets Only vs. Single Fixed Subset), and logs are indexed in [`docs/experiment_log.md`](../../docs/experiment_log.md).*

---

## 6. Repository Structure

```text
baselines/dpe4subpop/
├── README.md               # This reproduction guide
├── LICENSE                 # Upstream MIT License
├── requirements.txt        # Dependencies
├── main.py                 # Upstream CLI training entry point
├── docs/
│   ├── README.upstream.md  # Original upstream README
│   └── index.html          # Upstream ICML project page
├── src/dpe/                # Core DPE library
│   ├── core.py             # DPE high-level orchestrator
│   ├── isomaxplus.py       # IsoMax+ distance classifier and loss
│   ├── datasets.py         # Subsampling and feature dataset loaders
│   └── eval_helpers.py     # Evaluation metrics (WGA, balanced acc)
├── utils/                  # Models, optimizers, and SubpopBench loaders
└── scripts/                # Original SLURM launch scripts
```

---

## 7. Limitations & Practical Tips

1. **Feature Extraction Efficiency:** Because Stage 1+ trains only on pre-extracted feature vectors, running multiple ensemble members is extremely fast (seconds per epoch). Always ensure feature normalization (`norm_emb=yes`) is enabled.
2. **Hyperparameter Tuning:**
   - `--cov_reg`: Controls inter-prototype orthogonality. Optimal range is $10^4$ to $5 \times 10^5$. Setting it too high can impede convergence; setting it too low collapses diversity.
   - `--entropic_scale`: Controls prototype distance sharpness. For Waterbirds, `30` is recommended.
3. **Backbone Dependency:** DPE operates on representations extracted from the Stage-0 backbone. If the backbone completely eliminates non-spurious features, prototype heads cannot recover them.

---

## 8. References

```bibtex
@inproceedings{to2025diverse,
  title={Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift},
  author={To, Minh Nguyen Nhat and Wilson, Paul F. R. and Nguyen, Viet and Harmanani, Mohamed and Cooper, Michael and Fooladgar, Fahimeh and Abolmaesumi, Purang and Mousavi, Parvin and Krishnan, Rahul G.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```



## Reproduction integration corrections

The project notebook `notebooks/dpe_waterbirds_end_to_end.ipynb` contains the runtime fixes; preserved upstream Python files remain unchanged.

- ERM test evaluation now passes raw `feats_test.npy` to its linear head. DPE continues to use row-standardized features. Previously both used standardized features, which could change the ERM baseline prediction.
- Prototype inference preserves the batch dimension for a final batch containing one example. Prototype classification loss uses stable log-softmax cross entropy at the same entropic scale, and nonfinite loss is rejected.
- CE-to-prototype checkpoint loading checks that only the intended head keys differ. Backbone mismatches fail instead of being silently accepted.
- Existing feature caches require matching checkpoint, metadata, row-order and feature-file hashes. Legacy feature caches without this evidence are re-extracted. Backbone reuse requires `stage0_complete.json`, written only after successful training, with matching checkpoint/metadata hashes and command. Partial or legacy checkpoints without that record are rejected; preserve them and use a fresh `ckpt_dir`. Missing features no longer retrain a verified completed backbone.
- Standard descriptive YAML now matches the executed notebook: zero SGD weight decay, evaluation batch size 256, annotated prototype training, and matching checkpoint paths. These nested YAML files are descriptive; upstream `main.py` does not automatically ingest them.
- Upstream CLI `--data_dir` takes the parent `data` directory. Its parser accepts `--subsample_type group`, with `--train_attr no` selecting the class-balancing target internally; `--subsample_type class` is rejected by the parser. The displayed class-balanced command still uses group labels for WGA selection, so it is not an end-to-end annotation-free protocol.

After these fixes, retain old result directories and perform a fresh evaluation/run before updating reported numbers. A passing CPU regression test is not a new Waterbirds benchmark result. Tests execute the notebook's actual evaluation functions with raw-feature, singleton-batch, corrupt-cache and checkpoint-mismatch fixtures:

```bash
python -m pytest tests/methods/dpe_reproduction -q
```

To correct ERM metrics from a retained legacy backbone without retraining it or modifying the historical notebook outputs, use the shared image-based evaluator. It applies the original image preprocessing and forwards raw backbone embeddings to the CE head:

```bash
python scripts/evaluate_erm.py \
  --checkpoint checkpoints/waterbirds_stage0_standard_erm/ckpt_last.pt \
  --data-dir data/waterbirds --split test \
  --output results/dpe4subpop/erm_test_corrected.json
```

The dataset images and original checkpoint are required. This separate evaluation does not require a new completion record and does not certify the legacy checkpoint's training history; record that history from the retained run log. Do not overwrite the historical result with an unlabelled replacement.
