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
3. [Verified Scope & Quickstart](#3-verified-scope--quickstart)
4. [Environment Setup](#4-environment-setup)
5. [Step-by-Step Waterbirds Reproduction](#5-step-by-step-waterbirds-reproduction)
6. [Expected Benchmark Results](#6-expected-benchmark-results)
7. [Repository Structure](#7-repository-structure)
8. [Limitations & Practical Tips](#8-limitations--practical-tips)
9. [References](#9-references)

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

## 3. Verified Scope & Quickstart

We have verified that the core DPE pipeline runs locally on Windows and Linux without cloud GPU dependencies.

### One-Command Quickstart Demo (< 10 seconds)
Run the self-contained reproduction demo:
```bash
python baselines/dpe4subpop/reproduce_demo.py
```

**Expected output:**
```text
============================================================
  Diverse Prototypical Ensembles (DPE) - Quickstart Demo
============================================================
[*] Running on device: CPU (or CUDA)
[*] Generating synthetic subpopulation benchmark data...
[*] Initializing DPE model...
[*] Fitting Diverse Prototypical Ensemble (4 stages)...
[Stage 1]: wga=85.4, acc=91.2
[Stage 2]: wga=91.1, acc=92.2
[Stage 3]: wga=89.3, acc=92.0
[Stage 4]: wga=90.6, acc=92.2
[*] Evaluating on Test Set...
============================================================
Overall Accuracy:       92.25%
Balanced Accuracy:      92.24%
Worst-Group Accuracy:   90.62%
============================================================
[+] DPE pipeline successfully reproduced and verified!
```

### Automated Unit Test
Run the automated smoke test from the repository root:
```bash
python tests/test_dpe_smoke.py
```
Expected: `Ran 2 tests in ~2.7s. OK`

---

## 4. Environment Setup

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

## 5. Step-by-Step Waterbirds Reproduction

Waterbirds reproduction follows a **2-stage pipeline**:

### Step 5.1: Dataset Structure
Ensure the Waterbirds dataset follows the standard SubpopBench directory format:
```text
data/waterbirds/
├── metadata_waterbirds.csv
└── [image directories]
```

### Step 5.2: Stage-0 Backbone Training & Feature Pre-extraction
Train an ImageNet-pretrained ResNet-50 via standard ERM, then extract feature representations:
```bash
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds \
  --data_dir data/waterbirds \
  --ckpt_dir checkpoints/waterbirds_erm \
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
This generates `feats_val.npy` and `feats_test.npy` in `checkpoints/waterbirds_erm/`.

### Step 5.3: Stage-1+ Diverse Prototypical Ensemble Training
Train the 16-stage prototype ensemble on pre-extracted embeddings (runs in ~2 minutes):

**Attribute-Agnostic Setting (No Group Labels Used):**
```bash
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds \
  --data_dir data/waterbirds \
  --pretrained_path checkpoints/waterbirds_erm/ckpt_last.pt \
  --ckpt_dir checkpoints/waterbirds_dpe \
  --loss_name isomax \
  --stage 1 \
  --num_stages 16 \
  --epochs 20 \
  --batch_size 256 \
  --lr 0.001 \
  --train_attr no \
  --subsample_type class \
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
  --data_dir data/waterbirds \
  --pretrained_path checkpoints/waterbirds_erm/ckpt_last.pt \
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

## 6. Expected Benchmark Results

Reported results from the ICML 2025 paper:

| Method | Group Info (Train / Val) | Waterbirds Worst-Group Acc | Waterbirds Overall Acc |
| :--- | :---: | :---: | :---: |
| **Standard ERM** | None / None | 69.1% ± 1.8% | 97.2% ± 0.2% |
| **ERM\*** (Enhanced Backbone) | None / None | 77.9% ± 3.0% | 96.9% ± 0.3% |
| **GroupDRO** | Train & Val | 89.9% ± 0.9% | 93.5% ± 0.3% |
| **DFR** | Val only | 92.9% ± 0.4% | 94.0% ± 0.3% |
| **DPE (Ours, Unannotated)** | **None / None** | **94.1% ± 0.2%** | **96.8% ± 0.2%** |
| **DPE (Ours, Annotated Val)** | None / Val | **94.1% ± 0.4%** | **96.8% ± 0.2%** |

*Note: DPE outperforms or matches group-supervised methods while requiring zero subgroup metadata during training.*

---

## 7. Repository Structure

```text
baselines/dpe4subpop/
├── README.md               # This reproduction guide
├── LICENSE                 # Upstream MIT License
├── requirements.txt        # Dependencies
├── main.py                 # Upstream CLI training entry point
├── reproduce_demo.py       # Self-contained verification script (<10s)
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

## 8. Limitations & Practical Tips

1. **Feature Extraction Efficiency:** Because Stage 1+ trains only on pre-extracted feature vectors, running multiple ensemble members is extremely fast (seconds per epoch). Always ensure feature normalization (`norm_emb=yes`) is enabled.
2. **Hyperparameter Tuning:**
   - `--cov_reg`: Controls inter-prototype orthogonality. Optimal range is $10^4$ to $5 \times 10^5$. Setting it too high can impede convergence; setting it too low collapses diversity.
   - `--entropic_scale`: Controls prototype distance sharpness. For Waterbirds, `30` is recommended.
3. **Backbone Dependency:** DPE operates on representations extracted from the Stage-0 backbone. If the backbone completely eliminates non-spurious features, prototype heads cannot recover them.

---

## 9. References

```bibtex
@inproceedings{to2025diverse,
  title={Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift},
  author={To, Minh Nguyen Nhat and Wilson, Paul F. R. and Nguyen, Viet and Harmanani, Mohamed and Cooper, Michael and Fooladgar, Fahimeh and Abolmaesumi, Purang and Mousavi, Parvin and Krishnan, Rahul G.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
