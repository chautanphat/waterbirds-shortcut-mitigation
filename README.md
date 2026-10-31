# POTER — Local Reproduction Guide (Windows + NVIDIA GPU)

**Shared branch:** [`phat/poter-windows-cmnist`](https://github.com/chautanphat/waterbirds-shortcut-mitigation/tree/phat/poter-windows-cmnist) in [chautanphat/waterbirds-shortcut-mitigation](https://github.com/chautanphat/waterbirds-shortcut-mitigation).

This branch includes the source changes described in Section 5. The local smoke-test observations below come from the supplied reproduction report. See also the [Vietnamese overview](docs/README.vi.md) and the [original upstream README](docs/README.upstream.md).

Clone this branch, including the Windows fixes:

```cmd
git clone --branch phat/poter-windows-cmnist --single-branch https://github.com/chautanphat/waterbirds-shortcut-mitigation.git
cd POTER
```

> **Community reproduction notes, not the official POTER implementation.**  
> Based on the official code for **“Optimal Transport Reweighting for Robust Learning under Spurious Correlations and Label Noise”** by **Sung Ho Jo, Seonghwi Kim, Wonsang Yun, and Minwoo Chae** (NeurIPS 2026).  
> **Original repository:** https://github.com/Sung-Ho-Jo/POTER  
> **Paper:** https://arxiv.org/abs/2610.01028

This README documents how to **set up, debug, and run POTER locally on Windows**, including the small source-code changes needed in our local test. The goal is to help another developer reproduce the working environment **without needing a cloud GPU**.

**Verified scope:** the local CMNIST pipeline completed **one training epoch** and printed `Training complete.` on the configuration below. This is a **successful smoke test**, not a reproduction of the paper's 300-epoch benchmark or its reported accuracy.

---

## Table of contents

1. [What POTER does](#1-what-poter-does)
2. [What was tested](#2-what-was-tested)
3. [Repository structure](#3-repository-structure)
4. [Windows setup](#4-windows-setup)
5. [Required local source changes](#5-required-local-source-changes)
6. [Download and prepare Colored MNIST](#6-download-and-prepare-colored-mnist)
7. [Extract pretrained features](#7-extract-pretrained-features)
8. [Train and evaluate POTER](#8-train-and-evaluate-poter)
9. [Understand the logs and metrics](#9-understand-the-logs-and-metrics)
10. [Run an ERM baseline](#10-run-an-erm-baseline)
11. [Optional: label-noise experiments](#11-optional-label-noise-experiments)
12. [Troubleshooting](#12-troubleshooting)
13. [Reproducibility and limitations](#13-reproducibility-and-limitations)
14. [Sharing the repository](#14-sharing-the-repository)
15. [Credits and references](#15-credits-and-references)

---

## 1. What POTER does

### The problem

In **spurious-correlation learning**, a classifier may rely on a shortcut instead of the feature we actually care about. In **Colored MNIST (CMNIST)**, for example:

- The prediction target is whether a digit belongs to **0–4** or **5–9**.
- The color (**red** or **green**) is a spurious attribute correlated with the label in training.
- That correlation can change at evaluation time.
- Some training labels may also be noisy, making high-loss samples an unreliable signal of which examples deserve more attention.

### The idea

**POTER = Potentials from Optimal Transport for Empirical Reweighting.** It compares training representations with a **reference distribution** built from a limited validation set with group annotations. Class-conditioned optimal transport produces a dual-potential score for each training example. POTER converts these scores into importance weights: samples that align better with the reference are generally assigned larger weights.

Conceptually:

```text
Training images                         Validation images + group information
       |                                               |
       +------ pretrained feature extractor -----------+
                               |
                        Feature vectors
                               |
                  Class-conditioned Optimal Transport
                               |
                       OT dual potentials
                               |
                   Per-sample importance weights
                               |
                    WeightedRandomSampler
                               |
               Full fine-tuning of a ResNet-50 model
                               |
                Validation / test group accuracies
```

**Important implementation detail:** in this repository, POTER weights are passed into PyTorch's **`WeightedRandomSampler`** to change how frequently training examples are sampled. The POTER path does **not** simultaneously pass those weights directly into the loss: `run_expt.py` explicitly sets `sample_importance=None` for the training loop. This is different from saying the code directly multiplies every loss by its weight.

The main training model starts with **ImageNet-pretrained ResNet-50**, replaces its final classification layer, and **fine-tunes the full network** with SGD. This is **not** random-initialization training from scratch.

POTER itself is a **data reweighting procedure**, not a new neural-network architecture.

---

## 2. What was tested

This local test was performed with:

| Component | Local setup |
|---|---|
| Computer | MSI Thin GF63 12VE laptop |
| Operating system | Windows 64-bit |
| CPU | Intel Core i5-12450H |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Dedicated VRAM | **6 GB** |
| RAM | **16 GB** |
| Python | **3.10**, managed with Conda |
| PyTorch | **2.5.1** with CUDA 12.1 packages |
| torchvision | **0.20.1** |
| Dataset | CMNIST |
| Feature extraction model | Pretrained ResNet-50, **2,048-dimensional** features |
| Training batch size | **16** (chosen for the 6 GB GPU) |
| Smoke test | **1 epoch completed** |

The paper's reference hardware was an **RTX 3090**. A laptop GPU with 6 GB VRAM is therefore a **reduced-resource configuration**; a smaller batch size is recommended, and full 300-epoch reproduction may take considerably longer.

The working folder structure in this guide is illustrative. You can use other drive letters or locations by replacing the paths consistently:

```text
C:\POTER\                      # Source code (example)
D:\conda_envs\poter\           # Conda environment
D:\conda_pkgs\                # Conda download/extraction cache (optional)
D:\POTER_data\CMNIST\         # Dataset; created automatically
D:\POTER_features\            # Precomputed feature cache
D:\POTER_logs\                # Training output
D:\torch_cache\               # Downloaded pretrained PyTorch weights
D:\temp\                      # Optional temporary directory
```

> In the original successful run, the source checkout was at `C:\Users\ADMIN\POTER`. The commands below assume you have opened a terminal **inside your own cloned POTER repository**; they do not hard-code another person's Windows username.

---

## 3. Repository structure

The following files come from the official implementation:

```text
POTER/
├── README.md                 # This local reproduction guide, if added to your fork
├── extract_features.py       # Create pretrained feature cache
├── run_expt.py               # Experiment entry point
├── ot_reweight.py            # OT potentials and sample weights
├── train.py                  # Training + validation + test loop
├── loss.py                   # Loss computation and group statistics
├── models.py                 # Model configuration
├── noise.py                  # Training-label corruption experiments
├── utils.py                  # Logging, accuracy, seeding
├── data/
│   ├── cmnist_dataset.py     # Generates and loads Colored MNIST
│   ├── dro_dataset.py        # Dataset wrapper; WeightedRandomSampler
│   └── ...
└── scripts/
    └── run_experiments.sh    # Bash scripts in original project
```

The examples in this README use **Windows CMD / Anaconda Prompt**, not Bash. In Windows CMD, a long command can be placed on a **single line**. Do not copy Unix-style `\` line continuations into CMD.

---

## 4. Windows setup

### 4.1. Install Miniconda

Install **Miniconda for Windows (64-bit)** from:

https://www.anaconda.com/docs/getting-started/miniconda/install

Open **Anaconda Prompt / Miniconda Prompt** (not a terminal where `conda` is unavailable).

```cmd
conda --version
```

If Conda reports `CondaToSNonInteractiveError`, read and, **if you agree to the terms**, accept the requested channels:

```cmd
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### 4.2. Keep Conda packages off a nearly full C: drive (optional)

Use this section if the C: drive has little available space but D: has enough room. Run in **Anaconda Prompt**:

```cmd
mkdir D:\conda_pkgs
mkdir D:\conda_envs
mkdir D:\temp
conda config --add pkgs_dirs D:\conda_pkgs
set "TEMP=D:\temp"
set "TMP=D:\temp"
```

- `conda config --add pkgs_dirs ...` changes the package cache configuration.
- `set TEMP=...` and `set TMP=...` affect **only the current terminal session**.
- Do **not** manually move/delete a live Conda environment.
- Leave free space on both the system and data drives. CUDA, Conda extraction, logs, and model checkpoints need additional disk space.

### 4.3. Create and activate Python environment on D:

```cmd
conda create -p D:\conda_envs\poter python=3.10 -y
conda activate D:\conda_envs\poter
python --version
```

If `D:\conda_envs\poter` already exists, **do not create it again**; just activate it.

Expected prompt resembles:

```text
(D:\conda_envs\poter) C:\path\to\POTER>
```

### 4.4. Install PyTorch with GPU support

The **successful setup used Conda**:

```cmd
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Do not install a second copy of PyTorch using pip over this Conda installation. If you choose a different PyTorch/CUDA build, ensure it is compatible with your NVIDIA driver.

Check that PyTorch can see the GPU:

```cmd
nvidia-smi
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
```

Expected key output:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

The exact GPU model will depend on your own computer.

### 4.5. Install additional Python packages

With the Conda environment activated:

```cmd
python -m pip install --no-cache-dir "numpy<2" scipy pandas tqdm pillow scikit-learn transformers==4.44.2 pytorch-transformers
```

`numpy<2` was used as a compatibility precaution in the local environment; it is **not** a requirement explicitly pinned by the paper. The official repository lists package names but not a complete pinned Windows environment.

Check imports:

```cmd
python -c "import torch,torchvision,numpy,pandas,scipy,sklearn,transformers,tqdm,PIL; print('Dependencies imported successfully')"
```

### 4.6. Clone the upstream code

For this shared version, clone the branch with the Windows fixes already applied:

```cmd
git clone --branch phat/poter-windows-cmnist --single-branch https://github.com/chautanphat/waterbirds-shortcut-mitigation.git
cd POTER
```

If you already have the code, simply open Anaconda Prompt in the repository folder. **Do not reclone over existing uncommitted edits.**

---

## 5. Required local source changes

The changes in this section document the edits that enabled the **successful Windows CMNIST smoke test**. **Apply them to a fresh upstream clone**, unless your fork already contains the fixes. Preserve these changes when committing your modified code; a README alone cannot fix the upstream source.

### 5.1. Fix target-label dtype for CrossEntropyLoss (`train.py`)

**Observed error:**

```text
RuntimeError: "nll_loss_forward_no_reduce_cuda_kernel_index" not implemented for 'Int'
```

`CrossEntropyLoss` with class-index targets requires `torch.int64` (`LongTensor`), but one CMNIST path supplied an `Int` tensor.

In `train.py`, inside `run_epoch()`, find:

```python
batch = tuple(t.cuda() for t in batch)
x = batch[0]
y = batch[1]
g = batch[2]
idx = batch[3]
```

Replace it with:

```python
batch = tuple(t.cuda() for t in batch)
x = batch[0]
y = batch[1].long()    # target labels: required by CrossEntropyLoss
g = batch[2].long()    # group indices
idx = batch[3].long()  # sampling / dataset indices
```

**Essential fix:** `y = batch[1].long()`; `g`/`idx` casts also make the index types explicit. In the reported local run, applying this fix allowed training to proceed and complete the first epoch.

### 5.2. Use `num_workers=0` on Windows (`run_expt.py`)

In `run_expt.py`, locate the DataLoader configuration:

```python
loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4,
                 'pin_memory': torch.cuda.is_available()}
```

Change to:

```python
loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0,
                 'pin_memory': torch.cuda.is_available()}
```

This was used to simplify Windows DataLoader execution and diagnostics. It **may be slower** than using parallel workers. It was not independently proven to be the sole cause of the earlier stalled log.

### 5.3. Enable visible progress and checkpoints (`run_expt.py`)

In `parse_args()`'s fixed configuration block, change:

```python
args.show_progress = False
args.save_last = False
```

to:

```python
args.show_progress = True
args.save_last = True
```

Effects:

- A progress bar becomes visible during training.
- A `last_model.pth` file is written **after a completed epoch** to that run's log folder.
- In the current upstream code, the checkpoint is saved as a **full PyTorch model object** with `torch.save(model, ...)`. Only load checkpoint files you trust. Optimizer state is **not** saved for full training resumption.

For more frequent training log updates, pass `--log_every 50` in the command line. The upstream default `--log_every 10000` may show no mid-epoch statistics on CMNIST.

---

## 6. Download and prepare Colored MNIST

CMNIST is generated automatically by `data/cmnist_dataset.py` using MNIST. No manual zip download is required.

In **Anaconda Prompt**, while in the repository root:

```cmd
mkdir D:\POTER_data\CMNIST
mkdir D:\POTER_features
mkdir D:\POTER_logs
mkdir D:\torch_cache
set "TORCH_HOME=D:\torch_cache"
```

`TORCH_HOME` keeps downloaded ImageNet ResNet-50 weights on D: instead of `C:\Users\<user>\.cache\torch`. Set it **before running feature extraction and training** in each newly opened terminal.

On the first run, PyTorch may try an older MNIST URL that returns `HTTP Error 404`, then automatically fall back to a working mirror. The 404 message is harmless **if the fallback successfully downloads the files**.

The generated files should look similar to:

```text
D:\POTER_data\CMNIST\
├── MNIST\raw\...
└── ColoredMNIST\
    ├── train1.pt
    ├── val.pt
    └── test.pt
```

### Why reference group counts may differ

The upstream example asks for:

```text
--ref_per_group 0:2591,1:0,2:0,3:2431
```

However, the local dataset generated **2,588 validation samples in group 0**, so that request failed with:

```text
ValueError: Group 0 has only 2588 val samples, but ref_per_group requests 2591.
```

This is possible because the dataset-generation step contains randomness and the stored group counts can differ between generated datasets. **Do not fabricate reference samples or blindly use counts from another machine.**

Inspect the actual group counts with:

```cmd
python -c "import torch,collections; p=r'D:\POTER_data\CMNIST\ColoredMNIST\val.pt'; d=torch.load(p,weights_only=False); print(dict(sorted(collections.Counter(2*int(c)+int(y) for _,y,c in d).items())))"
```

Use this command **only for your own locally generated/trusted dataset**, because the file is loaded with Python pickle support.

In the tested environment, these were the validation counts:

| Group ID | Label / color | Validation count |
|---|---|---:|
| 0 | `y=0`, green (`isred=0`) | 2,588 |
| 1 | `y=1`, green (`isred=0`) | 2,474 |
| 2 | `y=0`, red (`isred=1`) | 2,500 |
| 3 | `y=1`, red (`isred=1`) | 2,438 |

Group IDs are derived as `group = 2 * isred + y`.

**Local smoke-test reference:** `0:2300,1:0,2:0,3:2300`. This is deliberately smaller than the reference sizes in the official README, so the local smoke test **is not an exact paper reproduction**.

---

## 7. Extract pretrained features

Feature extraction must complete **before** POTER training begins. It creates an offline cache for computing the OT potentials.

**Working CMNIST command (Windows CMD, single line):**

```cmd
python extract_features.py --dataset CMNIST --data_dir D:\POTER_data --ref_per_group 0:2300,1:0,2:0,3:2300 --output D:\POTER_features\cmnist_features.pt --gpu 0 --batch_size 32
```

Expected final message:

```text
Saved feature cache to: D:\POTER_features\cmnist_features.pt
```

The successful run produced:

```text
Train features: (30000, 2048)
Reference features: (4600, 2048)
```

In other words, the code computes **2,048-dimensional ResNet-50 embeddings** for **30,000 training** and **4,600 reference** examples. The feature extractor itself uses pretrained ImageNet weights; the cache is **not a trained POTER classifier**.

Check whether the cache exists:

```cmd
dir D:\POTER_features\cmnist_features.pt
```

> If extraction raises `ValueError: Group ... has only ...`, reduce the selected group's count based on your actual validation split and re-run extraction. If the command fails, **do not start `run_expt.py` yet**; it will report `FileNotFoundError` for the missing `.pt` cache.

---

## 8. Train and evaluate POTER

### 8.1. One-epoch smoke test (recommended first)

Activate the environment, enter the repo, and set the model cache path:

```cmd
conda activate D:\conda_envs\poter
cd /d C:\path\to\POTER
set "TORCH_HOME=D:\torch_cache"
```

Replace `C:\path\to\POTER` with your own repository location. Or omit the `cd` command when your prompt is already inside the repo.

Start the smoke test:

```cmd
python -u run_expt.py --dataset CMNIST --data_dir D:\POTER_data --ot_cache D:\POTER_features\cmnist_features.pt --f_power 0.6 --clip_cap 3 --n_epochs 1 --lr 1e-2 --weight_decay 1e-2 --batch_size 16 --seed 1 --gpu 0 --log_every 50 --log_dir D:\POTER_logs
```

Expected stages:

```text
[POTER] Loading feature cache: ...
[POTER] Computing OT potentials ...
[POTER] Computing sample weights ...
[POTER] Weight stats ...

Epoch [0]:
Training:
  ... progress bar ...
Validation:
  ... evaluation metrics ...
Training complete.
```

**Local smoke test's OT statistics:**

```text
[POTER] Train features: (30000, 2048), Val features: (4600, 2048)
[POTER] Weight stats - mean: 1.000, max: 3.011, min: 0.378
```

The exact weights depend on the generated dataset, chosen reference samples, and implementation settings.

**Validated outcome:** one epoch completed. The recorded **validation** metrics from that epoch were:

| Metric | Local 1-epoch value |
|---|---:|
| Average validation loss | 0.820 |
| Average validation accuracy | 53.0% |
| Group 0 validation accuracy | 11.5% |
| Group 1 validation accuracy | 96.0% |
| Group 2 validation accuracy | 53.3% |
| Group 3 validation accuracy | 52.6% |
| **Worst-group validation accuracy** | **11.5%** |

These are **early smoke-test numbers**, **not final performance claims** and **not test-set accuracy**. Large group disparities after one epoch are not proof that the method succeeds or fails.

### 8.2. Longer local experiment

Once a full epoch works, try a fresh 10-epoch run:

```cmd
python -u run_expt.py --dataset CMNIST --data_dir D:\POTER_data --ot_cache D:\POTER_features\cmnist_features.pt --f_power 0.6 --clip_cap 3 --n_epochs 10 --lr 1e-2 --weight_decay 1e-2 --batch_size 16 --seed 1 --gpu 0 --log_every 50 --log_dir D:\POTER_logs
```

To request the paper's **300 epochs**, change `--n_epochs 10` to `--n_epochs 300`.

**Caution:** a smaller batch size (`16` rather than the official CMNIST example's `128`) changes optimization. Combined with the modified reference set, this run **should not be presented as a bit-for-bit reproduction**. On 6 GB VRAM, test `16` first; reduce to `8` if CUDA reports out-of-memory.

Each invocation of `run_expt.py` **starts a new training run from pretrained initialization**. Changing `--n_epochs 1` to `10` does not resume the previous checkpoint. The upstream training loop does not implement a complete optimizer-state resume workflow.

### 8.3. Watch GPU memory

In a second Anaconda Prompt:

```cmd
nvidia-smi -l 2
```

The GPU utilization and memory usage can show whether training is active. PyTorch's CUDA memory usage may fluctuate; `0%` GPU utilization at a single instant is not conclusive evidence of a hang.

---

## 9. Understand the logs and metrics

Each run creates a timestamped output folder:

```text
D:\POTER_logs\
└── CMNIST\
    └── Clean\
        └── POTER_CMNIST_f0.6_cap3.0_reg0.03_wd0.01_lr0.01_ep1_nseed0_seed1_<timestamp>\
            ├── log.txt
            ├── train.csv
            ├── val.csv
            ├── test.csv
            └── last_model.pth   # only if args.save_last=True and an epoch completed
```

- `log.txt`: console-style setup and training/validation messages.
- `train.csv`: training statistics logged periodically and at epoch end.
- `val.csv`: per-epoch validation statistics.
- `test.csv`: per-epoch test statistics (test details need not appear in `log.txt`).
- `last_model.pth`: optional last completed-epoch full-model checkpoint.

**Metric definitions:**

- **Average accuracy:** fraction of correctly classified samples across the whole split.
- **Group accuracy:** fraction correct within a particular `(target, spurious attribute)` group.
- **Worst-group accuracy (WGA):** `min(group_accuracy_0, ..., group_accuracy_3)` for CMNIST. **This is the central robustness metric.**

To inspect the **newest non-empty test results** in CMD:

```cmd
python -c "import glob,os,pandas as pd; fs=glob.glob(r'D:\POTER_logs\CMNIST\Clean\**\test.csv',recursive=True); a=[(p,pd.read_csv(p)) for p in fs if os.path.getsize(p)>0]; a=[(p,d) for p,d in a if len(d)>0]; a.sort(key=lambda z:os.path.getmtime(z[0]),reverse=True); print('No completed test results yet' if not a else 'Run: '+a[0][0]); print(a[0][1].tail(1).to_string(index=False) if a else '')"
```

This prints the last test row with its **actual CSV column names**. For a specific completed run, open its `test.csv` or `val.csv` in a spreadsheet editor.

**Avoid a common error:** a `test.csv` that exists but has **zero data rows** does not mean the model achieved 0% accuracy. It generally indicates that evaluation has not written a completed epoch. In our earlier failed runs, the log stopped at `Epoch [0]: Training:` and the CSV files were empty.

**Do not accidentally peek at the test set during hyperparameter selection.** Use validation WGA to decide training settings, then report held-out test WGA separately.

### Checkpoint limitations

With `args.save_last=True`, the source calls:

```python
torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))
```

This saves a **serialized model object**, not a portable `state_dict`-only checkpoint. It may depend on matching Python/package source definitions. **Load only trusted checkpoints.** The optimizer and scheduler states are not stored, so the checkpoint alone is not sufficient to resume the original training trajectory exactly.

---

## 10. Run an ERM baseline

To determine whether POTER helps, compare it against **ordinary empirical risk minimization (ERM)** under the **same dataset, model, training epochs, batch size, learning rate, weight decay, and seed**.

Run ERM by **omitting `--ot_cache`**:

```cmd
python -u run_expt.py --dataset CMNIST --data_dir D:\POTER_data --n_epochs 10 --lr 1e-2 --weight_decay 1e-2 --batch_size 16 --seed 1 --gpu 0 --log_every 50 --log_dir D:\POTER_logs
```

Compare the **validation** curves and final held-out test results:

| Metric | ERM | POTER |
|---|---:|---:|
| Average validation accuracy | To measure | To measure |
| Worst-group validation accuracy | To measure | To measure |
| Average test accuracy | To measure | To measure |
| Worst-group test accuracy | To measure | To measure |

**Do not fill this table with the paper's published scores as though they were local measurements.** For a fair comparison, rerun both methods under the **same local configuration**, then evaluate across multiple seeds when practical.

---

## 11. Optional: label-noise experiments

The repository also supports artificially corrupting training labels.

**Symmetric noise** example with 20% flips:

```cmd
python -u run_expt.py --dataset CMNIST --data_dir D:\POTER_data --ot_cache D:\POTER_features\cmnist_features.pt --f_power 0.6 --clip_cap 3 --n_epochs 10 --lr 1e-2 --weight_decay 1e-2 --batch_size 16 --noise_type symmetric --noise_fraction 0.2 --noise_seed 0 --seed 1 --gpu 0 --log_dir D:\POTER_logs
```

**But:** this is an **exploratory example**, not a verified symmetric-noise reproduction setting. The official README explicitly requests **different reference compositions for symmetric noise** on Waterbirds, CelebA, and CivilComments. It does **not** provide a corresponding symmetric-noise reference recipe for CMNIST. Choose/justify the reference composition and re-extract the feature cache before making publication-quality claims.

For subgroup-concentrated noise, the official code supports `--noise_type subgroup` and `--group_flip_fractions`, but the subgroup choices and reference setup must match the experiment you intend to run. A 6 GB GPU has **not** been verified here for the full Waterbirds/CelebA/CivilComments benchmark suites.

---

## 12. Troubleshooting

| Symptom | Reason / solution |
|---|---|
| `'conda' is not recognized` | Miniconda installer was downloaded but not yet installed, or CMD cannot find Conda. Complete the installation and use **Anaconda Prompt**. |
| `CondaToSNonInteractiveError` | Review/accept the Terms of Service for the listed channels, or configure alternative channels you are permitted to use. |
| `EnvironmentNameNotFound: poter` | The environment was not created (for example, the ToS check blocked the command). Activate the path-based environment **only after** creation: `conda activate D:\conda_envs\poter`. |
| pip's huge PyTorch `.whl` download fails / `WinError 32` | Download was interrupted or the temporary file is locked. Use one PyTorch installation method; this guide's working environment used **Conda**. |
| `[Errno 28] No space left on device` / `InvalidArchiveError` | Not enough disk space while downloading/unpacking CUDA. Put environments, caches and temp files on a drive with enough space; run `conda clean --all -y` after failed installs when appropriate. |
| `HTTP Error 404` during MNIST download | Can be harmless if the dataset loader automatically retries a working mirror and successfully downloads MNIST. |
| `Group 0 has only 2588 ... requests 2591` | Reference sample count exceeds the **actual** generated validation group count. Inspect the groups and reduce `--ref_per_group`. |
| `FileNotFoundError: ...cmnist_features.pt` | Feature extraction failed before saving its output. Complete **Step 7** first. |
| `pretrained is deprecated` | A torchvision **warning**, not a fatal error. The upstream model constructor uses legacy `pretrained=True`. |
| `FutureWarning: torch.load(weights_only=False)` | Warning about pickle-based deserialization. Only load **trusted local files**. Do not blindly change the flag if the stored object requires full-object loading. |
| Training log stays at `Epoch [0]: Training:` | The upstream configuration disables progress and logs only every 10,000 batches. Enable `args.show_progress=True`, lower `--log_every`, and check whether the process is still active. |
| `nll_loss_forward_no_reduce_cuda_kernel_index not implemented for 'Int'` | Class-index labels have wrong dtype. Apply **`y = batch[1].long()` in `train.py`**. This was the critical proven local fix. |
| `torch.cuda.OutOfMemoryError` | Reduce training `--batch_size` from `16` to `8`, close other GPU-heavy applications, and inspect `nvidia-smi`. OT feature/reference sizes also affect memory. |
| `test.csv` exists but has no rows | No completed test evaluation has been written yet, or the run failed before that point. Check `log.txt` for `Training complete.`. |
| No `.pth` checkpoint | The official code uses `args.save_last=False` by default. Change to `True` **before** running, and finish an epoch. |
| `CUDA: False` | Verify NVIDIA driver and compatible CUDA-enabled PyTorch build inside the **active** environment. |

### General diagnostic commands

```cmd
conda info --envs
conda config --show pkgs_dirs
python --version
python -c "import torch; print(torch.__version__,torch.cuda.is_available())"
nvidia-smi
```

For an error, include the **full Python traceback**, the command used, the last 20–40 lines of the log, and the output of the diagnostic commands. This helps distinguish data/configuration bugs from CUDA or dependency problems.

---

## 13. Reproducibility and limitations

### What is established by this local run

- A Windows machine with **RTX 4050 Laptop GPU (6 GB)** can run the necessary CUDA operations.
- CMNIST was downloaded/generated and its data loader worked.
- ImageNet-pretrained ResNet-50 features were extracted and stored locally.
- POTER computed class-conditioned OT potentials and per-sample sampling weights.
- After the dtype fix in `train.py`, **one training epoch completed** and validation metrics were produced.

### What is **not** established yet

- Matching the paper's exact 300-epoch CMNIST accuracy.
- Reproduction using the paper's exact reference counts (the local data split differed).
- Equivalent optimization between local **batch size 16** and paper example **batch size 128**.
- Results averaged over multiple random seeds.
- Performance on other benchmarks (Waterbirds, CelebA, CivilComments) using this particular laptop.
- Evidence that POTER beats ERM locally: that requires a matched controlled experiment.
- A full interruption-safe training-resume system.

CMNIST's generated data are randomized. Preserve the generated split if you need reproducibility across runs, and record exact package versions, code commit, reference sample counts, and seeds. The feature-cache file must correspond to the **same dataset instance and sample ordering** used in training.

**Additional caution:** the code evaluates the test set after each epoch. For research reporting, avoid using repeated test feedback to tune hyperparameters.

---

## 14. Sharing the repository

If you intend to send this project to a friend via GitHub:

1. **Preserve the original authors' attribution** and the link to the upstream code and paper.
2. Put the **actual modified** `train.py` and `run_expt.py` in the repository; the README's instructions are not a substitute for committing those changes.
3. Prefer a **GitHub fork** of the original repo so the original provenance remains clear. If you keep the upstream README, this guide can instead be named `README_LOCAL.md` and linked near the top of the primary README.
4. **Check the original repository's license and redistribution permissions before publicly republishing modified source.** A publicly viewable GitHub repository does not automatically mean its contents have an open-source license.
5. Do not commit Conda environments, datasets, pretrained weights, `.pt` feature caches, `.pth` checkpoints, or large generated logs by accident. These can be large, and some datasets have their own usage requirements.

Suggested `.gitignore` additions (combine with the original project's `.gitignore`):

```gitignore
# Local datasets / generated artifacts
datasets/
features/
logs/
POTER_data/
POTER_features/
POTER_logs/
*.pt
*.pth
*.ckpt

# Local Conda / Python
.conda/
.venv/
__pycache__/
*.pyc

# Local experiment metadata
.env
```

To inspect changes before you upload:

```cmd
git status
git diff -- train.py run_expt.py
```

On **your own fork** (after verifying `git remote -v` points to your intended repository):

```cmd
git add README.md train.py run_expt.py .gitignore
git commit -m "Document Windows CMNIST reproduction and fix label dtype"
git push
```

If your checkout is still connected only to the original authors' repository, first create your **own GitHub fork** and configure the appropriate remote. Do not assume you have permission to push to upstream.

---

## 15. Credits and references

**Original paper:**  
Sung Ho Jo, Seonghwi Kim, Wonsang Yun, Minwoo Chae. *Optimal Transport Reweighting for Robust Learning under Spurious Correlations and Label Noise*. NeurIPS 2026.  
https://arxiv.org/abs/2610.01028

**Official implementation:**  
https://github.com/Sung-Ho-Jo/POTER

**PyTorch historical installation instructions (2.5.1):**  
https://pytorch.org/get-started/previous-versions/

**Miniconda installation:**  
https://www.anaconda.com/docs/getting-started/miniconda/install

**Additional dataset references** are listed in the upstream POTER README.

---

### Quick-start summary

After completing the source edits in **Section 5** and installing dependencies:

```cmd
conda activate D:\conda_envs\poter
set "TORCH_HOME=D:\torch_cache"

python extract_features.py --dataset CMNIST --data_dir D:\POTER_data --ref_per_group 0:2300,1:0,2:0,3:2300 --output D:\POTER_features\cmnist_features.pt --gpu 0 --batch_size 32

python -u run_expt.py --dataset CMNIST --data_dir D:\POTER_data --ot_cache D:\POTER_features\cmnist_features.pt --f_power 0.6 --clip_cap 3 --n_epochs 1 --lr 1e-2 --weight_decay 1e-2 --batch_size 16 --seed 1 --gpu 0 --log_every 50 --log_dir D:\POTER_logs
```

**Success condition:** the final training log contains `Training complete.` and the run has non-empty validation/test CSVs. A completed smoke test means the local pipeline works; meaningful research conclusions require longer, controlled comparisons.
