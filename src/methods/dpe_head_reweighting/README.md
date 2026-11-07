# DPE Head Reweighting

## Overview

This project-owned adaptation of Diverse Prototypical Ensembles (DPE) evaluates whether validation-derived head weights improve worst-group accuracy (WGA) on Waterbirds relative to uniform probability averaging. WGA is the minimum accuracy over the four bird/background groups. Both aggregation conditions use the same frozen ResNet-50 representation and the same 15 trained prototype heads.

The implementation includes sequential head training, reliability estimation, aggregation selection, artifact validation and held-out evaluation. Synthetic CPU checks have passed. Real Waterbirds experiments, GPU parity and any performance improvement remain unverified.

## Experimental protocol

Stage 0 consists of backbone training on the official Waterbirds training split. The present implementation starts from an existing final-epoch checkpoint and precomputed raw features; backbone training and feature extraction are external prerequisites.

The official validation split is partitioned within each bird/background group using split seed 0:

| Subset | Proportion | Role |
| --- | --- | --- |
| A | 60% | Head fitting and epoch selection. |
| B | 20% | Per-head reliability estimation and candidate weight construction. |
| C | 20% | Aggregation configuration selection. |

Integer counts use largest-remainder allocation with A/B/C tie order. The partition is identical across training seeds and across the standard ERM and enhanced ERM* tracks. Every subset must contain all four groups. Stage-0 checkpoint selection must exclude B, C and test labels; the supported provenance contract requires final-epoch selection without validation-based checkpoint selection.

The test split is reserved for evaluation after the aggregation configuration has been frozen. Heads trained on the complete validation split in the historical reproduction are unsuitable for this protocol. Comparisons use the matched A-trained uniform control, rather than historical DPE results with a different head-training data budget.

Each track comprises seeds 0, 1 and 2, with a matching Stage-0 backbone for each seed. Standard and enhanced results are reported separately. Track summaries contain means and sample standard deviations, including paired WGA differences.

## Detailed mechanism

### Feature representation and preprocessing

A Stage-0 ResNet-50 backbone produces a raw feature vector $f(x)\in\mathbb{R}^{2048}$. The backbone remains frozen throughout the experiment. Each image's feature coordinates are standardized before L2 normalization:

```math
u(x)=\frac{f(x)-\mathrm{mean}(f(x))}{\mathrm{std}(f(x))},
\qquad
z(x)=\frac{u(x)}{\lVert u(x)\rVert_2}.
```

The standard deviation uses `ddof=0`. Zero-variance rows are rejected; L2 normalization uses PyTorch's numerical safeguard for small norms. This sequence is applied once to raw features. Validation and test use the same preprocessing.

### Prototypical classification heads

Each head $h$ learns two 2048-dimensional vectors: a landbird prototype $v_{h,0}$ and a waterbird prototype $v_{h,1}$. It also learns one distance scale $d_h$. The IsoMax+ class logit is the negative Euclidean distance to the normalized class prototype, multiplied by the absolute scale:

```math
\ell_{h,c}(x)=-|d_h|\left\lVert z(x)-\frac{v_{h,c}}{\lVert v_{h,c}\rVert_2}\right\rVert_2,
\qquad c\in\{0,1\}.
```

A closer prototype gives a larger logit. Applying softmax across the two classes gives the head's inference probabilities:

```math
p_h(c\mid x)=\frac{\exp(\ell_{h,c}(x))}{\sum_{k=0}^{1}\exp(\ell_{h,k}(x))}.
```

Inference temperature is 1. The training entropic scale of 30, described below, is **not** multiplied into these inference logits.

### Sequential head training

Heads are trained sequentially. Each head receives an independently sampled, group-balanced subset of A, with the number of examples per group determined by A's smallest group. The subset remains fixed for that head's 20 epochs. Previously selected heads remain frozen.

The training loss contains three terms:

```math
\mathcal{L}_h=\mathcal{L}_{\mathrm{classification}}
+\mathcal{L}_{\mathrm{prototype}}
+\lambda_h R_{\mathrm{diversity}}.
```

The classification term is the mean negative log probability of the correct class, using the training-only entropic scale:

```math
\mathcal{L}_{\mathrm{classification}}
=-\frac{1}{M}\sum_{i=1}^{M}\log\left[\mathrm{softmax}(30\ell_h(x_i))\right]_{y_i}.
```

The implementation evaluates this objective with `log_softmax`, avoiding numerical underflow from taking the logarithm of a rounded-to-zero softmax probability. This changes numerical evaluation of the reference expression, not its mathematical objective.

The prototype penalty limits the size of the unnormalized prototype vectors:

```math
\mathcal{L}_{\mathrm{prototype}}=\frac{10}{2}\sum_{c=0}^{1}\lVert v_{h,c}\rVert_2^2.
```

The diversity term compares the current prototypes with previous heads' prototypes for the same class. With $D=2048$ and heads numbered from 1, the implemented expression is:

```math
R_{\mathrm{diversity}}
=\left|\frac{1}{2}\sum_{c=0}^{1}\frac{1}{h}\sum_{j=1}^{h-1}
\frac{v_{h,c}^{\mathsf T}v_{j,c}}{D-1}\right|,
\qquad
\lambda_h=\frac{500000}{1+0.05\max(0,h-2)}.
```

This preserves the reference notebook's regularizer, including its order of summing and taking the absolute value. It uses raw prototype dot products; it does not center the vectors or average absolute pairwise similarities. For the first head there are no previous prototypes, so the diversity term is absent. Different balanced samples and this regularizer aim to produce heads with different decision rules.

Only the current head is optimized, using SGD with learning rate `0.001`, momentum `0.9`, and batch size `256`. Incomplete training batches are dropped; a subset smaller than one complete batch is rejected. Epoch selection maximizes WGA over all of A, retaining the later epoch on ties. Thus, A is reused for both fitting and epoch selection, while B and C remain excluded from both operations.

### Reliability estimation

Waterbirds has four groups, defined by bird class $y$ and background $a$:

| Group | Bird | Background |
| --- | --- | --- |
| `00` (G1) | Landbird | Land |
| `01` (G2) | Landbird | Water |
| `10` (G3) | Waterbird | Land |
| `11` (G4) | Waterbird | Water |

For each frozen head, predictions are obtained by class-index argmax. Accuracy is calculated separately for each group in B. The reliability score is the minimum group accuracy:

```math
s_h=\min_{g\in\{00,01,10,11\}}\mathrm{accuracy}(h,B_g).
```

Scores are fractions between 0 and 1. This favors heads that perform reasonably across all groups rather than heads that succeed mainly on common backgrounds. Every group must be present; a missing group is an error.

### Weight construction

For $H=15$ heads, a score-based distribution $q_h$ is interpolated with uniform weights:

```math
q_h=\frac{s_h^\gamma}{\sum_{j=1}^{H}s_j^\gamma},
\qquad
w_h=\frac{1-\alpha}{H}+\alpha q_h.
```

`alpha` controls how far the weights move away from equal weighting. At `alpha = 0`, every head receives `1/15`; at `alpha = 1`, weights depend entirely on the scores. `gamma` controls how strongly higher scores dominate. Weights are nonnegative and sum to one. If all scores are zero, use equal weights and record the fallback.

For illustration, two heads with scores `[0.25, 0.75]` and `gamma = 2` receive score-based weights `[0.1, 0.9]`. Interpolation with uniform weights at `alpha = 0.5` produces `[0.3, 0.7]`.

### Aggregation selection

Weights are estimated exclusively on B. The following fixed configurations are evaluated on C:

- Equal weighting: `alpha = 0`, recorded with `gamma = 1`.
- Weighted averaging: `alpha ∈ {0.25, 0.5, 0.75, 1}` crossed with `gamma ∈ {1, 2, 4}`.

The grid contains 13 configurations. Selection maximizes C WGA, followed by C overall accuracy, smaller `alpha`, and smaller `gamma`, using numerical tolerance `1e-12`. The selected weights and input identities are persisted before test evaluation. Heads remain trained on A; weights are not subsequently refitted on B+C.

### Held-out evaluation and interpretation

The selected aggregation and uniform control combine the same frozen head probabilities:

```math
p_{\mathrm{weighted}}(c\mid x)=\sum_{h=1}^{H}w_h p_h(c\mid x),
\qquad
p_{\mathrm{uniform}}(c\mid x)=\frac{1}{H}\sum_{h=1}^{H}p_h(c\mid x).
```

Predictions use argmax over the combined probabilities, with exact ties assigned to class 0. Aggregation operates on probabilities. Backbone parameters, head parameters, features and example order are identical for both conditions; only the weights differ.

Evaluation reports overall accuracy, per-group accuracy, WGA and paired changes for each seed. The effective head count, $1/\sum_h w_h^2$, shows how concentrated the weights are: 15 for equal weighting, approaching 1 when one head dominates.

The diagnostic plots compare weights with $\min(\mathrm{acc}_{01},\mathrm{acc}_{10})$, accuracy on the two background-conflicting groups. An association on B is partly expected because B also determines the weights; C and test provide separate descriptive checks. Low weights alone do not establish that a head learned background features. Weighting can also hurt if individually weaker heads supply useful complementary predictions, which is why equal averaging remains a selection candidate.

The implementation is split between [ensemble.py](ensemble.py) (head predictions), [train_heads.py](train_heads.py) (A-only training), [reweighting.py](reweighting.py) (B scores and C selection), and [experiment.py](experiment.py) (data roles, saved records and evaluation).


## Input specification

### Metadata and features

The required artifacts are `metadata.csv`, `ckpt_last.pt`, `val_raw.npz`, `test_raw.npz` and `provenance.json`, all associated with the same track and Stage-0 seed. The checkpoint is hashed for provenance; the feature-only runner does not load it for inference. Training and selection do not consume test features. The notebook's full-run readiness check validates the test-feature file before training to detect input errors early; test predictions are computed only during final evaluation.

Metadata must contain `img_filename` (or `filename`), `y`, `place` (or `a`), and `split`. Split values are `0` for train, `1` for validation and `2` for test, with counts 4,795 / 1,199 / 5,794. Filenames must be unique.

Each feature NPZ must contain exactly two arrays:

- `ids`: filenames in the same order as that split's metadata rows.
- `features`: raw floating-point features, shaped `[1199, 2048]` for validation or `[5794, 2048]` for test.

Features must be raw backbone outputs, before row standardization or normalization. The runner performs preprocessing and rejects reordered, missing or duplicate IDs.

Existing `feats_val.npy` and `feats_test.npy` files may be converted after their checkpoint identity and metadata ordering have been verified:

```python
import csv
import numpy as np

with open("metadata.csv", newline="") as f:
    metadata = list(csv.DictReader(f))

for split, name in [(1, "val"), (2, "test")]:
    rows = [r for r in metadata if int(r["split"]) == split]
    ids = [r.get("filename", r.get("img_filename")) for r in rows]
    features = np.load(f"feats_{name}.npy", allow_pickle=False)
    assert features.shape == (len(rows), 2048)
    np.savez_compressed(f"{name}_raw.npz", ids=np.array(ids), features=features)
```

### Stage-0 provenance

`provenance.json` records the origin of the input artifacts and must contain exactly the fields below. Capitalized placeholders must be replaced with verified values. File hashes establish artifact identity; the training and checkpoint-selection history requires an independent provenance audit.

```json
{
  "track": "standard",
  "seed": 0,
  "checkpoint_sha256": "SHA256_OF_CKPT_LAST_PT",
  "val_features_sha256": "SHA256_OF_VAL_RAW_NPZ",
  "model": "resnet50",
  "pretrained_weights": "EXACT_IMAGENET_WEIGHTS_IDENTIFIER",
  "transforms": "ACTUAL_TRAIN_AND_FEATURE_EXTRACTION_TRANSFORMS",
  "extraction_order": "metadata_split_order",
  "feature_format": "raw",
  "checkpoint_selection": "final_epoch_no_validation_selection",
  "source_commit": "COMMIT_USED_TO_TRAIN_STAGE0"
}
```

SHA-256 hashes can be calculated with `sha256sum ckpt_last.pt val_raw.npz`. The validation-feature hash must refer to the final NPZ. Enhanced ERM* provenance uses `track: enhanced` and records the actual augmentation recipe in `transforms`.

## Execution

### Environment

Python 3.10 or later, PyTorch, NumPy, PyYAML and Matplotlib are required. Pytest is used for validation. Execution requires a writable repository checkout and access to the input artifacts. CPU execution is supported; CUDA is optional.

From the repository root:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install torch numpy pyyaml matplotlib pytest
```

### Notebook interface

[dpe_head_reweighting.ipynb](../../../notebooks/dpe_head_reweighting.ipynb) provides a general Jupyter interface to the experiment. Its configuration specifies the repository location, input directory, output directory, track and seed. Repository discovery searches the current directory and its parents; `REPO_PATH` supports an explicit location. The notebook uses the selected checkout without changing its Git revision. Network access is needed only if dependencies must be installed.

The notebook first checks the required files, feature shapes and IDs, provenance, device availability and writable output paths. The execution order is preparation, head training, weight fitting and selection, followed by held-out evaluation. Completed stages are reused only after input validation. A final cell creates a portable ZIP archive containing the run records, trained heads, metrics and diagnostics. Input checkpoints and raw feature files are excluded from the archive.

### Command-line interface

The [standard configuration](../../../configs/dpe_head_reweighting/waterbirds_standard.yaml) and [enhanced configuration](../../../configs/dpe_head_reweighting/waterbirds_enhanced.yaml) declare input paths, output paths and device settings. Paths are resolved from the repository root; `{track}` and `{seed}` are expanded from the run configuration.

```bash
CONFIG=configs/dpe_head_reweighting/waterbirds_standard.yaml
.venv/bin/python scripts/dpe_head_reweighting.py check --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py prepare --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py train-heads --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py fit-select --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py evaluate --config "$CONFIG" --seed 0
```

After completing seeds 0, 1 and 2 with matching artifacts:

```bash
.venv/bin/python scripts/dpe_head_reweighting.py summarize --config "$CONFIG"
```

The `check` action validates readiness for a complete run, including the test-feature file, without fitting or scoring models. It reports missing inputs before training. Stage 0 remains an external prerequisite. The CLI rejects existing stage outputs rather than overwriting them. Independent experiments require distinct output paths. Changes to configuration, features, heads, partitions or implementation files invalidate saved selection records. JSON records, ensemble checkpoints and prediction caches are written atomically so an interrupted write does not replace a complete artifact. An interrupted head-training stage restarts from its beginning; completed stages can be reused by the notebook.

## Output specification

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Validation partition, artifact provenance, input hashes and implementation identity. |
| `ensemble.pt` | Head prototypes, distance scales, selected epochs, preprocessing identity and provenance. |
| `selection.json` | Candidate validation metrics, selected hyperparameters, exact weights and input identities. |
| `metrics.json` | Uniform and selected test metrics, per-group counts, paired differences and changed error IDs. |
| `per_head.csv` | Head weights and group accuracies on B, C and test. |
| `head_diagnostics.png` | Head weights versus background-conflicting-group accuracy. |
| `summary.json` | Per-track means and sample standard deviations over three seeds. |

`delta_pp` denotes selected-minus-uniform differences in percentage points. A positive WGA difference indicates an improvement over the matched control. Selection of uniform averaging is retained as a valid experimental outcome.

The default CLI configuration places records in `experiments/exp-dpe-h2-waterbirds-<track>-seed<seed>/`, heads in `checkpoints/dpe_head_reweighting/`, and reports in `results/dpe_head_reweighting/<track>/seed<seed>/`. Prediction caches reside in the experiment directory's `predictions/` subdirectory, isolating independent runs with the same track and seed. Notebook runs place these artifacts under the configured output directory.

## Validation and provenance

```bash
.venv/bin/python -m pytest tests/methods/dpe_head_reweighting -q
```

The checks cover reference inference, atomic artifact writes, numerical stability, weight calculations, partition isolation, changed-input rejection and input-readiness failures. Full notebook integration checks are enabled when the optional Jupyter test dependencies are installed:

```bash
.venv/bin/pip install nbclient ipykernel
.venv/bin/python -m pytest tests/methods/dpe_head_reweighting -q
```

The notebook checks execute both standard and enhanced tracks in separate Jupyter kernels, with 15 heads, 20 epochs, 2048-dimensional synthetic features and official split/group counts. They verify final evaluation, archive creation and reuse of completed stages. Without the optional dependencies, these two integration checks are skipped. Synthetic results establish software behavior only; the fixture checkpoints are placeholders, not trained backbones. Real Waterbirds performance and GPU parity remain unverified.

Feature-only initialization consumes random numbers differently from the original reproduction notebook; the implementation does not reproduce its historical head parameters exactly. The original baseline and reproduction notebook are preserved. Source attribution, adaptation history and license notices are recorded in [ORIGINAL_SOURCE.md](ORIGINAL_SOURCE.md) and [LICENSE](LICENSE).
