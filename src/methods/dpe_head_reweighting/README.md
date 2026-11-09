# DPE Head Reweighting

## Overview

This project-owned adaptation of Diverse Prototypical Ensembles (DPE) evaluates whether validation-derived head weights improve worst-group accuracy (WGA) on Waterbirds relative to uniform probability averaging. WGA is the minimum accuracy over the four bird/background groups. Both aggregation conditions use the same frozen ResNet-50 representation and the same 15 trained prototype heads.

The end-to-end workflow reuses the shared ERM trainer and feature extractor, then performs sequential head training, reliability estimation, aggregation selection, artifact validation and held-out evaluation. A completed exploratory standard-track seed-0 power-rule run is documented in the [original run report](../../../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md). The subsequent [multi-rule seed-0 report](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/report.md) records a C-selected top-k10 test WGA of 89.6231% versus 89.7561% for uniform; no selected new procedure improves test WGA in that run. Multi-seed conclusions and GPU parity remain unverified.

## Experimental protocol

Stage 0 fine-tunes ImageNet-pretrained ResNet-50 on the official Waterbirds training split using the [shared ERM implementation](../erm/README.md). The notebook and Stage-0 adapter perform training and feature extraction automatically. The feature-only H2 runner can also consume existing, audited artifacts.

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

The Stage-0 adapter is [stage0.py](stage0.py); it delegates training, model loading and feature extraction to shared modules. The H2 implementation is split between [ensemble.py](ensemble.py) (head predictions), [train_heads.py](train_heads.py) (A-only training), [reweighting.py](reweighting.py) (B scores and C selection), and [experiment.py](experiment.py) (data roles, saved records and evaluation).


## Input specification

### Metadata and features

The required artifacts are `metadata.csv`, `ckpt_last.pt`, `val_raw.npz`, `test_raw.npz` and `provenance.json`, all associated with the same track and Stage-0 seed. The checkpoint is hashed for provenance; the feature-only runner does not load it for inference. Training and selection do not consume test features. The notebook's full-run readiness check validates the test-feature file before training to detect input errors early; test predictions are computed only during final evaluation.

Metadata must contain `img_filename` (or `filename`), `y`, `place` (or `a`), and `split`. Split values are `0` for train, `1` for validation and `2` for test, with counts 4,795 / 1,199 / 5,794. Filenames must be unique.

Each feature NPZ must contain exactly two arrays:

- `ids`: filenames in the same order as that split's metadata rows.
- `features`: raw floating-point features, shaped `[1199, 2048]` for validation or `[5794, 2048]` for test.

Features must be raw backbone outputs, before row standardization or normalization. The runner performs preprocessing and rejects reordered, missing or duplicate IDs.

The Stage-0 adapter produces these files automatically from a completed shared ERM run. Manual conversion is only needed for external feature caches. Existing `feats_val.npy` and `feats_test.npy` files may be converted after their checkpoint identity and metadata ordering have been verified:

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

`provenance.json` records the origin of the input artifacts and must contain exactly the fields below. The Stage-0 adapter fills these fields from the shared run records and actual file hashes. For externally supplied artifacts, capitalized placeholders must be replaced with verified values. File hashes establish artifact identity; the training and checkpoint-selection history requires an independent provenance audit.

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

Python 3.10 or later and a compatible PyTorch/torchvision pair are required. Shared training dependencies are in the repository `requirements.txt`; Matplotlib provides H2 diagnostics. Pytest and optional Jupyter packages support validation. GPU execution is recommended for full backbone training; CPU execution is supported.

From the repository root, using a PyTorch/torchvision pair appropriate for the runtime:

```bash
python -m pip install -r requirements.txt matplotlib tqdm
```

### Notebook interface

[dpe_head_reweighting.ipynb](../../../notebooks/dpe_head_reweighting.ipynb) runs the complete experiment: dataset preparation, shared ERM fine-tuning, raw-feature extraction, validation partitioning, head training, weight selection and held-out evaluation. Set `DATA_DIR`, `TRACK` and `SEED`; the remaining paths have defaults. `REPO_PATH` supports an explicit checkout location. The notebook uses that checkout without changing its Git revision.

The default Stage-0 recipe uses ImageNet `IMAGENET1K_V2`, full-backbone cross-entropy training, class-balanced sampling with replacement, SGD at learning rate 0.001 and momentum 0.9, no weight decay, batch size 128 and 100 epochs. Standard ERM uses square resize, center crop and ImageNet normalization. Enhanced ERM* applies random resized crop, photometric distortion and horizontal flip to training; validation/test feature extraction remains deterministic. The checkpoint is always `ckpt_last.pt`. Validation metrics are recorded by the shared trainer, but do not select the H2 backbone.

Existing completed shared ERM runs can be reused by setting `BACKBONE_DIR` to their directory. The adapter verifies the saved configuration, seed, completed update count and metadata identity. Random initialization, a shorter training duration or a best-validation checkpoint is rejected for this declared experiment. No manual feature conversion or provenance editing is required.

Default reusable artifact locations are:

```text
checkpoints/waterbirds_stage0_<track>_erm/seed<seed>/
    ckpt_last.pt, config.yaml, run.json, summary.json, ...
outputs/dpe_features/<track>/seed<seed>/
    metadata.csv, val_raw.npz, test_raw.npz, provenance.json, handoff.json
outputs/dpe_head_reweighting_runs/h2-<track>-seed<seed>/
    config.yaml, experiment/, checkpoints/, results/, stage0_records/, ...
```

The feature bundle references the checkpoint at its original location. `handoff.json` binds its files to checkpoint, metadata, shared-run records and extraction-source hashes. It is published only after both feature splits are complete; changed inputs are rejected on reuse. The final notebook ZIP includes H2 outputs and compact Stage-0 records. Retain the backbone and feature directories separately, since large inputs are excluded from that ZIP.

Network access is required for missing dependencies, initial dataset preparation and initial ImageNet weights. Existing datasets may be read-only. A completed matching backbone, feature bundle or H2 stage is reused after checks. The original notebook retains fresh-run behavior for partial backbones. The automated study below enables opt-in epoch-boundary resume for new checkpoints; completed backbones are reusable by either workflow. An interrupted feature extraction can be retried; an interrupted head-training stage restarts from its beginning.

### Command-line interface

The [standard configuration](../../../configs/dpe_head_reweighting/waterbirds_standard.yaml) and [enhanced configuration](../../../configs/dpe_head_reweighting/waterbirds_enhanced.yaml) declare input paths, output paths and device settings. Paths are resolved from the repository root; `{track}` and `{seed}` are expanded from the run configuration.

Prepare data and generate Stage-0 inputs first:

```bash
python scripts/prepare_waterbirds.py --data_dir data
python scripts/dpe_head_reweighting_stage0.py all --track standard --seed 0 --device cuda
```

The Stage-0 entry point also supports `train` and `export` separately. To export from an existing compatible shared run, use `export --backbone-dir <run-directory>`. `--data-dir` takes the Waterbirds directory itself; `--output-dir` changes the feature bundle location. If custom paths are used, update the corresponding H2 configuration. For ERM*, use `--track enhanced` and the enhanced H2 configuration.

The H2 stages consume the generated inputs:

```bash
CONFIG=configs/dpe_head_reweighting/waterbirds_standard.yaml
python scripts/dpe_head_reweighting.py check --config "$CONFIG" --seed 0
python scripts/dpe_head_reweighting.py prepare --config "$CONFIG" --seed 0
python scripts/dpe_head_reweighting.py train-heads --config "$CONFIG" --seed 0
python scripts/dpe_head_reweighting.py fit-select --config "$CONFIG" --seed 0
python scripts/dpe_head_reweighting.py evaluate --config "$CONFIG" --seed 0
```

After completing seeds 0, 1 and 2 with matching artifacts:

```bash
python scripts/dpe_head_reweighting.py summarize --config "$CONFIG"
```

The `check` action validates readiness for a complete run, including the test-feature file, without fitting or scoring models. It reports missing inputs before training. The Stage-0 entry point or notebook generates the required inputs before these H2 commands. The CLI rejects existing stage outputs rather than overwriting them. Independent experiments require distinct output paths. Changes to configuration, features, heads, partitions or implementation files invalidate saved selection records. JSON records, ensemble checkpoints and prediction caches are written atomically so an interrupted write does not replace a complete artifact. An interrupted head-training stage restarts from its beginning; completed stages can be reused by the notebook.

## One end-to-end weighting study

Use [dpe_weighting_rules.ipynb](../../../notebooks/dpe_weighting_rules.ipynb) as the
single entry point. **No original H2 ZIP is required.** The default
[study configuration](../../../configs/dpe_head_reweighting/study.yaml) runs
standard-track seeds 0, 1 and 2 with the unchanged weighting grid.

1. **Kaggle:** import the notebook, select a GPU runtime, enable Internet and
   choose Run All. An attached original H2 ZIP or Waterbirds dataset is optional.
2. **Colab:** open the notebook, select a GPU runtime and choose Run all.
   No archive-upload prompt or manual handoff between notebooks is required.

The notebook clones `experiment/dpe-multiple-weighting-rules` when no compatible
checkout exists, installs missing dependencies, discovers or downloads official
Waterbirds once, trains a separate 100-epoch ERM backbone and 15 heads for each
missing seed, compares all weighting rules and creates one combined results ZIP.
Existing checkouts are used as supplied without changing their Git revision.
GPU time is still required for new backbone training; CPU is supported but slow.
The orchestrator is [study.py](study.py), using the shared ERM/Stage-0/H2 modules.
The numerical prototype and weighting recipes remain unchanged.

Completed source bundles found among attached inputs skip backbone and head
training for the matching track/seed. For example, attaching the original seed-0
H2 ZIP lets the default study reuse seed 0 and train seeds 1/2 automatically.
`SOURCE=None` discovers attached bundles; an explicit ZIP/directory or list is
also accepted. ZIPs may contain an enclosing directory. Inputs are read only;
extraction checks path escapes, symbolic links and duplicate members. Several
different bundles for the same seed require an explicit choice rather than an
arbitrary selection. A summary-only suite ZIP cannot replace a prediction bundle.

Defaults normally need no edits. Optional settings in the first cell are:

| Setting | Purpose |
| --- | --- |
| `MODE="auto"` | Reuse original H2 bundles where available and train missing seeds |
| `MODE="cached"` | ZIP-only comparison; infer available seeds when `SEEDS=None` |
| `SEEDS=None` | Default [0,1,2]; set [1,2] to omit seed 0 |
| `TRACK=None` | Default standard; enhanced is also supported |
| `DATA_DIR=None` | Discover an attached dataset or download it once |
| `WORK_DIR=None` | Writable session storage; optionally choose persistent storage |
| `REPO_PATH`, `SEARCH_PATHS`, `DEVICE`, `WORKERS` | Local/offline paths and execution overrides |

Within each seed, B fits weights and C freezes every family/overall decision
**before new test predictions are generated**. Test IDs/labels/groups must agree
across seeds. No family is chosen using test scores. Historical seed-0 reuse is
still exploratory because that test split has already been inspected.

### Interruption and checkpoint reuse

New backbone runs save atomic version-2 training states at initialization and
completed epoch boundaries, including model, optimizer, best model, metric
history, Python/NumPy/Torch/CUDA RNG states, metadata and training-code identities.
The study resumes with the same scientific recipe. An interruption inside an
epoch replays that epoch, not its partially completed minibatches. CPU tests
verify exact resumed/uninterrupted checkpoint agreement; cross-hardware GPU
bitwise equivalence is not claimed. Completed compatible legacy backbones and
H2 prediction ZIPs remain reusable; incomplete legacy states lacking the new RNG
and history fields cannot be resumed. Existing non-resume entry points still
reject nonempty incomplete runs by default.

Feature extraction is published atomically. Interrupted head training restarts
its short feature-only stage. Changed extraction or H2 code gets separate
feature/head directories; completed, matching stages are reused after checks.
Completed frozen heads with B/C caches can also be reused across a weighting-only
code change, even when the old run stopped before test inference.
Keep WORK_DIR persistent to survive deletion of a hosted VM. Colab users can
point it at an already mounted Drive directory; Kaggle users must retain saved
outputs/checkpoints. A runtime restart with the same files can resume, but the
compact ZIP does not include large backbone/resume checkpoints or raw features.

**Recover the joint-solver failure from a saved Kaggle version:** attach that
failed version's notebook outputs as an input, import the updated notebook,
restart the kernel and Run All with the defaults. Keep both `h2_sources/` and
`features/` from that version; its partial study ZIP alone does not contain the
unfinished seed's heads/raw features. Discovery finds completed seeds and frozen
pretest bundles automatically. Recovery verifies the backbone, metadata, raw
validation features, test features and provenance against the saved handoff,
fits/freezes the new rules on B/C, then generates only the missing test predictions
from the original 15 heads. It writes a separate managed copy and preserves the
attached outputs. Missing seeds still train normally. The bootstrap uses a new
managed checkout when the old checkout lacks the recovery module, avoiding stale
imports after a kernel restart. An explicit `REPO_PATH` must contain this update.

### Study outputs

Results are saved under `WORK_DIR/studies/<study-id>/` and zipped alongside it.
Progress and the ZIP update after **each completed seed**, so an interrupted
multi-seed study can still provide partial results:

- `plan.json` and `progress.json`: fixed settings, source/code identities and completed seeds.
- `per_seed.csv`: each family/overall C-selected candidate, test metrics and paired changes.
- `summary.json` and `report.md`: means and sample SDs, including paired deltas.
  A one-seed SD is pending; the study is marked incomplete until all planned seeds finish.
- `comparisons/seedN/`: all C candidates, exact weights, solver diagnostics,
  selected test metrics, group counts and changed errors.
- `source_archives/h2-<track>-seedN.zip`: original prediction bundles and trained
  heads, automatically retained for reuse without another notebook handoff.

The final cell provides a download link and starts a Colab download. It can also
be run separately to retrieve the latest partial ZIP after interrupting the
study. Dataset images, large backbones/training states and raw features remain
outside the compact download in WORK_DIR. Per-seed comparison outputs also
remain available under `WORK_DIR/results/`. No new benchmark result is claimed
by this orchestration change; record actual executions in the experiment log.

The [fixed rule configuration](../../../configs/dpe_head_reweighting/weighting_rules.yaml)
declares seven additional candidates and the solver settings:

| Family | Weight fit on B | Candidate settings |
| --- | --- | --- |
| Uniform | Exactly `1/15` per head | Included globally and as every family's fallback |
| Original power control | Existing score powers and uniform shrinkage | Original 12 nonuniform configurations, unchanged |
| Softmax reliability | `exp(beta * (s_h - max(s)))`, normalized | beta 10 and 30; WGA scores are fractions |
| Top-k averaging | Equal weights on the k highest B-WGA heads | k 10 and 5; tied scores use original head order |
| Regularized joint weights | Fit the combined probabilities with a worst-group log-loss objective | lambda 1, 0.1 and 0.01 |

There are 20 distinct candidate records including uniform. Joint fitting solves
the following convex problem on B, using an epigraph variable and SciPy SLSQP
with a deterministic SciPy `trust-constr` recovery solver:

```math
\min_{w\geq0,\ \sum_h w_h=1}
\max_g\frac{1}{|B_g|}\sum_{i\in B_g}
-\log\left(\sum_h w_h p_h(y_i\mid x_i)\right)
+\lambda\sum_h(w_h-1/15)^2.
```

The solver starts at uniform. Correct-class head probabilities are floored at
`1e-12` for the fitting objective to avoid undefined logarithms; reported
inference probabilities remain unchanged. The YAML declares `maxiter=1000`,
`ftol=1e-10` and feasibility tolerance `1e-7`. If SLSQP fails its success,
finite-weight, simplex, epigraph or objective checks (including its status-8
line-search failure), `trust-constr` solves the same objective and constraints
from uniform, with analytic gradients/Hessians. Its optimality residual must also
be at most `1e-7`. Neither C nor test results decide whether to retry. Failure
of both solvers remains an error; no candidate is silently omitted. The solver
method, attempt messages, success, iteration count, objective and B-group log
losses are saved with each joint candidate. This changes numerical recovery,
not the candidate grid or the scientific objective.
Only the weights are optimized; there is no fitted intercept, feature-dependent
gate, probability calibration or change to the heads.

All candidates are evaluated on C. Each family selects one configuration with
an exact uniform fallback, and the suite also selects an overall configuration
on C. Selection maximizes C WGA and then C overall accuracy using tolerance
`1e-12`. Uniform wins a complete tie; otherwise configuration declaration order
breaks ties. This retains the original power rule's smaller-alpha/smaller-gamma
ordering and prefers stronger joint regularization when its C metrics tie.
Weights are not refitted on B+C.

Every family winner and the overall winner are persisted **before** opening the
test cache. Test evaluation scores only those frozen choices and uniform; it
does not sweep all hyperparameters on test. Selecting a family or reporting only
its best seed after looking at test WGA would be test-driven selection. Keep
seed-0 follow-ups exploratory because that test split has already been inspected.
Lock the protocol before evaluating later seeds, and report failures to improve.

The suite checks unique/disjoint A/B/C IDs, cache row alignment, class/group
consistency, preprocessing and ensemble identities. It recomputes the original
power decision from B/C before reusing old bundles. If original test metrics are
present, their uniform metrics must agree with the test cache. New output binds
the resolved rule configuration, implementation/notebook hashes, SciPy/NumPy versions,
source hashes, selection hash and test-cache hash. The notebook identity hashes
code cells only: autosaved outputs, execution counts and editor metadata do not
invalidate a run. Code edits do invalidate it. Re-running verifies the
frozen choices and returns identical completed results; changed inputs require
a new output directory. An interrupted candidate CSV export is regenerated from
the frozen selection on rerun. Notebook setup checks the required archive files
before fitting and reports missing files together. These checks establish consistency of saved predictions,
not independent verification of the original images or feature extraction.

Additional dependencies and the equivalent CLI are:

```bash
python -m pip install -r src/methods/dpe_head_reweighting/requirements-weighting.txt
python scripts/dpe_weighting_rules.py select --source-run outputs/dpe_head_reweighting_runs/h2-standard-seed0 --output-dir outputs/dpe_weighting_rule_runs/standard-seed0-v1
python scripts/dpe_weighting_rules.py evaluate --source-run outputs/dpe_head_reweighting_runs/h2-standard-seed0 --output-dir outputs/dpe_weighting_rule_runs/standard-seed0-v1
```

Use `--config` to specify a separately declared YAML grid. Output includes
`selection.json`, `candidate_comparison.csv`, `metrics.json`, `run.json` and
`report.md`. They contain exact weights, all C candidate metrics, per-family
winners, overall winner, effective head counts, probability/prediction changes,
per-group test accuracies/counts, paired deltas and fixed/new error IDs. The
notebook exports a ZIP of this new comparison without duplicating the source
predictions or heads; retain the original H2 ZIP separately for replay.

The first completed standard seed-0 suite is documented in the
[multi-rule run report](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/report.md).
Its uploaded aggregate records were checked for consistency, not independently
recomputed from prediction caches. Add a dated experiment entry and lightweight
result summary for subsequent real runs, following the existing experiment-log
template. Synthetic regression results must not be reported as Waterbirds accuracy.

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

The [implementation audit](AUDIT.md) records findings, fixes and verification
limits. To independently recompute the original power-grid decision and test
metrics from an extracted end-to-end H2 archive, including earlier-runner archives
(this checker does not accept the new multi-rule comparison ZIP):

```bash
python scripts/audit_dpe_head_reweighting.py --run-dir path/to/extracted/run
```

The checker uses saved per-head predictions and does not import production
aggregation, selection or metric functions. It reads the bundle without changing
it. Raw-image inference requires the separately retained dataset and backbone.
The notebook displays all candidate C scores, probability differences and changed
predictions. The selection stage also exports `candidate_comparison.csv` beside
`selection.json`. Equal weights are reported explicitly when selected.

Final reports include a `selection_hash` binding them to the frozen decision.
Before evaluation, the saved decision is checked against the B/C prediction
caches. Modified selection records and stale final reports are rejected. Keep
older completed runs intact for the independent checker; implementation changes
require a fresh H2 run directory for normal execution.

```bash
python -m pip install -r src/methods/dpe_head_reweighting/requirements-weighting.txt
python -m pytest tests/methods/dpe_head_reweighting -q
```

The checks cover reference inference, atomic artifact writes, numerical stability, weight calculations, partition isolation, changed-input rejection and input-readiness failures. Multi-rule checks include a known complementary-head optimum, original power-grid parity, failed-solver rejection, B-only fitting, uniform fallback, frozen-source checks and notebook execution/reuse with autosaving after every cell. Hosted setup tests exercise fresh local clones, read-only attached ZIPs, nested extraction and simulated Colab downloads. Study tests cover a complete two-seed notebook with no H2 ZIP, real 100-epoch optimization loops using tiny synthetic models, completed-stage reuse, mixed cached/new seeds, automatic dataset preparation, test-set consistency, freeze-before-test ordering and paired mean/sample-SD calculations. Resume tests check interruption before epoch 1 and after epoch 2 against uninterrupted CPU checkpoints. These tests do not run inside the live Kaggle or Colab service and do not establish ResNet-50 benchmark accuracy. Full notebook integration checks are enabled when the optional Jupyter test dependencies are installed:

```bash
python -m pip install nbclient ipykernel
python -m pytest tests/methods/dpe_head_reweighting -q
```

The notebook checks execute both standard and enhanced tracks in separate Jupyter kernels, with 15 heads, 20 epochs, 2048-dimensional synthetic features and official split/group counts. They verify final evaluation, archive creation and reuse of completed stages. Without the optional dependencies, these two integration checks are skipped. The notebook tests run the shared 100-epoch optimization loop with a small synthetic classifier in place of ResNet-50, then use the real feature handoff and H2 stages. They also verify completed-backbone and feature-bundle reuse. Synthetic results establish software behavior only; the fixture models are not Waterbirds ResNet-50 backbones. Real Waterbirds performance and GPU parity remain unverified.

Feature-only initialization consumes random numbers differently from the original reproduction notebook; the implementation does not reproduce its historical head parameters exactly. The original baseline and reproduction notebook are preserved.

## Source and attribution

This project-owned H2 adaptation was introduced on 2026-11-07. Its reference is
To et al., *Diverse Prototypical Ensembles Improve Robustness to Subpopulation
Shift* (ICML 2025), [paper](https://arxiv.org/abs/2505.23027),
[upstream repository](https://github.com/minhto2802/dpe4subpop), preserved commit
`68b811461e238d44cced6ae6c8ce75b66cad5449` (copyright 2025 Minh To, MIT).
The behavioral reference is `notebooks/dpe_waterbirds_end_to_end.ipynb`, cells
11, 13 and 14 at project commit `fc6c267`.

`train_heads.py` adapts its group-balanced feature-only training, covariance and
prototype penalties, entropic loss and last-on-tie epoch selection.
`ensemble.py` follows the IsoMax+ Euclidean-distance implementation preserved in
`baselines/dpe4subpop/utils/isomaxplus.py`, originally David Macêdo (2022),
Apache 2.0 ([source](https://github.com/dlmacedo/entropic-out-of-distribution-detection)).
The dataset subsampling reference is Yuzhe Yang (2023), MIT
([SubpopBench](https://github.com/YyzHarry/SubpopBench)). Applicable MIT and
third-party license texts remain in [LICENSE](LICENSE).

Weight fitting, split manifests, experiment locks and diagnostic reporting are
project-owned H2 additions. The multi-rule extension added on 2026-11-08 provides
softmax reliability, top-k averaging and constrained regularized worst-group
log-loss weighting in `weighting_rules.py`, with cached-run orchestration in
`rule_suite.py`. These experiment rules and their results are not attributed to
the DPE paper. The original power rule, preserved baseline sources and completed
result records remain intact.
