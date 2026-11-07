# DPE head reweighting

This experiment asks: **does giving more weight to reliable DPE heads improve worst-group accuracy on Waterbirds?**

DPE combines 15 small classifiers, called **heads**. Normally, each head contributes equally to the final prediction. Here, we measure how well each head performs on validation data and try giving stronger heads more weight. We compare this with equal weighting using the same trained heads.

**Current status:** the code passes synthetic CPU tests. Real Waterbirds runs and any accuracy improvement are still unverified.

## What you need before starting

The notebook starts **after Stage 0**, when a ResNet-50 backbone has already been trained and used to extract features. Features are the 2048 numbers the backbone produces for each image.

You need these five files from one matching backbone run:

| File | What it contains |
| --- | --- |
| `metadata.csv` | Waterbirds filenames, labels, backgrounds and official splits. |
| `ckpt_last.pt` | The final Stage-0 backbone checkpoint. |
| `val_raw.npz` | Raw features and filenames for the 1,199 validation images. |
| `test_raw.npz` | Raw features and filenames for the 5,794 test images. |
| `provenance.json` | A record of how the checkpoint and features were produced. |

Start with **standard ERM, seed 0**. The checkpoint, features and provenance must all belong to that run. Enhanced ERM* and other seeds are separate runs.

If you only have images, you still need to train or recover the backbone and extract features first. This runner does not perform those steps. If you have the original notebook's `feats_val.npy` and `feats_test.npy`, see **Preparing the input files** below.

## Run on Kaggle

1. Upload [dpe_head_reweighting.ipynb](../../../notebooks/dpe_head_reweighting.ipynb) as a Kaggle notebook.
2. Enable **Internet**. A GPU is optional; the notebook uses one when available.
3. Attach a Kaggle Dataset containing the five input files above.
4. In the notebook's settings cell, change `INPUT_DIR` to your dataset folder. Adjust the individual filenames if needed. Leave `TRACK = "standard"` and `SEED = 0` for your first run.
5. Run the cells in order: **prepare → train heads → fit/select weights → evaluate**.
6. Run the last cell to create a ZIP, then download it from Kaggle's Output panel.

The notebook uses the tested runner at commit `10b25e2`. It saves your configuration, trained heads, selected weights and results. Re-running a completed cell reuses its saved output after checking that the inputs still match. Use a new `RUN_NAME` when starting a different experiment.

## What happens during a run

The official training split is used for the existing Stage-0 backbone. This experiment divides the official **validation split** into three parts:

| Part | Share of validation | Used for |
| --- | --- | --- |
| A | 60% | Training the 15 heads and choosing their best epochs. |
| B | 20% | Measuring each head's reliability and calculating candidate weights. |
| C | 20% | Choosing between equal weighting and the weighted candidates. |

Each group is divided in these proportions, with split seed 0 fixing which images go into each part. The same partition is used across training seeds and both ERM tracks. Head training uses group-balanced samples from A; A is also reused for epoch selection.

After selection is saved, a separate step evaluates the chosen weights and equal weighting on the held-out **test split**. Test results do not choose the weights.

The experiment trains new heads on A. Heads from the original reproduction used the full validation set, so they cannot be substituted for these heads in this experiment.

## Reading the results

**Worst-group accuracy (WGA)** is the accuracy of the least accurate of the four bird/background groups. It is the main comparison because overall accuracy can hide poor performance on birds with an unusual background.

| Output | What to look at |
| --- | --- |
| `metrics.json` | Equal-weight (`uniform`) and selected-method (`selected`) test accuracy, WGA and per-group accuracy. |
| `delta_pp` inside `metrics.json` | Selected minus uniform accuracy/WGA, in percentage points. Positive means improvement. |
| `selection.json` | The chosen `alpha`, `gamma` and head weights, plus validation scores for all candidates. |
| `per_head.csv` | Each head's weight and group accuracies on B, C and test. |
| `head_diagnostics.png` | Head weight versus accuracy on background-conflicting groups. |

For example, uniform WGA of 88% and selected WGA of 89% gives a WGA change of **+1 percentage point**. If equal weighting wins validation selection, retain and report that result.

Kaggle puts the run under the notebook's `RUN_DIR` and bundles it as a ZIP. With the default command-line configuration, experiment records go under `experiments/exp-dpe-h2-waterbirds-<track>-seed<seed>/`, trained heads under `checkpoints/dpe_head_reweighting/`, and results under `results/dpe_head_reweighting/<track>/seed<seed>/`. Prediction caches go under `outputs/dpe_head_reweighting/`.

Repeat the experiment with matching Stage-0 seeds 1 and 2. Report standard and enhanced results separately. Compare against the equal-weight control trained in the same run; the historical DPE results used a different head-training data budget.

## Preparing the input files

### Metadata and features

Metadata must contain `img_filename` (or `filename`), `y`, `place` (or `a`), and `split`. Split values are `0` for train, `1` for validation and `2` for test, with counts 4,795 / 1,199 / 5,794. Filenames must be unique.

Each feature NPZ must contain exactly two arrays:

- `ids`: filenames in the same order as that split's metadata rows.
- `features`: raw floating-point features, shaped `[1199, 2048]` for validation or `[5794, 2048]` for test.

The runner standardizes features itself. Supply raw features, before standardization or normalization. Reordered, missing or duplicate IDs are rejected.

If you recovered raw `.npy` files from the original notebook, first verify that they came from the matching checkpoint and follow metadata split order. Then package them with IDs:

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

### The provenance file

`provenance.json` records where the inputs came from. It must contain exactly the fields below. Replace the capitalized placeholders with the actual values; do not guess the training history. The runner checks file hashes and requires a final-epoch checkpoint that was not selected using validation or test performance.

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

To calculate a file's SHA-256, use `sha256sum ckpt_last.pt val_raw.npz` in a terminal. Compute the validation hash after creating the final NPZ. For enhanced ERM*, set `track` to `enhanced` and record its augmentation recipe in `transforms`.

## Run from a terminal instead

From the repository root, install the dependencies:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install torch numpy pyyaml matplotlib pytest
```

Edit the input paths and device in [waterbirds_standard.yaml](../../../configs/dpe_head_reweighting/waterbirds_standard.yaml). Then run:

```bash
CONFIG=configs/dpe_head_reweighting/waterbirds_standard.yaml
.venv/bin/python scripts/dpe_head_reweighting.py prepare --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py train-heads --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py fit-select --config "$CONFIG" --seed 0
.venv/bin/python scripts/dpe_head_reweighting.py evaluate --config "$CONFIG" --seed 0
```

Repeat for seeds 1 and 2 using matching input artifacts. The configuration expands `{track}` and `{seed}` in paths. After all three runs, calculate their means and sample standard deviations:

```bash
.venv/bin/python scripts/dpe_head_reweighting.py summarize --config "$CONFIG"
```

Use [waterbirds_enhanced.yaml](../../../configs/dpe_head_reweighting/waterbirds_enhanced.yaml) for the enhanced track. The terminal commands reject existing experiment outputs rather than overwrite them. Use fresh output paths for a new experiment. Changed configuration, features, heads, partitions or implementation files invalidate a saved selection.

## Detailed mechanism

### 1. Turn each image into a fixed feature vector

A Stage-0 ResNet-50 backbone produces a raw feature vector $f(x)\in\mathbb{R}^{2048}$. The backbone stays frozen throughout this experiment. For each image separately, standardize its 2048 coordinates and then normalize the vector to unit length:

$$
u(x)=\frac{f(x)-\operatorname{mean}(f(x))}{\operatorname{std}(f(x))},
\qquad
z(x)=\frac{u(x)}{\lVert u(x)\rVert_2}.
$$

The standard deviation uses `ddof=0`. Zero-variance rows are rejected; L2 normalization uses PyTorch's numerical safeguard for small norms. This sequence is applied once to raw features. Validation and test use the same preprocessing.

### 2. Give each head its own class prototypes

Each head $h$ learns two 2048-dimensional vectors: a landbird prototype $v_{h,0}$ and a waterbird prototype $v_{h,1}$. It also learns one distance scale $d_h$. The IsoMax+ class logit is the negative Euclidean distance to the normalized class prototype, multiplied by the absolute scale:

$$
\ell_{h,c}(x)=-|d_h|\left\lVert z(x)-\frac{v_{h,c}}{\lVert v_{h,c}\rVert_2}\right\rVert_2,
\qquad c\in\{0,1\}.
$$

A closer prototype gives a larger logit. Applying softmax across the two classes gives the head's inference probabilities:

$$
p_h(c\mid x)=\frac{\exp(\ell_{h,c}(x))}{\sum_{k=0}^{1}\exp(\ell_{h,k}(x))}.
$$

Inference temperature is 1. The training entropic scale of 30, described below, is **not** multiplied into these inference logits.

### 3. Train diverse heads using A

Train the heads sequentially. For every new head, independently sample the same number of examples from each of A's four groups, using the smallest group size. Keep this sampled set fixed for that head's 20 epochs. Previously selected heads remain frozen.

The training loss contains three terms:

$$
\mathcal{L}_h=\mathcal{L}_{\mathrm{classification}}
+\mathcal{L}_{\mathrm{prototype}}
+\lambda_h R_{\mathrm{diversity}}.
$$

The classification term is the mean negative log probability of the correct class, using the training-only entropic scale:

$$
\mathcal{L}_{\mathrm{classification}}
=-\frac{1}{M}\sum_{i=1}^{M}\log\left[\operatorname{softmax}(30\ell_h(x_i))\right]_{y_i}.
$$

The prototype penalty limits the size of the unnormalized prototype vectors:

$$
\mathcal{L}_{\mathrm{prototype}}=\frac{10}{2}\sum_{c=0}^{1}\lVert v_{h,c}\rVert_2^2.
$$

The diversity term compares the current prototypes with previous heads' prototypes for the same class. With $D=2048$ and heads numbered from 1, the implemented expression is:

$$
R_{\mathrm{diversity}}
=\left|\frac{1}{2}\sum_{c=0}^{1}\frac{1}{h}\sum_{j=1}^{h-1}
\frac{v_{h,c}^{\mathsf T}v_{j,c}}{D-1}\right|,
\qquad
\lambda_h=\frac{500000}{1+0.05\max(0,h-2)}.
$$

This preserves the reference notebook's regularizer, including its order of summing and taking the absolute value. It uses raw prototype dot products; it does not center the vectors or average absolute pairwise similarities. For the first head there are no previous prototypes, so the diversity term is absent. Different balanced samples and this regularizer aim to produce heads with different decision rules.

Optimize only the new head using SGD with learning rate `0.001`, momentum `0.9`, and batch size `256`. Drop incomplete training batches; reject a subset too small for one full batch. After each epoch, measure WGA on all of A and save the best head parameters. Equal WGA keeps the later epoch. A therefore serves both head fitting and epoch selection; B and C remain separate from this process.

### 4. Measure each frozen head's reliability on B

Waterbirds has four groups, defined by bird class $y$ and background $a$:

| Group | Bird | Background |
| --- | --- | --- |
| `00` (G1) | Landbird | Land |
| `01` (G2) | Landbird | Water |
| `10` (G3) | Waterbird | Land |
| `11` (G4) | Waterbird | Water |

For every head, predict the class with the largest probability and calculate accuracy separately for each B group. Its reliability score is its worst-group accuracy:

$$
s_h=\min_{g\in\{00,01,10,11\}}\operatorname{accuracy}(h,B_g).
$$

Scores are fractions between 0 and 1. This favors heads that perform reasonably across all groups rather than heads that succeed mainly on common backgrounds. Every group must be present; a missing group is an error.

### 5. Convert reliability scores into candidate weights

For $H=15$ heads, calculate a score-based distribution $q_h$ and blend it with equal weights:

$$
q_h=\frac{s_h^\gamma}{\sum_{j=1}^{H}s_j^\gamma},
\qquad
w_h=\frac{1-\alpha}{H}+\alpha q_h.
$$

`alpha` controls how far the weights move away from equal weighting. At `alpha = 0`, every head receives `1/15`; at `alpha = 1`, weights depend entirely on the scores. `gamma` controls how strongly higher scores dominate. Weights are nonnegative and sum to one. If all scores are zero, use equal weights and record the fallback.

For a small **two-head illustration**, suppose scores are `[0.25, 0.75]`. With `gamma = 2`, the score-based weights are `[0.1, 0.9]`. With `alpha = 0.5`, blending those with equal weights `[0.5, 0.5]` gives final weights `[0.3, 0.7]`.

### 6. Select the aggregation rule on C

Weights are calculated using B alone. Evaluate these fixed candidates on C:

- Equal weighting: `alpha = 0`, recorded with `gamma = 1`.
- Weighted averaging: `alpha ∈ {0.25, 0.5, 0.75, 1}` crossed with `gamma ∈ {1, 2, 4}`.

These are 13 configurations. Select the highest C WGA, breaking ties by higher C overall accuracy, then smaller `alpha`, then smaller `gamma`. Numerical ties use tolerance `1e-12`. Save the chosen weights and input identities before test evaluation. Do not retrain heads on A+B+C or refit weights on B+C afterward.

### 7. Compare weighted and equal averaging on test

For each test image, combine the same frozen heads' probabilities:

$$
p_{\mathrm{weighted}}(c\mid x)=\sum_{h=1}^{H}w_h p_h(c\mid x),
\qquad
p_{\mathrm{uniform}}(c\mid x)=\frac{1}{H}\sum_{h=1}^{H}p_h(c\mid x).
$$

Take the class with the largest combined probability; exact ties choose class 0. This averages probabilities, rather than logits or class votes. Backbone, head parameters, features and example order are identical for both predictions; only the aggregation weights differ.

Report overall accuracy, per-group accuracy, WGA and paired changes for each seed. The effective head count, $1/\sum_h w_h^2$, shows how concentrated the weights are: 15 for equal weighting, approaching 1 when one head dominates.

The diagnostic plots compare weights with $\min(\operatorname{acc}_{01},\operatorname{acc}_{10})$, accuracy on the two background-conflicting groups. An association on B is partly expected because B also determines the weights; C and test provide separate descriptive checks. Low weights alone do not establish that a head learned background features. Weighting can also hurt if individually weaker heads supply useful complementary predictions, which is why equal averaging remains a selection candidate.

The implementation is split between [ensemble.py](ensemble.py) (head predictions), [train_heads.py](train_heads.py) (A-only training), [reweighting.py](reweighting.py) (B scores and C selection), and [experiment.py](experiment.py) (data roles, saved records and evaluation).

## Validation and source

Run the focused checks with:

```bash
.venv/bin/python -m pytest tests/methods/dpe_head_reweighting -q
```

They cover reference inference, saved-head reloads, weights, partition separation, changed-input rejection and a complete synthetic CPU run. GPU behavior and real Waterbirds results remain unverified. A relationship between low weights and poor minority-group accuracy alone does not prove that a head uses background features.

See [experiment.py](experiment.py) for the split and selection implementation, [ORIGINAL_SOURCE.md](ORIGINAL_SOURCE.md) for adaptation history, and [LICENSE](LICENSE) for source notices. The original baseline and reproduction notebook are preserved.
