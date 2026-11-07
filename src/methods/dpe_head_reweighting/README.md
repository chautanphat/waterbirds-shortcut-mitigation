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

## Method details and checks

Each head's score is its WGA on B. Candidate weights are proportional to `score ** gamma`, blended with equal weights using `alpha`. A larger `alpha` gives the score-based weights more influence; a larger `gamma` favors the strongest heads more sharply.

The fixed grid contains equal weighting plus `alpha ∈ {0.25, 0.5, 0.75, 1}` crossed with `gamma ∈ {1, 2, 4}`: 13 candidates total. Selection uses C WGA, then C overall accuracy, then smaller `alpha`, then smaller `gamma`, with numerical tolerance `1e-12`. All-zero scores fall back to equal weights. The final prediction averages head probabilities, with class-index argmax for ties.

Training uses 15 IsoMax+ heads, 20 epochs, batch size 256 and the notebook's loss and covariance schedule. Inference applies row standardization (`ddof=0`) followed by IsoMax+ L2 normalization. The runner exports heads and handles a final batch of one. Feature-only initialization consumes random numbers differently from the original notebook, so training does not reproduce its historical heads exactly.

Run the focused checks with:

```bash
.venv/bin/python -m pytest tests/methods/dpe_head_reweighting -q
```

They cover reference inference, saved-head reloads, weights, partition separation, changed-input rejection and a complete synthetic CPU run. GPU behavior and real Waterbirds results remain unverified. A relationship between low weights and poor minority-group accuracy alone does not prove that a head uses background features.

See [experiment.py](experiment.py) for the split and selection implementation, [ORIGINAL_SOURCE.md](ORIGINAL_SOURCE.md) for adaptation history, and [LICENSE](LICENSE) for source notices. The original baseline and reproduction notebook are preserved.
