# DPE head reweighting (H2)

Project-owned experiment implementing [the H2 plan](../../../docs/dpe_h2_implementation_plan.md).
Implementation and synthetic validation are available; Waterbirds runs are pending.
This is an adaptation, not an upstream reproduction or a verified improvement.

The runner trains 15 IsoMax+ prototype heads on a frozen ResNet-50 representation.
Each head's worst-group accuracy on validation subset B determines its weight:
`q = score**gamma / sum(score**gamma)`, `w = (1-alpha)/15 + alpha*q`.
It selects among uniform averaging and 12 weighted candidates using subset C
(WGA, then overall accuracy, then smaller alpha, then smaller gamma; tolerance
`1e-12`). All-zero scores fall back to uniform weights and are recorded.
Predictions average probabilities. Classification ties use class-index argmax.

## Setup and input artifacts

Run from the repository root. Python 3.10+, NumPy, PyTorch, PyYAML and Matplotlib
are required; pytest is used for checks. A CPU-only install is sufficient for
feature-only operation. The preserved baseline dependencies can also be used.

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install torch numpy pyyaml matplotlib pytest
.venv/bin/python -m pytest tests/methods/dpe_head_reweighting -q
```

Edit the selected YAML's paths/device before preparing. All paths resolve from
the repository root. `{track}` and `{seed}` expand in path strings. Standard
and enhanced tracks are separate; each seed needs its own matching backbone.
The strict schema rejects unknown keys, nonstandard dimensions, and changes to
the predeclared training recipe. Small synthetic runs use the module API only.

Supply these external artifacts; the runner does not download or train Stage 0:

1. Official `metadata.csv`, with `img_filename` (or `filename`), `split` (0/1/2), `y` and `place`
   (or `a`). Split counts must be 4795/1199/5794. Filenames serve as stable,
   globally unique IDs. Both the official Waterbirds format and the baseline's
   renamed `a` column are accepted.
2. Final-epoch Stage-0 checkpoint trained on official train, without selection
   using validation/test. The checkpoint is hashed, not loaded by this runner.
3. Validation raw feature NPZ containing exactly `ids` (Unicode filenames) and
   `features` (finite floating-point `[1199, 2048]`). Rows must follow official
   validation metadata order. Standardization is performed here, once per row,
   with NumPy population standard deviation (`ddof=0`), then IsoMax+ L2
   normalization. Do not supply already standardized features.
4. Test NPZ with the same contract and `[5794, 2048]`, required only by `evaluate`.
5. Stage-0 `provenance.json`, with exactly these fields (replace descriptions
   with audited values and SHA-256 hashes of the actual files):

```json
{
  "track": "standard",
  "seed": 0,
  "checkpoint_sha256": "SHA256_OF_CHECKPOINT",
  "val_features_sha256": "SHA256_OF_VALIDATION_NPZ",
  "model": "resnet50",
  "pretrained_weights": "EXACT_IMAGENET_WEIGHTS_IDENTIFIER",
  "transforms": "EXACT_TRAIN_AND_EVALUATION_TRANSFORMS",
  "extraction_order": "metadata_split_order",
  "feature_format": "raw",
  "checkpoint_selection": "final_epoch_no_validation_selection",
  "source_commit": "STAGE0_IMPLEMENTATION_COMMIT"
}
```

For enhanced ERM*, record its actual augmentation recipe in `transforms` and
set `track` to `enhanced`. These declarations need a human provenance audit;
hashes cannot establish how a checkpoint was trained. Existing notebook
`feats_val.npy`/`feats_test.npy` are raw and can be wrapped in NPZ only after
verifying their row order and checkpoint provenance. For example:

```python
import csv
import numpy as np
with open("data/waterbirds/metadata.csv", newline="") as f:
    rows = [r for r in csv.DictReader(f) if int(r["split"]) == 1]
np.savez_compressed("val_raw.npz", ids=np.array([r.get("filename", r.get("img_filename")) for r in rows]),
                    features=np.load("feats_val.npy", allow_pickle=False))
```

The main experiment always retrains Stage 1 on A. Full-validation heads cannot
be used to claim unseen calibration results. Recovered-head exploratory pilots
are outside this CLI; no automatic conversion guesses their provenance.

## Execute

```bash
python scripts/dpe_head_reweighting.py prepare --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
python scripts/dpe_head_reweighting.py train-heads --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
python scripts/dpe_head_reweighting.py fit-select --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
python scripts/dpe_head_reweighting.py evaluate --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
```

Use `.venv/bin/python` if using the environment above. Repeat with seeds 1/2
and then the enhanced YAML; never pool the two tracks. After all three seeds:

```bash
python scripts/dpe_head_reweighting.py summarize --config configs/dpe_head_reweighting/waterbirds_standard.yaml
```

`prepare` checks metadata/artifacts and saves the group-stratified A/B/C split
(60/20/20, split seed 0, largest-remainder integer allocation with A/B/C tie
order). IDs are sorted for seeded allocation and restored to metadata order.
It reads official metadata once, but saves only validation labels. It requires
no test features. Training sees only A for both fitting and epoch selection;
selection reads prepared validation metadata and B/C features, with no test
metadata or feature access. The test file can remain absent until evaluation.

Each group is subsampled to the smallest A group each stage. The runner uses
20 epochs, SGD `.001`/`.9`, batch 256 with `drop_last=True`, entropic scale 30,
prototype penalty 10, and the notebook's covariance schedule (500000 at both
stages 1 and 2, then division by `1 + .05*(stage-2)`). A too-small balanced
training subset is an error. Equal epoch WGA retains the last epoch.

The notebook and upstream are preserved. Changes from their execution are:
A-only training/selection, exported heads, no per-stage test evaluation,
singleton-safe inference, explicit metadata alignment, and feature-only model
initialization. RNG consumption differs because a redundant ResNet model is
not constructed; this is reproducible new training, not bit-identical replay
of historical heads. Group-size ties in subsampling use group-key order.

## Outputs and validation

- `experiments/exp-dpe-h2-waterbirds-<track>-seed<seed>/`: resolved config,
  split/labels/provenance manifest, source hashes, and frozen selection record.
- Ignored `checkpoints/dpe_head_reweighting/`: versioned CPU tensor artifact,
  head IDs/epochs, scales, temperature, preprocessing and manifest identity.
- Ignored `outputs/dpe_head_reweighting/`: per-head B/C/test probability caches
  with IDs, labels, groups, roles and ensemble/preprocessing identities.
- `results/dpe_head_reweighting/<track>/seed<seed>/`: paired metrics, group
  counts/deltas, fixed/new error IDs, effective head count, per-head diagnostic
  CSV and B/C/test weight-versus-minority-accuracy plot. Track summary contains
  means and sample standard deviations, including paired WGA deltas.

The selection record locks configuration, manifest, validation features,
ensemble and implementation hashes. Changed inputs invalidate evaluation.
The official metadata checksum is rechecked when opening test labels.
Existing manifests, ensembles, selections and final results are not overwritten;
use fresh artifact/output paths for a separately recorded experiment. Commit
the implementation and finish the protocol before preparing a real run.

Checks cover reference inference parity (including negative learned scales and
singleton batches), export/reload, manually calculated weighting and selection,
partition isolation, invalid inputs and stale artifacts, and a complete synthetic
CPU lifecycle. Synthetic data establishes software behavior only. Real CPU/GPU
parity, runtime, Stage-0 provenance, and the six Waterbirds runs remain unverified.
Three seeds provide preliminary evidence; weight/minority-group associations
alone do not establish that a head uses background features. Keep historical
89.00%/92.70% results separate from these A-trained matched comparisons.

See [ORIGINAL_SOURCE.md](ORIGINAL_SOURCE.md) and [LICENSE](LICENSE) for attribution.
