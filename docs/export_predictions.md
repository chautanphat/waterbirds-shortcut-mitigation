# Export predictions for multi-method inspection

Use [`scripts/export_predictions.py`](../scripts/export_predictions.py) to write a **new run directory** containing `predictions.csv` and `provenance.json`. Exports contain every example in the requested split, including correct predictions. The [Waterbirds Prediction Viewer](viewer.md) can therefore compare predictions across methods without assuming that missing rows are correct.

All future method exports follow the [shared predictions.csv v1 protocol](prediction_protocol.md). It is the source of truth for column order, labels, group mapping, confidence, completeness and validation requirements.

Exporting does not train, tune, or choose a checkpoint. Supply the checkpoint already chosen by the method's validation procedure. Keep distinct training recipes and seeds in distinct run names. The original DPE baseline described here uses equal averaging of head probabilities; it does not use the project's head-reweighting variant.

For new experiments, document the frozen predictor in
`experiments/<experiment-id>/model.json` using the [model artifact contract](model_artifact_contract.md),
and retain execution/selection evidence in `run.json` using the
[experiment records contract](experiment_record_contract.md). The export commands
below do not generate or validate these records; retain their links in provenance
and review them separately. Missing historical artifacts must be disclosed.

## Available artifacts and recovery

A scan of the working tree, including ignored files, local Git refs, and reachable Git history found no trained checkpoints, feature caches, saved DPE heads, or Waterbirds benchmark images. The AutoBackSwap checkout includes dataset metadata. Locally cached Git ref `origin/method/autobackswap` contains ten per-image CSVs, recoverable **without a network request or branch switch**:

```sh
python scripts/export_predictions.py recover-autobackswap
```

The earlier recovery was executed during viewer setup and its historical copies remain under `outputs/failure_cases/`. New recovery runs now write `experiments/<run>/predictions/test/`:

| Export directories | Legacy source directory (`origin/method/autobackswap`) | Runs | What they represent |
| --- | --- | --- | --- |
| `autobackswap_seed0`, `seed1`, `seed2` | `ours_minority_false_seed0..2` | 3 | AutoBackSwap, excluding minority training groups |
| `autobackswap_w_minority_seed0` | `ours_minority_true_seed0` | 1 | AutoBackSwap, including minority training groups |
| `erm_baseline_seed0`, `seed1`, `seed2` | `erm_minority_false_seed0..2` | 3 | AutoBackSwap implementation's ERM baseline, excluding minority training groups |
| `erm_w_minority_seed0`, `seed1`, `seed2` | `erm_minority_true_seed0..2` | 3 | AutoBackSwap implementation's ERM baseline, including minority training groups |

Each has 5,794 held-out test rows, with groups of 2,255 / 2,255 / 642 / 642 examples. Recovery checks image identities, target/background labels, group mapping, and correctness against metadata. Each provenance file records the local ref, immutable source commit, source path/hash, metadata hash, and recomputed metrics. Recovery is not a new model evaluation or a new reproduction claim.

The recovered ERM runs are **not** the shared DPE-aligned standard/enhanced ERM implementation. Their training recipe remains identified by their original run records. No real POTER or original-DPE CSV has been generated: their required model artifacts are absent. Aggregate metrics cannot reconstruct individual predictions.

Exports refuse to overwrite any existing run directory. To repeat recovery, use a fresh experiment root, for example `--output-root outputs/recovery_check/experiments`. This retains the `<run>/predictions/test/` structure without replacing existing exports.

## Where to put files

All new exports use the general experiment artifact location, independently of the viewer:

```text
experiments/
├── dpe_seed0/predictions/test/
│   ├── predictions.csv
│   └── provenance.json
├── poter_seed1/predictions/test/
│   ├── predictions.csv
│   └── provenance.json
├── autobackswap_seed0/predictions/test/
│   ├── predictions.csv
│   └── provenance.json
└── erm_stage0_seed0/predictions/test/
    ├── predictions.csv
    └── provenance.json

data/waterbirds/waterbird_complete95_forest2water2/
├── metadata.csv
└── <species>/<image>.jpg
```

Each example directory is created only when its export runs. Keep dataset images in the existing dataset directory; an additional `images/` subfolder there also works. Canonical CSV paths are `<species>/<image>.jpg` relative image identities, without a dataset or `images/` prefix.

In the app, select **Open experiment folder** and choose `experiments/` (or one experiment directory), then use **Choose dataset / images folder** to select the dataset directory. The viewer can read predictions and images from separate locations. Existing recovered CSVs in `outputs/failure_cases/` still work; they are historical copies, not the destination for future exports.

If you need the dataset, the repository already provides a download/preparation script:

```sh
python scripts/prepare_waterbirds.py --data_dir data
```

This downloads a large dataset and requires that script's dependencies (including `tqdm`). It was **not run** as part of prediction export. Do not put paper figures or the sample Labrador images in the viewer's image folder; those are not benchmark examples.

## Image-based export: POTER, AutoBackSwap and ERM

Run from the project root, using the method's working Python environment or the shared dependencies in `requirements.txt`. No pretrained weights are downloaded when loading a saved checkpoint. CPU is the default; add `--device cuda` or `--device cuda:0` where available. These commands export test; use `--split val` for validation inspection.

Paths below are **expected example artifact locations**, not files found by the scan. Replace `--checkpoint`, `--metadata`, `--data-dir`, and `--config` with paths to your actual artifacts in Colab, Kaggle, or local storage. Relative command paths assume the project root.

### POTER

The Kaggle runner saves its validation-selected model as `best_validation.pt`. This is preferable to `last_training.pt` when reproducing the runner's best-validation result.

```sh
python scripts/export_predictions.py poter --checkpoint checkpoints/poter/seed1/best_validation.pt --metadata data/waterbirds/waterbird_complete95_forest2water2/metadata.csv --data-dir data/waterbirds --run-label poter_seed1 --seed 1 --output experiments/poter_seed1/predictions/test
```

Supported inputs: a torchvision ResNet-50 state dict, the runner's `{"model": state_dict, ...}` checkpoint, or its original full-model `best_model.pth`. For a trusted legacy full-model pickle or resumable training checkpoint, explicitly add `--trusted-pickle`; loading pickle can execute code, so use it only for a checkpoint you trust. The exporter otherwise uses `weights_only=True`.

Use the **original metadata file used during training** when the checkpoint records its hash. The exporter rejects a different hash or a conflicting saved seed. In the POTER training layout the equivalent dataset root may be `POTER_data/cub/data/waterbird_complete95_forest2water2`; pass that as `--data-dir`, and its `metadata.csv` as `--metadata`.

### AutoBackSwap

The ten recovered CSVs already cover the recorded runs. To export a different trained checkpoint:

```sh
python scripts/export_predictions.py autobackswap --checkpoint checkpoints/autobackswap/seed0/checkpoint.pt --config checkpoints/autobackswap/seed0/config.yaml --metadata data/waterbirds/waterbird_complete95_forest2water2/metadata.csv --data-dir data/waterbirds --run-label autobackswap_new_seed0 --seed 0 --output experiments/autobackswap_new_seed0/predictions/test
```

Supply the saved configuration for **that run**. It determines ResNet/ViT architecture, image size, and normalization. The exporter uses the upstream `Classifier` with pretrained downloads disabled, strictly loads the checkpoint, removes the compile wrapper prefix `_orig_mod.` when present, and checks the config seed. It does not load the background-swap generator or alter training settings.

The older `scripts/autobackswap/dump_all_predictions.py` now also uses the shared writer, includes `split`, writes companion provenance and refuses existing run directories. Its historical default-config inference behavior is retained; prefer the command above for new runs with explicit saved configurations. The separate errors-only helper remains an inspection tool.

### Shared ERM / DPE Stage-0 ERM

```sh
python scripts/export_predictions.py erm --checkpoint checkpoints/erm/waterbirds_standard/seed0/ckpt_last.pt --metadata data/waterbirds/metadata_waterbirds.csv --data-dir data/waterbirds --run-label erm_stage0_seed0 --seed 0 --output experiments/erm_stage0_seed0/predictions/test
```

This accepts the shared/DPE Stage-0 ResNet-50 state dict with `0.*` backbone and `2.*` classifier keys. It forwards the image through the backbone and CE head using **raw** embeddings, preserving the corrected ERM evaluation behavior. An AutoBackSwap ERM checkpoint instead uses the `autobackswap` command with that ERM run's saved config; do not pass it to this loader.

POTER and shared ERM apply square resize to 256, center crop to 224, and ImageNet normalization. AutoBackSwap uses its saved image size and normalization with the same upstream evaluation resize/crop rule. No test-time augmentation or training data is used.

## Original DPE: export from a live session or saved heads

A Stage-0 ERM checkpoint alone is **not** a trained DPE ensemble. The original reproduction notebook's `train_ensemble` returns aggregate metrics and discards its local prototype list; the preserved `DPE` library object retains it as `.ensemble`. If the training session has ended and neither heads nor per-image probabilities were saved, those predictions cannot be reconstructed from metrics. You need the saved ensemble, or a new faithfully configured head-training run on the original backbone/features. Never select the best ensemble size using test failures.

The helper imports below assume the project root is on Python's import path. If the notebook changed its working directory to a baseline checkout, first add the project checkout root to `sys.path`.

### Option A: a trained original `DPE` object is still alive

With the already-trained object named `dpe`, run this in that session:

```python
from src.evaluation.dpe_export import save_dpe_probabilities, export_dpe_probabilities

metadata = "data/waterbirds/metadata_waterbirds.csv"
artifact = "outputs/dpe_seed0_probabilities.npz"
save_dpe_probabilities(
    artifact,
    dpe.predict_proba(target="test", avg=True),
    dpe.datasets["test"].x,
    metadata=metadata,
    seed=0,
    source="Original DPE standard-track run; retain its training configuration and log",
)
export_dpe_probabilities(
    artifact, metadata, "experiments/dpe_seed0/predictions/test",
    run_label="dpe_seed0", seed=0,
)
```

The helpers align image identities to metadata, record the seed and metadata hash, and write the viewer CSV. The object's `predict_proba` uses its original probability averaging. If you already saved identified probabilities as an NPZ with `ids` and `probabilities` (shape `[N,2]` or `[heads,N,2]`), the equivalent CLI is:

```sh
python scripts/export_predictions.py dpe-probabilities --artifact outputs/dpe_seed0_probabilities.npz --metadata data/waterbirds/metadata_waterbirds.csv --run-label dpe_seed0 --seed 0 --output experiments/dpe_seed0/predictions/test
```

Columns must represent landbird `0`, waterbird `1`. A `[heads,N,2]` probability array is averaged equally. Do not supply logits or apply the training entropic scale of 30 at inference. Anonymous arrays without image IDs are rejected.

### Option B: capture the original reproduction notebook's selected heads

If starting a new original-DPE head-training session, execute this **after defining `train_prototypes` and before running its existing experiment call**:

```python
from src.evaluation.dpe_export import capture_notebook_heads

captured_heads = []
original_train_prototypes = train_prototypes
train_prototypes = capture_notebook_heads(original_train_prototypes, captured_heads)
```

Run the original, preconfigured experiment unchanged. The wrapper copies each returned validation-selected head, returns the same object to training, and consumes no randomness. Use a fresh capture list for every separate experiment. After that run finishes:

```python
from pathlib import Path
from src.evaluation.dpe_export import save_dpe_heads, package_notebook_features, export_dpe_heads

train_prototypes = original_train_prototypes
metadata = Path(data_dir) / "waterbirds/metadata_waterbirds.csv"
backbone = Path(ckpt_dir) / "ckpt_last.pt"

save_dpe_heads(
    "outputs/dpe_seed0_heads.pt",
    captured_heads,
    metadata=metadata, backbone=backbone, seed=0,
    source="Original reproduction notebook, standard track, original validation-selected heads",
    normalize_embeddings=True,
)
package_notebook_features(
    "outputs/dpe_seed0_features.npz",
    Path(ckpt_dir) / "feats_test.npy",
    Path(ckpt_dir) / "feature_manifest_test.json",
    metadata, backbone,
)
export_dpe_heads(
    "outputs/dpe_seed0_heads.pt",
    "outputs/dpe_seed0_features.npz",
    metadata,
    "experiments/dpe_seed0/predictions/test",
    run_label="dpe_seed0", seed=0,
)
```

For a live original library object you can instead pass `dpe.ensemble` to `save_dpe_heads`, using its original `norm_emb` setting. The head format supports the original temperature-1 IsoMax+ inference. Calibrated/custom-temperature heads need a separately documented adapter.

The feature packager verifies the corrected notebook's feature-file hash, metadata hash, checkpoint hash, preprocessing description, split, and image order. It refuses an anonymous/unverified legacy feature cache. The subsequent export verifies feature/backbone identity, applies the saved row-standardization setting once to **raw** features, uses L2-normalized IsoMax+ distances, and averages head softmax probabilities equally.

For a later session, the equivalent head-export command is:

```sh
python scripts/export_predictions.py dpe-heads --heads outputs/dpe_seed0_heads.pt --features outputs/dpe_seed0_features.npz --metadata data/waterbirds/metadata_waterbirds.csv --run-label dpe_seed0 --seed 0 --output experiments/dpe_seed0/predictions/test
```

## Output contract and verification

CSV columns are `image_path,true_label,predicted_label,correct,group,group_name,spurious_attribute,confidence,run_label,seed,split`. Group IDs use **1–4**, the viewer's default. Confidence is the predicted-class probability. Each export records input hashes and summary metrics in `provenance.json`; keep the original experiment configuration, command, selection record, and environment too.

The shared writer validates new CSVs automatically. Before reporting or accepting an export, also verify complete split coverage and metadata agreement:

```sh
python scripts/validate_predictions.py experiments --metadata data/waterbirds/metadata_waterbirds.csv
```

The viewer does not need a change or a redeployment to read these CSVs. Open its existing link and select the local output folder. Training and validation protocols remain unchanged.

Maintainers can run focused checks without downloads:

```sh
python -m unittest discover -s tests -p test_prediction_export.py -v
node --test tests/test_viewer.cjs
```

Tests check metadata identity/order, probability validation, group metrics, checkpoint format adapters, no pretrained downloads, singleton batches, and original DPE inference against the preserved upstream IsoMax+ implementation. They are CPU smoke/regression checks, not a verified model reproduction. Keep this guide and the adapters consistent when checkpoint formats change.
