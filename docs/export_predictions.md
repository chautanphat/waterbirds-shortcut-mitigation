# Export predictions for failure-case inspection

Use [`scripts/export_predictions.py`](../scripts/export_predictions.py) to write a **new run directory** containing `predictions.csv` and `provenance.json`. Exports contain every example in the requested split, including correct predictions. The [viewer](failure_case_viewer.md) can therefore compare failures without assuming that missing rows are correct.

Exporting does not train, tune, or choose a checkpoint. Supply the checkpoint already chosen by the method's validation procedure. Keep distinct training recipes and seeds in distinct run names. The original DPE baseline described here uses equal averaging of head probabilities; it does not use the project's head-reweighting variant.

## Available artifacts and recovery

A scan of the working tree, including ignored files, local Git refs, and reachable Git history found no trained checkpoints, feature caches, saved DPE heads, or Waterbirds benchmark images. The AutoBackSwap checkout includes dataset metadata. Locally cached Git ref `origin/method/autobackswap` contains ten per-image CSVs, recoverable **without a network request or branch switch**:

```sh
python scripts/export_predictions.py recover-autobackswap
```

This command was executed during setup. Its current outputs are under `outputs/failure_cases/`:

| Export directories | Runs | What they represent |
| --- | --- | --- |
| `erm_minority_false_seed0`, `seed1`, `seed2` (same prefix) | 3 | AutoBackSwap implementation's ERM, excluding minority training groups |
| `erm_minority_true_seed0`, `seed1`, `seed2` (same prefix) | 3 | AutoBackSwap implementation's ERM, including minority training groups |
| `ours_minority_false_seed0`, `seed1`, `seed2` (same prefix) | 3 | AutoBackSwap, excluding minority training groups |
| `ours_minority_true_seed0` | 1 | AutoBackSwap, including minority training groups |

Each has 5,794 held-out test rows, with groups of 2,255 / 2,255 / 642 / 642 examples. Recovery checks image identities, target/background labels, group mapping, and correctness against metadata. Each provenance file records the local ref, immutable source commit, source path/hash, metadata hash, and recomputed metrics. Recovery is not a new model evaluation or a new reproduction claim.

The recovered ERM runs are **not** the shared DPE-aligned standard/enhanced ERM implementation. Their training recipe remains identified by their original run records. No real POTER or original-DPE CSV has been generated: their required model artifacts are absent. Aggregate metrics cannot reconstruct individual predictions.

Exports refuse to overwrite any existing run directory. To repeat recovery, use a fresh destination, for example `--output-root outputs/failure_cases_recovered_again`.

## Where to put files

Recommended viewer bundle (all generated files stay in Git-ignored `outputs/`):

```text
outputs/failure_cases/
├── images/                               # add real Waterbirds images here
│   ├── 001.Black_footed_Albatross/
│   │   └── Black_Footed_Albatross_0046_18.jpg
│   └── <other species>/<image>.jpg
├── erm_minority_true_seed0/
│   ├── predictions.csv                   # already recovered
│   └── provenance.json
├── ours_minority_false_seed0/
│   ├── predictions.csv                   # already recovered
│   └── provenance.json
├── <other recovered runs>/
├── poter_seed1/                           # created only after POTER export
│   ├── predictions.csv
│   └── provenance.json
└── dpe_original_standard_seed0/           # created only after original DPE export
    ├── predictions.csv
    └── provenance.json
```

Select **Open experiment folder → `outputs/failure_cases/`** in the app. CSV paths are `<species>/<image>.jpg`; do not add an `images/` prefix to them. The viewer finds the matching files in `images/` automatically.

You can also avoid copying images. Keep an existing dataset in its original location, open the CSV bundle first, then use **Choose dataset / images folder** to select the original dataset folder. Both these layouts work:

```text
data/waterbirds/waterbird_complete95_forest2water2/<species>/<image>.jpg
data/waterbirds/waterbird_complete95_forest2water2/images/<species>/<image>.jpg
```

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
python scripts/export_predictions.py poter --checkpoint checkpoints/poter/seed1/best_validation.pt --metadata data/waterbirds/waterbird_complete95_forest2water2/metadata.csv --data-dir data/waterbirds --run-label poter_seed1 --seed 1 --output outputs/failure_cases/poter_seed1
```

Supported inputs: a torchvision ResNet-50 state dict, the runner's `{"model": state_dict, ...}` checkpoint, or its original full-model `best_model.pth`. For a trusted legacy full-model pickle or resumable training checkpoint, explicitly add `--trusted-pickle`; loading pickle can execute code, so use it only for a checkpoint you trust. The exporter otherwise uses `weights_only=True`.

Use the **original metadata file used during training** when the checkpoint records its hash. The exporter rejects a different hash or a conflicting saved seed. In the POTER training layout the equivalent dataset root may be `POTER_data/cub/data/waterbird_complete95_forest2water2`; pass that as `--data-dir`, and its `metadata.csv` as `--metadata`.

### AutoBackSwap

The ten recovered CSVs already cover the recorded runs. To export a different trained checkpoint:

```sh
python scripts/export_predictions.py autobackswap --checkpoint checkpoints/autobackswap/seed0/checkpoint.pt --config checkpoints/autobackswap/seed0/config.yaml --metadata data/waterbirds/waterbird_complete95_forest2water2/metadata.csv --data-dir data/waterbirds --run-label autobackswap_new_seed0 --seed 0 --output outputs/failure_cases/autobackswap_new_seed0
```

Supply the saved configuration for **that run**. It determines ResNet/ViT architecture, image size, and normalization. The exporter uses the upstream `Classifier` with pretrained downloads disabled, strictly loads the checkpoint, removes the compile wrapper prefix `_orig_mod.` when present, and checks the config seed. It does not load the background-swap generator or alter training settings.

### Shared ERM / DPE Stage-0 ERM

```sh
python scripts/export_predictions.py erm --checkpoint checkpoints/erm/waterbirds_standard/seed0/ckpt_last.pt --metadata data/waterbirds/metadata_waterbirds.csv --data-dir data/waterbirds --run-label erm_shared_standard_seed0 --seed 0 --output outputs/failure_cases/erm_shared_standard_seed0
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
artifact = "outputs/dpe_original_standard_seed0_probabilities.npz"
save_dpe_probabilities(
    artifact,
    dpe.predict_proba(target="test", avg=True),
    dpe.datasets["test"].x,
    metadata=metadata,
    seed=0,
    source="Original DPE standard-track run; retain its training configuration and log",
)
export_dpe_probabilities(
    artifact, metadata, "outputs/failure_cases/dpe_original_standard_seed0",
    run_label="dpe_original_standard_seed0", seed=0,
)
```

The helpers align image identities to metadata, record the seed and metadata hash, and write the viewer CSV. The object's `predict_proba` uses its original probability averaging. If you already saved identified probabilities as an NPZ with `ids` and `probabilities` (shape `[N,2]` or `[heads,N,2]`), the equivalent CLI is:

```sh
python scripts/export_predictions.py dpe-probabilities --artifact outputs/dpe_original_standard_seed0_probabilities.npz --metadata data/waterbirds/metadata_waterbirds.csv --run-label dpe_original_standard_seed0 --seed 0 --output outputs/failure_cases/dpe_original_standard_seed0
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
    "outputs/dpe_original_standard_seed0_heads.pt",
    captured_heads,
    metadata=metadata, backbone=backbone, seed=0,
    source="Original reproduction notebook, standard track, original validation-selected heads",
    normalize_embeddings=True,
)
package_notebook_features(
    "outputs/dpe_original_standard_seed0_features.npz",
    Path(ckpt_dir) / "feats_test.npy",
    Path(ckpt_dir) / "feature_manifest_test.json",
    metadata, backbone,
)
export_dpe_heads(
    "outputs/dpe_original_standard_seed0_heads.pt",
    "outputs/dpe_original_standard_seed0_features.npz",
    metadata,
    "outputs/failure_cases/dpe_original_standard_seed0",
    run_label="dpe_original_standard_seed0", seed=0,
)
```

For a live original library object you can instead pass `dpe.ensemble` to `save_dpe_heads`, using its original `norm_emb` setting. The head format supports the original temperature-1 IsoMax+ inference. Calibrated/custom-temperature heads need a separately documented adapter.

The feature packager verifies the corrected notebook's feature-file hash, metadata hash, checkpoint hash, preprocessing description, split, and image order. It refuses an anonymous/unverified legacy feature cache. The subsequent export verifies feature/backbone identity, applies the saved row-standardization setting once to **raw** features, uses L2-normalized IsoMax+ distances, and averages head softmax probabilities equally.

For a later session, the equivalent head-export command is:

```sh
python scripts/export_predictions.py dpe-heads --heads outputs/dpe_original_standard_seed0_heads.pt --features outputs/dpe_original_standard_seed0_features.npz --metadata data/waterbirds/metadata_waterbirds.csv --run-label dpe_original_standard_seed0 --seed 0 --output outputs/failure_cases/dpe_original_standard_seed0
```

## Output contract and verification

CSV columns are `image_path,true_label,predicted_label,correct,group,group_name,spurious_attribute,confidence,run_label,seed,split`. Group IDs use **1–4**, the viewer's default. Confidence is the predicted-class probability. Each export records input hashes and summary metrics in `provenance.json`; keep the original experiment configuration, command, selection record, and environment too.

The viewer does not need a change or a redeployment to read these CSVs. Open its existing link and select the local output folder. Training and validation protocols remain unchanged.

Maintainers can run focused checks without downloads:

```sh
python -m unittest discover -s tests -p test_prediction_export.py -v
node --test tests/test_failure_case_viewer.cjs
```

Tests check metadata identity/order, probability validation, group metrics, checkpoint format adapters, no pretrained downloads, singleton batches, and original DPE inference against the preserved upstream IsoMax+ implementation. They are CPU smoke/regression checks, not a verified model reproduction. Keep this guide and the adapters consistent when checkpoint formats change.
