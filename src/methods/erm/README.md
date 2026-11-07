# Shared standard ERM

Repository-wide, project-owned Waterbirds ERM, using the **DPE standard stage-0 recipe**. Training, evaluation, and feature export run without importing or installing any preserved baseline. This is the shared starting point for future methods that need a fine-tuned backbone.

## Recipe and source

The source of truth for the defaults is [`configs/erm/waterbirds_standard.yaml`](../../../configs/erm/waterbirds_standard.yaml), aligned with [`configs/dpe4subpop/waterbirds_stage0_erm.yaml`](../../../configs/dpe4subpop/waterbirds_stage0_erm.yaml) and the implementation in `baselines/dpe4subpop/utils/`:

| Setting | Shared default |
| --- | --- |
| Backbone / initialization | ResNet-50 / torchvision `IMAGENET1K_V2` |
| Trainable parameters | Entire backbone and biased linear 2048→2 head |
| BatchNorm | Update running statistics during training, as in DPE |
| Loss | Mean cross entropy; no group weighting or prototype regularization |
| Optimizer | SGD, LR 0.001, momentum 0.9, weight decay 0.0001 |
| Schedule / duration | Constant LR, 100 epochs |
| Training batches | 128; inverse-class-frequency sampling with replacement; drop final incomplete batch |
| Evaluation batches | 512; no sampling/shuffling; retain every example |
| Preprocessing, all splits | Square `Resize((256,256))`, `CenterCrop(224)`, RGB tensor, ImageNet normalization |
| Augmentation | None (DPE's standard `--no_augmentation` track) |
| Seed / workers | 0 / 4 |
| Checkpoint selection | Validation worst-group accuracy each epoch; latest wins ties |

**Weight-decay discrepancy:** DPE's tracked YAML sets `optim_weight_decay: 0.0001`, but `main.py` defaults to `0` when the flag is omitted. This implementation follows the tracked YAML. Reproducing an older command that omitted weight decay requires an explicit config copy with `weight_decay: 0`; such a run differs from this shared default.

This recipe inherits SubpopBench preprocessing, but **does not claim to use SubpopBench's default hyperparameters**. SubpopBench uses batch size 108, 5,001 updates, uniform replacement sampling, ImageNet V1 weights, and frozen BatchNorm statistics. DPE differs on those points. Full upstream attribution and references are in [ORIGINAL_SOURCE.md](ORIGINAL_SOURCE.md).

## Setup and data

From the repository root, install a matching PyTorch/torchvision pair for your CPU or CUDA runtime, then:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_waterbirds.py --data_dir data
```

Pass `--data-dir data/waterbirds` to shared scripts. It must contain `metadata_waterbirds.csv` and `waterbird_complete95_forest2water2/<images>`. The loader also accepts original `metadata.csv` inside the archive directory (columns `img_filename`, `y`, `place`, `split`), or flattened metadata/images. DPE's original CLI takes the **parent** `--data_dir data`; the shared scripts take the Waterbirds directory itself.

Labels: landbird=0, waterbird=1; backgrounds: land=0, water=1. Evaluation groups use `g = 2*y + a`: 0=landbird/land, 1=landbird/water, 2=waterbird/land, 3=waterbird/water. The official train/val/test counts are 4,795 / 1,199 / 5,794. Dataset checks reject missing images and invalid metadata; all four validation groups are required for selection.

## Train, evaluate, and export

```bash
# Inspect the effective config without downloading weights or opening data.
python scripts/train_erm.py --print-config

# Full shared run; choose a fresh output directory per seed/configuration.
python scripts/train_erm.py --config configs/erm/waterbirds_standard.yaml \
  --data-dir data/waterbirds --output-dir checkpoints/waterbirds_erm_seed0 --seed 0

# Evaluate the validation-selected checkpoint once on the held-out test set.
python scripts/evaluate_erm.py \
  --checkpoint checkpoints/waterbirds_erm_seed0/ckpt_best_wga.pt \
  --data-dir data/waterbirds --output checkpoints/waterbirds_erm_seed0/test.json

# Export features from the same selected checkpoint for later methods.
python scripts/extract_erm_features.py \
  --checkpoint checkpoints/waterbirds_erm_seed0/ckpt_best_wga.pt \
  --data-dir data/waterbirds --splits train val test
```

Paths in YAML and CLI overrides resolve from the working directory. Scripts locate their Python imports independently of that directory. Edit a config copy for experimental hyperparameters; unknown fields are rejected. CLI overrides support data/output paths, device, seed, and duration (`--epochs` or `--steps`). The effective config is saved with every run; a nonempty output directory is rejected. Step-based runs are optional experiments and do not use the default DPE epoch recipe.

For a short local smoke run, set `model.weights: null`, reduce batch size/workers in a config copy, and use `--device cpu --steps 2`. This uses random initialization and is not a benchmark run. No automatic weight download occurs during checkpoint evaluation/export.

Training only constructs train and validation datasets. It writes `ckpt_last.pt`, `ckpt_best_wga.pt`, `training_state.pt` (model, optimizer, update count and PyTorch RNG state), `config.yaml`, `run.json` (command, code revision/dirty state, software versions and metadata hash), `metrics.jsonl`, and `summary.json`. Resume is not implemented. All accuracies are fractions in [0,1]; reports include group counts, per-group accuracy, overall accuracy, and WGA. WGA is null if an evaluated split lacks any of the four groups.

## Reuse from another method

```python
from src.datasets.waterbirds import Waterbirds
from src.datasets.loaders import evaluation_loader
from src.models.resnet import load_resnet_checkpoint
from src.evaluation.classification import evaluate
from src.evaluation.features import extract_features
from src.methods.erm.train import train_epoch

model = load_resnet_checkpoint("checkpoints/waterbirds_erm_seed0/ckpt_best_wga.pt", device="cuda")
data = Waterbirds("data/waterbirds", "val")
loader = evaluation_loader(data)
features = extract_features(model, loader, device="cuda")
metrics = evaluate(model, loader, device="cuda")
```

Run Python from the repository root (or add the root to `PYTHONPATH`). Dataset batches are `(split_local_index, input, target, background)`; `train_epoch` ignores background. Model `extract_features` returns raw `(N,2048)` embeddings, including when a batch has one example. Shared components live under `src/datasets/`, `src/models/`, `src/evaluation/`, and `src/utils/`; ERM orchestration/configuration stays here.

## DPE handoff

Checkpoints are plain state dictionaries with the same sequential backbone/head keys (`0.*`, `2.*`) as DPE's CE stage-0 model; load them strictly. Feature export writes `feats_<split>.npy` in split-local CSV order and **does not normalize** embeddings. DPE performs its own per-example normalization on load. `.npz` sidecars record indices, labels, attributes and filenames; `.json` sidecars record checkpoint/metadata SHA-256 hashes and preprocessing. Existing feature caches are rejected to prevent mixing checkpoints. Use a distinct export directory for each checkpoint when comparing last versus best.

To use the original DPE last-epoch handoff after exporting a best-checkpoint cache, keep the last checkpoint and its matching cache together in a separate directory:

```bash
mkdir -p checkpoints/waterbirds_erm_seed0_last_features
cp checkpoints/waterbirds_erm_seed0/ckpt_last.pt checkpoints/waterbirds_erm_seed0_last_features/
python scripts/extract_erm_features.py \
  --checkpoint checkpoints/waterbirds_erm_seed0_last_features/ckpt_last.pt \
  --data-dir data/waterbirds

# Example annotated DPE stage-1+ run; DPE's --data_dir takes the parent directory.
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds --data_dir data \
  --stage 1 --num_stages 16 --train_mode freeze --loss_name isomax \
  --epochs 20 --batch_size 256 --lr 0.001 --train_attr yes --subsample_type group \
  --cov_reg 5.e5 --entropic_scale 30 --ensemble_criterion wga_val \
  --pretrained_path checkpoints/waterbirds_erm_seed0_last_features/ckpt_last.pt \
  --ckpt_dir checkpoints/waterbirds_dpe --no_wandb
```

The existing DPE notebook and configs remain historical reproduction entry points. Their upstream flow is preserved. Reusing a shared backbone is a new experiment; record the checkpoint choice and feature source explicitly. The shared trainer selects only on validation, whereas the original DPE loop also evaluates test each epoch and can overwrite some best-checkpoint files using test metrics. No old results are relabeled as shared-ERM results.

## Validation and limitations

```bash
python -m unittest discover -s tests -v
```

CPU tests cover configuration parity, metadata formats, class-balanced sampling, BatchNorm behavior, strict DPE checkpoint layout, feature ordering/singleton batches, held-out-test isolation, validation selection and exact update counts. This validates the implementation contracts, not published accuracy. A full Waterbirds GPU run and multi-seed benchmark comparison are still required. Dataset-specific logic currently supports Waterbirds and ResNet-50 only; no unrelated baseline is forced onto this infrastructure.
