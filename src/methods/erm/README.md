# Shared standard ERM

Repository-wide Waterbirds ERM extracted from the **existing DPE stage-0 workflow**. Training, evaluation, and feature export run without importing or installing any preserved baseline. This is the shared starting point for future methods that need a fine-tuned backbone.

## Recipe and source

Defaults in [`configs/erm/waterbirds_standard.yaml`](../../../configs/erm/waterbirds_standard.yaml) match the stage-0 command in [`notebooks/dpe_waterbirds_end_to_end.ipynb`](../../../notebooks/dpe_waterbirds_end_to_end.ipynb), including effective defaults from `baselines/dpe4subpop/main.py` and behavior from its `utils/` modules:

| Setting | Shared default |
| --- | --- |
| Backbone / initialization | ResNet-50 / torchvision `IMAGENET1K_V2` |
| Trainable parameters | Entire backbone and biased linear 2048→2 head |
| BatchNorm | Update running statistics during training, as in DPE |
| Loss | Mean cross entropy; no group weighting or prototype regularization |
| Optimizer | SGD, LR 0.001, momentum 0.9, weight decay 0 |
| Schedule / duration | Constant LR, 100 epochs |
| Training batches | 128; inverse-class-frequency sampling with replacement; drop final incomplete batch |
| Evaluation batches | 256; no sampling/shuffling; retain every example |
| Preprocessing, all splits | Square `Resize((256,256))`, `CenterCrop(224)`, RGB tensor, ImageNet normalization |
| Augmentation | None (DPE's standard `--no_augmentation` track) |
| Seed / workers | 0 / 4 |
| Default checkpoint | Last epoch (`ckpt_last.pt`), matching DPE's stage-1 handoff |
| Additional checkpoint | Best validation WGA each epoch (`ckpt_best_wga.pt`); latest wins ties |

An [enhanced configuration](../../../configs/erm/waterbirds_enhanced.yaml) provides ERM*: it sets `dataset.augmentation: true` for training-only random resized crops, photometric distortion (probability 0.5) and horizontal flips (probability 0.5), matching DPE. Evaluation and feature extraction retain deterministic preprocessing. The standard default remains `augmentation: false`.

The notebook does not load `configs/dpe4subpop/waterbirds_stage0_erm.yaml`: it passes CLI arguments directly. In particular, it uses evaluation batch size 256 and omits `--optim_weight_decay`, so the effective weight decay is 0. The shared defaults follow this executed workflow.

Sources and citations:

- [Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift](https://arxiv.org/abs/2505.23027), ICML 2025; [official code](https://github.com/minhto2802/dpe4subpop), preserved under `baselines/dpe4subpop/` from upstream commit `68b811461e238d44cced6ae6c8ce75b66cad5449`. MIT License, copyright 2025 Minh To. Shared model layout, sampling and training behavior follow this implementation and the project notebook above.
- [SubpopBench](https://github.com/YyzHarry/SubpopBench), MIT License, copyright 2023 Yuzhe Yang. The deterministic Waterbirds preprocessing is inherited through DPE. This shared default targets the existing DPE workflow rather than SubpopBench's standalone default run.

Applicable notices are retained in [LICENSE](LICENSE). The preserved baseline keeps its upstream README at `baselines/dpe4subpop/docs/README.upstream.md`.

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

# Evaluate the last-epoch checkpoint used by DPE on the held-out test set.
python scripts/evaluate_erm.py \
  --checkpoint checkpoints/waterbirds_erm_seed0/ckpt_last.pt \
  --data-dir data/waterbirds --output checkpoints/waterbirds_erm_seed0/test.json

# Export features from the same checkpoint for later methods.
python scripts/extract_erm_features.py \
  --checkpoint checkpoints/waterbirds_erm_seed0/ckpt_last.pt \
  --data-dir data/waterbirds --splits train val test
```

Paths in YAML and CLI overrides resolve from the working directory. Scripts locate their Python imports independently of that directory. Edit a config copy for experimental hyperparameters; unknown fields are rejected. CLI overrides support data/output paths, device, seed, and duration (`--epochs` or `--steps`). The effective config is saved with every run; a nonempty output directory is rejected. Step-based runs are optional experiments and do not use the default DPE epoch recipe.

For a short local smoke run, set `model.weights: null`, reduce batch size/workers in a config copy, and use `--device cpu --steps 2`. This uses random initialization and is not a benchmark run. No automatic weight download occurs during checkpoint evaluation/export.

Training only constructs train and validation datasets. It writes `ckpt_last.pt`, `ckpt_best_wga.pt`, `training_state.pt` (model, optimizer, best model, metric history, update count and RNG states), `config.yaml`, `run.json` (command, code revision/dirty state, software versions and metadata hash), `metrics.jsonl`, and `summary.json`. Epoch-based Python/API callers can opt into `run_training(config, resume=True)`.
New atomic version-2 states restore optimizer, model, best model, metric history
and Python/NumPy/Torch/CUDA RNG states at completed epoch boundaries (including
initialization before epoch 1). Recipe, metadata and training-source changes are
rejected. Non-resume callers retain the nonempty-directory guard; step-based
resume and incomplete legacy states are unsupported. The automated multi-seed
DPE study enables epoch resume by default. Files must remain available across
hosted runtime restarts; deleting the VM without persistent storage loses them. All accuracies are fractions in [0,1]; reports include group counts, per-group accuracy, overall accuracy, and WGA. WGA is null if an evaluated split lacks any of the four groups.

## Reuse from another method

```python
from src.datasets.waterbirds import Waterbirds
from src.datasets.loaders import evaluation_loader
from src.models.resnet import load_resnet_checkpoint
from src.evaluation.classification import evaluate
from src.evaluation.features import extract_features
from src.methods.erm.train import train_epoch

model = load_resnet_checkpoint("checkpoints/waterbirds_erm_seed0/ckpt_last.pt", device="cuda")
data = Waterbirds("data/waterbirds", "val")
loader = evaluation_loader(data)
features = extract_features(model, loader, device="cuda")
metrics = evaluate(model, loader, device="cuda")
```

Run Python from the repository root (or add the root to `PYTHONPATH`). Dataset batches are `(split_local_index, input, target, background)`; `train_epoch` ignores background. Model `extract_features` returns raw `(N,2048)` embeddings, including when a batch has one example. Shared components live under `src/datasets/`, `src/models/`, `src/evaluation/`, and `src/utils/`; ERM orchestration/configuration stays here.

## DPE handoff

Checkpoints are plain state dictionaries with the same sequential backbone/head keys (`0.*`, `2.*`) as DPE's CE stage-0 model; load them strictly. Feature export writes `feats_<split>.npy` in split-local CSV order and **does not normalize** embeddings. DPE performs its own per-example normalization on load. `.npz` sidecars record indices, labels, attributes and filenames; `.json` sidecars record checkpoint/metadata SHA-256 hashes and preprocessing. Existing feature caches are rejected to prevent mixing checkpoints. Use a distinct export directory for each checkpoint when comparing last versus best.

The export command above places the last checkpoint and matching caches in the same directory, ready for DPE:

```bash
# Example annotated DPE stage-1+ run; DPE's --data_dir takes the parent directory.
python baselines/dpe4subpop/main.py \
  --dataset_name Waterbirds --data_dir data \
  --stage 1 --num_stages 16 --train_mode freeze --loss_name isomax \
  --epochs 20 --batch_size 256 --lr 0.001 --train_attr yes --subsample_type group \
  --cov_reg 5.e5 --entropic_scale 30 --ensemble_criterion wga_val \
  --pretrained_path checkpoints/waterbirds_erm_seed0/ckpt_last.pt \
  --ckpt_dir checkpoints/waterbirds_dpe --no_wandb
```

The shared default handoff uses `ckpt_last.pt`, as in the DPE notebook. `ckpt_best_wga.pt` is an additional option for experiments that explicitly use validation selection; export its features to a separate directory to avoid mixing caches. Training only evaluates validation, with held-out test evaluation in a separate script. The original DPE loop also evaluates test each epoch; this reporting is not needed for the last-checkpoint handoff. Record which checkpoint and feature source each downstream run uses.

The [DPE head-reweighting notebook](../../../notebooks/dpe_head_aggregation.ipynb) reuses this trainer end to end. Its [Stage-0 adapter](../dpe_head_reweighting/stage0.py) can train or reuse a completed matching shared run, then generate ordered feature NPZ files and provenance for H2 automatically. The adapter requires the declared 100-epoch recipe and final checkpoint; shortened smoke runs and best-validation checkpoints are excluded from that handoff.

## Validation and limitations

```bash
python -m unittest discover -s tests -v
```

CPU tests cover configuration parity, metadata formats, class-balanced sampling, BatchNorm behavior, strict DPE checkpoint layout, feature ordering/singleton batches, held-out-test isolation, validation selection and exact update counts. This validates the implementation contracts, not published accuracy. A full Waterbirds GPU run and multi-seed benchmark comparison are still required. Dataset-specific logic currently supports Waterbirds and ResNet-50 only; no unrelated baseline is forced onto this infrastructure.
