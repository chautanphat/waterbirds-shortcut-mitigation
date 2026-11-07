"""Shared ERM training and an audited, atomic handoff to the H2 experiment."""

import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from src.datasets.loaders import evaluation_loader
from src.datasets.waterbirds import Waterbirds
from src.evaluation.features import extract_features
from src.methods.erm.config import ERMConfig, load_config
from src.methods.erm.train import run_training
from src.models.resnet import load_resnet_checkpoint
from .experiment import digest, feature_file, metadata, read_json, write_json


EVAL_TRANSFORMS = "Resize((256,256)), CenterCrop(224), RGB tensor, ImageNet normalization"
ENHANCED_TRANSFORMS = ("RandomResizedCrop(224), RandomPhotometricDistort(p=0.5), "
                       "RandomHorizontalFlip(p=0.5), RGB tensor, ImageNet normalization")


def backbone_config(data_dir, output_dir, track="standard", seed=0, device="cpu", workers=4):
    """The fixed 100-epoch stage-0 recipe for one track/seed."""
    if track not in ("standard", "enhanced") or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Choose standard/enhanced and seed 0, 1 or 2")
    c = ERMConfig()
    c.profile = f"dpe_{track}_erm"
    c.dataset.data_dir = str(Path(data_dir).resolve())
    c.dataset.workers = workers
    c.dataset.augmentation = track == "enhanced"
    c.training.seed, c.training.device = seed, device
    c.output_dir = str(Path(output_dir).resolve())
    return c.validate()


def validate_dataset(data_dir):
    """Check counts, IDs and every image path before a long backbone run."""
    datasets = [Waterbirds(data_dir, name) for name in ("train", "val", "test")]
    rows = metadata(datasets[0].metadata_path)
    if [len(d) for d in datasets] != [4795, 1199, 5794]:
        raise ValueError("Expected official Waterbirds split counts 4795/1199/5794")
    if any(set(d.g) != {0, 1, 2, 3} for d in datasets):
        raise ValueError("All four groups must be present in each official split")
    return datasets[0].metadata_path, rows


def _validate_recipe(c):
    c.validate()
    expected = backbone_config(c.dataset.data_dir, c.output_dir,
                               "enhanced" if c.dataset.augmentation else "standard",
                               c.training.seed, c.training.device, c.dataset.workers)
    # Resolve paths before comparison; they do not define the scientific recipe.
    actual = c.as_dict()
    actual["dataset"]["data_dir"] = expected.dataset.data_dir
    actual["output_dir"] = expected.output_dir
    if actual != expected.as_dict():
        raise ValueError("H2 requires the declared ImageNet-initialized 100-epoch backbone recipe")


def completed_backbone(c):
    """Accept only a completed shared run using the declared final epoch."""
    _validate_recipe(c)
    out = Path(c.output_dir)
    required = [out / name for name in ("ckpt_last.pt", "config.yaml", "run.json", "summary.json")]
    if any(not p.is_file() for p in required):
        raise ValueError(f"Incomplete shared ERM run: {out}. The shared trainer does not resume "
                         "partial runs; preserve it and choose a fresh BACKBONE_DIR.")
    saved = load_config(out / "config.yaml")
    # Operational paths, workers and device may differ when reusing a completed run.
    actual, expected = saved.as_dict(), c.as_dict()
    for value in (actual, expected):
        value.pop("output_dir")
        value["dataset"].pop("data_dir")
        value["dataset"].pop("workers")
        value["training"].pop("device")
    if actual != expected:
        raise ValueError("Saved backbone recipe differs from the declared track/seed/100-epoch recipe")
    run, summary = read_json(out / "run.json"), read_json(out / "summary.json")
    recorded_config = run["config"]
    recorded_config["dataset"].setdefault("augmentation", False)
    if recorded_config != saved.as_dict():
        raise ValueError("Backbone run.json and config.yaml disagree")
    if (summary["selected_checkpoint"] != "ckpt_last.pt" or
            summary["updates"] != (4795 // c.dataset.batch_size) * c.training.epochs):
        raise ValueError("Backbone has not completed the declared final-epoch recipe")
    if not run.get("git_commit"):
        raise ValueError("Backbone training source commit is missing from run.json")
    return run


def train_or_reuse(c):
    """Use the shared trainer unchanged; reuse only completed, matching runs."""
    _validate_recipe(c)
    path, _ = validate_dataset(c.dataset.data_dir)
    torch.empty(1, device=c.training.device)
    out = Path(c.output_dir)
    if out.exists() and any(out.iterdir()):
        run = completed_backbone(c)
        if run["metadata_sha256"] != digest(path):
            raise ValueError("Backbone training metadata changed")
        print(f"Reusing completed shared ERM: {out}", flush=True)
    else:
        print(f"Training shared {c.profile}: 100 epochs, seed {c.training.seed}", flush=True)
        run_training(c)
        completed_backbone(c)
    return out / "ckpt_last.pt"


def export_inputs(c, output_dir):
    """Extract raw features with shared utilities; publish only a complete bundle.

    No checkpoint is duplicated. A handoff record binds all generated artifacts
    to the completed shared run, metadata, checkpoint and extraction source.
    """
    metadata_path, rows = validate_dataset(c.dataset.data_dir)
    run = completed_backbone(c)
    if run["metadata_sha256"] != digest(metadata_path):
        raise ValueError("Backbone training metadata changed")
    backbone = Path(c.output_dir) / "ckpt_last.pt"
    out = Path(output_dir).resolve()
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), root / "datasets/waterbirds.py", root / "datasets/loaders.py",
                    root / "evaluation/features.py", root / "models/resnet.py"]
    identity = dict(checkpoint_sha256=digest(backbone), metadata_sha256=digest(metadata_path),
                    run_sha256=digest(Path(c.output_dir) / "run.json"),
                    config_sha256=digest(Path(c.output_dir) / "config.yaml"),
                    summary_sha256=digest(Path(c.output_dir) / "summary.json"),
                    implementation={str(p.relative_to(root)): digest(p) for p in source_paths})
    inputs = dict(metadata=str(out / "metadata.csv"), backbone=str(backbone.resolve()),
                  val_features=str(out / "val_raw.npz"), test_features=str(out / "test_raw.npz"),
                  provenance=str(out / "provenance.json"))
    if out.exists():
        record = read_json(out / "handoff.json")
        if record["identity"] != identity:
            raise ValueError("Stage-0 handoff inputs or extraction implementation changed; use a fresh INPUT_DIR")
        for name, checksum in record["artifacts"].items():
            if digest(out / name) != checksum:
                raise ValueError(f"Stage-0 handoff artifact changed: {name}")
        print(f"Reusing verified raw features: {out}", flush=True)
        return inputs
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        shutil.copyfile(metadata_path, staging / "metadata.csv")
        model = load_resnet_checkpoint(backbone, device=c.training.device)
        for split, name in ((1, "val"), (2, "test")):
            print(f"Extracting raw {name} features", flush=True)
            data = Waterbirds(c.dataset.data_dir, name)
            loader = evaluation_loader(data, c.dataset.batch_size_eval, c.dataset.workers)
            features = extract_features(model, loader, c.training.device)["features"]
            path = staging / f"{name}_raw.npz"
            np.savez_compressed(path, ids=np.asarray(data.filenames), features=features)
            feature_file(path, [r for r in rows if r["split"] == split], 2048)
        provenance = dict(track="enhanced" if c.dataset.augmentation else "standard",
                          seed=c.training.seed, checkpoint_sha256=identity["checkpoint_sha256"],
                          val_features_sha256=digest(staging / "val_raw.npz"), model="resnet50",
                          pretrained_weights=c.model.weights,
                          transforms=f"train: {ENHANCED_TRANSFORMS if c.dataset.augmentation else EVAL_TRANSFORMS}; "
                                     f"val/test: {EVAL_TRANSFORMS}",
                          extraction_order="metadata_split_order", feature_format="raw",
                          checkpoint_selection="final_epoch_no_validation_selection",
                          source_commit=run["git_commit"])
        write_json(staging / "provenance.json", provenance)
        names = ("metadata.csv", "val_raw.npz", "test_raw.npz", "provenance.json")
        write_json(staging / "handoff.json", dict(identity=identity,
                   artifacts={name: digest(staging / name) for name in names}, backbone_run=run))
        # A failed/interrupted extraction cannot leave an apparently complete bundle.
        os.replace(staging, out)
    return inputs
