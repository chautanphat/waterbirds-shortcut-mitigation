"""Run the official Waterbirds configuration with resumable epoch checkpoints."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision

from data.data import prepare_data
from loss import LossComputer
from ot_reweight import compute_ot_potentials, compute_sample_weights
from run_expt import parse_args as upstream_args
from train import run_epoch
from training_checkpoint import load_checkpoint, restore_training, save_checkpoint
from utils import CSVBatchLogger, Logger, log_args, set_seed


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def trim_csv(path, completed_epoch):
    """Discard rows written after the last committed checkpoint, before appending."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= completed_epoch]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=9.5)
    options = parser.parse_args()
    if min(options.epochs, options.batch_size) < 1 or options.max_hours <= 0:
        parser.error("Epochs, batch size and max hours must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required. Enable GPU T4 x2 in Kaggle Settings.")
    torch.cuda.set_device(options.gpu)

    # Reuse the original argument defaults, dataset setup, transforms and losses.
    original_argv = sys.argv
    sys.argv = [str(ROOT / "run_expt.py"), "--dataset", "CUB", "--data_dir",
                str(options.data_dir), "--ot_cache", str(options.cache),
                "--f_power", "0.1", "--clip_cap", "20", "--reg_scale", "0.03",
                "--n_epochs", str(options.epochs), "--lr", "1e-5",
                "--weight_decay", "1.0", "--batch_size", str(options.batch_size),
                "--augment_data", "--seed", str(options.seed), "--gpu", str(options.gpu)]
    try:
        args = upstream_args()
    finally:
        sys.argv = original_argv
    args.log_dir = str(options.run_dir)
    args.log_every = 50
    metadata = options.data_dir / "cub/data/waterbird_complete95_forest2water2/metadata.csv"
    config = {
        "dataset": "CUB", "seed": options.seed, "epochs": options.epochs,
        "batch_size": options.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "f_power": args.f_power, "clip_cap": args.clip_cap, "reg_scale": args.reg_scale,
        "augment_data": True, "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": np.__version__,
        "metadata_sha256": digest(metadata), "cache_sha256": digest(options.cache),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / "train.py", ROOT / "loss.py", ROOT / "ot_reweight.py",
            ROOT / "utils.py", ROOT / "run_expt.py", ROOT / "scripts/kaggle_waterbirds.py",
            ROOT / "scripts/training_checkpoint.py", *sorted((ROOT / "data").glob("*.py")),
        ]},
    }
    options.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.run_dir / "last_training.pt"
    state = load_checkpoint(checkpoint_path, config) if checkpoint_path.exists() else None
    if state is None and (options.run_dir / "log.txt").exists():
        parser.error("This run directory has logs but no checkpoint. Use a new run directory.")
    start_epoch = state["epoch"] + 1 if state else 0
    if state:
        for name in ["train", "val", "test"]:
            trim_csv(options.run_dir / f"{name}.csv", state["epoch"])
    logger = Logger(str(options.run_dir / "log.txt"), mode="a" if state else "w")
    if state is None:
        log_args(args, logger)
    logger.write(f"Starting at epoch {start_epoch} / {options.epochs}.\n")
    (options.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    set_seed(args.seed, strict=False, warn_only=False)
    train_data, val_data, test_data = prepare_data(args, train=True)
    cache = torch.load(options.cache, map_location="cpu", weights_only=False)
    if cache.get("dataset") != "CUB" or cache.get("ref_per_group") != {0: 0, 1: 466, 2: 133, 3: 0}:
        raise ValueError("Expected the clean Waterbirds reference cache (0,466,133,0)")
    if len(cache["y_train"]) != len(train_data) or not np.array_equal(
        np.asarray(cache["y_train"]), np.asarray(train_data._y_array)
    ):
        raise ValueError("Feature cache does not match the training dataset")
    potentials = compute_ot_potentials(cache["X_train"], cache["y_train"],
                                      cache["X_val"], cache["y_val"],
                                      reg_scale=args.reg_scale, device=f"cuda:{args.gpu}")
    weights = torch.from_numpy(compute_sample_weights(
        potentials, cache["y_train"], f_power=args.f_power, clip_cap=args.clip_cap)).float()
    del cache
    loader_args = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": True}
    train_loader = train_data.get_loader(train=True, reweight_groups=False,
                                        sample_weights=weights, **loader_args)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_args)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_args)
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, train_data.n_classes)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    train_loss = LossComputer(criterion, False, train_data, alpha=args.alpha,
                              gamma=args.gamma, adj=np.zeros(train_data.n_groups),
                              step_size=args.robust_step_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    best_wga = state["best_wga"] if state else -1.0
    if state:
        restore_training(state, model, optimizer, train_loss)
        del state
    csv_logs = {name: CSVBatchLogger(str(options.run_dir / f"{name}.csv"),
                                    train_data.n_groups, mode="a" if start_epoch else "w")
                for name in ["train", "val", "test"]}
    deadline = time.monotonic() + options.max_hours * 3600
    completed = start_epoch >= options.epochs
    for epoch in range(start_epoch, options.epochs):
        started = time.monotonic()
        logger.write(f"\nEpoch [{epoch}]:\nTraining:\n")
        torch.cuda.reset_peak_memory_stats()
        run_epoch(epoch, model, optimizer, train_loader, train_loss, logger,
                  csv_logs["train"], args, is_training=True, show_progress=True,
                  log_every=args.log_every, sample_importance=None)
        logger.write("\nValidation:\n")
        val_loss = LossComputer(criterion, False, val_data, alpha=args.alpha,
                                 step_size=args.robust_step_size)
        run_epoch(epoch, model, optimizer, val_loader, val_loss, logger,
                  csv_logs["val"], args, is_training=False)
        test_loss = LossComputer(criterion, False, test_data, alpha=args.alpha,
                                  step_size=args.robust_step_size)
        run_epoch(epoch, model, optimizer, test_loader, test_loss, None,
                  csv_logs["test"], args, is_training=False)
        wga = float(val_loss.avg_group_acc.min().item())
        if wga > best_wga:
            best_wga = wga
            temporary = options.run_dir / "best_validation.pt.tmp"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_wga": wga, "config": config}, temporary)
            os.replace(temporary, options.run_dir / "best_validation.pt")
        for csv_logger in csv_logs.values():
            csv_logger.flush()
        elapsed = time.monotonic() - started
        logger.write(f"Epoch seconds: {elapsed:.1f}; peak allocated VRAM MiB: "
                     f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}\n")
        logger.flush()
        save_checkpoint(checkpoint_path, model, optimizer, epoch, config, best_wga, train_loss)
        completed = epoch + 1 >= options.epochs
        if not completed and time.monotonic() + elapsed * 1.5 >= deadline:
            logger.write("Session budget reached. Epoch checkpoint saved; rerun to continue.\n")
            break
    for csv_logger in csv_logs.values():
        csv_logger.close()
    logger.write("\nTraining complete.\n" if completed else "\nTraining paused.\n")
    logger.flush()
    logger.close()


if __name__ == "__main__":
    main()
