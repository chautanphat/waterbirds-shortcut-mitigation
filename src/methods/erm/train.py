"""Full-backbone cross-entropy ERM; group annotations are evaluation-only."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import yaml

from src.datasets.loaders import evaluation_loader, training_loader
from src.datasets.waterbirds import Waterbirds
from src.evaluation.classification import evaluate
from src.models.resnet import ResNetClassifier
from src.utils.reproducibility import file_sha256, seed_everything


def train_epoch(model, loader, optimizer, device):
    """One pass over a loader, usable with other datasets/models and sampling."""
    model.train()
    loss_sum = count = correct = 0
    for _, inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(targets)
        count += len(targets)
        correct += int((logits.argmax(1) == targets).sum())
    if not count:
        raise ValueError("Training loader is empty")
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


def _cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def run_training(config):
    """Train DPE-style ERM and track validation WGA without loading test data.

    Plain state dictionaries retain the original DPE checkpoint layout.
    Separate training_state.pt stores optimizer/RNG state for downstream tooling.
    Existing output artifacts are rejected to avoid accidental run mixing.
    """
    config.validate()
    d, m, t = config.dataset, config.model, config.training
    seed_everything(t.seed)
    device = torch.device(t.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a CPU smoke test")
    out = Path(config.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Use a new run directory.")
    train_data = Waterbirds(d.data_dir, "train")
    val_data = Waterbirds(d.data_dir, "val")
    if set(val_data.g) != {0, 1, 2, 3}:
        raise ValueError("Validation requires all four groups for checkpoint selection")
    loader = training_loader(train_data, d.batch_size, d.workers, t.sampling, steps=t.steps)
    val_loader = evaluation_loader(val_data, d.batch_size_eval, d.workers)
    model = ResNetClassifier(m.weights, m.freeze_bn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=t.lr, momentum=t.momentum,
                                weight_decay=t.weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[3], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    metadata = {"config": config.as_dict(), "command": sys.argv,
                "git_commit": commit, "git_dirty": dirty,
                "python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "device": str(device),
                "metadata_sha256": file_sha256(train_data.metadata_path),
                "split_counts": {"train": len(train_data), "val": len(val_data)},
                "selection": "last checkpoint, matching the DPE stage-0 handoff",
                "best_checkpoint_selection": "validation worst-group accuracy; ties choose the latest checkpoint"}
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_score, best_position = -1.0, None
    updates = 0

    def validate_and_save(position, train_metrics):
        nonlocal best_score, best_position
        metrics = evaluate(model, val_loader, device)
        record = {**position, "updates": updates, "train": train_metrics, "val": metrics}
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        state = _cpu_state_dict(model)
        score = metrics["worst_group_accuracy"]
        if score >= best_score:
            best_score, best_position = score, position
            torch.save(state, out / "ckpt_best_wga.pt")
        torch.save(state, out / "ckpt_last.pt")
        torch.save({"model": state, "optimizer": optimizer.state_dict(), "updates": updates,
                    "position": position, "best_wga": best_score,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "config": config.as_dict()}, out / "training_state.pt")
        print(json.dumps(record), flush=True)

    if t.epochs is not None:
        for epoch in range(t.epochs):
            train_metrics = train_epoch(model, loader, optimizer, device)
            updates += len(loader)
            validate_and_save({"epoch": epoch + 1}, train_metrics)
    else:
        # A finite replacement stream yields exactly the requested number of updates.
        iterator = iter(loader)
        while updates < t.steps:
            chunk_size = min(t.eval_every, t.steps - updates)
            chunk = (next(iterator) for _ in range(chunk_size))
            train_metrics = train_epoch(model, chunk, optimizer, device)
            updates += chunk_size
            validate_and_save({"step": updates}, train_metrics)
    summary = {"updates": updates, "best_val_wga": best_score, "best_position": best_position,
               "selected_checkpoint": "ckpt_last.pt", "last_checkpoint": "ckpt_last.pt",
               "best_checkpoint": "ckpt_best_wga.pt"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
