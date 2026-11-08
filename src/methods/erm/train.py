"""Full-backbone cross-entropy ERM; group annotations are evaluation-only."""

import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml

from src.datasets.loaders import evaluation_loader, training_loader
from src.datasets.waterbirds import Waterbirds, waterbirds_training_transform
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
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _atomic_save(value, path):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}-')
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_text(text, path):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}-')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _recipe(value):
    value = json.loads(json.dumps(value))
    value.pop('output_dir')
    value['dataset'].pop('data_dir')
    value['dataset'].pop('workers')
    value['training'].pop('device')
    return value


def _training_sources():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / 'datasets/waterbirds.py', root / 'datasets/loaders.py',
             root / 'models/resnet.py', root / 'utils/reproducibility.py',
             root / 'evaluation/classification.py', Path(__file__).with_name('config.py')]
    return {str(p.relative_to(root)): file_sha256(p) for p in paths}


def run_training(config, *, resume=False):
    """Train DPE-style ERM and track validation WGA without loading test data.

    Plain state dictionaries retain the original DPE checkpoint layout.
    Separate training_state.pt stores optimizer/RNG state for downstream tooling.
    Existing output artifacts are rejected unless epoch-boundary resume is
    explicitly enabled. Resume restores optimizer, model, RNG and metric history.
    """
    config.validate()
    d, m, t = config.dataset, config.model, config.training
    seed_everything(t.seed)
    device = torch.device(t.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a CPU smoke test")
    out = Path(config.output_dir)
    saved = None
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory is not empty: {out}. Use a new run directory.")
        if t.epochs is None or not (out / 'training_state.pt').is_file():
            raise ValueError('Epoch resume requires a saved training_state.pt checkpoint')
        saved = torch.load(out / 'training_state.pt', map_location='cpu', weights_only=True)
        if (saved.get('version') != 2 or _recipe(saved['config']) != _recipe(config.as_dict()) or
                saved['implementation'] != _training_sources()):
            raise ValueError('Resume checkpoint recipe or training implementation differs')
    train_data = Waterbirds(d.data_dir, "train", transform=(
        waterbirds_training_transform() if d.augmentation else None))
    val_data = Waterbirds(d.data_dir, "val")
    if set(val_data.g) != {0, 1, 2, 3}:
        raise ValueError("Validation requires all four groups for checkpoint selection")
    loader = training_loader(train_data, d.batch_size, d.workers, t.sampling, steps=t.steps)
    val_loader = evaluation_loader(val_data, d.batch_size_eval, d.workers)
    model = ResNetClassifier(None if saved else m.weights, m.freeze_bn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=t.lr, momentum=t.momentum,
                                weight_decay=t.weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    if saved is None:
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
    best_score, best_position = -1.0, None
    updates = 0
    start_epoch, history, best_state = 0, [], None
    if saved is not None:
        original = json.loads((out / 'run.json').read_text())
        if (original['config'] != saved['config'] or
                original['metadata_sha256'] != metadata['metadata_sha256'] or
                saved['metadata_sha256'] != metadata['metadata_sha256'] or
                saved['config'] != yaml.safe_load((out / 'config.yaml').read_text())):
            raise ValueError('Resume data or saved configuration changed')
        metadata = original
        metadata.setdefault('resumes', []).append(dict(device=str(device), workers=d.workers,
                                                       start_epoch=saved['position']['epoch'],
                                                       torch=str(torch.__version__)))
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        updates, start_epoch = saved['updates'], saved['position']['epoch']
        if not 0 <= start_epoch <= t.epochs or updates != start_epoch * len(loader):
            raise ValueError('Invalid epoch/update resume position')
        history = saved['history']
        if len(history) != start_epoch or (history and history[-1]['updates'] != updates):
            raise ValueError('Resume metric history is incomplete')
        best_score, best_position, best_state = saved['best_wga'], saved['best_position'], saved['best_model']
        (out / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        _atomic_save(saved['model'], out / 'ckpt_last.pt')
        if best_state is not None:
            _atomic_save(best_state, out / 'ckpt_best_wga.pt')
        random.setstate(saved['python_rng_state'])
        nr = saved['numpy_rng_state']
        np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type == 'cuda' and saved['cuda_rng_state']:
            for index, state in enumerate(saved['cuda_rng_state'][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, index)
        print(f'Resuming after epoch {start_epoch}; target {t.epochs}', flush=True)
    _atomic_text(json.dumps(metadata, indent=2) + "\n", out / 'run.json')

    def save_training_state(position, state):
        nr = np.random.get_state()
        _atomic_save({"version": 2, "model": state, "optimizer": optimizer.state_dict(), "updates": updates,
                    "position": position, "best_wga": best_score,
                    "best_position": best_position, "best_model": best_state, "history": history,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": (nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "metadata_sha256": metadata['metadata_sha256'], "implementation": _training_sources(),
                    "config": metadata['config']}, out / "training_state.pt")

    def validate_and_save(position, train_metrics):
        nonlocal best_score, best_position, best_state
        metrics = evaluate(model, val_loader, device)
        record = {**position, "updates": updates, "train": train_metrics, "val": metrics}
        history.append(record)
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        state = _cpu_state_dict(model)
        score = metrics["worst_group_accuracy"]
        if score >= best_score:
            best_score, best_position = score, position
            best_state = state
            _atomic_save(state, out / "ckpt_best_wga.pt")
        _atomic_save(state, out / "ckpt_last.pt")
        save_training_state(position, state)
        print(json.dumps(record), flush=True)

    if t.epochs is not None:
        if saved is None:
            # A disconnect during the first epoch can restart from epoch zero.
            save_training_state({'epoch': 0}, _cpu_state_dict(model))
        for epoch in range(start_epoch, t.epochs):
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
    _atomic_text(json.dumps(summary, indent=2) + "\n", out / 'summary.json')
    return summary
