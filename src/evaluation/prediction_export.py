"""Shared prediction CSVs from selected Waterbirds predictions/checkpoints.

Export adapters leave training, checkpoint selection and upstream files untouched.
Imports for image inference are lazy so cached CSV recovery needs only NumPy.
"""

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tempfile

import numpy as np

from .prediction_protocol import (BACKGROUNDS, FIELDS, GROUP_NAMES, LABELS, SCHEMA_VERSION,
                                  SPLITS, image_path, read_metadata, validate_prediction_csv)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def align_ids(ids, rows):
    """Align identified artifacts to metadata, accepting DPE absolute dataset paths."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != len(rows):
        raise ValueError("IDs must be one-dimensional and cover the entire requested split")
    expected = {r["image_path"]: i for i, r in enumerate(rows)}
    matched = {}
    for position, value in enumerate(ids.astype(str)):
        value = value.replace("\\", "/")
        candidates = ["/".join(value.split("/")[i:]) for i in range(len(value.split("/")))]
        matches = [name for name in candidates if name in expected]
        if len(matches) != 1 or matches[0] in matched:
            raise ValueError(f"Unknown, ambiguous or duplicate artifact image ID: {value}")
        matched[matches[0]] = position
    if set(matched) != set(expected):
        raise ValueError("Artifact IDs do not match the requested metadata split")
    return np.array([matched[r["image_path"]] for r in rows])


def validate_probabilities(probabilities, count):
    p = np.asarray(probabilities)
    if p.shape != (count, 2) or p.dtype.kind != "f" or not np.isfinite(p).all():
        raise ValueError("Expected finite floating-point probabilities of shape [N, 2]")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("Probabilities must be in [0, 1] and sum to one")
    return p


def summary(rows, predictions):
    correct = np.asarray(predictions) == np.array([r["y"] for r in rows])
    groups = {}
    for group in range(4):
        mask = np.array([2 * r["y"] + r["a"] == group for r in rows])
        count = int(mask.sum())
        groups[str(group + 1)] = dict(name=GROUP_NAMES[group], count=count,
                                     accuracy=float(correct[mask].mean()) if count else None)
    complete = all(g["count"] for g in groups.values())
    return dict(count=len(rows), errors=int((~correct).sum()), accuracy=float(correct.mean()),
                worst_group_accuracy=min(g["accuracy"] for g in groups.values()) if complete else None,
                per_group=groups)


def write_predictions(output, rows, probabilities, *, run_label, seed, provenance, ids=None,
                      predicted_labels=None, confidence=None):
    """Write CSV plus provenance in a new directory; refuse existing outputs.

    Methods with label-only output can pass ``probabilities=None`` and
    ``predicted_labels``. Unavailable confidences are explicitly ``None``/blank.
    """
    output = Path(output)
    if not run_label.strip() or type(seed) is not int or seed < 0 or not rows:
        raise ValueError("A run label, nonnegative integer seed and nonempty split are required")
    if len({r["image_path"] for r in rows}) != len(rows) or len({r["split"] for r in rows}) != 1:
        raise ValueError("Rows must have unique image paths and one split")
    if probabilities is not None:
        p = validate_probabilities(probabilities, len(rows))
        if ids is not None:
            p = p[align_ids(ids, rows)]
        predictions = p.argmax(1)
        conf = p[np.arange(len(p)), predictions]
    else:
        predictions = np.asarray(predicted_labels)
        if predictions.shape != (len(rows),) or not np.isin(predictions, (0, 1)).all():
            raise ValueError("Invalid predicted labels")
        values = [None] * len(rows) if confidence is None else list(confidence)
        if len(values) != len(rows) or any(v is not None and (not np.isfinite(v) or not 0 <= v <= 1) for v in values):
            raise ValueError("Confidences must be finite probabilities or explicitly None")
        conf = values
    report = dict(format_version=1, prediction_schema_version=SCHEMA_VERSION,
                  run_label=run_label, seed=seed, split=rows[0]["split"],
                  group_convention="1 + 2*y + background", confidence="predicted-class probability; blank if unavailable",
                  metrics=summary(rows, predictions), provenance=provenance,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                  command=sys.argv, exporter_sha256=sha256(__file__))
    # Stage both files outside the destination, then rename the directory atomically.
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Preserve the existing export: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        destination = stage / "predictions.csv"
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row, prediction, certainty in zip(rows, predictions, conf):
                group = 2 * row["y"] + row["a"]
                writer.writerow(dict(image_path=row["image_path"], true_label=LABELS[row["y"]],
                    predicted_label=LABELS[int(prediction)], correct=int(prediction == row["y"]),
                    group=group + 1, group_name=GROUP_NAMES[group], spurious_attribute=BACKGROUNDS[row["a"]],
                    confidence="" if certainty is None else format(float(certainty), ".10g"),
                    run_label=run_label, seed=seed, split=row["split"]))
        validate_prediction_csv(destination)
        report["csv_sha256"] = sha256(destination)
        (stage / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    finally:
        # Only remove our own two staged files, never an existing export.
        if stage.exists():
            for filename in ("predictions.csv", "provenance.json"):
                (stage / filename).unlink(missing_ok=True)
            stage.rmdir()
    return report


def resolve_images(data_dir, rows):
    root = Path(data_dir).resolve()
    candidates = [root, root / "images", root / "waterbird_complete95_forest2water2",
                  root / "waterbird_complete95_forest2water2" / "images"]
    paths = []
    for row in rows:
        options = [base / row["image_path"] for base in candidates]
        found = next((p for p in options if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"Image missing under {root}: {row['image_path']}")
        paths.append(found)
    return paths


def load_image_model(method, checkpoint, *, trusted_pickle=False, config_path=None):
    import torch
    from torch import nn
    from torchvision.models import resnet50

    value = torch.load(checkpoint, map_location="cpu", weights_only=not trusted_pickle)
    config = value.get("config", {}) if isinstance(value, dict) else {}
    if isinstance(value, nn.Module):
        state = value.state_dict()
    elif isinstance(value, dict) and "model" in value:
        state = value["model"]
    else:
        state = value
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint must contain a model state dict or a supported POTER model wrapper")
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if method == "erm":
        from src.models.resnet import ResNetClassifier
        model = ResNetClassifier(weights=None)
    elif method == "poter":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif method == "autobackswap":
        import importlib.util
        from types import SimpleNamespace
        import yaml
        source = Path(__file__).resolve().parents[2] / "baselines/autobackswap/methods/common/model.py"
        spec = importlib.util.spec_from_file_location("_export_autobackswap_model", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if config_path is None:
            raise ValueError("AutoBackSwap requires the saved run config via --config")
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if config.get("dataset_type") != "waterbird":
            raise ValueError("AutoBackSwap export requires a Waterbirds config")
        model = module.Classifier(SimpleNamespace(**config), pre_trained=False)
    else:
        raise ValueError(f"Unknown image method: {method}")
    model.load_state_dict(state, strict=True)
    return model, config


def infer_images(model, paths, *, batch_size=64, device="cpu", image_size=224,
                 mean=(.485, .456, .406), std=(.229, .224, .225)):
    import torch
    from PIL import Image
    from torchvision import transforms
    if batch_size < 1 or image_size < 1:
        raise ValueError("Batch size and image size must be positive")
    transform = transforms.Compose([transforms.Resize((int(image_size * 256 / 224),) * 2),
        transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(mean, std)])
    result = []
    model = model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            logits = model(torch.stack(images).to(device))
            if logits.shape != (len(images), 2) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite two-class logits")
            result.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(result)


def export_checkpoint(method, checkpoint, metadata, data_dir, output, *, run_label, seed,
                      split="test", device="cpu", batch_size=64, config_path=None, trusted_pickle=False):
    checkpoint = Path(checkpoint)
    if Path(output).exists():
        raise FileExistsError(output)
    rows = read_metadata(metadata, split)
    paths = resolve_images(data_dir, rows)
    model, config = load_image_model(method, checkpoint, trusted_pickle=trusted_pickle, config_path=config_path)
    if method == "poter" and config.get("metadata_sha256") and config["metadata_sha256"] != sha256(metadata):
        raise ValueError("POTER checkpoint metadata hash differs; use its original metadata.csv")
    if "seed" in config and int(config["seed"]) != seed:
        raise ValueError("Requested seed disagrees with checkpoint/config")
    if method == "autobackswap" and int(config.get("random_seed", seed)) != seed:
        raise ValueError("Requested seed disagrees with AutoBackSwap config")
    kwargs = {} if method != "autobackswap" else {k: config[k] for k in ("image_size", "mean", "std")}
    p = infer_images(model, paths, batch_size=batch_size, device=device, **kwargs)
    import torch
    import torchvision
    provenance = dict(method=method, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
        metadata=str(Path(metadata).resolve()), metadata_sha256=sha256(metadata),
        preprocessing=dict(resize="square 256/224 * image_size", crop="center", image_size=kwargs.get("image_size", 224),
                           mean=kwargs.get("mean", [.485, .456, .406]), std=kwargs.get("std", [.229, .224, .225])),
        torch=str(torch.__version__), torchvision=str(torchvision.__version__), device=device,
        selection="User-supplied checkpoint; no selection or fitting during export")
    if config_path:
        provenance.update(config=str(Path(config_path).resolve()), config_sha256=sha256(config_path))
    return write_predictions(output, rows, p, run_label=run_label, seed=seed, provenance=provenance)


AUTOBACKSWAP_NAME_MAP = {
    "ours_minority_false_seed0": "autobackswap_seed0",
    "ours_minority_false_seed1": "autobackswap_seed1",
    "ours_minority_false_seed2": "autobackswap_seed2",
    "ours_minority_true_seed0": "autobackswap_w_minority_seed0",
    "erm_minority_false_seed0": "erm_baseline_seed0",
    "erm_minority_false_seed1": "erm_baseline_seed1",
    "erm_minority_false_seed2": "erm_baseline_seed2",
    "erm_minority_true_seed0": "erm_w_minority_seed0",
    "erm_minority_true_seed1": "erm_w_minority_seed1",
    "erm_minority_true_seed2": "erm_w_minority_seed2",
}


def recover_autobackswap(repo, metadata, output_root, ref="origin/method/autobackswap"):
    """Recover existing test exports from a local Git ref; no fetch or checkout."""
    repo = Path(repo)
    commit = subprocess.check_output(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo, text=True).strip()
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=repo, text=True).splitlines()
    files = [p for p in files if p.startswith("results/autobackswap/predictions/") and p.endswith("/predictions.csv")]
    if not files:
        raise ValueError(f"No saved prediction CSVs in the locally cached ref {ref}")
    rows = read_metadata(metadata, "test")
    reports = []
    for path in files:
        blob = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo)
        records = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
        order = align_ids([r["image_path"] for r in records], rows)
        records = [records[i] for i in order]
        labels, seeds = {r["run_label"] for r in records}, {r["seed"] for r in records}
        if len(labels) != 1 or len(seeds) != 1:
            raise ValueError(f"Expected one run and seed in {path}")
        label, seed = next(iter(labels)), int(next(iter(seeds)))
        raw_dir = PurePosixPath(path).parent.name
        exp_id = AUTOBACKSWAP_NAME_MAP.get(raw_dir, raw_dir)
        run_label = AUTOBACKSWAP_NAME_MAP.get(label, label)
        predictions, confidences = [], []
        for row, r in zip(rows, records):
            pred = LABELS.index(r["predicted_label"])
            if (r["true_label"] != LABELS[row["y"]] or r["spurious_attribute"] != BACKGROUNDS[row["a"]]
                or int(r["group"]) != 1 + 2 * row["y"] + row["a"] or int(r["correct"]) != (pred == row["y"])):
                raise ValueError(f"Saved predictions disagree with metadata in {path}")
            predictions.append(pred)
            confidences.append(float(r["confidence"]))
        destination = Path(output_root) / exp_id / "predictions" / "test"
        reports.append(write_predictions(destination, rows, None, run_label=run_label, seed=seed,
            predicted_labels=predictions, confidence=confidences,
            provenance=dict(method="erm-autobackswap" if label.startswith("erm_") else "autobackswap",
                action="Recovered existing predictions; no inference or retraining", source_ref=ref,
                source_commit=commit, source_path=path, source_csv_sha256=hashlib.sha256(blob).hexdigest(),
                metadata_sha256=sha256(metadata), checkpoint="Not present in this checkout; see original run records")))
    return reports
