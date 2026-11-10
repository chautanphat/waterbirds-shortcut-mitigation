"""Export the original DPE's equally averaged prototype probabilities.

No head reweighting, refitting, or checkpoint selection is performed here.
"""

from functools import wraps
import json
from pathlib import Path

import numpy as np

from .prediction_export import (align_ids, read_metadata, sha256, validate_probabilities,
                                write_predictions)


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mean_probabilities(value, count):
    p = as_numpy(value)
    if p.ndim == 3:
        if not len(p):
            raise ValueError("Empty DPE ensemble")
        validate_probabilities(p.reshape(-1, 2), len(p) * count)
        p = p.mean(0)
    return validate_probabilities(p, count)


def save_dpe_probabilities(path, probabilities, ids, *, metadata, seed, source, split="test"):
    """Call while the trained DPE object is still in memory.

    Example: save_dpe_probabilities(..., dpe.predict_proba('test'),
        dpe.datasets['test'].x, metadata=..., seed=0, source='original DPE run ...')
    """
    rows = read_metadata(metadata, split)
    order = align_ids(ids, rows)
    p = mean_probabilities(probabilities, len(rows))[order]
    if not source or type(seed) is not int or seed < 0:
        raise ValueError("Record the original run source and nonnegative integer seed")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, probabilities=p, ids=np.array([r["image_path"] for r in rows]),
            metadata_sha256=sha256(metadata), seed=seed, split=split, source=source,
            method="original-dpe", aggregation="equal mean of per-head probabilities")


def export_dpe_probabilities(artifact, metadata, output, *, run_label, seed, split="test"):
    rows = read_metadata(metadata, split)
    with np.load(artifact, allow_pickle=False) as data:
        if not {"ids", "probabilities"}.issubset(data.files):
            raise ValueError("DPE NPZ needs ids and probabilities; metrics cannot reconstruct predictions")
        for key, expected in (("metadata_sha256", sha256(metadata)), ("seed", str(seed)),
                              ("split", split), ("method", "original-dpe")):
            if key in data and str(data[key]) != expected:
                raise ValueError(f"DPE artifact {key} disagrees with the requested export")
        p = mean_probabilities(data["probabilities"], len(rows))
        ids = data["ids"]
        source = str(data["source"]) if "source" in data else "User-supplied saved probabilities"
    return write_predictions(output, rows, p, ids=ids, run_label=run_label, seed=seed,
        provenance=dict(method="original-dpe", artifact=str(Path(artifact).resolve()),
            artifact_sha256=sha256(artifact), metadata_sha256=sha256(metadata), source=source,
            aggregation="equal mean of per-head probabilities; no temperature/entropic rescaling",
            exporter_sha256=sha256(__file__)))


def save_dpe_heads(path, prototype_ensemble, *, metadata, backbone, seed, source,
                   normalize_embeddings=True):
    """Persist the original [(prototypes [2,1,D], scale [1]), ...] head list."""
    import torch
    if not prototype_ensemble or type(seed) is not int or seed < 0 or not source:
        raise ValueError("A trained ensemble, source description and seed are required")
    prototypes, scales = [], []
    for item in prototype_ensemble:
        p, s = item
        p, s = torch.as_tensor(p).detach().cpu(), torch.as_tensor(s).detach().cpu()
        if p.ndim == 3 and p.shape[1] == 1:
            p = p[:, 0, :]
        if p.ndim != 2 or p.shape[0] != 2 or p.shape[1] < 2 or s.numel() != 1:
            raise ValueError("Unsupported original DPE head shape")
        if not p.is_floating_point() or not s.is_floating_point() or not torch.isfinite(p).all() or not torch.isfinite(s).all():
            raise ValueError("Invalid DPE head parameters")
        prototypes.append(p)
        scales.append(s.reshape(()))
    record = dict(version=1, method="original-dpe", classes=[0, 1], seed=seed,
        prototypes=torch.stack(prototypes), distance_scales=torch.stack(scales), temperature=1.,
        normalize_embeddings=bool(normalize_embeddings), metadata_sha256=sha256(metadata),
        backbone_sha256=sha256(backbone), source=source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(record, stream)


def capture_notebook_heads(function, destination):
    """Wrap notebook train_prototypes without changing its result or RNG state.

    The notebook's train_ensemble discards its head list. Install this before the
    training call to collect each returned validation-selected head.
    """
    @wraps(function)
    def capture(*args, **kwargs):
        result = function(*args, **kwargs)
        if result is None:
            raise ValueError("DPE head training returned no validation-selected head")
        destination.append(tuple(t.detach().cpu().clone() for t in result))
        return result
    return capture


def predict_heads(record, raw, *, batch_size=256, device="cpu"):
    import torch
    from torch.nn import functional as F
    if record.get("version") != 1 or record.get("method") != "original-dpe" or record.get("classes") != [0, 1] or record.get("temperature") != 1.:
        raise ValueError("Unsupported original DPE head artifact")
    p, s = record["prototypes"], record["distance_scales"]
    if not isinstance(p, torch.Tensor) or not isinstance(s, torch.Tensor):
        raise ValueError("DPE heads must be saved tensors")
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[1] != 2 or s.shape != (len(p),):
        raise ValueError("Misaligned DPE head dimensions")
    if not p.is_floating_point() or not s.is_floating_point() or not torch.isfinite(p).all() or not torch.isfinite(s).all():
        raise ValueError("Invalid DPE head parameters")
    x = np.asarray(raw)
    if x.ndim != 2 or not len(x) or x.shape[1] != p.shape[2] or x.dtype.kind != "f" or not np.isfinite(x).all() or batch_size < 1:
        raise ValueError("Invalid or misaligned raw backbone features")
    if type(record.get("normalize_embeddings")) is not bool:
        raise ValueError("Missing DPE embedding-normalization setting")
    if record["normalize_embeddings"]:
        sd = x.std(axis=1, keepdims=True)
        if (sd == 0).any() or not np.isfinite(sd).all():
            raise ValueError("DPE cannot standardize a zero-variance/nonfinite feature row")
        x = (x - x.mean(axis=1, keepdims=True)) / sd
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite standardized DPE features")
    batches = []
    with torch.inference_mode():
        for batch in x.split(batch_size):
            normalized = F.normalize(batch.to(device), dim=1)
            heads = []
            for prototypes, scale in zip(p, s):
                # Upstream IsoMax+ inference uses temperature 1, not training scale 30.
                logits = -scale.to(device).abs() * torch.cdist(normalized,
                    F.normalize(prototypes.to(device), dim=1), p=2.,
                    compute_mode="donot_use_mm_for_euclid_dist")
                heads.append(logits.softmax(1))
            batches.append(torch.stack(heads).mean(0).cpu().numpy())
    return validate_probabilities(np.concatenate(batches), len(x))


def export_dpe_heads(heads, features, metadata, output, *, run_label, seed, split="test",
                     device="cpu", batch_size=256):
    """Use identified raw-feature NPZs, never anonymous legacy feature-row order."""
    import torch
    rows = read_metadata(metadata, split)
    record = torch.load(heads, map_location="cpu", weights_only=True)
    if record.get("metadata_sha256") != sha256(metadata) or record.get("seed") != seed:
        raise ValueError("DPE head metadata/seed disagrees with this export")
    with np.load(features, allow_pickle=False) as data:
        required = {"ids", "features", "backbone_sha256"}
        if not required.issubset(data.files):
            raise ValueError("Raw feature NPZ requires ids, features and backbone_sha256")
        if str(data["backbone_sha256"]) != record.get("backbone_sha256"):
            raise ValueError("Features were extracted from a different backbone")
        order = align_ids(data["ids"], rows)
        x = data["features"][order]
    probabilities = predict_heads(record, x, batch_size=batch_size, device=device)
    return write_predictions(output, rows, probabilities, run_label=run_label, seed=seed,
        provenance=dict(method="original-dpe", heads=str(Path(heads).resolve()), heads_sha256=sha256(heads),
            features=str(Path(features).resolve()), features_sha256=sha256(features),
            metadata_sha256=sha256(metadata), backbone_sha256=record["backbone_sha256"],
            normalize_embeddings=record["normalize_embeddings"], aggregation="equal mean of head softmax at temperature 1",
            source=record["source"], exporter_sha256=sha256(__file__)))


def package_notebook_features(output, features, manifest, metadata, backbone, *, split="test"):
    """Validate the reproduction notebook's hashes/order before packaging NPY."""
    rows = read_metadata(metadata, split)
    record = json.loads(Path(manifest).read_text(encoding="utf-8"))
    expected_format = "raw;Resize((256,256));CenterCrop(224);ImageNet normalization"
    if (record.get("checkpoint_sha256") != sha256(backbone) or record.get("metadata_sha256") != sha256(metadata)
        or record.get("features_sha256") != sha256(features) or record.get("split") != split
        or record.get("format") != expected_format):
        raise ValueError("Feature manifest does not verify this raw cache/backbone/metadata/split")
    ids = record.get("image_paths", [])
    order = align_ids(ids, rows)
    x = np.load(features, allow_pickle=False)
    if x.ndim != 2 or len(x) != len(rows) or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError("Invalid raw feature cache")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, ids=np.array([r["image_path"] for r in rows]), features=x[order],
                            backbone_sha256=sha256(backbone))
