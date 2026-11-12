"""Ordered raw backbone embeddings for reuse, including DPE's .npy cache format."""

import json
from pathlib import Path

import numpy as np
import torch

from src.utils.reproducibility import file_sha256


@torch.inference_mode()
def extract_features(model, loader, device="cpu"):
    model.eval()
    indices, features, targets, attributes = [], [], [], []
    for ids, inputs, labels, attrs in loader:
        features.append(model.extract_features(inputs.to(device)).cpu())
        indices.append(ids)
        targets.append(labels)
        attributes.append(attrs)
    if not features:
        raise ValueError("Cannot extract features from an empty loader")
    ids = torch.cat(indices)
    if not torch.equal(ids, torch.arange(len(loader.dataset))):
        raise ValueError("Feature extraction requires all rows in metadata order without sampling")
    return {"features": torch.cat(features).numpy(), "indices": ids.numpy(),
            "targets": torch.cat(targets).numpy(), "attributes": torch.cat(attributes).numpy()}


def save_features(model, loader, checkpoint, output_dir, device="cpu"):
    dataset = loader.dataset
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"feats_{dataset.split}"
    paths = [stem.with_suffix(suffix) for suffix in (".npy", ".npz", ".json")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Feature cache already exists: {stem}; choose a fresh output directory")
    data = extract_features(model, loader, device)
    # Keep raw embeddings. DPE applies its own per-example normalization on load.
    np.save(paths[0], data["features"])
    np.savez(paths[1], indices=data["indices"], targets=data["targets"],
             attributes=data["attributes"], filenames=np.asarray(dataset.filenames))
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                "metadata_sha256": file_sha256(dataset.metadata_path), "split": dataset.split,
                "shape": list(data["features"].shape), "normalization": "none (raw embeddings)",
                "row_order": "split-local CSV order", "preprocessing": "Resize((256,256)), CenterCrop(224), ImageNet normalization"}
    paths[2].write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
