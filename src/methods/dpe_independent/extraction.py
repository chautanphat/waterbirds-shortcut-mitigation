"""Optional missing-feature recovery from an exact saved checkpoint; never train."""
from pathlib import Path

import numpy as np

from src.evaluation.prediction_export import sha256
from .storage import atomic_json, atomic_npz, frozen_json


def extract_saved(checkpoint, expected_sha256, data, output, device="cpu"):
    checkpoint, output = Path(checkpoint), Path(output)
    if not checkpoint.is_file() or sha256(checkpoint) != expected_sha256:
        raise ValueError("Exact saved backbone missing or hash-mismatched; replacement training is forbidden")
    # Model construction happens only after exact checkpoint acceptance.
    from src.datasets.waterbirds import Waterbirds
    from src.datasets.loaders import evaluation_loader
    from src.evaluation.features import extract_features
    from src.models.resnet import load_resnet_checkpoint
    datasets = {split: Waterbirds(data, split) for split in ("val", "test")}
    if [len(datasets[s]) for s in ("val", "test")] != [1199, 5794]:
        raise ValueError("Official validation/test coverage required")
    identity = dict(checkpoint_sha256=expected_sha256, metadata_sha256=sha256(datasets["val"].metadata_path),
                    preprocessing="RGB; Resize((256,256)); CenterCrop(224); ImageNet normalization; raw 2048 features",
                    device=device)
    frozen_json(output/"extraction_inputs.json", identity)
    model = None
    for split, dataset in datasets.items():
        path = output/(split+".npz")
        complete = output/(split+".complete.json")
        if path.exists() and complete.exists():
            from .storage import read_json
            if sha256(path) != read_json(complete)["sha256"]:
                raise ValueError("Recovered features changed")
            continue
        if model is None:
            model = load_resnet_checkpoint(checkpoint, device=device)
        features = extract_features(model, evaluation_loader(dataset, 256, 0), device)["features"]
        atomic_npz(path, features=features, ids=np.asarray(dataset.filenames), y=dataset.y,
                   g=np.array([f"{y}{a}" for y, a in zip(dataset.y, dataset.a)]))
        atomic_json(complete, dict(sha256=sha256(path), identity=identity))
    return identity
