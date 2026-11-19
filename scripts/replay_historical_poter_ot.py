"""Replay POTER seed 1 sampling propensities from the hash-verified Kaggle cache."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from audit_ot_vulnerability_complementarity import OUTPUT as PRIOR, ROOT, RUN, inspect_code
from src.counterfactual.common import read_csv, sha256, write_csv, write_json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
POTER = ROOT / "baselines/poter"
sys.path.insert(0, str(POTER))
import torch
import torchvision
from data.cub_dataset import CUBDataset
from extract_features import _parse_ref_per_group, make_reference_subset
from ot_reweight import compute_ot_potentials, compute_sample_weights
from utils import set_seed

EXPECTED_CACHE = "8a7636923bffaddce16f843b38afe0a97743f19a151339ddb96e200d2c4d2258"
RECOVERY = ROOT / "outputs/kaggle_cache_recovery"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _reconstruct(*args):
    # The Kaggle NumPy 2 cache stores this symbol under numpy._core.
    return np.core.multiarray._reconstruct(*args)


_reconstruct.__module__ = "numpy._core.multiarray"


def load_cache(path):
    require(sha256(path) == EXPECTED_CACHE, "Cache SHA-256 differs from the historical fingerprint")
    allowed = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        cache = torch.load(path, map_location="cpu", weights_only=True)
    shapes = {"X_train": (4795, 2048), "X_val": (599, 2048), "y_train": (4795,), "y_val": (599,)}
    require(set(cache) == {*shapes, "dataset", "ref_per_group"}, "Unexpected feature cache keys")
    for key, shape in shapes.items():
        array = cache[key]
        dtype = np.float32 if key.startswith("X") else np.int64
        require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype == dtype
                and np.isfinite(array).all(), f"Unexpected feature cache array: {key}")
    return cache


def check_order(cache, manifest, data_root):
    target, places = "waterbird_complete95", ["forest2water2"]
    meta = data_root / "data" / "_".join([target, *places]) / "metadata.csv"
    require(sha256(meta) == manifest["source_dataset"]["metadata_sha256"], "Metadata SHA-256 differs")
    dataset = CUBDataset(data_root, target, places, model_type="resnet50")
    splits = dataset.get_splits(["train", "val"])
    train, val = splits["train"].indices, splits["val"].indices
    ids = dataset.metadata_df["img_id"].to_numpy()
    mapping = PRIOR / "expected_train_order.csv"
    require(sha256(mapping) == manifest["train_id_mapping"]["sha256"], "Train mapping SHA-256 differs")
    order = read_csv(mapping)
    train_ids = ids[train]
    require(len(set(train_ids)) == len(train_ids) == 4795, "Training IDs are incomplete or duplicated")
    require(np.array_equal(train_ids, [int(row["sample_id"]) for row in order]), "Canonical train order differs")
    require(np.array_equal(dataset.y_array[train], cache["y_train"]), "Training label order differs")
    groups = dataset.group_array[train]
    require(np.array_equal(groups, [int(row["original_group"]) for row in order]), "Train group order differs")

    quotas = _parse_ref_per_group("0:0,1:466,2:133,3:0")
    require(cache["dataset"] == "CUB" and cache["ref_per_group"] == quotas, "Reference cache recipe differs")
    val_groups = SimpleNamespace(_group_array=torch.as_tensor(dataset.group_array[val]))
    subset, ref_groups = make_reference_subset(val_groups, quotas, seed=0)
    ref = val[subset.indices]
    ref_ids = ids[ref].tolist()
    require(len(set(ref_ids)) == len(ref_ids) == 599, "Reference IDs are incomplete or duplicated")
    require(set(ref_ids) == set(manifest["dataset_assets"]["reference_ids_derived_from_manifest"]),
            "Reference membership differs")
    require(not set(train_ids).intersection(ref_ids), "Training and reference IDs overlap")
    require(np.array_equal(dataset.y_array[ref], cache["y_val"]), "Cached reference label order differs")
    require(np.array_equal(np.bincount(ref_groups, minlength=4), list(quotas.values())), "Reference quotas differ")
    return train_ids, groups, ref_ids


def replay(cache, config, device):
    raw = np.empty(len(cache["y_train"]), dtype=np.float64)
    captured = 0

    def capture_raw(frame, event, result):
        # Read the upstream function's pre-normalization duals without changing its code.
        nonlocal captured
        if event == "return" and frame.f_code is compute_ot_potentials.__code__ and result is not None:
            for indices, potential in frame.f_locals["raw_by_class"].values():
                raw[indices] = potential
                captured += len(indices)

    prior_profiler = sys.getprofile()
    try:
        sys.setprofile(capture_raw)
        normalized = compute_ot_potentials(cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"],
                                           reg_scale=config["reg_scale"], device=device)
    finally:
        sys.setprofile(prior_profiler)
    require(captured == len(raw), "The original solver did not return raw duals for all rows")
    weights = torch.from_numpy(compute_sample_weights(normalized, cache["y_train"],
                               f_power=config["f_power"], clip_cap=config["clip_cap"])).float()
    # WeightedRandomSampler converts the runner's float32 weights to double before drawing.
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    probability = sampler.weights.numpy() / sampler.weights.sum().item()
    require(all(np.isfinite(a).all() for a in (raw, normalized, weights.numpy(), probability)),
            "OT replay produced nonfinite values")
    require(np.all(probability > 0) and np.isclose(probability.sum(), 1), "Invalid sampling probabilities")
    require(np.isclose(float(weights.mean()), 1, atol=1e-6), "Unexpected final mean sampler weight")
    return raw, normalized, weights.numpy(), probability


def run(cache_path, data_root, output):
    require(not output.exists(), f"Choose a new output directory to preserve existing results: {output}")
    manifest = json.loads((PRIOR / "weights_manifest.json").read_text(encoding="utf-8"))
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(sha256(config_path) == manifest["run_config_sha256"], "Historical run config SHA-256 differs")
    require(sha256(RUN / "best_validation.pt") == manifest["checkpoint"]["sha256"], "Checkpoint SHA-256 differs")
    require(config["seed"] == 1 and config["cache_sha256"] == EXPECTED_CACHE, "Unexpected training run")
    source_checks = inspect_code(config["source_sha256"])
    for name in ("ot_reweight.py", "data/cub_dataset.py", "data/confounder_dataset.py", "data/dro_dataset.py",
                 "scripts/kaggle_waterbirds.py"):
        require(source_checks[name]["same_after_newline_normalization"], f"Historical POTER source differs: {name}")
    cache = load_cache(cache_path)
    ids, groups, ref_ids = check_order(cache, manifest, data_root)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(config["seed"], strict=False, warn_only=False)
    raw, normalized, weights, probability = replay(cache, config, device)
    parity = {"checked": False}
    if device != "cpu":
        _, _, cpu_weights, cpu_probability = replay(cache, config, "cpu")
        parity = dict(checked=True, maximum_absolute_weight_difference=float(np.max(np.abs(weights - cpu_weights))),
                      maximum_absolute_probability_difference=float(np.max(np.abs(probability - cpu_probability))))
        require(np.allclose(probability, cpu_probability, rtol=0.002, atol=1e-9), "CUDA and CPU replay diverge")

    class_mass = np.bincount(cache["y_train"], weights=probability, minlength=2)
    group_mass = np.bincount(groups, weights=probability, minlength=4)
    empirical = manifest["original_sampler"]["sampler_empirical"]
    draws = empirical["draws"]
    require(draws == config["epochs"] * len(ids), "Historical draw count differs")
    group_count = np.asarray(empirical["group_counts"])
    expected_count = draws * group_mass
    z = (group_count - expected_count) / np.sqrt(expected_count * (1 - group_mass))
    require(np.max(np.abs(z)) < 5, "Replayed group mass is inconsistent with historical draw totals")

    rows = [dict(train_image_id=int(sample_id), official_y=int(label), official_group=int(group),
                 source_split="train", reference_membership=False, raw_OT_potential=float(raw[i]),
                 final_sampler_weight=float(weights[i]), normalized_sampling_probability=float(probability[i]),
                 normalized_OT_potential=float(normalized[i]))
            for i, (sample_id, label, group) in enumerate(zip(ids, cache["y_train"], groups))]
    new_candidate = dict(source="user-supplied Kaggle cache", kaggle_version="", path=str(cache_path),
                         size_bytes=cache_path.stat().st_size, sha256=EXPECTED_CACHE, status="VERIFIED_HISTORICAL_CACHE")
    candidates = [*read_csv(RECOVERY / "candidate_files.csv"), new_candidate]
    audit = dict(decision="VERIFIED_AND_REPLAYED", checkpoint=manifest["checkpoint"], run_config_sha256=sha256(config_path),
                 cache=new_candidate, config=config, source_code_checks=source_checks,
                 split_metadata=manifest["source_dataset"], train_order=manifest["train_id_mapping"],
                 verified_train_rows=len(rows), reference_ids_in_replayed_order=ref_ids, reference_seed=0,
                 train_reference_labels_exact=True, train_reference_ids_disjoint=True,
                 row_order_evidence="Canonical metadata, original ordered CUB splits and seeded reference selection, with exact label alignment. The cache has no embedded IDs.",
                 restricted_load=True, all_values_finite=True, all_weights_positive=True,
                 local_runtime=dict(device=device, gpu=torch.cuda.get_device_name(0) if device != "cpu" else None,
                                    torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                                    torchvision=torchvision.__version__, allow_tf32=False),
                 cpu_cuda_comparison=parity, historical_bitwise_weight_identity_verified=False,
                 original_sampler=manifest["original_sampler"], class_probability_mass=class_mass.tolist(),
                 group_probability_mass=group_mass.tolist(), historical_group_draws=group_count.tolist(),
                 replay_expected_group_draws=expected_count.tolist(), group_draw_z=z.tolist(),
                 weight_summary=dict(minimum=float(weights.min()), mean=float(weights.mean()), maximum=float(weights.max())),
                 train_dev_vulnerability="BLOCKED_TRAIN_DEV_VULNERABILITY", training_started=False,
                 model_inference_started=False, embeddings_recomputed=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        stage = Path(staging) / output.name
        stage.mkdir()
        weights_path = stage / "historical_ot_weights_seed1.csv"
        write_csv(weights_path, rows)
        write_csv(stage / "candidate_files.csv", candidates)
        audit["export_sha256"] = sha256(weights_path)
        write_json(stage / "replay_audit.json", audit)
        stage.rename(output)
    print(json.dumps(dict(decision=audit["decision"], rows=len(rows), class_mass=class_mass.tolist(),
                          group_mass=group_mass.tolist(), historical_draw_z=z.tolist(), parity=parity), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--cub-root", type=Path, default=Path("D:/POTER_data/cub"))
    parser.add_argument("--output", type=Path, default=RECOVERY / "verified_replay")
    args = parser.parse_args()
    run(args.cache.resolve(), args.cub_root, args.output.resolve())
