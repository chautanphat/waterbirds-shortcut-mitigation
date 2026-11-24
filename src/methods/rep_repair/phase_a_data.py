"""Validation-only data loading, label cleaning, and CV utilities for Direction 2 Phase A.

Strict zero-test-leakage policy: every file access rejects test paths and verifies
that all loaded Waterbirds records come exclusively from the validation split (split == 1).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.methods.dpe_head_reweighting import ensemble
from src.methods.dpe_head_reweighting.aggregation_source import load_cache, partition
from src.methods.dpe_head_reweighting.reweighting import GROUPS, probabilities

TERN_SPECIES = (
    "021.Eastern_Towhee",
    "088.Western_Meadowlark",
    "102.Western_Wood_Pewee",
)
VAL_SPLIT_ID = 1
VAL_EXPECTED_ROWS = 1199
HELD_OUT_NAME = "combiner_training"
FIT_NAME = "head_training"
HEADS = 15
DIMENSION = 2048
GROUP_TO_INT = {g: i for i, g in enumerate(GROUPS)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_val_file(root: Path, relative_path: str, allowed_names: tuple[str, ...]) -> Path:
    """Resolve a validation-only input under root; reject traversal and any test filename."""
    root = Path(root).resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes configured root: {path}")
    lowered = path.name.lower()
    if "test" in lowered or path.name not in allowed_names:
        raise ValueError(f"Blocked non-allowlisted or test file: {path}")
    return path


def resolve_asset_root(config: dict, repo_root: Path) -> Path:
    raw = config.get("asset_root")
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.is_dir():
            return candidate.resolve()
    return repo_root.resolve()


def load_val_metadata(metadata_path: Path) -> list[dict]:
    """Load official Waterbirds validation rows (split == 1) in canonical CSV order."""
    path = safe_val_file(metadata_path.parent, metadata_path.name, ("metadata.csv", "metadata_waterbirds.csv"))
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["split"]) != VAL_SPLIT_ID:
                continue
            y = int(row["y"])
            a = int(row["place"])
            rows.append(
                {
                    "img_id": int(row["img_id"]),
                    "id": row["img_filename"],
                    "y": y,
                    "a": a,
                    "group": f"{y}{a}",
                    "split": VAL_SPLIT_ID,
                }
            )
    if len(rows) != VAL_EXPECTED_ROWS:
        raise ValueError(f"Expected {VAL_EXPECTED_ROWS} validation rows, found {len(rows)}")
    return rows


def tern_mask(ids, species: tuple[str, ...] = TERN_SPECIES) -> np.ndarray:
    species_set = set(species)
    return np.array([str(item).split("/")[0] in species_set for item in ids], dtype=bool)


def apply_label_condition(rows: list[dict], condition: str, species: tuple[str, ...] = TERN_SPECIES):
    """Return (keep_indices, cleaned_rows) for condition in {'official', 'relabel', 'drop'}."""
    if condition not in ("official", "relabel", "drop"):
        raise ValueError(f"Unsupported label condition: {condition}")
    mask = tern_mask([r["id"] for r in rows], species=species)
    if condition == "official":
        return np.arange(len(rows), dtype=int), [dict(r) for r in rows]
    if condition == "drop":
        keep = np.flatnonzero(~mask)
        return keep, [dict(rows[i]) for i in keep]
    out = []
    for r, is_tern in zip(rows, mask):
        item = dict(r)
        if is_tern:
            item["y"] = 0
            item["group"] = "0" + str(item["group"])[1]
        out.append(item)
    return np.arange(len(rows), dtype=int), out


def stratified_group_folds(rows: list[dict], n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Deterministic group-stratified fold assignment ordered by canonical image id."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    rng = np.random.RandomState(seed)
    folds = np.full(len(rows), -1, dtype=int)
    offset = 0
    for group in GROUPS:
        members = sorted([i for i, r in enumerate(rows) if r["group"] == group], key=lambda i: rows[i]["id"])
        order = rng.permutation(members)
        folds[order] = (np.arange(len(order)) + offset) % n_splits
        offset = (offset + len(order)) % n_splits
    if (folds < 0).any():
        raise ValueError("Unassigned rows in stratified_group_folds")
    return folds


def group_metrics(y_true, y_pred, groups) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups).astype(str)
    per_group = {}
    counts = {}
    for g in GROUPS:
        mask = groups == g
        counts[g] = int(mask.sum())
        per_group[g] = float((y_pred[mask] == y_true[mask]).mean()) * 100.0 if mask.any() else float("nan")
    return {
        "wga": float(min(per_group.values())),
        "overall": float((y_pred == y_true).mean()) * 100.0,
        "groups": per_group,
        "counts": counts,
    }


def balanced_group_weights(groups) -> np.ndarray:
    """Inverse-group-frequency weights normalized to mean 1.0."""
    groups = np.asarray(groups).astype(str)
    weights = np.zeros(len(groups), dtype=np.float64)
    present = [g for g in GROUPS if np.any(groups == g)]
    for g in present:
        mask = groups == g
        weights[mask] = len(groups) / (len(present) * float(mask.sum()))
    return weights


def load_cohort_source(
    kind: str,
    seed: int,
    cohort_root: Path,
    source_dir: Path,
    feature_path: Path,
    val_rows: list[dict],
) -> dict:
    """Load validation-only 1,199 raw features and 479 held-out cached DPE head predictions."""
    feature_name = "val_raw.npz" if kind == "standard_erm" else "val.npz"
    feature_file = safe_val_file(cohort_root, str(feature_path.relative_to(cohort_root)), (feature_name,))

    if kind == "standard_erm":
        model_file = safe_val_file(source_dir, "checkpoints/ensemble.pt", ("ensemble.pt",))
        split_file = safe_val_file(source_dir, "experiment/manifest.json", ("manifest.json",))
        record_file = safe_val_file(source_dir, "experiment/heads.json", ("heads.json",))
        cache_file = safe_val_file(source_dir, f"experiment/predictions/{HELD_OUT_NAME}.npz", (f"{HELD_OUT_NAME}.npz",))
    elif kind == "erm_star":
        model_file = safe_val_file(source_dir, "neural_split/ensemble.pt", ("ensemble.pt",))
        split_file = safe_val_file(source_dir, "neural_split/partition.json", ("partition.json",))
        record_file = None
        cache_file = safe_val_file(source_dir, f"neural_split/{HELD_OUT_NAME}.npz", (f"{HELD_OUT_NAME}.npz",))
    else:
        raise ValueError(f"Unsupported cohort: {kind}")

    state = ensemble.load(model_file, dim=DIMENSION, count=HEADS)
    manifest = json.loads(split_file.read_text(encoding="utf-8"))
    model_hash = file_sha256(model_file)

    if kind == "standard_erm":
        record_data = json.loads(record_file.read_text(encoding="utf-8"))
        split_hash = file_sha256(split_file)
        if (
            record_data["lock"] != {"ensemble": model_hash, "manifest": split_hash}
            or state["provenance"]["manifest"] != split_hash
            or manifest["provenance"]["seed"] != seed
            or manifest["parts"] != partition(manifest["validation"])
            or manifest["val_features_hash"] != file_sha256(feature_file)
        ):
            raise ValueError(f"Standard ERM provenance mismatch: {source_dir}")
        parts = manifest["parts"]
        held_scores, held_y, held_g, held_ids = load_cache(
            source_dir, HELD_OUT_NAME, manifest, state["head_ids"], model_hash
        )
        inputs = [model_file, split_file, record_file, cache_file, feature_file]
    else:
        parts = manifest
        if state["provenance"]["seed"] != seed:
            raise ValueError(f"ERM* seed mismatch: {source_dir}")
        with np.load(cache_file, allow_pickle=False) as saved:
            held_scores = probabilities(saved["probabilities"])
            held_y, held_g, held_ids = saved["labels"], saved["groups"], saved["ids"]
        inputs = [model_file, split_file, cache_file, feature_file]

    held_rows = parts[HELD_OUT_NAME]
    if (
        len(parts[FIT_NAME]) != 720
        or len(held_rows) != 479
        or held_scores.shape != (HEADS, 479, 2)
        or held_ids.tolist() != [r["id"] for r in held_rows]
        or held_y.tolist() != [r["y"] for r in held_rows]
        or held_g.tolist() != [r["group"] for r in held_rows]
    ):
        raise ValueError(f"Invalid held-out validation cache: {source_dir}")

    # Load all 1,199 raw features and align strictly to canonical val_rows order.
    with np.load(feature_file, allow_pickle=False) as data:
        feat_ids = data["ids"].tolist()
        raw_feats = data["features"].astype(np.float32)
        if len(feat_ids) != len(val_rows) or raw_feats.shape != (len(val_rows), DIMENSION):
            raise ValueError(f"Unexpected feature array shape in {feature_file}: {raw_feats.shape}")
        if set(feat_ids) != {r["id"] for r in val_rows}:
            raise ValueError(f"Feature IDs do not match official validation metadata: {feature_file}")
        if kind == "erm_star":
            by_id = {r["id"]: r for r in val_rows}
            if (
                data["y"].tolist() != [by_id[k]["y"] for k in feat_ids]
                or data["g"].tolist() != [by_id[k]["group"] for k in feat_ids]
            ):
                raise ValueError(f"ERM* feature metadata disagrees with official validation CSV: {feature_file}")
    feat_index = {k: i for i, k in enumerate(feat_ids)}
    aligned_feats = raw_feats[[feat_index[r["id"]] for r in val_rows]]
    val_index = {r["id"]: i for i, r in enumerate(val_rows)}
    held_val_indices = np.array([val_index[k] for k in held_ids.tolist()], dtype=int)

    return {
        "cohort": kind,
        "seed": seed,
        "source_id": f"{kind}_s{seed}",
        "features": aligned_feats,
        "held_val_indices": held_val_indices,
        "held_scores": held_scores,
        "inputs": inputs,
    }


def load_all_sources(
    config: dict, repo_root: Path, asset_root_override: Path | None = None
) -> tuple[Path, list[dict], list[dict]]:
    asset_root = asset_root_override.resolve() if asset_root_override is not None else resolve_asset_root(config, repo_root)
    meta_candidate = asset_root / config["metadata_path"]
    if meta_candidate.is_file():
        metadata_path = safe_val_file(asset_root, config["metadata_path"], ("metadata.csv", "metadata_waterbirds.csv"))
    else:
        fallback_rel = "baselines/autobackswap/data/waterbird_complete95_forest2water2/metadata.csv"
        metadata_path = safe_val_file(repo_root, fallback_rel, ("metadata.csv",))
    val_rows = load_val_metadata(metadata_path)
    sources = []
    seen = set()
    for cohort in config["cohorts"]:
        kind = cohort["name"]
        cohort_root = (asset_root / cohort["root"]).resolve(strict=True)
        for fs in cohort["feature_sets"]:
            for seed in fs["seeds"]:
                if (kind, seed) in seen:
                    raise ValueError(f"Duplicate source: {kind} seed {seed}")
                seen.add((kind, seed))
                source_dir = (cohort_root / cohort["source_pattern"].format(seed=seed)).resolve(strict=True)
                feature_path = (cohort_root / fs["pattern"].format(seed=seed)).resolve(strict=True)
                sources.append(load_cohort_source(kind, seed, cohort_root, source_dir, feature_path, val_rows))
    return asset_root, val_rows, sources
