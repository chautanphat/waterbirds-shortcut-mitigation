"""Spawrious o2o_hard dataset downloader, zero-test-leakage loader, and 8-group validation metrics."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.methods.rep_repair.phase_b_assets import eval_image_transform

SPAWRIOUS_O2O_HARD_URL = "https://www.dropbox.com/s/p1ry121m2gjj158/spawrious224__o2o_hard.tar.gz?dl=1"

CLASSES = ("bulldog", "corgi", "dachshund", "labrador")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Exact Spawrious o2o_hard training vs OOD shifted test assignments from aengusl/spawrious/torch.py.
TRAIN_SPURIOUS_BG = {
    "bulldog": "jungle",
    "dachshund": "mountain",
    "labrador": "snow",
    "corgi": "desert",
}
FILLER_BG = "beach"
TEST_SHIFTED_BG = {
    "bulldog": "mountain",
    "dachshund": "snow",
    "labrador": "desert",
    "corgi": "jungle",
}
# int(0.97 * 3168) = 3072, filler = 96; int(0.87 * 3168) = 2756, filler = 412.
TRAIN_LIMITS = {
    0: {"spurious": 3072, "filler": 96},
    1: {"spurious": 2756, "filler": 412},
}
GROUPS_8 = tuple(f"{y}_{bg}" for y in range(4) for bg in ("spur", "beach"))
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def assert_not_o2o_hard_test(location: str, cls_name: str) -> None:
    """Block any access to Spawrious o2o_hard shifted OOD test (location, class) combinations."""
    if TEST_SHIFTED_BG.get(cls_name) == location:
        raise ValueError(f"Blocked access to Spawrious o2o_hard OOD test combination: ({location}, {cls_name})")


def _has_images(directory: Path) -> bool:
    return directory.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in directory.iterdir())


def _safe_stream_extract_gz(url: str, dest_dir: Path) -> None:
    """Download and stream-extract .tar.gz directly without storing a duplicate archive on disk."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe tar member path: {member.name}")
                tar.extract(member, path=dest_dir)


def ensure_spawrious_o2o_hard(work_dir: Path) -> Path:
    """Return path to extracted spawrious224/ root, stream-downloading o2o_hard if absent."""
    work_dir = Path(work_dir).resolve()
    for cand in (work_dir / "spawrious224", work_dir):
        if _has_images(cand / "0" / "jungle" / "bulldog") and _has_images(cand / "1" / "beach" / "corgi"):
            return cand.resolve()

    print("Streaming Spawrious o2o_hard (~7.03 GB tar.gz) directly into", work_dir, "...", flush=True)
    _safe_stream_extract_gz(SPAWRIOUS_O2O_HARD_URL, work_dir)
    root = work_dir / "spawrious224"
    if not _has_images(root / "0" / "jungle" / "bulldog"):
        raise FileNotFoundError(f"Expected Spawrious o2o_hard images under {root}")
    return root.resolve()


def load_o2o_hard_train_val_rows(
    data_root: Path,
    *,
    val_fraction: float = 0.15,
    split_seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Load official o2o_hard training-domain rows ONLY (zero OOD test folders), stratified by 8 groups."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    data_root = Path(data_root).resolve()
    if (data_root / "spawrious224").is_dir() and not (data_root / "0").is_dir():
        data_root = data_root / "spawrious224"

    all_rows: list[dict] = []
    for domain in (0, 1):
        for cls_name in CLASSES:
            y = CLASS_TO_IDX[cls_name]
            for kind, location in (("spurious", TRAIN_SPURIOUS_BG[cls_name]), ("filler", FILLER_BG)):
                assert_not_o2o_hard_test(location, cls_name)
                folder = data_root / str(domain) / location / cls_name
                if not folder.is_dir():
                    raise FileNotFoundError(f"Missing Spawrious o2o_hard folder: {folder}")
                files = sorted(
                    [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                    key=lambda p: p.name,
                )[: TRAIN_LIMITS[domain][kind]]
                is_beach = int(location == FILLER_BG)
                group = f"{y}_{'beach' if is_beach else 'spur'}"
                for p in files:
                    all_rows.append(
                        {
                            "id": f"{domain}/{location}/{cls_name}/{p.name}",
                            "path": str(p),
                            "y": y,
                            "bg": location,
                            "is_beach": is_beach,
                            "domain": domain,
                            "group": group,
                        }
                    )

    rng = np.random.RandomState(split_seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for group in GROUPS_8:
        members = sorted([i for i, r in enumerate(all_rows) if r["group"] == group], key=lambda i: all_rows[i]["id"])
        if not members:
            continue
        order = rng.permutation(members)
        n_val = max(1, int(round(len(order) * val_fraction))) if len(order) > 1 else 0
        for idx in order[:n_val]:
            row = dict(all_rows[int(idx)])
            row["split"] = "val"
            val_rows.append(row)
        for idx in order[n_val:]:
            row = dict(all_rows[int(idx)])
            row["split"] = "train"
            train_rows.append(row)

    train_rows.sort(key=lambda r: r["id"])
    val_rows.sort(key=lambda r: r["id"])
    return train_rows, val_rows


class SpawriousImageDataset(Dataset):
    """Dataset over Spawrious train or validation rows returning (index, image, y, group)."""

    def __init__(self, rows: list[dict], transform=None):
        self.rows = rows
        self.y = np.array([int(r["y"]) for r in rows], dtype="int64")
        self.groups = np.array([str(r["group"]) for r in rows], dtype=str)
        self.transform = transform if transform is not None else eval_image_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        with Image.open(r["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return idx, x, int(r["y"]), str(r["group"])


def stratified_8group_folds(rows: list[dict], n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Deterministic group-stratified fold assignment across all 8 (class x spur/beach) groups."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    rng = np.random.RandomState(seed)
    folds = np.full(len(rows), -1, dtype=int)
    offset = 0
    for group in GROUPS_8:
        members = sorted([i for i, r in enumerate(rows) if r["group"] == group], key=lambda i: rows[i]["id"])
        if not members:
            continue
        order = rng.permutation(members)
        folds[order] = (np.arange(len(order)) + offset) % n_splits
        offset = (offset + len(order)) % n_splits
    if (folds < 0).any():
        raise ValueError("Unassigned rows in stratified_8group_folds")
    return folds


def spawrious_balanced_weights(groups) -> np.ndarray:
    """Inverse-group-frequency weights over present Spawrious groups normalized to mean 1.0."""
    groups = np.asarray(groups).astype(str)
    weights = np.zeros(len(groups), dtype=np.float64)
    present = [g for g in GROUPS_8 if np.any(groups == g)]
    if not present:
        raise ValueError("No valid Spawrious groups found")
    for g in present:
        mask = groups == g
        weights[mask] = len(groups) / (len(present) * float(mask.sum()))
    return weights


def spawrious_group_metrics(y_true, y_pred, groups) -> dict:
    """Compute per-group accuracy (8 groups), WGA, minority_acc (beach), majority_acc (spur), and overall."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups).astype(str)
    per_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    for g in GROUPS_8:
        mask = groups == g
        counts[g] = int(mask.sum())
        if mask.any():
            per_group[g] = float((y_pred[mask] == y_true[mask]).mean()) * 100.0
    beach_accs = [per_group[g] for g in GROUPS_8 if g.endswith("_beach") and g in per_group]
    spur_accs = [per_group[g] for g in GROUPS_8 if g.endswith("_spur") and g in per_group]
    return {
        "wga": float(min(per_group.values())),
        "minority_acc": float(np.mean(beach_accs)) if beach_accs else float("nan"),
        "majority_acc": float(np.mean(spur_accs)) if spur_accs else float("nan"),
        "overall": float((y_pred == y_true).mean()) * 100.0,
        "groups": per_group,
        "counts": counts,
    }


def load_o2o_hard_ood_test_rows(data_root: Path) -> list[dict]:
    """Load official Spawrious o2o_hard shifted-background OOD test split (Step 7c final confirmation ONLY)."""
    data_root = Path(data_root).resolve()
    if (data_root / "spawrious224").is_dir() and not (data_root / "0").is_dir():
        data_root = data_root / "spawrious224"

    rows: list[dict] = []
    for domain in (0, 1):
        for cls_name in CLASSES:
            y = CLASS_TO_IDX[cls_name]
            location = TEST_SHIFTED_BG[cls_name]
            folder = data_root / str(domain) / location / cls_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing Spawrious o2o_hard OOD test folder: {folder}")
            files = sorted(
                [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                key=lambda p: p.name,
            )
            for p in files:
                rows.append(
                    {
                        "id": f"{domain}/{location}/{cls_name}/{p.name}",
                        "path": str(p),
                        "y": y,
                        "bg": location,
                        "domain": domain,
                        "group": f"{cls_name}@{location}",
                        "split": "test_ood",
                    }
                )
    rows.sort(key=lambda r: r["id"])
    return rows
