"""Feature caches, groups, stratified folds and the 60/40 partition.

Ported from src/methods/dpe_independent/sources.py. Groups are "<y><a>" strings found in the
data, iterated in sorted order; for Waterbirds that is ("00", "01", "10", "11") as before.

Duplicated inputs: CivilCommentsFine repeats a comment once per identity it mentions, as
separate rows. Rows that share a `cluster` value are forced into the same fold / partition
side (the side of the cluster's first row in id order), so no comment is used to fit a stage
that then scores a copy of it. Without duplicates (cluster == id) nothing changes.
"""
import hashlib
from pathlib import Path

import numpy as np

FEATURE_KEYS = ("ids", "features", "y", "a", "cluster")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def group_names(rows):
    return tuple(sorted({r["group"] for r in rows}))


def _keep_clusters_together(rows, assignment):
    """Give every row of a cluster the value of the cluster's first row (sorted by id)."""
    first = {}
    for i in sorted(range(len(rows)), key=lambda i: rows[i]["id"]):
        first.setdefault(rows[i].get("cluster", rows[i]["id"]), i)
    return np.array([assignment[first[r.get("cluster", r["id"])]] for r in rows])


def folds(rows, count=5, seed=0):
    rng = np.random.RandomState(seed)
    result = np.full(len(rows), -1, dtype=int)
    offset = 0
    for group in group_names(rows):
        members = sorted([i for i, r in enumerate(rows) if r["group"] == group], key=lambda i: rows[i]["id"])
        shuffled = rng.permutation(members)
        result[shuffled] = (np.arange(len(shuffled)) + offset) % count
        offset = (offset + len(shuffled)) % count
    if (result < 0).any():
        raise ValueError("Missing fold assignments")
    return _keep_clusters_together(rows, result)


def partition(rows, seed=0):
    """Same stable largest-remainder 60/40 allocation as the preserved source."""
    rng = np.random.RandomState(seed)
    head = set()
    for group in group_names(rows):
        members = sorted([r for r in rows if r["group"] == group], key=lambda r: r["id"])
        quotas = np.array([.6, .4]) * len(members)
        counts = np.floor(quotas).astype(int)
        for i in np.argsort(-(quotas - counts), kind="stable")[:len(members) - counts.sum()]:
            counts[i] += 1
        order = rng.permutation(len(members))
        head.update(members[i]["id"] for i in order[:counts[0]])
    side = _keep_clusters_together(rows, np.array([r["id"] in head for r in rows]))
    return dict(head_training=[r for r, h in zip(rows, side) if h],
                combiner_training=[r for r, h in zip(rows, side) if not h])


def stratified_subsample(rows, size, seed=0):
    """Smaller validation set for PLAN Sect. 7: every row of the smallest group is kept, the other
    groups share the remaining size in proportion to their sizes (largest remainder). Keeping the
    smallest group whole keeps DPE's group-balanced head pools able to fill a 256-row batch at every
    nesting level (a proportional 1,199-image CelebA subsample would leave 11 blond-male rows).
    Clusters partly drawn are completed. Returns sorted indices."""
    rng = np.random.RandomState(seed)
    names = group_names(rows)
    sizes = np.array([sum(r["group"] == g for r in rows) for g in names])
    smallest = int(np.argmin(sizes))
    if size <= sizes[smallest] * len(names) or size >= sizes.sum():
        raise ValueError(f"Subsample size {size} must exceed {len(names)} x the smallest group ({sizes[smallest]})")
    others = np.arange(len(names)) != smallest
    quotas = np.zeros(len(names))
    quotas[others] = sizes[others] / sizes[others].sum() * (size - sizes[smallest])
    counts = np.floor(quotas).astype(int)
    counts[smallest] = sizes[smallest]
    for i in np.argsort(-(quotas - counts), kind="stable")[:size - counts.sum()]:
        counts[i] += 1
    if (counts[others] < sizes[smallest] * 0.5).any():
        raise ValueError(f"Subsample leaves a group much smaller than the kept smallest group: {dict(zip(names, counts))}")
    chosen = set()
    for g, k in zip(names, counts):
        members = sorted([i for i, r in enumerate(rows) if r["group"] == g], key=lambda i: rows[i]["id"])
        chosen.update(np.asarray(members)[rng.permutation(len(members))[:k]].tolist())
    clusters = {rows[i].get("cluster", rows[i]["id"]) for i in chosen}
    return np.array(sorted(i for i, r in enumerate(rows) if r.get("cluster", r["id"]) in clusters))


def load_split(path, dim):
    with np.load(path, allow_pickle=False) as value:
        missing = [k for k in FEATURE_KEYS if k not in value]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        out = {k: value[k] for k in FEATURE_KEYS}
    n = len(out["ids"])
    if out["features"].shape != (n, dim) or not np.isfinite(out["features"]).all():
        raise ValueError(f"{path}: expected finite features [{n}, {dim}]")
    if len(set(out["ids"].astype(str))) != n:
        raise ValueError(f"{path}: duplicate ids")
    for k in ("y", "a", "cluster"):
        if out[k].shape != (n,):
            raise ValueError(f"{path}: misaligned {k}")
    return out


class Source:
    """One backbone: validation and test features, labels and groups.

    spec: dict(id, dataset, seed, features=dict(val=path, test=path), feature_dim,
               [val_subsample=int, val_subsample_seed=int]).
    """

    def __init__(self, spec):
        self.spec = spec
        dim = spec["feature_dim"]
        val, test = load_split(spec["features"]["val"], dim), load_split(spec["features"]["test"], dim)
        rows = [dict(id=str(i), y=int(y), group=f"{int(y)}{int(a)}", cluster=str(c))
                for i, y, a, c in zip(val["ids"], val["y"], val["a"], val["cluster"])]
        keep = np.arange(len(rows))
        if spec.get("val_subsample"):
            keep = stratified_subsample(rows, int(spec["val_subsample"]), spec.get("val_subsample_seed", 0))
        self.rows = [rows[i] for i in keep]
        self._val = val["features"][keep]
        self._test = test["features"]
        self.val_y = np.array([r["y"] for r in self.rows])
        self.val_g = np.array([r["group"] for r in self.rows])
        self.test_y = test["y"].astype(int)
        self.test_g = np.array([f"{int(y)}{int(a)}" for y, a in zip(test["y"], test["a"])])
        self.test_ids = test["ids"].astype(str)
        if set(r["id"] for r in self.rows) & set(self.test_ids):
            raise ValueError("Validation/test overlap")
        self.groups = group_names(self.rows)
        if set(self.test_g) - set(self.groups):
            raise ValueError("Test has groups that validation lacks")
        self.classes = int(spec.get("classes", 2))
        if set(self.val_y) != set(range(self.classes)):
            raise ValueError("Validation must contain every class")
