"""Read-only historical adapters and explicitly feature-only lightweight head fits."""
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.prediction_export import align_ids, read_metadata, sha256
from src.methods.dpe_head_reweighting import ensemble


def partition(rows, seed=0):
    """Same stable largest-remainder 60/40 allocation as the preserved source."""
    rng = np.random.RandomState(seed)
    head, fit = set(), set()
    for group in ("00", "01", "10", "11"):
        members = sorted([r for r in rows if r["group"] == group], key=lambda r: r["id"])
        quotas = np.array([.6, .4])*len(members)
        counts = np.floor(quotas).astype(int)
        for i in np.argsort(-(quotas-counts), kind="stable")[:len(members)-counts.sum()]:
            counts[i] += 1
        order = rng.permutation(len(members))
        head.update(members[i]["id"] for i in order[:counts[0]])
        fit.update(members[i]["id"] for i in order[counts[0]:])
    return dict(head_training=[r for r in rows if r["id"] in head],
                combiner_training=[r for r in rows if r["id"] in fit])


def folds(rows, count=5, seed=0):
    rng = np.random.RandomState(seed)
    result = np.full(len(rows), -1, dtype=int)
    offset = 0
    for group in ("00", "01", "10", "11"):
        members = sorted([i for i, r in enumerate(rows) if r["group"] == group], key=lambda i: rows[i]["id"])
        shuffled = rng.permutation(members)
        result[shuffled] = (np.arange(len(shuffled))+offset) % count
        offset = (offset+len(shuffled)) % count
    if (result < 0).any():
        raise ValueError("Missing fold assignments")
    return result


def one(paths, label):
    paths = list(paths)
    identities = {}
    for path in paths:
        identities.setdefault(sha256(path), path)
    if len(identities) != 1:
        raise ValueError(f"Expected one unambiguous {label}; found {len(identities)}")
    return str(next(iter(identities.values())).resolve())


def discover(roots):
    """Known saved normal-ERM and ERM* layouts; no inferred seed from row order."""
    roots = [Path(r).resolve() for r in roots]
    metadata = []
    for root in roots:
        metadata.extend(root.rglob("metadata.csv"))
        metadata.extend(root.rglob("metadata_waterbirds.csv"))
    metadata = [p for p in metadata if len(read_metadata(p, "val")) == 1199]
    if not metadata:
        return [], ["No official Waterbirds metadata found"]
    reference = one(metadata, "metadata")
    found, blockers = {}, []
    for root in roots:
        for manifest_path in root.rglob("manifest.json"):
            if manifest_path.parent.name != "experiment":
                continue
            m = json.loads(manifest_path.read_text())
            if m.get("protocol") != "independent_comparison" or m.get("provenance", {}).get("track") != "standard":
                continue
            seed = m["provenance"]["seed"]
            identifier = f"normal-erm-seed{seed}"
            base = manifest_path.parent.parent
            model = base/"checkpoints/ensemble.pt"
            feature_candidates = []
            for search_root in roots:
                for handoff in search_root.rglob("handoff.json"):
                    h = json.loads(handoff.read_text())
                    if h.get("identity", {}).get("checkpoint_sha256") == m["backbone_hash"]:
                        if sha256(handoff.parent/"val_raw.npz") != m["val_features_hash"]:
                            raise ValueError("Validation features differ from historical manifest")
                        if sha256(handoff.parent/"test_raw.npz") != h["artifacts"]["test_raw.npz"]:
                            raise ValueError("Test feature handoff changed")
                        feature_candidates.append(handoff.parent)
            if not feature_candidates or not model.is_file():
                blockers.append(f"{identifier}: missing features or ensemble")
                continue
            feature_dir = Path(one([p/"val_raw.npz" for p in feature_candidates], "validation feature cache")).parent
            # Immutable head record binds the recovered model to the source.
            head_record = json.loads((base/"experiment/heads.json").read_text())
            if head_record["lock"]["ensemble"] != sha256(model):
                raise ValueError("Historical ensemble identity changed")
            controls = []
            for search_root in roots:
                controls.extend(search_root.rglob(f"seed{seed}-full_validation_20epochs.pt"))
            ensembles = {"partition0": str(model)}
            hashes = {"partition0": head_record["lock"]["ensemble"]}
            manifests = {"partition0": str(manifest_path)}
            if controls:
                ensembles["full_validation"] = one(controls, "full-validation heads")
            for part in (1, 2):
                choices = [p for sr in roots for p in sr.rglob(f"seed{seed}-partition{part}.pt")]
                if choices:
                    ensembles[f"partition{part}"] = one(choices, f"partition-{part} heads")
                    pm = [p for sr in roots for p in sr.rglob(f"seed{seed}-split{part}-manifest.json")]
                    manifests[f"partition{part}"] = one(pm, f"partition-{part} manifest")
            for arm, model_path in ensembles.items():
                if arm == "partition0":
                    continue
                marker = Path(model_path).with_suffix(".identity.json")
                recorded = json.loads(marker.read_text())
                if recorded["sha256"] != sha256(model_path) or recorded["identity"]["source_seed"] != seed:
                    raise ValueError("Saved control head identity changed")
                hashes[arm] = recorded["sha256"]
            found[identifier] = dict(id=identifier, recipe="normal_erm", seed=seed, metadata=reference,
                features=dict(val=str(feature_dir/"val_raw.npz"), test=str(feature_dir/"test_raw.npz")),
                ensembles=ensembles, manifest=str(manifest_path), ensemble_hashes=hashes, partition_manifests=manifests,
                historical_predictions=dict(partition0=str(base/"experiment/predictions/test.npz")),
                backbone_sha256=m["backbone_hash"], provenance=str(feature_dir/"provenance.json"))
        for marker in root.rglob("complete.json"):
            if marker.parent.name != "features":
                continue
            m = json.loads(marker.read_text())
            recipe = m.get("identity", {}).get("recipe", {})
            if recipe.get("training", {}).get("epochs") != 400:
                continue
            seed = recipe["training"]["seed"]
            base = marker.parent.parent
            identifier = f"ermstar-seed{seed}"
            if not all((base/p).is_file() for p in ("paper_reference/ensemble.pt", "neural_split/ensemble.pt", "features/val.npz", "features/test.npz")):
                blockers.append(f"{identifier}: incomplete full saved jobs directory")
                continue
            found[identifier] = dict(id=identifier, recipe="ermstar", seed=seed, metadata=reference,
                features=dict(val=str(base/"features/val.npz"), test=str(base/"features/test.npz")),
                ensembles=dict(full_validation=str(base/"paper_reference/ensemble.pt"), partition0=str(base/"neural_split/ensemble.pt")),
                manifest=str(base/"neural_split/partition.json"), provenance=str(marker),
                historical_predictions={arm: str(base/p) for arm, p in (("partition0", "neural_split/test_heads.npz"), ("full_validation", "paper_reference/test_heads.npz"))},
                backbone_sha256=None)
    return sorted(found.values(), key=lambda s: (s["recipe"] != "ermstar", s["seed"])), blockers


class Source:
    def __init__(self, spec):
        self.spec = spec
        self.val_rows = read_metadata(spec["metadata"], "val")
        self.test_rows = read_metadata(spec["metadata"], "test")
        if len(self.val_rows) != 1199 or len(self.test_rows) != 5794:
            raise ValueError("Official validation/test coverage required")
        self.rows = [dict(id=r["image_path"], y=r["y"], group=f"{r['y']}{r['a']}", split=1) for r in self.val_rows]
        if set(r["id"] for r in self.rows) & set(r["image_path"] for r in self.test_rows):
            raise ValueError("Validation/test overlap")

    def features(self, split):
        rows = self.val_rows if split == "val" else self.test_rows
        with np.load(self.spec["features"][split], allow_pickle=False) as value:
            order = align_ids(value["ids"], rows)
            features = value["features"][order]
            if "y" in value and not np.array_equal(value["y"][order], [r["y"] for r in rows]):
                raise ValueError("Feature labels differ from reference metadata")
            if "g" in value and not np.array_equal(value["g"][order].astype(str), [f"{r['y']}{r['a']}" for r in rows]):
                raise ValueError("Feature groups differ from reference metadata")
        if features.shape != (len(rows), 2048) or not np.isfinite(features).all():
            raise ValueError("Expected raw ResNet-50 features [N,2048]")
        return features

    def head_indices(self, arm):
        if arm == "full_validation":
            return np.arange(1199), np.array([], dtype=int)
        part = partition(self.rows, int(arm.removeprefix("partition")))
        index = {r["id"]: i for i, r in enumerate(self.rows)}
        head = np.array([index[r["id"]] for r in part["head_training"]])
        fit = np.array([index[r["id"]] for r in part["combiner_training"]])
        if len(head) != 720 or len(fit) != 479 or set(head) & set(fit):
            raise ValueError("Invalid 720/479 split")
        manifest = self.spec.get("partition_manifests", {}).get(arm)
        if arm == "partition0" or manifest:
            saved = json.loads(Path(manifest or self.spec["manifest"]).read_text())
            if self.spec["recipe"] == "normal_erm":
                saved = saved["parts"]
            if saved != part:
                raise ValueError("Primary partition differs from historical inputs")
        return head, fit

    def heads(self, arm, work, config, device="cpu"):
        head_idx, _ = self.head_indices(arm)
        if arm in self.spec["ensembles"]:
            path = Path(self.spec["ensembles"][arm])
            if arm in self.spec.get("ensemble_hashes", {}) and sha256(path) != self.spec["ensemble_hashes"][arm]:
                raise ValueError("Saved ensemble changed")
            heads = ensemble.load(path, 2048, 15)
            if "training_pool_ids" in heads["provenance"]:
                if heads["provenance"]["training_pool_ids"] != [self.rows[i]["id"] for i in head_idx]:
                    raise ValueError("Saved heads use a different training pool")
            return heads, path
        return self.train_heads(head_idx, Path(work)/arm/"ensemble.pt", config, device)

    def train_heads(self, indices, path, config, device="cpu"):
        """Deliberately cannot import or call a backbone trainer."""
        from .storage import frozen_json
        rows = [self.rows[i] for i in indices]
        settings = config["head_training"]
        identity = dict(source=self.spec, training_ids=[r["id"] for r in rows], settings=settings,
                        raw_features_sha256=sha256(self.spec["features"]["val"]), device=device)
        frozen_json(path.with_suffix(".inputs.json"), identity)
        if path.exists():
            return ensemble.load(path, 2048, 15), path
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.features("val")[indices]
        if self.spec["recipe"] == "ermstar":
            from src.methods.dpe_ermstar.run import train_heads
            heads = train_heads(raw, rows, path.with_name("head_progress.pt"), self.spec["seed"], settings, device)
        else:
            from src.methods.dpe_head_reweighting.train_heads import train
            heads = train(raw, np.array([r["y"] for r in rows]), np.array([r["group"] for r in rows]),
                          settings, self.spec["seed"], identity, device, progress=True)
        ensemble.save(path, heads)
        return heads, path
