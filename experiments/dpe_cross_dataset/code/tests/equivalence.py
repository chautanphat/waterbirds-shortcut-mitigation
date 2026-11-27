"""Bit-for-bit check: the ported pipeline reproduces the original Waterbirds pipeline.

Runs one (source, split) cell through experiments/dpe_nested_corrected/code/corrected_nested.py
(the code of the strict nested Waterbirds run) and through xdpe.nested on the same Waterbirds
features, then compares every arm, every setting, every inner score and every cached head
score. DFR-lite exists only in the original and is ignored.

    python tests/equivalence.py --val FEATS_VAL.npy --val-manifest MANIFEST.json \
        --test FEATS_TEST.npy --test-manifest MANIFEST.json --work DIR [--epochs 2] [--split 0]

`--epochs 20` is the full recipe (about 10-20 CPU minutes per side); fewer epochs test the
same code paths faster.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPO = CODE.parents[2]
ORIGINAL = REPO / "experiments/dpe_nested_corrected/code"
METADATA = ORIGINAL / "data/metadata.csv"


def relative(path):
    path = path.replace("\\", "/")
    marker = "waterbird_complete95_forest2water2/"
    return path.split(marker, 1)[1] if marker in path else path


def prepare(a, work):
    with METADATA.open(newline="") as f:
        meta = list(csv.DictReader(f))
    out = {}
    for split, code, feats, manifest in (("val", "1", a.val, a.val_manifest), ("test", "2", a.test, a.test_manifest)):
        rows = [r for r in meta if r["split"] == code]
        paths = [relative(p) for p in json.loads(Path(manifest).read_text())["image_paths"]]
        raw = np.load(feats)
        index = {p: i for i, p in enumerate(paths)}
        order = np.array([index[r["img_filename"]] for r in rows])
        ids = np.array([r["img_filename"] for r in rows])
        y = np.array([int(r["y"]) for r in rows])
        attr = np.array([int(r["place"]) for r in rows])
        target = work / f"{split}.npz"
        np.savez(target, ids=ids, features=raw[order].astype(np.float32), y=y, a=attr, cluster=ids)
        out[split] = target
    return out


def run_original(work, epochs, split):
    sys.path.insert(0, str(ORIGINAL))
    import nested_selection as old
    import corrected_nested as cn
    cfg = old.adapters()["grid"].load_config(ORIGINAL / "configs/dpe_independent/study.yaml")
    cfg["head_training"]["epochs"] = epochs
    with np.load(work / "val.npz") as v, np.load(work / "test.npz") as t:
        spec = dict(id="wb", recipe="normal_erm", seed=0, metadata=str(METADATA),
                    features=dict(val=str(work / "val.npz"), test=str(work / "test.npz")))

        class S:
            pass
        s = S()
        s.spec = spec
        s.rows = [dict(id=str(i), y=int(y), group=f"{int(y)}{int(a)}", split=1) for i, y, a in zip(v["ids"], v["y"], v["a"])]
        s._val, s._test = v["features"], t["features"]
        s.val_y = np.array([r["y"] for r in s.rows])
        s.val_g = np.array([r["group"] for r in s.rows])
        s.test_y = t["y"].astype(int)
        s.test_g = np.array([f"{int(y)}{int(a)}" for y, a in zip(t["y"], t["a"])])
    old._SOURCE_CACHE["wb"] = s
    out = work / "original"
    result = cn.run_task((spec, split, str(out), cfg, "equivalence", time.time() + 1e6))
    print("original:", result, flush=True)
    return result


def run_port(work, epochs, split):
    sys.path.insert(0, str(CODE))
    from xdpe import nested
    cfg = nested.protocol_config("Waterbirds", False)
    cfg["head_training"]["epochs"] = epochs
    spec = dict(id="wb", dataset="Waterbirds", seed=0, feature_dim=2048, classes=2,
                features=dict(val=str(work / "val.npz"), test=str(work / "test.npz")))
    out = work / "port"
    result = nested.run_task((spec, split, str(out), cfg, "equivalence", time.time() + 1e6, True))
    print("port:", result, flush=True)
    return result


def compare(work, split):
    a = json.loads((work / f"original/results/wb/split{split:02d}.json").read_text())
    b = json.loads((work / f"port/results/wb/split{split:02d}.json").read_text())
    problems = []
    arms = sorted(set(a["arms"]) - {"dfr_lite"})
    if sorted(b["arms"]) != arms:
        problems.append(f"arm sets differ: {sorted(set(arms) ^ set(b['arms']))}")
    for arm in arms:
        x, y = a["arms"][arm], b["arms"].get(arm, {})
        for key in ("wga", "overall", "groups", "selected", "threshold"):
            if x.get(key) != y.get(key):
                problems.append(f"{arm}.{key}: {x.get(key)} != {y.get(key)}")
    for family, settings in a["grid"].items():
        for setting, value in settings.items():
            other = b["grid"][family][setting]
            if (value["wga"], value["overall"], value["groups"]) != (other["wga"], other["overall"], other["groups"]):
                problems.append(f"grid {family}/{setting} differs")
    for family, settings in a["inner"].items():
        if b["inner"].get(family) != settings:
            problems.append(f"inner {family} differs")
    if a["diagnostics"]["threshold_folds"] != b["diagnostics"]["threshold_folds"]:
        problems.append("threshold folds differ")
    caches = sorted((work / "original/checkpoints").rglob("*.npz"))
    for path in caches:
        twin = work / "port/checkpoints" / path.relative_to(work / "original/checkpoints")
        with np.load(path) as x, np.load(twin) as y:
            for key in x.files:
                if not np.array_equal(x[key], y[key]):
                    problems.append(f"cache {path.relative_to(work)}:{key} differs (max {np.abs(x[key] - y[key]).max()})")
    print(f"compared {len(arms)} arms, {sum(len(s) for s in a['grid'].values())} settings, {len(caches)} score caches")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val"), p.add_argument("--val-manifest"), p.add_argument("--test"), p.add_argument("--test-manifest")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--side", choices=["original", "port"])
    a = p.parse_args()
    work = a.work.resolve()
    if a.side:
        (run_original if a.side == "original" else run_port)(work, a.epochs, a.split)
        return 0
    if (work / "original").exists() or (work / "port").exists():
        raise SystemExit(f"Use a fresh --work directory: {work}")
    work.mkdir(parents=True, exist_ok=True)
    prepare(a, work)
    base = [sys.executable, __file__, "--work", str(work), "--epochs", str(a.epochs), "--split", str(a.split)]
    procs = [subprocess.Popen(base + ["--side", side], cwd=REPO) for side in ("original", "port")]
    if any(p.wait() for p in procs):
        raise SystemExit("A side failed")
    problems = compare(work, a.split)
    for line in problems[:50]:
        print("MISMATCH", line)
    print("EQUIVALENT: bit-for-bit" if not problems else f"NOT EQUIVALENT: {len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
