"""End-to-end run of the nested pipeline on synthetic backbone outputs (no real data).

Covers: discover, `run --smoke` (CLI), cells with test scoring (2 epochs to stay fast),
summarize.py, a binary 4-group dataset (CelebA layout), a 16-group dataset whose rows repeat
the same input under different groups (CivilCommentsFine layout), and a 3-class dataset
(MultiNLI layout, PLAN Sect. 9 calibrators).

    python tests/integration.py WORK_DIR [DATASET ...]

Groups must be large enough that the 60% head pool fills one 256-row balanced batch.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from xdpe import data, nested  # noqa: E402
from xdpe.datasets import DATASETS  # noqa: E402

LAYOUTS = {  # dataset: (classes, attributes, rows per group in val, duplicated fraction)
    "CelebA": (2, 2, [260, 240, 150, 110], 0.0),
    "CivilCommentsFine": (2, 8, [70] * 8 + [40] * 8, 0.15),
    "MultiNLI": (3, 2, [130, 90, 130, 90, 120, 90], 0.0),
}


def synthetic_split(rng, dataset, split, sizes, seed):
    classes, attrs, _, dup = LAYOUTS[dataset]
    dim = DATASETS[dataset]["feature_dim"]
    centres = np.random.RandomState(1000 + seed).normal(size=(classes + attrs, dim))
    ids, y, a, cluster, feats = [], [], [], [], []
    for g, n in enumerate(sizes):
        yy, aa = divmod(g, attrs)
        for i in range(n):
            ids.append(f"{split}-{yy}{aa}-{i}")
            y.append(yy), a.append(aa)
            cluster.append(ids[-1])
            feats.append(centres[yy] + 0.8 * centres[classes + aa] + rng.normal(scale=2.0, size=dim))
    feats = np.asarray(feats, dtype=np.float32)
    if dup:  # repeat some inputs under another identity, like CivilCommentsFine
        k = int(dup * len(ids))
        src = rng.choice(len(ids), size=k, replace=False)
        for j, s in enumerate(src):
            new_a = (a[s] + 1) % attrs
            ids.append(f"{split}-dup-{j}")
            y.append(y[s]), a.append(new_a), cluster.append(cluster[s])
        feats = np.concatenate([feats, feats[src]])
    return dict(ids=np.array(ids), features=feats, y=np.array(y), a=np.array(a), cluster=np.array(cluster))


def make_backbones(root, dataset, seeds):
    rng = np.random.RandomState(0)
    sizes = LAYOUTS[dataset][2]
    for seed in seeds:
        base = root / f"{dataset}-seed{seed}"
        (base / "features").mkdir(parents=True)
        hashes = {}
        for split, scale in (("val", 1), ("test", 2)):
            arrays = synthetic_split(rng, dataset, split, [n * scale for n in sizes], seed)
            np.savez(base / "features" / f"{split}.npz", **arrays)
            hashes[split] = data.sha256(base / "features" / f"{split}.npz")
        (base / "backbone.json").write_text(json.dumps(dict(
            dataset=dataset, seed=seed, status="complete", checkpoint_sha256="synthetic", features_sha256=hashes,
            erm_metrics=dict(test=dict(wga=0.5)))))


def check_cluster_folds(source):
    for seed in range(3):
        assignment = data.folds(source.rows, 5, seed)
        by = {}
        for r, f in zip(source.rows, assignment):
            by.setdefault(r["cluster"], set()).add(int(f))
        assert all(len(v) == 1 for v in by.values()), "a cluster was split across folds"
        part = data.partition(source.rows, seed)
        head = {r["cluster"] for r in part["head_training"]}
        fit = {r["cluster"] for r in part["combiner_training"]}
        assert not head & fit, "a cluster is on both sides of the 60/40 partition"


def main():
    work = Path(sys.argv[1]).resolve()
    if work.exists():
        raise SystemExit("use a fresh directory")
    seeds = [0, 1, 2]
    reports = []
    for dataset in (sys.argv[2:] or LAYOUTS):
        t0 = time.time()
        roots = work / "backbones" / dataset
        make_backbones(roots, dataset, seeds)
        out = work / "nested" / dataset
        subprocess.run([sys.executable, "-m", "xdpe.nested", "discover", "--roots", str(roots), "--dataset", dataset,
                        "--out", str(out)], cwd=CODE, check=True)
        specs = json.loads((out / "sources.json").read_text())["sources"]
        assert [s["seed"] for s in specs] == seeds
        source = nested.get_source(specs[0])
        if LAYOUTS[dataset][3]:
            assert len(set(source.val_g)) == 16 and len({r["cluster"] for r in source.rows}) < len(source.rows)
            check_cluster_folds(source)
        smoke = work / "smoke" / dataset
        smoke.mkdir(parents=True)
        (smoke / "sources.json").write_text((out / "sources.json").read_text())
        subprocess.run([sys.executable, "-m", "xdpe.nested", "run", "--out", str(smoke), "--splits", "0", "--only",
                        specs[0]["id"], "--smoke", "--workers", "1"], cwd=CODE, check=True)
        smoke_cell = json.loads((smoke / "smoke" / specs[0]["id"] / "split00.json").read_text())
        assert not (smoke / "results").exists(), "smoke run wrote test results"
        # Scored cells, 2 epochs per head to keep the test short.
        cfg, protocol_id = nested.prepare_protocol(out, dataset, False)
        cfg["head_training"]["epochs"] = 2
        for spec in specs:
            for split in (0, 1):
                result = nested.run_task((spec, split, str(out), cfg, protocol_id, time.time() + 1e6, True))
                if result[2] != "done":
                    print((out / "errors" / f"{spec['id']}-split{split:02d}.txt").read_text())
                    raise SystemExit(f"{dataset}: {result}")
        cell = json.loads((out / "results" / specs[0]["id"] / "split00.json").read_text())
        expected = 27 if LAYOUTS[dataset][0] == 2 else 25
        assert len(cell["arms"]) == expected, (dataset, sorted(cell["arms"]))
        assert sorted(smoke_cell["arms"]) == sorted(cell["arms"]), "smoke and scored runs build different arms"
        assert len(next(iter(cell["arms"].values()))["groups"]) == len(LAYOUTS[dataset][2])
        reports.append(out)
        print(f"{dataset}: {len(cell['arms'])} arms, {len(cell['arms']['fullval_uniform']['groups'])} groups, "
              f"fullval WGA {cell['arms']['fullval_uniform']['wga']:.1f}, {time.time() - t0:.0f} s")
    subprocess.run([sys.executable, "summarize.py", *map(str, reports), "--report", str(work / "report")],
                   cwd=CODE, check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((work / "report/summary.json").read_text())
    for r in summary["results"]:
        assert r["missing"], "only 2 of 10 splits ran, so the run must be flagged partial"
        assert all(c["n"] == 3 for c in r["contrasts"])
    print("integration: all checks passed; report at", work / "report/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
