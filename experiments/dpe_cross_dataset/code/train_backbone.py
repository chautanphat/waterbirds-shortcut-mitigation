#!/usr/bin/env python3
"""Train one SubpopBench ERM backbone and save frozen validation/test features (PLAN.md Sect. 3).

Runs inside the pinned environment on exactly one visible GPU:

    python train_backbone.py --subpop-root SUBPOP --data-dir DATA --dataset CelebA --seed 0 --output-dir OUT
        [--steps 300]            timing smoke only; the real run keeps SubpopBench's default (30,001)
        [--resume MODEL.pkl]     continue an interrupted run (recorded as a deviation, see below)

Training: `subpopbench.train --algorithm ERM --train_attr no --hparams_seed 0` at the pinned
commit, default hyper-parameters and steps, unmodified. The backbone is SubpopBench's last
checkpoint (`model.pkl`, written at the final step), as for the Waterbirds fresh backbones.

Features: the ERM featurizer output (ResNet-50: 2,048-d pooled; BERT: 768-d pooler output, with
dropout off in eval mode) on the SubpopBench validation and test datasets, same transforms and
tokenisation as SubpopBench evaluation. Saved as features/{val,test}.npz with ids, y, a and a
`cluster` key (CivilComments: SHA-1 of the comment text, since one comment appears once per
identity it mentions; otherwise the id). The ERM classifier's own validation/test metrics are
stored in backbone.json for context.

Resume: SubpopBench's --resume restores weights and step count, but not optimizer state, the
text learning-rate schedule or the data order. A resumed backbone is therefore NOT the same as an
uninterrupted one; backbone.json records `resumed_from_step` so the deviation is reported.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUBPOP_COMMIT = "4d3dbbe21029666ef19d040e110ec22908640c5b"
FEATURE_DIM = {"CelebA": 2048, "CivilCommentsFine": 768, "MultiNLI": 768}
# Expected (split -> group counts in (y, a) order 00, 01, 10, 11) for CelebA, from GroupDRO/SubpopBench.
# A mismatch is recorded and printed, not fatal (PLAN Sect. 2: verified counts are used).
EXPECTED = {"CelebA": {1: [8535, 8276, 2874, 182], 2: [9767, 7535, 2480, 180]}}
METADATA = {"CelebA": "celeba/metadata_celeba.csv", "MultiNLI": "multinli/metadata_multinli.csv",
            "CivilCommentsFine": "civilcomments/metadata_civilcomments_fine.csv"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def verify_repo(path):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != SUBPOP_COMMIT:
        raise RuntimeError(f"Expected SubpopBench {SUBPOP_COMMIT}, found {actual}")
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise RuntimeError("SubpopBench tracked files have been modified")


def verify_data(data_dir, dataset):
    import pandas as pd
    meta = Path(data_dir) / METADATA[dataset]
    df = pd.read_csv(meta)
    counts = {}
    for split in (0, 1, 2):
        part = df[df.split == split]
        counts[split] = {f"{y}{a}": int(n) for (y, a), n in part.groupby(["y", "a"]).size().items()}
    out = dict(metadata=str(meta), metadata_sha256=sha256(meta), rows=len(df), group_counts=counts)
    if dataset in EXPECTED:
        observed = {s: [counts[s].get(g, 0) for g in ("00", "01", "10", "11")] for s in EXPECTED[dataset]}
        out["expected_group_counts"] = EXPECTED[dataset]
        out["matches_expected"] = observed == EXPECTED[dataset]
        if not out["matches_expected"]:
            print(f"WARNING: {dataset} group counts {observed} differ from expected {EXPECTED[dataset]}", flush=True)
    return out


def train(args, out):
    bootstrap = ("import runpy; from subpopbench.dataset import datasets as D; "
                 f"D.{args.dataset}.N_WORKERS = {args.workers}; "
                 "runpy.run_module('subpopbench.train', run_name='__main__')")
    cmd = [sys.executable, "-c", bootstrap, "--algorithm", "ERM", "--dataset", args.dataset, "--train_attr", "no",
           "--data_dir", str(args.data_dir), "--output_dir", str(out / "subpopbench"),
           "--output_folder_name", "standard", "--seed", str(args.seed), "--hparams_seed", "0"]
    if args.steps:
        cmd += ["--steps", str(args.steps)]
    if args.resume:
        cmd += ["--resume", str(args.resume)]
    write_json(out / "train_command.json", dict(argv=cmd, cwd=str(args.subpop_root)))
    started = time.time()
    with open(out / "train.log", "a") as log:
        subprocess.run(cmd, cwd=args.subpop_root, stdout=log, stderr=subprocess.STDOUT, check=True)
    run_dir = out / "subpopbench/standard_attrNo" / f"{args.dataset}_ERM_hparams0_seed{args.seed}"
    if not (run_dir / "done").exists():
        raise RuntimeError("SubpopBench did not finish")
    return run_dir, time.time() - started


def timing(run_dir, wall_seconds, steps_requested):
    """Per-step time from SubpopBench's results.json; evaluation time is the remainder."""
    lines = [json.loads(l) for l in (run_dir / "results.json").read_text().splitlines() if l.strip()]
    steps = lines[-1]["step"] + 1
    # Each line averages step_time over the steps since the previous checkpoint.
    spans, prev = [], -1
    for line in lines:
        spans.append((line["step"] - prev, line["step_time"]))
        prev = line["step"]
    step_seconds = sum(n * t for n, t in spans) / sum(n for n, _ in spans)
    evals = len(lines) + 1  # every checkpoint plus the final evaluation
    eval_seconds = max(0., wall_seconds - steps * step_seconds) / evals
    return dict(steps=steps, seconds_per_step=step_seconds, evaluations=evals,
                seconds_per_evaluation=eval_seconds, wall_seconds=wall_seconds)


def extract(args, run_dir, out, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(args.subpop_root))
    from subpopbench.dataset import datasets
    from subpopbench.learning import algorithms
    ckpt = torch.load(run_dir / "model.pkl", map_location="cpu", weights_only=False)  # our own SubpopBench output
    hparams = ckpt["model_hparams"]
    cls = datasets.get_dataset_class(args.dataset)
    splits = {name: cls(str(args.data_dir), split, hparams) for name, split in (("val", "va"), ("test", "te"))}
    erm = algorithms.ERM(splits["val"].data_type, ckpt["model_input_shape"], ckpt["num_labels"],
                         ckpt["num_attributes"], 0, hparams)
    erm.load_state_dict(ckpt["model_dict"], strict=True)
    erm.to(device).eval()
    results, metrics = {}, {}
    for name, ds in splits.items():
        loader = DataLoader(ds, batch_size=256 if ds.data_type == "images" else 128, shuffle=False,
                            num_workers=args.workers)
        feats, logits, index = [], [], []
        with torch.no_grad():
            for i, x, y, a in loader:
                z = erm.featurizer(x.to(device))
                feats.append(z.float().cpu().numpy())
                logits.append(erm.classifier(z).float().cpu().numpy())
                index.append(i.numpy())
        feats, logits, index = np.concatenate(feats), np.concatenate(logits), np.concatenate(index)
        if not np.array_equal(index, np.arange(len(ds))):
            raise RuntimeError("Unexpected loader order")
        if feats.shape != (len(ds), FEATURE_DIM[args.dataset]) or not np.isfinite(feats).all() or (feats.std(1) == 0).any():
            raise ValueError("Invalid features")
        y = np.asarray(ds.y, dtype=int)
        a = np.asarray(ds.a, dtype=int)
        ids = np.array([os.path.basename(p) if ds.data_type == "images" else str(p) for p in ds.x])
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate ids")
        if args.dataset.startswith("CivilComments"):
            cluster = np.array([hashlib.sha1(ds.text_array[int(p)].encode("utf-8")).hexdigest() for p in ds.x])
        else:
            cluster = ids
        target = out / "features" / f"{name}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, ids=ids, features=feats.astype(np.float32), y=y, a=a, cluster=cluster)
        results[name] = target
        pred = logits.argmax(1)
        groups = np.array([f"{u}{v}" for u, v in zip(y, a)])
        per_group = {g: dict(n=int((groups == g).sum()), accuracy=float((pred == y)[groups == g].mean()))
                     for g in sorted(set(groups))}
        metrics[name] = dict(overall=float((pred == y).mean()), wga=min(v["accuracy"] for v in per_group.values()),
                             per_group=per_group, duplicated_rows=int(len(cluster) - len(set(cluster))))
    return results, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subpop-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=sorted(FEATURE_DIM), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--skip-features", action="store_true", help="timing smoke: train only")
    args = p.parse_args()
    for name in ("subpop_root", "data_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    verify_repo(args.subpop_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import torchvision
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Launch with CUDA_VISIBLE_DEVICES=<one GPU>")
    manifest_path = args.output_dir / "backbone.json"
    manifest = dict(dataset=args.dataset, seed=args.seed, status="running", smoke_steps=args.steps or None,
                    subpopbench=SUBPOP_COMMIT, runner_sha256=sha256(__file__),
                    environment=dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
                    data=verify_data(args.data_dir, args.dataset))
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)["start_step"]
        manifest["resumed_from_step"] = int(resumed)
        manifest["deviation"] = ("resumed with SubpopBench --resume: optimizer state, text LR schedule and "
                                 "data order restart; not identical to an uninterrupted run")
    write_json(manifest_path, manifest)
    try:
        run_dir, wall = train(args, args.output_dir)
        manifest["timing"] = timing(run_dir, wall, args.steps)
        manifest["checkpoint"] = str(run_dir / "model.pkl")
        manifest["checkpoint_sha256"] = sha256(run_dir / "model.pkl")
        write_json(manifest_path, manifest)
        if not args.skip_features:
            files, metrics = extract(args, run_dir, args.output_dir, "cuda")
            manifest["features_sha256"] = {k: sha256(v) for k, v in files.items()}
            manifest["erm_metrics"] = metrics
        manifest["status"] = "complete" if not args.steps else "smoke-complete"
    except BaseException as e:
        manifest["status"] = "failed"
        manifest["error"] = repr(e)
        raise
    finally:
        write_json(manifest_path, manifest)
    print(json.dumps({k: manifest.get(k) for k in ("status", "timing", "erm_metrics")}, default=str)[:2000], flush=True)


if __name__ == "__main__":
    main()
