"""Train the missing ResNet-50 backbones (resumable, up to two GPUs) and describe them as sources.

Used only when the saved features of the earlier studies cannot be recovered. It reuses the repository's
own trainers unchanged:
  normal ERM : src.methods.erm.train.run_training with configs/erm/waterbirds_standard.yaml (100 epochs, no augmentation)
  ERM*       : src.methods.dpe_ermstar.run worker (400 epochs, augmentation, then DPE heads), one seed per GPU

Commands (run from the folder that holds `src/`):
  python train_backbones.py data   --work DIR --out OUT [--roots DIR ...] [--data-dir DIR]
  python train_backbones.py train  --out OUT --data DATA --normal-seeds 0 1 2 3 --ermstar-seeds 0 [--gpus 0 1] [--time-limit-hours H]
  python train_backbones.py specs  --out OUT --data DATA
  python train_backbones.py job    --kind normal --seed S --out OUT --data DATA --workers 2      (called by `train`)
"""
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

GROUPS = ("00", "01", "10", "11")


def log(*parts):
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def job_dir(out, kind, seed):
    return Path(out) / ("ermstar_jobs/seed%d" % seed if kind == "ermstar" else "normal/seed%d" % seed)


def is_done(out, kind, seed):
    d = job_dir(out, kind, seed)
    return (d / "results.json").is_file() if kind == "ermstar" else (d / "done.json").is_file()


# ----------------------------------------------------------------------------- data

def cmd_data(a):
    from src.methods.dpe_head_reweighting.study import ensure_dataset
    path = ensure_dataset(Path(a.work), [Path(r) for r in a.roots], a.data_dir, True)
    write_json(Path(a.out) / "data.json", dict(data=str(path)))
    log("Waterbirds at", path)
    return 0


# ----------------------------------------------------------------------------- one job

def erm_test_wga(backbone_ckpt, test_npz):
    """Worst-group test accuracy of the ERM linear head itself, from saved raw features (no images)."""
    import numpy as np
    import torch
    state = torch.load(backbone_ckpt, map_location="cpu", weights_only=True)
    w, b = state["2.weight"].double().numpy(), state["2.bias"].double().numpy()
    with np.load(test_npz, allow_pickle=False) as f:
        x, y, g = f["features"].astype(np.float64), f["y"], f["g"].astype(str)
    pred = (x @ w.T + b).argmax(1)
    acc = {k: float((pred[g == k] == y[g == k]).mean()) * 100 for k in GROUPS}
    return dict(wga=min(acc.values()), overall=float((pred == y).mean()) * 100, groups=acc)


def cmd_job(a):
    if a.kind != "normal":
        raise SystemExit("ERM* jobs run through src.methods.dpe_ermstar.run --worker")
    from src.methods.erm.config import load_config
    from src.methods.erm.train import run_training
    from src.methods.dpe_ermstar.run import features
    d = job_dir(a.out, "normal", a.seed)
    d.mkdir(parents=True, exist_ok=True)
    c = load_config(HERE / "configs/erm/waterbirds_standard.yaml")
    c.training.seed = a.seed
    c.training.device = "cpu" if a.smoke else "cuda:0"
    c.dataset.data_dir = str(a.data)
    c.dataset.workers = a.workers
    c.dataset.batch_size_eval = 512
    c.output_dir = str(d / "backbone")
    if a.smoke:
        c.training.epochs, c.model.weights, c.dataset.batch_size = a.smoke, None, 16
    c.validate()
    summary = d / "backbone/summary.json"
    if not summary.exists():
        resume = (d / "backbone/training_state.pt").exists()
        if (d / "backbone").exists() and not resume:
            shutil.rmtree(d / "backbone")  # a run that died before its first epoch checkpoint
        run_training(c, resume=resume)
    saved = read_json(summary)
    if not a.smoke and saved["updates"] != 37 * 100:
        raise ValueError("Backbone completion does not match the 100-epoch recipe: %s" % saved)
    values = features(a.data, d / "backbone/ckpt_last.pt", d / "features", c, c.training.device)
    erm = erm_test_wga(d / "backbone/ckpt_last.pt", d / "features/test.npz")
    write_json(d / "done.json", dict(seed=a.seed, updates=saved["updates"], erm_test=erm,
                                     val_rows=len(values["val"]["ids"]), test_rows=len(values["test"]["ids"])))
    log("normal ERM seed", a.seed, "done; ERM-only test WGA %.1f" % erm["wga"])
    return 0


# ----------------------------------------------------------------------------- queue over GPUs

def job_command(kind, seed, a, gpus_workers):
    if kind == "ermstar":
        return [sys.executable, "-m", "src.methods.dpe_ermstar.run", "--worker", "--work", str(Path(a.out) / "ermstar_jobs"),
                "--data", str(a.data), "--seed", str(seed), "--config", str(HERE / "configs/dpe_ermstar/study.json")]
    return [sys.executable, "train_backbones.py", "job", "--kind", "normal", "--seed", str(seed), "--out", str(a.out),
            "--data", str(a.data), "--workers", "2"] + (["--smoke", str(a.smoke)] if a.smoke else [])


def cmd_train(a):
    jobs = [("ermstar", s) for s in a.ermstar_seeds] + [("normal", s) for s in a.normal_seeds]
    pending = [j for j in jobs if not is_done(a.out, *j)]
    log(f"{len(jobs)} backbone jobs, {len(jobs) - len(pending)} already done: pending {pending}")
    gpus = a.gpus
    if gpus is None:
        import torch
        gpus = list(range(max(1, torch.cuda.device_count())))
    running, failed, started = {}, [], time.monotonic()
    logs = Path(a.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    last_report = 0
    try:
        while pending or running:
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                kind, seed = pending.pop(0)
                handle = (logs / f"{kind}-seed{seed}.log").open("a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                process = subprocess.Popen(job_command(kind, seed, a, 1 if len(gpus) > 1 else 2), cwd=HERE, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                running[gpu] = (process, handle, kind, seed)
                log(f"started {kind} seed {seed} on GPU {gpu} (log {logs / f'{kind}-seed{seed}.log'})")
            for gpu, (process, handle, kind, seed) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[gpu]
                ok = code == 0 and is_done(a.out, kind, seed)
                log(f"{'finished' if ok else 'FAILED (code %s)' % code} {kind} seed {seed}")
                if not ok:
                    failed.append((kind, seed))
            if time.monotonic() - started > a.time_limit_hours * 3600:
                log("time limit reached; jobs are stopped and resume from their epoch checkpoints on the next run")
                return 3
            if running and time.monotonic() - last_report > 600:
                last_report = time.monotonic()
                for gpu, (process, handle, kind, seed) in running.items():
                    tail = ""
                    p = logs / f"{kind}-seed{seed}.log"
                    if p.exists():
                        lines = [l for l in p.read_text(errors="ignore").replace("\r", "\n").splitlines() if l.strip()]
                        tail = lines[-1][-160:] if lines else ""
                    log(f"  GPU {gpu}: {kind} seed {seed}: {tail}")
            time.sleep(20)
    finally:
        for gpu, (process, handle, kind, seed) in running.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for gpu, (process, handle, kind, seed) in running.items():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            handle.close()
    return 1 if failed else 0


# ----------------------------------------------------------------------------- describe as sources

def metadata_for(data):
    from src.evaluation.prediction_export import read_metadata
    for candidate in (Path(data) / "metadata_waterbirds.csv", Path(data) / "metadata.csv",
                      Path(data) / "waterbird_complete95_forest2water2/metadata.csv", HERE / "data/metadata.csv"):
        if candidate.is_file():
            try:
                if len(read_metadata(candidate, "val")) == 1199 and len(read_metadata(candidate, "test")) == 5794:
                    return str(candidate.resolve())
            except ValueError:
                continue
    raise FileNotFoundError("No usable official Waterbirds metadata")


def cmd_specs(a):
    metadata = metadata_for(a.data)
    specs = []
    for kind in ("normal", "ermstar"):
        base = Path(a.out) / ("normal" if kind == "normal" else "ermstar_jobs")
        for d in sorted(base.glob("seed*")):
            seed = int(d.name[4:])
            if not is_done(a.out, kind, seed):
                continue
            features = d / "features"
            spec = dict(id=f"fresh-{'normal-erm' if kind == 'normal' else 'ermstar'}-seed{seed}", fresh=True,
                        recipe="normal_erm" if kind == "normal" else "ermstar", seed=seed, metadata=metadata,
                        features=dict(val=str(features / "val.npz"), test=str(features / "test.npz")),
                        ensembles={}, manifest="", provenance=str(features / "complete.json"))
            try:
                spec["erm_test"] = read_json(d / "done.json")["erm_test"] if kind == "normal" else erm_test_wga(
                    d / "backbone/ckpt_last.pt", features / "test.npz")
            except Exception as error:
                spec["erm_test_error"] = str(error)
            specs.append(spec)
    write_json(Path(a.out) / "sources_fresh.json", dict(sources=specs, blockers=[]))
    log("fresh sources:", [s["id"] for s in specs])
    return 0 if specs else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data")
    d.add_argument("--work", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--roots", nargs="*", default=["/kaggle/input"])
    d.add_argument("--data-dir", default=None)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--normal-seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    t.add_argument("--ermstar-seeds", type=int, nargs="*", default=[0])
    t.add_argument("--gpus", type=int, nargs="*", default=None)
    t.add_argument("--time-limit-hours", type=float, default=11.0)
    t.add_argument("--smoke", type=int, default=0, help="CPU smoke test: this many epochs, no pretrained weights")
    s = sub.add_parser("specs")
    s.add_argument("--out", required=True)
    s.add_argument("--data", required=True)
    j = sub.add_parser("job")
    j.add_argument("--kind", required=True)
    j.add_argument("--seed", type=int, required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--data", required=True)
    j.add_argument("--workers", type=int, default=2)
    j.add_argument("--smoke", type=int, default=0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    sys.exit({"data": cmd_data, "train": cmd_train, "specs": cmd_specs, "job": cmd_job}[args.command](args))
