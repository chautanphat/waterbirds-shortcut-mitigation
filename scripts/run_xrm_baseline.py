"""Configurable launcher for the authors' Waterbirds ERM and GroupDRO code."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / "baselines" / "xrm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("erm", "groupdro"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--num-seeds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Short override for smoke runs; omit for the full upstream schedule")
    parser.add_argument("--checkpoint-frequency", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    adapter = importlib.import_module(f"src.methods.{args.method}.xrm")
    config_path = args.config or ROOT / adapter.CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if config.get("method") != args.method:
        parser.error(f"{config_path} is for method {config.get('method')!r}, not {args.method!r}")

    upstream = config["upstream"]
    output_root = args.output_root or ROOT / "outputs" / f"xrm_{args.method}"
    data_path = args.data_path.resolve()
    output_root = output_root.resolve()
    steps = args.max_steps or upstream["steps"]
    checkpoint_frequency = args.checkpoint_frequency or upstream["checkpoint_frequency"]
    workers = upstream["workers"] if args.workers is None else args.workers
    if steps < 1 or checkpoint_frequency < 1 or workers < 0:
        parser.error("steps/checkpoint-frequency must be positive and workers nonnegative")

    if args.num_seeds is not None and args.num_seeds < 1:
        parser.error("--num-seeds must be positive")
    seed_args = ["--num_seeds", str(args.num_seeds)] if args.num_seeds is not None else ["--seed", str(0 if args.seed is None else args.seed)]
    command = [
        sys.executable, "main.py", "--phase", "2", "--data_path", str(data_path),
        "--datasets", upstream["dataset"], "--group_labels", upstream["group_labels"],
        "--algorithms", adapter.METHOD, "--out_dir", str(output_root),
        "--hparams_comb", str(upstream["hparams_comb"]), *seed_args,
        "--num_steps_override", str(steps),
        "--checkpoint_freq_override", str(checkpoint_frequency),
        "--num_workers_override", str(workers),
    ]
    if args.resume:
        command.append("--resume")
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=BASELINE, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

