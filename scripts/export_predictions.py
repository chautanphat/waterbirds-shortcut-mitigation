#!/usr/bin/env python
"""Create failure-viewer CSVs without training or selecting models."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="method", required=True)
    for method in ("erm", "poter", "autobackswap", "dpe-probabilities", "dpe-heads"):
        p = commands.add_parser(method)
        p.add_argument("--metadata", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path, help="New run directory, not a filename")
        p.add_argument("--run-label", required=True)
        p.add_argument("--seed", required=True, type=int)
        p.add_argument("--split", choices=("val", "test"), default="test")
        if method in ("erm", "poter", "autobackswap"):
            p.add_argument("--checkpoint", required=True, type=Path)
            p.add_argument("--data-dir", required=True, type=Path)
            if method == "poter":
                p.add_argument("--trusted-pickle", action="store_true", help="Allow a trusted legacy full-model/training checkpoint")
            if method == "autobackswap":
                p.add_argument("--config", required=True, type=Path, help="Saved config.yaml for this run")
        if method != "dpe-probabilities":
            p.add_argument("--device", default="cpu")
            p.add_argument("--batch-size", type=int, default=64)
        if method == "dpe-probabilities":
            p.add_argument("--artifact", required=True, type=Path, help="NPZ with ids and probabilities")
        if method == "dpe-heads":
            p.add_argument("--heads", required=True, type=Path)
            p.add_argument("--features", required=True, type=Path, help="Identified raw feature NPZ")
    p = commands.add_parser("recover-autobackswap", help="Recover 10 saved CSVs from the cached Git ref, with no network")
    p.add_argument("--ref", default="origin/method/autobackswap")
    p.add_argument("--metadata", type=Path, default=ROOT / "baselines/autobackswap/data/waterbird_complete95_forest2water2/metadata.csv")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/failure_cases")
    p = commands.add_parser("package-dpe-features", help="Package a notebook NPY using its verified feature manifest")
    for name in ("features", "manifest", "metadata", "backbone", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--split", choices=("val", "test"), default="test")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.method == "recover-autobackswap":
            from src.evaluation.prediction_export import recover_autobackswap
            reports = recover_autobackswap(ROOT, args.metadata, args.output_root, args.ref)
            print(json.dumps([dict(run=r["run_label"], **r["metrics"]) for r in reports], indent=2))
            return
        if args.method == "package-dpe-features":
            from src.evaluation.dpe_export import package_notebook_features
            package_notebook_features(args.output, args.features, args.manifest, args.metadata, args.backbone, split=args.split)
            print(f"Saved identified raw features: {args.output}")
            return
        common = dict(run_label=args.run_label, seed=args.seed, split=args.split)
        if args.method == "dpe-probabilities":
            from src.evaluation.dpe_export import export_dpe_probabilities
            report = export_dpe_probabilities(args.artifact, args.metadata, args.output, **common)
        elif args.method == "dpe-heads":
            from src.evaluation.dpe_export import export_dpe_heads
            report = export_dpe_heads(args.heads, args.features, args.metadata, args.output,
                                      device=args.device, batch_size=args.batch_size, **common)
        else:
            from src.evaluation.prediction_export import export_checkpoint
            report = export_checkpoint(args.method, args.checkpoint, args.metadata, args.data_dir,
                args.output, device=args.device, batch_size=args.batch_size,
                config_path=getattr(args, "config", None), trusted_pickle=getattr(args, "trusted_pickle", False), **common)
        print(f"Saved {args.output / 'predictions.csv'}")
        print(json.dumps(report["metrics"], indent=2))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
