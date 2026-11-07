#!/usr/bin/env python
"""Evaluate an explicitly selected ERM checkpoint on validation or held-out test."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.loaders import evaluation_loader
from src.datasets.waterbirds import Waterbirds
from src.evaluation.classification import evaluate
from src.models.resnet import load_resnet_checkpoint
from src.utils.reproducibility import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional JSON report (must not already exist)")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    dataset = Waterbirds(args.data_dir, args.split)
    model = load_resnet_checkpoint(args.checkpoint, device=args.device)
    metrics = evaluate(model, evaluation_loader(dataset, args.batch_size, args.workers), args.device)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
              "metadata_sha256": file_sha256(dataset.metadata_path), "split": args.split, "metrics": metrics}
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
