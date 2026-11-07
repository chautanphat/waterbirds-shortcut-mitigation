#!/usr/bin/env python
"""Export reusable, ordered ERM features without importing a baseline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.loaders import evaluation_loader
from src.datasets.waterbirds import Waterbirds
from src.evaluation.features import save_features
from src.models.resnet import load_resnet_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to checkpoint directory (DPE cache lookup)")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    model = load_resnet_checkpoint(args.checkpoint, device=args.device)
    for split in args.splits:
        dataset = Waterbirds(args.data_dir, split)
        loader = evaluation_loader(dataset, args.batch_size, args.workers)
        print(json.dumps(save_features(model, loader, args.checkpoint,
                                       args.output_dir or args.checkpoint.parent, args.device)), flush=True)


if __name__ == "__main__":
    main()
