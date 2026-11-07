#!/usr/bin/env python
"""Train shared ERM and export matching inputs for DPE head reweighting."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.methods.dpe_head_reweighting.stage0 import backbone_config, export_inputs, train_or_reuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "export", "all"))
    parser.add_argument("--track", choices=("standard", "enhanced"), default="standard")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--data-dir", default="data/waterbirds")
    parser.add_argument("--backbone-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Raw-feature and provenance bundle directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    backbone = args.backbone_dir or Path(f"checkpoints/waterbirds_stage0_{args.track}_erm/seed{args.seed}")
    output = args.output_dir or Path(f"outputs/dpe_features/{args.track}/seed{args.seed}")
    c = backbone_config(args.data_dir, backbone, args.track, args.seed, args.device, args.workers)
    if args.action in ("train", "all"):
        train_or_reuse(c)
    if args.action in ("export", "all"):
        print(json.dumps(export_inputs(c, output), indent=2))


if __name__ == "__main__":
    main()
