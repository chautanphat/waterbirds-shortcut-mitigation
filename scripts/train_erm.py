#!/usr/bin/env python
"""Repository-wide DPE-aligned standard ERM entry point."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods.erm.config import load_config
from src.methods.erm.train import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/erm/waterbirds_standard.yaml")
    parser.add_argument("--data-dir", help="Waterbirds directory containing metadata and archive folder")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--epochs", type=int)
    duration.add_argument("--steps", type=int)
    parser.add_argument("--print-config", action="store_true", help="Print resolved settings without training")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config.dataset.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    for name in ("device", "seed"):
        if getattr(args, name) is not None:
            setattr(config.training, name, getattr(args, name))
    if args.epochs is not None:
        config.training.epochs, config.training.steps = args.epochs, None
    if args.steps is not None:
        config.training.steps, config.training.epochs = args.steps, None
    config.validate()
    if args.print_config:
        import yaml
        print(yaml.safe_dump(config.as_dict(), sort_keys=False))
        return
    run_training(config)


if __name__ == "__main__":
    main()
