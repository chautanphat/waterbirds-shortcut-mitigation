#!/usr/bin/env python
"""Validate predictions.csv v1 files and optionally their complete metadata split."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.prediction_protocol import validate_prediction_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or directories to search for predictions.csv")
    parser.add_argument("--metadata", type=Path, help="Reference metadata; verifies every image, truth and split")
    args = parser.parse_args(argv)
    files = []
    for path in args.paths:
        files.extend(sorted(path.rglob("predictions.csv")) if path.is_dir() else [path])
    if not files:
        parser.error("No predictions.csv files found")
    failures = False
    for path in dict.fromkeys(files):
        try:
            report = validate_prediction_csv(path, args.metadata)
            print(json.dumps(dict(file=str(path), valid=True, **report)))
        except (OSError, UnicodeError, ValueError) as error:
            failures = True
            print(json.dumps(dict(file=str(path), valid=False, error=str(error))))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
