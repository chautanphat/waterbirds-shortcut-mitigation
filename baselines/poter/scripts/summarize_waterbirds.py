"""Select epochs by validation worst-group accuracy, then summarize held-out tests."""

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


GROUP_COLUMNS = [f"avg_acc_group:{group}" for group in range(4)]
COUNT_COLUMNS = [f"processed_data_count_group:{group}" for group in range(4)]
TRAIN_GROUP_COUNTS = [3498, 184, 56, 1057]
CONFIG_KEYS = [
    "Dataset", "Model", "N epochs", "Batch size", "Lr", "Weight decay",
    "Augment data", "Ot cache", "F power", "Clip cap", "Reg scale",
    "Noise type", "Noise fraction", "Group flip fractions", "Noise seed",
]


def read_log(path):
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252", errors="backslashreplace")


def read_epochs(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"epoch", "avg_acc", *GROUP_COLUMNS, *COUNT_COLUMNS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing evaluation columns: {path}")
        epochs = {}
        for row in reader:
            epoch = int(row["epoch"])
            if epoch in epochs:
                raise ValueError(f"Duplicate epoch {epoch}: {path}")
            group_acc = [float(row[column]) for column in GROUP_COLUMNS]
            group_counts = [int(float(row[column])) for column in COUNT_COLUMNS]
            overall_acc = float(row["avg_acc"])
            if not all(math.isfinite(value) and 0 <= value <= 1
                       for value in [overall_acc, *group_acc]):
                raise ValueError(f"Invalid accuracy at epoch {epoch}: {path}")
            if not all(count > 0 for count in group_counts):
                raise ValueError(f"Empty evaluation group at epoch {epoch}: {path}")
            epochs[epoch] = {
                "overall_acc": overall_acc,
                "group_acc": group_acc,
                "group_counts": group_counts,
            }
    return epochs


def summarize_run(run_dir, expected_epochs):
    validation = read_epochs(run_dir / "val.csv")
    tests = read_epochs(run_dir / "test.csv")
    required = set(range(expected_epochs))
    if set(validation) != required or set(tests) != required:
        raise ValueError(f"Expected all {expected_epochs} epochs in val/test CSVs: {run_dir}")
    # On a tie, keep the earliest epoch. Test scores never influence selection.
    best_epoch = max(sorted(validation),
                     key=lambda epoch: min(validation[epoch]["group_acc"]))
    val_groups = validation[best_epoch]["group_acc"]
    test = tests[best_epoch]
    result = {
        "epoch_zero_based": best_epoch,
        "val_wga_percent": 100 * min(val_groups),
        "test_avg_percent": 100 * test["overall_acc"],
        "test_wga_percent": 100 * min(test["group_acc"]),
        "test_adjusted_avg_percent": 100 * sum(
            accuracy * count
            for accuracy, count in zip(test["group_acc"], TRAIN_GROUP_COUNTS)
        ) / sum(TRAIN_GROUP_COUNTS),
    }
    for group, (accuracy, count) in enumerate(
            zip(test["group_acc"], test["group_counts"])):
        result[f"test_group{group}_acc_percent"] = 100 * accuracy
        result[f"test_group{group}_count"] = count
    result["run_directory"] = str(run_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path, help="The --log_dir passed to run_expt.py")
    parser.add_argument("--case", default="Clean", help="Noise bucket directory, e.g. Clean")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if args.epochs < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Epochs must be positive and seeds must be unique")

    results = {}
    configurations = []
    bucket = args.log_root / "CUB" / args.case
    for val_path in sorted(bucket.glob("*/val.csv")):
        run_dir = val_path.parent
        match = re.search(r"_seed(\d+)_", run_dir.name)
        if not match or int(match.group(1)) not in args.seeds:
            continue
        log_path = run_dir / "log.txt"
        log_text = read_log(log_path) if log_path.is_file() else ""
        if "Training complete." not in log_text:
            print(f"Skipping unfinished run: {run_dir.name}")
            continue
        seed = int(match.group(1))
        if seed in results:
            parser.error(f"Multiple completed runs for seed {seed}; use a dedicated log root")
        log_header = log_text.split("\n\n", 1)[0]
        config = dict(line.split(": ", 1) for line in log_header.splitlines() if ": " in line)
        if config.get("Dataset") != "CUB" or int(config.get("N epochs", "0")) != args.epochs:
            parser.error(f"Unexpected dataset/epoch configuration in {log_path}")
        configurations.append({key: config.get(key) for key in CONFIG_KEYS})
        try:
            results[seed] = {"seed": seed, **summarize_run(run_dir, args.epochs)}
        except (ValueError, OSError) as error:
            parser.error(str(error))
    missing = sorted(set(args.seeds) - results.keys())
    if missing:
        parser.error(f"No completed {args.epochs}-epoch run for seeds {missing} in {bucket}")
    if any(config != configurations[0] for config in configurations[1:]):
        parser.error("Runs have different settings; do not average different configurations")

    for seed in args.seeds:
        row = results[seed]
        groups = ", ".join(
            f"g{group}={row[f'test_group{group}_acc_percent']:.2f}% "
            f"(n={row[f'test_group{group}_count']})"
            for group in range(4)
        )
        print(f"Seed {seed}: epoch={row['epoch_zero_based']} (zero-based), "
              f"val WGA={row['val_wga_percent']:.2f}%, "
              f"test average={row['test_avg_percent']:.2f}%, "
              f"test WGA={row['test_wga_percent']:.2f}%, "
              f"adjusted average={row['test_adjusted_avg_percent']:.2f}%")
        print(f"  Test groups: {groups}")
    for key, label in [("test_avg_percent", "Test average"),
                       ("test_wga_percent", "Test WGA"),
                       ("test_adjusted_avg_percent", "Test adjusted average")]:
        values = [results[seed][key] for seed in args.seeds]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{label}: {statistics.mean(values):.2f}% +/- {deviation:.2f}% "
              f"(sample SD, n={len(values)})")
    output = args.log_root / f"summary_{args.case}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[args.seeds[0]]))
        writer.writeheader()
        writer.writerows(results[seed] for seed in args.seeds)
    print(f"Saved: {output}")
    print(f"Training batch size: {configurations[0].get('Batch size')}; official setting: 128")


if __name__ == "__main__":
    main()
