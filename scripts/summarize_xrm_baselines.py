"""Aggregate XRM logs into traceable, per-method reports; incomplete runs are excluded."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_run(path: Path, method: str, requested_seed: int, hparams_comb: int, expected_steps: int) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty result log: {path}")
    config = json.loads(lines[0])
    if config.get("algorithm_name") != method or int(config.get("seed", -1)) != requested_seed or int(config.get("hparams_comb", -1)) != hparams_comb:
        raise ValueError(f"Run identity mismatch in {path}")
    records = [json.loads(line) for line in lines[1:] if line.strip()]
    records = [row for row in records if int(row.get("step", 0)) > 0]
    final_step = max((int(row["step"]) for row in records), default=0)
    complete = final_step >= expected_steps - 1
    row = None
    if complete:
        best = max(float(record["va_wga"]) for record in records)
        row = next(record for record in reversed(records) if float(record["va_wga"]) == best)
    return {
        "method": method, "seed": requested_seed, "status": "complete" if complete else "incomplete",
        "last_step": final_step, "expected_steps": expected_steps,
        "selected_step": int(row["step"]) if row else None,
        "va_wga": row.get("va_wga") if row else None,
        "te_wga": row.get("te_wga") if row else None,
        "te_avg_acc": row.get("te_avg_acc") if row else None,
        "log": path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.as_posix(),
    }


def summarize(config_path: Path, archive_root: Path, results_root: Path) -> tuple[Path, list[dict]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    method_key = config["method"]
    method = config["upstream"]["algorithm"]
    combination = int(config["upstream"]["hparams_comb"])
    expected_steps = int(config["upstream"]["steps"])
    seeds = config["selection"]["requested_seeds"]
    method_archive = archive_root / method_key
    rows = []
    for seed in seeds:
        path = method_archive / f"results_hpcomb_{combination}_seed{seed}.json"
        if not path.is_file():
            rows.append({"method": method, "seed": seed, "status": "missing", "last_step": None, "expected_steps": expected_steps, "selected_step": None, "va_wga": None, "te_wga": None, "te_avg_acc": None, "log": path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.as_posix()})
            continue
        rows.append(read_run(path, method, int(seed), combination, expected_steps))

    complete = [row for row in rows if row["status"] == "complete"]
    metrics = ("va_wga", "te_avg_acc", "te_wga")
    summary = {key: {"mean": statistics.mean(float(row[key]) for row in complete), "sd": statistics.stdev(float(row[key]) for row in complete) if len(complete) > 1 else 0.0} for key in metrics} if complete else {}
    result_dir = results_root / method_key
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "xrm_waterbirds_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "status", "last_step", "expected_steps", "selected_step", "va_wga", "te_avg_acc", "te_wga", "log"])
        writer.writeheader()
        writer.writerows(rows)

    try:
        config_rel = config_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        config_rel = config_path.as_posix()
    lines = [
        f"# XRM Waterbirds {method}", "",
        f"Generated from archived run logs by [`scripts/summarize_xrm_baselines.py`](../../scripts/summarize_xrm_baselines.py).",
        f"Config: [`{config_rel}`](../../{config_rel}). Source commit: `{config['source']['commit']}`.",
        f"Raw logs: [`results/archive/xrm_waterbirds_initial/{method_key}/`](../archive/xrm_waterbirds_initial/{method_key}/). Per-seed trace: [`xrm_waterbirds_runs.csv`](xrm_waterbirds_runs.csv).", "",
        f"Selected hyperparameter combination: {combination}; checkpoint chosen by the last occurrence of maximum validation WGA, matching upstream `utils.py`. Completed: {len(complete)}/{len(seeds)} requested seeds.", "",
        "| Metric | Mean ± sample SD |", "|---|---:|",
    ]
    labels = {"va_wga": "Validation worst-group accuracy", "te_avg_acc": "Test average accuracy", "te_wga": "Test worst-group accuracy"}
    for key in metrics:
        value = summary.get(key)
        lines.append(f"| {labels[key]} | {100*value['mean']:.2f}% ± {100*value['sd']:.2f}% |" if value else f"| {labels[key]} | unavailable |")
    lines.extend(["", "| Seed | Status | Last step | Selected step | Test accuracy | Test WGA |", "|---:|---|---:|---:|---:|---:|"])
    for row in rows:
        avg = f"{100*row['te_avg_acc']:.2f}%" if row["te_avg_acc"] is not None else "—"
        wga = f"{100*row['te_wga']:.2f}%" if row["te_wga"] is not None else "—"
        selected = str(row["selected_step"]) if row["selected_step"] is not None else "—"
        last = str(row["last_step"]) if row["last_step"] is not None else "—"
        lines.append(f"| {row['seed']} | {row['status']} | {last}/{expected_steps-1} | {selected} | {avg} | {wga} |")
    lines.extend([
        "", "## Interpretation", "",
        "These are historical results from the selected upstream configuration, not a new run. Incomplete and missing seeds are listed and excluded from the aggregate. The source logs contain overall accuracy and worst-group accuracy, but no per-group accuracies; per-group accuracy is therefore not inferred here. The source report's combined multi-seed table included an incomplete GroupDRO run, so this artifact recomputes aggregates from each run's JSON history.",
        "", "The handoff records Windows CPU execution with Python 3.9, PyTorch 1.13.0, torchvision 0.14.0, and an 11th Gen Intel Core i7-11390H. Package versions were not captured in each individual raw run log.", "",
    ])
    report_path = result_dir / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("erm", "groupdro", "all"), default="all")
    parser.add_argument("--configs-root", type=Path, default=ROOT / "configs")
    parser.add_argument("--archive-root", type=Path, default=ROOT / "results" / "archive" / "xrm_waterbirds_initial")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    methods = ("erm", "groupdro") if args.method == "all" else (args.method,)
    for method in methods:
        path, rows = summarize(args.configs_root / method / "waterbirds_xrm.yaml", args.archive_root, args.results_root)
        print(f"{path}: {sum(row['status'] == 'complete' for row in rows)}/{len(rows)} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
