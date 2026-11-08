"""Evaluate one XRM checkpoint and report every Waterbirds test group."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "xrm"
GROUP_NAMES = ("landbird_land", "landbird_water", "waterbird_land", "waterbird_water")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("erm", "groupdro"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / args.method / "waterbirds_xrm.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    spec = config["upstream"]
    sys.path.insert(0, str(BASELINE))
    from algorithms import get_algorithm
    from datasets import get_loaders
    from hparams_registry import get_hparams
    from networks import get_network, get_optim

    hparams = get_hparams({
        "phase": 2, "data_path": str(args.data_path.resolve()),
        "dataset": spec["dataset"], "group_labels": spec["group_labels"],
        "algorithm": spec["algorithm"], "hparams_comb": spec["hparams_comb"],
        "seed": 0, "out_dir": str((ROOT / "outputs" / "evaluate").resolve()),
        "quick_run": False, "resume": False, "adam": False,
    })
    loaders = get_loaders(hparams)
    network = get_network(hparams)
    optimizer = get_optim(hparams, network)
    algorithm = get_algorithm(hparams, network, optimizer)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=hparams["device"], weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=hparams["device"])
    algorithm.net.load_state_dict(checkpoint["net_state_dict"])
    ys, logits, attributes = algorithm.evaluate(loaders["te"])
    predictions = logits.argmax(1)
    groups = hparams["num_m"] * ys + attributes
    rows = []
    for group_id, name in enumerate(GROUP_NAMES):
        mask = groups == group_id
        rows.append({"group": name, "count": int(mask.sum()), "accuracy": float((predictions[mask] == ys[mask]).float().mean()) if mask.any() else None})
    result = {"method": args.method, "checkpoint": str(args.checkpoint), "overall_accuracy": float((predictions == ys).float().mean()), "worst_group_accuracy": min(row["accuracy"] for row in rows if row["accuracy"] is not None), "groups": rows}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
