"""CLI with no backbone-training command or automatic backbone fallback."""
import argparse
from pathlib import Path

from .grid import DEFAULT, candidates, crossfit_candidates, load_config
from .sources import discover
from .storage import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recovery = sub.add_parser("recover")
    recovery.add_argument("--output", type=Path, required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--sha256", required=True)
    extract.add_argument("--data", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--device", default="cpu")
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--roots", type=Path, nargs="+", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    grid = sub.add_parser("grid")
    grid.add_argument("--config", type=Path, default=DEFAULT)
    grid.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--sources", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path, default=DEFAULT)
    run.add_argument("--device", default="cpu")
    run.add_argument("--source-id", nargs="+")
    # Operational staging only: all default notebook runs include crossfit.
    run.add_argument("--defer-crossfit", action="store_true")
    launch = sub.add_parser("launch")
    launch.add_argument("--sources", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--config", type=Path, default=DEFAULT)
    launch.add_argument("--devices", nargs="+", default=["cpu"])
    summary = sub.add_parser("report")
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "recover":
        from .recovery import recover
        print(f"Recovered {len(recover(args.output))} saved artifacts")
    elif args.command == "extract":
        from .extraction import extract_saved
        print(extract_saved(args.checkpoint, args.sha256, args.data, args.output, args.device))
    elif args.command == "inventory":
        sources, blockers = discover(args.roots)
        atomic_json(args.output, dict(sources=sources, blockers=blockers))
        print(f"Recovered {len(sources)} sources; {len(blockers)} blockers")
    elif args.command == "grid":
        c = load_config(args.config)
        atomic_json(args.output, dict(full_validation=candidates(c, False), split_pool=candidates(c), crossfit=crossfit_candidates(c)))
        print("Fixed configurations:", 85, 240, 35)
    elif args.command == "report":
        from .reporting import report
        print(report(args.output))
    elif args.command == "launch":
        from .launcher import launch
        print(launch(args.sources, args.output, args.devices, args.config))
    else:
        from .runner import run_study
        sources = read_json(args.sources)["sources"]
        if args.source_id:
            requested = set(args.source_id)
            if not requested <= {s["id"] for s in sources}:
                parser.error("Unknown source ID")
            sources = [s for s in sources if s["id"] in requested]
        if not sources:
            parser.error("No recoverable saved sources; backbones will not be trained")
        print(run_study(sources, args.output, load_config(args.config), args.device, not args.defer_crossfit))


if __name__ == "__main__":
    main()
