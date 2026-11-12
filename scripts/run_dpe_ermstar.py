"""Run the ERM* + DPE neural comparison on two GPUs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods.dpe_ermstar.run import DEFAULT, prepare_and_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--hours', type=float, default=10)
    args = parser.parse_args()
    prepare_and_launch(args.work, seeds=args.seeds, data_dir=args.data,
                       resume_from=args.resume_from, config_path=args.config,
                       timeout_hours=args.hours)


if __name__ == '__main__':
    main()
