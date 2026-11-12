"""Run the project-owned DPE H2 experiment from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods.dpe_head_reweighting.experiment import main

if __name__ == '__main__':
    main()
