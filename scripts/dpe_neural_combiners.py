"""Compare B-fitted neural combiners over cached DPE predictions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods.dpe_head_reweighting.neural_suite import main

if __name__ == '__main__':
    main()
