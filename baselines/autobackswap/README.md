# AutoBackSwap (reproduction on Waterbirds)

Upstream: https://github.com/cesro7/spur-corr, pinned commit `b060532c2632039ffc2918828780d2b696763af6` (paper arXiv 2606.32018). The upstream source is not copied here; clone it at that commit to run.

- `patches/augment_waterbirds_chunk.patch`: memory fix for the generate step (chunked augmentation) applied on top of the pinned commit.
- `analysis/`: scripts that dump per-example predictions (`dump_all_predictions.py`) and build the error viewer. They do not modify upstream code.
- Results, run log and comparison with the paper: `results/autobackswap/` (`report.md`, `runs_log.csv`, `runs_log.md`, per-example `predictions/`).

Runs were done on Kaggle (2x T4, fp32), seeds 0 to 2, without the README default for cuda optimizations; see `runs_log.csv` for each command. Misclassified test images are not in Git; they are shared separately.
