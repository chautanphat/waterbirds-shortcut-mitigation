# XRM Waterbirds ERM

Generated from archived run logs by [`scripts/summarize_xrm_baselines.py`](../../scripts/summarize_xrm_baselines.py).
Config: [`configs/erm/waterbirds_xrm.yaml`](../../configs/erm/waterbirds_xrm.yaml). Source commit: `5e90815250aedea546b9e17ffd5d7e0a7c34c20f`.
Raw logs: [`results/archive/xrm_waterbirds_initial/erm/`](../archive/xrm_waterbirds_initial/erm/). Per-seed trace: [`xrm_waterbirds_runs.csv`](xrm_waterbirds_runs.csv).

Selected hyperparameter combination: 4; checkpoint chosen by the last occurrence of maximum validation WGA, matching upstream `utils.py`. Completed: 7/10 requested seeds.

| Metric | Mean ± sample SD |
|---|---:|
| Validation worst-group accuracy | 64.12% ± 0.72% |
| Test average accuracy | 83.79% ± 0.35% |
| Test worst-group accuracy | 66.73% ± 0.81% |

| Seed | Status | Last step | Selected step | Test accuracy | Test WGA |
|---:|---|---:|---:|---:|---:|
| 0 | complete | 5000/5000 | 2700 | 83.19% | 67.29% |
| 1 | complete | 5000/5000 | 3850 | 83.69% | 67.60% |
| 2 | complete | 5000/5000 | 4200 | 84.26% | 65.73% |
| 3 | complete | 5000/5000 | 5000 | 84.16% | 66.51% |
| 4 | complete | 5000/5000 | 4550 | 83.78% | 67.13% |
| 5 | complete | 5000/5000 | 3700 | 83.71% | 67.29% |
| 6 | complete | 5000/5000 | 1900 | 83.72% | 65.58% |
| 7 | incomplete | 0/5000 | — | — | — |
| 8 | missing | —/5000 | — | — | — |
| 9 | missing | —/5000 | — | — | — |

## Interpretation

These are historical results from the selected upstream configuration, not a new run. Incomplete and missing seeds are listed and excluded from the aggregate. The source logs contain overall accuracy and worst-group accuracy, but no per-group accuracies; per-group accuracy is therefore not inferred here. The source report's combined multi-seed table included an incomplete GroupDRO run, so this artifact recomputes aggregates from each run's JSON history.

The handoff records Windows CPU execution with Python 3.9, PyTorch 1.13.0, torchvision 0.14.0, and an 11th Gen Intel Core i7-11390H. Package versions were not captured in each individual raw run log.
