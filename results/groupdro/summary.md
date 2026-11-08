# XRM Waterbirds GroupDRO

Generated from archived run logs by [`scripts/summarize_xrm_baselines.py`](../../scripts/summarize_xrm_baselines.py).
Config: [`configs/groupdro/waterbirds_xrm.yaml`](../../configs/groupdro/waterbirds_xrm.yaml). Source commit: `5e90815250aedea546b9e17ffd5d7e0a7c34c20f`.
Raw logs: [`results/archive/xrm_waterbirds_initial/groupdro/`](../archive/xrm_waterbirds_initial/groupdro/). Per-seed trace: [`xrm_waterbirds_runs.csv`](xrm_waterbirds_runs.csv).

Selected hyperparameter combination: 8; checkpoint chosen by the last occurrence of maximum validation WGA, matching upstream `utils.py`. Completed: 6/10 requested seeds.

| Metric | Mean ± sample SD |
|---|---:|
| Validation worst-group accuracy | 85.59% ± 0.30% |
| Test average accuracy | 90.44% ± 0.29% |
| Test worst-group accuracy | 86.70% ± 0.30% |

| Seed | Status | Last step | Selected step | Test accuracy | Test WGA |
|---:|---|---:|---:|---:|---:|
| 0 | complete | 5000/5000 | 500 | 90.97% | 86.60% |
| 1 | complete | 5000/5000 | 800 | 90.21% | 86.45% |
| 2 | complete | 5000/5000 | 600 | 90.49% | 86.92% |
| 3 | complete | 5000/5000 | 700 | 90.20% | 87.05% |
| 4 | complete | 5000/5000 | 800 | 90.47% | 86.29% |
| 5 | complete | 5000/5000 | 850 | 90.32% | 86.92% |
| 6 | incomplete | 3150/5000 | — | — | — |
| 7 | missing | —/5000 | — | — | — |
| 8 | missing | —/5000 | — | — | — |
| 9 | missing | —/5000 | — | — | — |

## Interpretation

These are historical results from the selected upstream configuration, not a new run. Incomplete and missing seeds are listed and excluded from the aggregate. The source logs contain overall accuracy and worst-group accuracy, but no per-group accuracies; per-group accuracy is therefore not inferred here. The source report's combined multi-seed table included an incomplete GroupDRO run, so this artifact recomputes aggregates from each run's JSON history.

The handoff records Windows CPU execution with Python 3.9, PyTorch 1.13.0, torchvision 0.14.0, and an 11th Gen Intel Core i7-11390H. Package versions were not captured in each individual raw run log.
