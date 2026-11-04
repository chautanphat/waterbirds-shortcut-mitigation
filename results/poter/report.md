# POTER on Waterbirds: Seed 1 Artifact Audit

- **Method:** Potentials from Optimal Transport for Empirical Reweighting (POTER)
- **Paper target:** held-out **test WGA** (90.9 \pm 0.3\%), reported over three seeds
- **Run:** seed 1, ResNet-50, batch size 128, 300 epochs on one NVIDIA Tesla T4
- **Current status:** **partial reproduction; held-out test result not yet verified**
- **Tracked metric source:** [seed1_full_metrics.csv](seed1_full_metrics.csv)

## What the tracked artifacts establish

The tracked metric table contains exactly 300 rows for epochs 0–299. Recomputing the selection criterion from its four validation-group columns gives a unique maximum validation WGA at epoch 258:

| Quantity | Verified value | Evidence |
| --- | ---: | --- |
| Selected epoch | 258 (zero-based) | Maximum of `min(val_g0_acc, ..., val_g3_acc)` |
| Validation WGA | 90.10% | Minimum validation-group accuracy at epoch 258 |
| Validation average accuracy | 91.60% | `val_avg_acc`; also consistent with the validation group counts |
| Validation group accuracies | [92.50%, 90.10%, 92.50%, 92.50%] | Epoch-258 row |
| Training budget represented | 300 epochs | Epochs 0–299 are present |

The run log records that training finished on a Tesla T4 and that `best_validation.pt` and `last_training.pt` were produced in the untracked Kaggle output. Those files and the raw `log.txt` are not present in this PR, so checkpoint existence and the terminal `Training complete.` line are not independently verifiable from the repository.

## What is not established

The paper number and the tracked seed-1 number are not the same quantity:

| Paper | This PR currently contains |
| --- | --- |
| Held-out **test** WGA | **Validation** WGA |
| Mean and standard deviation over 3 seeds | One seed |
| Test overall/adjusted average and test group accuracies | No test metrics |

Therefore, the 90.10% validation WGA must not be described as reproducing the paper's 90.9% test WGA. The 0.8 percentage-point difference is descriptive only; it is not evidence of agreement within stochastic variation. Likewise, the tracked 91.60% is validation average accuracy and cannot be compared directly with a published test average.

The validation trajectory does show that the selected model performs similarly across the four validation groups. It does not by itself establish held-out test performance or a verified benchmark reproduction.

## Checkpoint selection

Selection is implemented correctly in [summarize_waterbirds.py](../../baselines/poter/scripts/summarize_waterbirds.py):

1. Read validation and test rows for all 300 epochs.
2. Select the earliest epoch attaining the maximum validation WGA.
3. Read test metrics from that exact epoch.
4. Never use a test metric to select the epoch.

For the tracked validation table, step 2 selects epoch 258. The runner also evaluates and logs the test split once per epoch, following the upstream implementation, but those test rows were not committed.

## Artifact sufficiency for seed 1

The artifacts currently in the PR are **not sufficient** to calculate seed-1 test WGA or test accuracy. In particular, [seed1_full_metrics.csv](seed1_full_metrics.csv) has no test columns, and [runs_log.csv](runs_log.csv) contains only validation results.

No retraining is needed if the original Kaggle output still exists. Recover this directory from `poter-results-backup.zip`:

```text
CUB/Clean/POTER_CUB_seed1_kaggle/
├── log.txt
├── val.csv
├── test.csv
├── config.json
├── best_validation.pt
└── last_training.pt
```

The minimum raw inputs for the existing summarizer are `log.txt`, `val.csv`, and `test.csv` in that directory layout. Then run:

```bash
python baselines/poter/scripts/summarize_waterbirds.py /path/to/poter-results --seeds 1
```

The revised summarizer writes `summary_Clean.csv` and reports:

- selected epoch and validation WGA;
- held-out test overall accuracy;
- held-out test WGA;
- held-out test adjusted average accuracy;
- all four test-group accuracies and group counts.

If `test.csv` was lost but `best_validation.pt` remains, evaluate that checkpoint once on the official Waterbirds test split. This is evaluation only, not retraining. The checkpoint records epoch 258 and the model state needed for this step. If only `last_training.pt` (epoch 299) remains, it cannot reconstruct the epoch-258 model; recover `best_validation.pt` or the original `test.csv`.

## Benchmark verdict

- **Training execution:** reported complete for seed 1; the 300-row tracked metric trajectory is internally consistent.
- **Validation-based checkpoint selection:** verified from the tracked table; epoch 258 is correct.
- **Held-out seed-1 evaluation:** pending recovery of `test.csv` or evaluation of `best_validation.pt`.
- **Three-seed paper comparison:** pending seeds 2 and 3 plus the same held-out evaluation and aggregation.
- **Repository status:** **partial reproduction**, not verified reproduction.

This classification follows [the project research protocol](../../docs/research_protocol.md), which requires overall test accuracy, every test-group accuracy and count, and WGA before a verified-reproduction claim.
