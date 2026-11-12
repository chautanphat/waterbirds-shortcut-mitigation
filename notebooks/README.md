# Notebooks

| Notebook | Purpose |
| --- | --- |
| [dpe_head_aggregation.ipynb](dpe_head_aggregation.ipynb) | **Current DPE weighting comparison:** all 30 global-rule/neural configurations, fitted on a dedicated combiner-training split and evaluated independently across three seeds. |
| [dpe_waterbirds_end_to_end.ipynb](dpe_waterbirds_end_to_end.ipynb) | Original DPE Waterbirds workflow. |
| [poter_waterbirds_kaggle.ipynb](poter_waterbirds_kaggle.ipynb) | POTER Waterbirds workflow. |

For the current weighting comparison, select a GPU runtime, enable Internet on
Kaggle, and Run All. No attachments or code edits are required.
[Detailed instructions](../src/methods/dpe_head_reweighting/HEAD_AGGREGATION.md).

`archive/dpe_head_reweighting.ipynb` and `archive/dpe_weighting_rules.ipynb`
preserve the earlier historical power-only and global-rule selection experiments. They are historical
reproduction interfaces, superseded by the current head-aggregation notebook.
Recorded reports retain the filenames used at the executed Git commits.

The completed [five-seed scientific report](../results/dpe/five_seed_verification_20261112/README.md) and its [lightweight audit](../scripts/analysis/audit_dpe_five_seed.py) do not require a GPU run. Historical verification artifacts are in the DPE supporting-evidence archive linked from the report.
