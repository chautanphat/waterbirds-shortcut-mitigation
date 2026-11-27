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

## ERM* + DPE neural aggregation, two GPUs

[dpe_ermstar_neural_two_gpu_kaggle.ipynb](dpe_ermstar_neural_two_gpu_kaggle.ipynb)
uses the upstream **400-epoch** enhanced recipe, with a full-validation DPE uniform
reference and a separate-pool neural arm with matched uniform. Enable Internet and
select **GPU T4 ×2** to run seeds0/1 concurrently, one GPU each. Full-output resume
is supported. The notebook is a short launcher importing repository modules; it writes no Python
source files. It checks out `experiment/dpe-ermstar-neural` by default. [Recipe audit, code reuse and tests](../src/methods/dpe_ermstar/README.md).

## Corrected nested DPE selection

The self-contained corrected notebook is in
[the corrected experiment package](../experiments/dpe_nested_corrected/README.md).
It fixes inner-CV nesting and threshold controls. Use CPU, enable Internet and
Save & Run All. The corrected Waterbirds rerun is pending; synthetic tests passed.
[Audit and interpretation](../docs/dpe_nested_audit_and_rerun.md).

## Direction 2 · Representation Repair (`rep_repair`) Kaggle Notebooks

All Direction 2 notebooks clone branch `spurious_correlation_mechanisms` cleanly on Kaggle (`Internet: On`) and run the corresponding script under `scripts/analysis/`:

| Step | Notebook | Hardware | Approx Runtime | Public Checkpoint Inputs | Output Archive |
|---|---|---|---|---|---|
| **Phase A** | [rep_repair_phase_a_kaggle.ipynb](rep_repair_phase_a_kaggle.ipynb) | CPU / 1× T4 | ~5 min | `noerula/dpe-mlp-dropout-verification`, `noerula/dpe-exploration` | `results/rep_repair/phase_a_20261123/` |
| **Phase B · Controls** | [rep_repair_phase_b_controls_kaggle.ipynb](rep_repair_phase_b_controls_kaggle.ipynb) | 1× T4 | ~25 min | Same ResNet-50 backbones | `results/rep_repair/phase_b_controls_20261123/` |
| **Phase B · Counterfactuals** | [rep_repair_phase_b_counterfactuals_kaggle.ipynb](rep_repair_phase_b_counterfactuals_kaggle.ipynb) | 1× T4 | ~15 min | Same ResNet-50 backbones + CUB masks | `results/rep_repair/phase_b_counterfactuals_20261123/` |
| **Phase C · Step 5** | [rep_repair_phase_c_patching_kaggle.ipynb](rep_repair_phase_c_patching_kaggle.ipynb) | 1× T4 | ~25 min | Same ResNet-50 backbones + CUB masks | `results/rep_repair/phase_c_step5_20261123/` |
| **Phase C · Step 6** | [rep_repair_phase_c_step6_kaggle.ipynb](rep_repair_phase_c_step6_kaggle.ipynb) | 1× T4 | ~30 min | Same ResNet-50 backbones | `results/rep_repair/phase_c_step6_20261124/` |
| **Phase C · Step 6b** | [rep_repair_phase_c_step6b_kaggle.ipynb](rep_repair_phase_c_step6b_kaggle.ipynb) | 1× T4 | ~30 min | Same ResNet-50 backbones | `results/rep_repair/phase_c_step6b_20261124/` |
| **Phase C · Step 7a** | [rep_repair_step7a_vit_kaggle.ipynb](rep_repair_step7a_vit_kaggle.ipynb) | 1× T4 | ~45 min | Trains & saves `ViT-B/16` seeds 0–2 | `results/rep_repair/step7a_vit_20261124/` |
| **Phase C · Step 7b** | [rep_repair_step7b_spawrious_kaggle.ipynb](rep_repair_step7b_spawrious_kaggle.ipynb) | 2× T4 | ~45 min | Trains & saves Spawrious ResNet-50 seeds 0–2 | `results/rep_repair/step7b_spawrious_20261124/` |
| **Phase C · Step 7c-1** | [rep_repair_step7c_waterbirds_test_kaggle.ipynb](rep_repair_step7c_waterbirds_test_kaggle.ipynb) | 2× T4 | ~12 min | Waterbirds ResNet (`std` + `star`) + `ViT-B/16` | `results/rep_repair/step7c_waterbirds_test_20261124/` |
| **Phase C · Step 7c-2** | [rep_repair_step7c_spawrious_test_kaggle.ipynb](rep_repair_step7c_spawrious_test_kaggle.ipynb) | 2× T4 | ~50 min | `noerula/rep-repair-step7b-spawrious-kaggle` | `results/rep_repair/step7c_spawrious_test_20261124/` |
