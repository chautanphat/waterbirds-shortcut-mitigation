# DPE Head Aggregation

Compare global weighting rules and neural combiners over the same 15 frozen
Diverse Prototypical Ensemble heads on Waterbirds.

Use [dpe_head_aggregation.ipynb](../../../notebooks/dpe_head_aggregation.ipynb)
as the single active entry point. Select a GPU runtime, enable Internet on
Kaggle and Run All. No attachments or code edits are required.

| Data | Role |
| --- | --- |
| Official training set: 4,795 images | Train the ResNet-50 backbone. |
| Head-training split: 720 validation images | Train the 15 prototype heads and choose their epochs. |
| Combiner-training split: 479 validation images | Fit every global weighting rule and neural combiner. |
| Official test set: 5,794 images | Evaluate every frozen configuration independently. |

The two training splits use direct group-stratified 60/40 allocation with fixed
split seed 0. Every configuration shares the same heads, fitting examples and
test images within a training seed. Report all 30 configurations across seeds
0, 1 and 2; do not choose hyperparameters using test results.

See [HEAD_AGGREGATION.md](HEAD_AGGREGATION.md) for the candidate grid, training
recipe, configuration, CLI/API usage, outputs and verification.

This is an exploratory project adaptation, not an exact paper reproduction.
Historical results and their original protocols remain in [reports/](reports/h2.md);
[archived usage documentation](archive/README.md) and `notebooks/archive/` are
for reproducing those experiments.
