# DPE aggregation verification results

**Follow-up:** the [user-run Kaggle verification](../kaggle_verification_20261112/README.md) subsequently completed all three image checks with exact numerical reproduction. The cloud-only partial status below is retained as the history of the stopped local run.

The cloud experiments were stopped at the user's request on 2026-11-12. Remaining heavy checks have been transferred to the [Kaggle notebook](../../../notebooks/archive/dpe_verification/dpe_aggregation_verification_kaggle.ipynb). This directory records completed checks, not a working plan. Waterbirds and the original ResNet-50 backbones remain fixed; no new backbone was trained in the cloud.

## Completed evidence

- Source was audited at pinned commit `8e14249f4b3510363be06f627182e2daa9926179`; locked method and shared extraction source hashes matched.
- Initial repository tests: 173 passed and nine optional checks skipped. Three current-notebook integration tests subsequently passed. Thirteen additional numerical/behavioral contracts passed.
- Fresh training reproduced all test decisions for 18 logistic/plain/dropout MLP fits in each of two CPU PyTorch environments. Both original dropout MLP settings retain the +0.4730 percentage-point mean test WGA gain over matched uniform averaging.
- All 45 prototype heads reproduced their selected epochs and had zero changed head test decisions on original features.
- Across ten combiner seeds per original head ensemble, mean dropout gains over uniform were +0.4464 pp (weight decay 0.01) and +0.4449 pp (0.001). Plain MLP means were slightly higher. These are conditional repetitions on three frozen ensembles, not 30 independently trained backbones.
- The original dropout-versus-uniform paired, group-stratified bootstrap interval is approximately [0.1963, 0.9089] pp. Against plain MLP at decay 0.01 it is [-0.4673, 0.2069] pp. These pointwise intervals describe test-example uncertainty conditional on the fitted models, not training-seed uncertainty.

## Image preprocessing finding

Pillow 10.3 and 13.0 decoded all 1,199 validation JPEGs identically but produced different resized/normalized inputs for 119 images. Diagnostic tests isolated the difference to Pillow rather than the tested PyTorch versions. Pillow 13.0.0 reproduced the original features closely. The original run did not record its Pillow version; matching version 13 is an empirical compatibility finding, not a recovered environment fact.

With Pillow 13, full image inference for source seed 0 reproduced every individual head decision and all five checked uniform/plain/dropout configuration decisions on 5,794 test images. Seeds 1 and 2 were not completed before cloud computation was stopped. See [seed-0 image evidence](image_replay_seed0.json). The notebook pins Pillow 13 in a separate worker environment and records versions.

## Interpretation and additional controls

The saved gain over uniform is reproducible. A consistent dropout-specific advantage over plain MLP is not established. The tested affine-calibration and training-fitted threshold controls did not improve mean WGA over uniform. An additional uniform-head control using all 1,199 validation examples reached 89.7715% mean WGA, narrowing the original dropout advantage to +0.2211 pp. That control also performs two head-training updates per epoch rather than one, so it is a data-allocation/recipe control, not an equal-update ablation.

Each configuration is independently evaluated; no candidate-selection split is required. Original and supplementary results remain separate. Additional independent backbone/head seeds have not yet been run.

CSV accuracy values are fractions in [0,1]; changes are percentage points. The CSV files retain replay discrepancies, conditional combiner repetitions, bootstrap intervals, calibration controls and the full-validation uniform control. [Status record](status.json) identifies completed and uncompleted work.
