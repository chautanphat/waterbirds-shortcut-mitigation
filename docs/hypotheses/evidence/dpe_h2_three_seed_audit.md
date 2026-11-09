# Supporting audit: three-seed DPE head weighting

The [H2 hypothesis summary](../dpe_h2.md) is the primary narrative. This audit retains detailed methods, diagnostics and provenance.

**Status:** completed exploratory adaptation, standard track, seeds 0/1/2.
**Documentation date:** 2026-11-09; the execution calendar date is not recorded.

The C-selected procedure produced **zero test-WGA improvements, two ties and one
decrease** relative to the matched uniform ensemble. Uniform achieved
**89.5196 ± 0.6538% WGA**; overall C selection achieved **89.4752 ± 0.6339%**.
The paired change was **−0.0443 ± 0.0768 percentage points (pp)**.
These observations do not support H2 for the tested global weighting rules and
selection protocol. They do not establish a general disadvantage of weighting.

## Experiment history

This record consolidates the DPE/H2 experiments discussed in this investigation.
Earlier records remain the source for their full settings, group metrics and ablations.

| Experiment | Evidence and result | Relationship to this study |
| --- | --- | --- |
| [Historical standard DPE](../../experiment_log.md#exp-dpe-waterbirds-standard-002--2026-11-02) | Seed 0: reported test accuracy/WGA 91.40/89.00%. | Different validation-data budget; not a matched H2 control. |
| [Historical enhanced DPE](../../experiment_log.md#exp-dpe-waterbirds-ermstar-001--2026-11-02) | Seed 0: reported test accuracy/WGA 93.30/92.70%. | Augmented 100-epoch backbone; not the upstream 400-epoch ERM* recipe. |
| [Original H2 power grid](dpe_h2_power_seed0.md) | Seed 0: uniform selected; test accuracy/WGA 91.2323/89.7561%. All 13 candidates give identical C labels. | Establishes the A-trained matched uniform control. |
| [Expanded weighting grid](dpe_h2_rules_seed0.md) | Same seed-0 heads: top-10 selected; test accuracy/WGA 91.1978/89.6231%. | Reuses seed 0; not another independent training replicate. |
| [Failed multi-seed attempt](../../experiment_log.md#exp-dpe-h2-rules-waterbirds-standard-seeds012-002--2026-11-09) | Seed 1 finished training, then joint fitting stopped with an SLSQP line-search error. No final seed-1 test comparison. | Software failure, not a scientific negative result. |
| This study, `…seeds012-003` | Seeds 0/1/2 completed with the same 20-candidate grid. | Seed 0 reused; seeds 1/2 trained. All three comparisons re-evaluated. |

Historical DPE scores are retained as reported, without a new artifact audit.
Their ERM-only evaluations have a documented raw/standardized-feature mismatch
and require correction. They must not be used to estimate the effect of H2.

## Question and controlled procedure

**H2:** Do validation-derived weights improve worst-group accuracy over equal
probability averaging of the same 15 frozen DPE heads?

The official Waterbirds train/validation/test sizes are 4,795/1,199/5,794.
Group annotations are used for validation partitioning, head fitting and selection.
Within each seed, all weighting candidates share one backbone, features and heads.

Validation A/B/C sizes are 720/241/238: A fits heads and selects their epochs,
B fits weights, and C selects aggregation. Each waterbird group has only 27 B
examples and 26 C examples. All three training seeds share partition seed 0.
Backbones use ImageNet V2 initialization and 100 epochs without augmentation;
the final epoch supplies frozen features. Each of 15 heads trains for 20 epochs
on 320 balanced A examples. Optimizer/loss settings are preserved in the
[per-seed configurations](../../../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/run.json)
and [method README](../../../src/methods/dpe_head_reweighting/README.md).

The fixed grid contains uniform, 12 power rules, softmax β=10/30, top-k=10/5 and
joint λ=1/0.1/0.01. Joint fitting minimizes worst B-group negative log likelihood
(NLL), plus λ times squared distance from uniform, over simplex weights.
Probability floor is 1e-12; solver tolerances and limits are in the
[recorded grid](../../../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/config.yaml).

C selects by WGA, then overall accuracy; complete ties prefer uniform and then
declaration order. Every family includes uniform. Weights are not refitted on
B+C. Only frozen family/overall winners and uniform are evaluated on test;
unselected candidates are not swept on test.

## Backbone validation diagnostics

| Seed | Final-epoch validation accuracy (%) | Validation WGA (%) |
| --- | ---: | ---: |
| 0, reused | 86.4887 | 60.1504 |
| 1 | 86.4887 | 62.4060 |
| 2 | 86.5721 | 59.3985 |

Seed 0 comes from the [earlier backbone log](../../../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/backbone_epochs.csv);
seeds 1/2 are in [backbone_epochs.csv](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/backbone_epochs.csv). These are backbone
validation scores, not DPE or ERM-only test scores; differences across those
stages/splits are not matched test improvements.

## Test results

| Seed | Uniform accuracy (%) | Uniform WGA (%) | Overall C choice | Selected accuracy (%) | Selected WGA (%) | Paired WGA change (pp) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 91.2323 | 89.7561 | Top-10 | 91.1978 | 89.6231 | −0.1330 |
| 1 | 91.6120 | 90.0222 | Uniform | 91.6120 | 90.0222 | 0.0000 |
| 2 | 90.9562 | 88.7805 | Uniform | 90.9562 | 88.7805 | 0.0000 |

All ± values below are **sample SD across three training seeds**, not confidence
intervals. Paired changes are computed within seed before aggregation.

| C-selected procedure | Test accuracy mean ± SD (%) | Test WGA mean ± SD (%) | Paired WGA change mean ± SD (pp) |
| --- | ---: | ---: | ---: |
| Uniform | 91.2668 ± 0.3293 | 89.5196 ± 0.6538 | 0.0000 ± 0.0000 |
| Power | 91.2668 ± 0.3293 | 89.5196 ± 0.6538 | 0.0000 ± 0.0000 |
| Softmax | 91.2668 ± 0.3293 | 89.5196 ± 0.6538 | 0.0000 ± 0.0000 |
| Top-k | 91.2553 ± 0.3317 | 89.4752 ± 0.6339 | −0.0443 ± 0.0768 |
| Joint | 91.2668 ± 0.3293 | 89.5196 ± 0.6538 | 0.0000 ± 0.0000 |
| Overall selection | 91.2553 ± 0.3317 | 89.4752 ± 0.6339 | −0.0443 ± 0.0768 |

Power, softmax and joint select uniform in all three seeds; top-k selects uniform
in seeds 1/2. Their equal scores describe the selected procedures, not the test
performance of their unselected nonuniform candidates. Mean paired accuracy
change for top-k/overall selection is −0.0115 ± 0.0199 pp.

| Test group | Count per seed | Seed 0 uniform (%) | Seed 0 top-10 (%) | Seed 1 uniform/selected (%) | Seed 2 uniform/selected (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00: landbird/land | 2,255 | 92.7273 | 92.5055 | 93.5255 | 93.0820 |
| 01: landbird/water | 2,255 | 89.7561 | 89.6231 | 90.0222 | 88.7805 |
| 10: waterbird/land | 642 | 91.1215 | 91.5888 | 90.3427 | 89.8754 |
| 11: waterbird/water | 642 | 91.2773 | 91.7445 | 91.7445 | 92.2118 |

Group 01 is the test worst group in every seed. Seed-0 top-10 corrects one extra
C image (C WGA 89.2473→90.3226%), but fixes 27 test errors and introduces 29.
Counts and exact values are retained in [group_metrics.csv](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/group_metrics.csv),
[per_seed.csv](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/per_seed.csv), [summary.json](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/summary.json) and `comparisons/seed*/`.

## Diagnostics and interpretation

The weighting computation is active: fitted probabilities and some classifications
change. All 12 power rules change zero C labels in seeds 0/1; in seed 2 they
change at most two. Joint λ=1 changes zero C labels in every seed, so uniform wins
the complete accuracy ties.

Stronger joint weighting improves NLL without improving WGA. The table compares
uniform with **joint λ=0.01**; NLL excludes the regularization term.

| Seed | Worst B-group NLL, uniform→joint | Worst C-group NLL, uniform→joint | C WGA (%), uniform→joint |
| --- | ---: | ---: | ---: |
| 0 | 0.6558→0.6316 | 0.6595→0.6418 | 89.2473→88.1720 |
| 1 | 0.6566→0.6283 | 0.6563→0.6418 | 89.2473→84.9462 |
| 2 | 0.6524→0.6258 | 0.6555→0.6361 | 87.0968→86.0215 |

Lower NLL rewards probability changes even when a correct prediction becomes
more confident without changing class. Consequently, its improvement need not
increase finite-sample WGA. Seed 1's B WGA also falls, 88.89→85.19%, so the
observed trade-off already occurs on the fitting set. Improved C NLL makes a
pure B-overfitting explanation insufficient; this is not proof of a unique cause.

For λ=0.01, effective head counts are 1.8355/1.3964/3.2733 for seeds 0/1/2,
versus 15 for uniform. Seed 1 assigns approximately 82.87% to `head-00` and
17.13% to `head-11`. Such concentration reduces averaging across distinct heads.
Mean pairwise test disagreement is 11.4910/11.4631/10.7089%; identical heads
therefore cannot explain the selected-score equality. B-to-C per-head WGA rank
correlations are −0.0276/0.4954/0.7796, so poor ranking transfer is not universal.
See [joint_diagnostics.csv](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/joint_diagnostics.csv) and [artifact_checks.json](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/artifact_checks.json).

The seed-1 λ=0.01 SLSQP line-search failure recurred; `trust-constr` converged in
24 iterations with objective-gap bound 5.53e-10, below tolerance 1e-7. All nine
joint fits completed. Software recovery succeeded independently of whether the
resulting weights improved classification.

## Follow-up hypotheses — untested

| Hypothesis | Evidence motivating it | Proposed discriminating test |
| --- | --- | --- |
| Objective sensitivity | Joint NLL improves while WGA ties or decreases. | Compare predeclared losses or decision-threshold variants on B/C, retaining the matched uniform control. Assess whether C-WGA gains recur across independent validation partitions. |
| Weight/selection instability | B has only 27 examples in each waterbird group; one C example changes group accuracy by 1.08 or 3.85 pp. | Group-stratified B resampling and repeated A/B/C partitions; measure weight dispersion, choice frequency and C performance before any new test evaluation. |
| Conditional complementarity | Heads disagree, while one global weight vector gives no selected test gain. | Compare a regularized probability/logit stacker with a small input-dependent softmax gate and uniform. Fit on B, select on C, and prevent gate collapse with predeclared regularization. |

These are proposed explanations and tests, not established causal findings.
A neural gate, stacking model and DFR baseline have **not** been run. Equal
averaging is an established deep-ensemble baseline [1]; stacking [2] and gating
[3] motivate the proposed combiners. DFR [4] retrains a classifier on frozen
features using group-balanced held-out data; it is a separate feature-level
baseline rather than an equivalent head-weighting experiment.

## Provenance and limits

The study ran `notebooks/dpe_weighting_rules.ipynb` at commit `dbeba0f`, calling
`study.run_study`, `rule_suite.select` and `rule_suite.evaluate`. Kaggle runtime:
Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3, torch 2.11.0+cu130, torchvision
0.26.0+cu130, CUDA 13.0; physical GPU model/utilization is not recorded. Scientific
completion took 11,789.7 s; final notebook export finished at 11,799.5 s.
See [run metadata](../../../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/run.json)
for archive/log checksums, source identities, commands and portable per-seed configurations.

Review recomputed all 60 B/C candidates, refitted all nine joint fits, and checked selected test metrics/error changes,
group counts, CSVs, means, sample SDs and paired deltas. Refitted weights match
exactly on NumPy 2.1.3/SciPy 1.16.3. Source/code/selection/cache hashes and test
IDs/labels/groups agree. The 200 backbone epoch records are internally consistent;
seed 1's logged trajectory matches the prior failed run. These are cache-level
checks, not a rerun of image-to-feature inference: raw images, backbone states
and raw features are excluded. Original failed-version checkpoints remain unavailable.

There are three training seeds, one shared validation partition and one shared
test set. Seed 0 was reused and had informed follow-up design. This is exploratory
evidence, not an independent confirmatory study or a paper reproduction. SD across
training seeds does not quantify uncertainty from choosing another partition or
test sample. No statistical-significance or general superiority claim is made.
Further runs on the already-inspected test set must remain labeled exploratory.

## References

1. Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474). NeurIPS.
2. Wolpert, D. H. (1992). [Stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1). Neural Networks 5(2), 241–259.
3. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., Hinton, G. E. (1991). [Adaptive Mixtures of Local Experts](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts-bitmap.pdf). Neural Computation 3(1), 79–87.
4. Kirichenko, P., Izmailov, P., Wilson, A. G. (2023). [Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations](https://arxiv.org/abs/2204.02937). ICLR.
