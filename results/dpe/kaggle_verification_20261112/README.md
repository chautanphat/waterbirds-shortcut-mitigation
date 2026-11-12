> Historical three-source evidence. The [completed five-seed report](../five_seed_verification_20261112/README.md) is the current assessment; the dropout-specific advantage is not established.

# Kaggle verification of DPE head aggregation

**The existing three-seed result reproduced exactly on the reported numerical values and class decisions.** The uploaded verification archive reports completion for source seeds 0, 1 and 2 on a Tesla T4. It used the original ResNet-50 backbone checkpoints and official Waterbirds splits. No new backbone seeds were trained.

## Main result

| Configuration | Mean test WGA | Change versus matched uniform |
| --- | ---: | ---: |
| Uniform DPE | 89.5196% | Reference |
| Plain MLP, weight decay 0.01 | 89.9630% | +0.4435 pp |
| Dropout MLP, weight decay 0.01 or 0.001 | 89.9926% | +0.4730 pp |

The dropout gains by source seed are +0.1330, +0.0443 and +1.2417 percentage points. These are the same results as the original experiment, not an additional independent benchmark sample.

## Reproduction and audit evidence

- All 90 configuration/seed decisions reproduced without any changed test classifications.
- Numeric saved parameters or global weights matched exactly for all 87 nonuniform configuration/seed state comparisons. Uniform has no fitted state.
- All 45 prototype heads reproduced their selected epochs; reported maximum prototype and distance-scale differences are zero for every source seed.
- Full inference from all 5,794 test images per seed reproduced the cached raw features and head probabilities with reported maximum differences of zero. All 30 configurations per seed also reproduced their image-based class decisions.
- The repository checks report 169 passed and 14 deselected. The three additional embedded numerical contracts passed. These software checks do not substitute for the image and result comparisons above.
- Independent inspection of the archive recomputed all 90 WGA values from group accuracies/correct counts, all 30 summary means and sample standard deviations, and the saved state differences. These checks passed. The review did not execute uploaded code or rerun training.

This completes the previously pending image-reproduction checks for source seeds 1 and 2. The cloud-only partial record in [the earlier verification directory](../verification_20261112/README.md) is historical.

## What the additional combiner runs show

The notebook froze 120 fits: three original head ensembles × ten combiner seeds × two weight decays × plain/dropout MLP. These are not 120 independent backbone/head runs.

| Weight decay | Plain MLP mean WGA | Dropout MLP mean WGA | Dropout minus plain |
| --- | ---: | ---: | ---: |
| 0.01 | 90.0296% | 89.9660% | −0.0636 pp |
| 0.001 | 90.0177% | 89.9645% | −0.0532 pp |

Dropout MLP retains an average gain over uniform of +0.4464 and +0.4449 pp, respectively. However, the advantage of dropout over plain MLP is not consistent: its effect changes across the frozen head ensembles. The existing evidence supports studying neural aggregation; it does not establish that prediction dropout increases expected WGA over plain MLP.

## Paired uncertainty analysis

The uploaded run used 10,000 group-stratified paired bootstrap replicates, sharing resampled examples across configurations and source seeds and recomputing the worst group each time.

| Contrast, weight decay 0.01 | Observed mean WGA change | Conditional pointwise 95% interval |
| --- | ---: | ---: |
| Dropout MLP versus uniform | +0.4730 pp | [+0.2069, +0.8899] pp |
| Dropout MLP versus plain MLP | +0.0296 pp | [−0.4673, +0.2069] pp |

These intervals concern test-example uncertainty conditional on the existing fitted models. They do not establish uncertainty across new backbone/head training seeds, and they are not simultaneous confidence intervals across the full 30-configuration grid. Small differences from earlier cloud bootstrap quantiles reflect a different jointly resampled configuration set and Monte Carlo draws; the observed point estimates agree.

## Environment and logs

The recorded environment is Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.1.3, SciPy 1.16.3 and Pillow 13.0.0, at source commit `8e14249f4b3510363be06f627182e2daa9926179`.

Pillow 13 is material: earlier local tests found that older Pillow versions changed resized inputs on 119 validation images even though decoded RGB pixels matched. The successful full-image replay establishes that the pinned verification environment matches the original cached computation. It does not recover an originally recorded Pillow version, because that version was absent from the original metadata.

The worker log spans approximately six minutes from environment logging to the last seed's image verification, excluding notebook setup and finalization. All three GPU image checks finish within roughly 70 seconds of their first image progress message, including intervening loading and archiving. This is measured log evidence for this run, not a universal runtime guarantee.

The warnings concern a scalar conversion in the numerical test and future Python tar extraction behavior. The joint lambda-0.01 fit used its declared solver fallback. Completion, matching states and results give no evidence of an unresolved fitting failure in this run.

## Current conclusion

The observed +0.47 pp test WGA improvement over matched uniform DPE is now supported by fresh fitting and full raw-image reproduction across the three original source seeds. It is a valid result for this fixed-backbone experiment. Additional independent backbone/head seeds remain the main unfinished stability check before making a broader expected-performance claim. A dropout-specific benefit over plain MLP is not established.

The dataset, architecture and independent-configuration protocol remain unchanged; no extra candidate-selection split or background-intervention experiment is required for this claim.

## Artifacts

- [Summary](summary.csv), [all 90 seed/configuration results](per_seed.csv), and [120 combiner-randomness results](combiner_randomness.csv).
- [Saved state comparisons](model_state_comparison.csv), [head reproduction](head_replay.json), and [all three image replays](image_replay.json).
- [Bootstrap results](paired_bootstrap.json), [environment](environment.json), [software test output](repository_tests.txt), and [review provenance](review_record.json).

The original uploaded ZIP, including full model records and its worker log, is retained in `outputs/kaggle_verification_20261112/` outside Git. CSV accuracy values are fractions; WGA changes are percentage points.
