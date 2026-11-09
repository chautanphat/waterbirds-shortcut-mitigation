# H2: multiple weighting rules, standard ERM, seed 0

**Follow-up (2026-11-09):** The [consolidated three-seed report](../main.tex) records completed seeds 0/1/2, prediction-cache verification and revised follow-up hypotheses. This earlier record retains its original measurements and review scope.

## Status and provenance

Completed exploratory cached-prediction comparison, documented on 2026-11-08.
The supplied archive is `h2-standard-seed0-rules-6452ea10e0.zip`; its execution
calendar date, elapsed time, hosting service and accelerator are not recorded.
This is one seed and is not a final multi-seed result or a paper reproduction.

- Executed commit: `3ee1c6e7a9cfad351d69e75c92ad0d9c85d4c424`.
- Interface: `notebooks/dpe_weighting_rules.ipynb`, calling `rule_suite.select`
  and then `rule_suite.evaluate`; no shell training command is claimed.
- [Recorded grid](../../experiments/exp-dpe-h2-rules-waterbirds-standard-seed0-001/config.yaml),
  [runtime, archive checksums and review scope](../../experiments/exp-dpe-h2-rules-waterbirds-standard-seed0-001/run.json).
- Runtime: Linux, Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3,
  PyTorch 2.11.0+cu130; CUDA 13.0 is recorded, but GPU use is not established.
- Inputs: the same saved ensemble and prediction caches as the
  [original H2 seed-0 run](dpe_h2_power_seed0.md).
  Source ensemble, manifest, selection and original test-result identities match
  the existing record. No backbone, raw features or heads were retrained.

## Procedure

Waterbirds uses the official 4,795/1,199/5,794 train/validation/test split.
Inherited validation roles are A=720 for head fitting/epoch selection, B=241 for
weight fitting, and C=238 for aggregation selection, with partition seed 0.
The group-annotated setting and A-only head-training budget are unchanged.

All 20 candidates use the same 15 heads: exact uniform, 12 original power-rule
configurations, softmax beta 10/30, top-k 10/5, and joint lambda 1/0.1/0.01.
Joint fitting minimizes worst B-group log loss plus a squared-distance-to-uniform
penalty under nonnegative, sum-to-one weights. Every family includes uniform as
a fallback. Selection maximizes C WGA, then C overall accuracy; complete ties
retain uniform and then declared candidate order. The overall C winner and each
family winner are frozen before their test evaluation. Unselected candidates
are not evaluated on test in this archive.

## Recorded results

| C-selected procedure | Chosen candidate | Test accuracy | Test WGA | WGA change vs uniform |
| --- | --- | ---: | ---: | ---: |
| Uniform control | uniform | 91.2323% | 89.7561% | 0.0000 pp |
| Power family | uniform | 91.2323% | 89.7561% | 0.0000 pp |
| Softmax family | uniform | 91.2323% | 89.7561% | 0.0000 pp |
| Joint family | uniform | 91.2323% | 89.7561% | 0.0000 pp |
| Top-k family | top-k10 | 91.1978% | 89.6231% | -0.1330 pp |
| Overall C-selected procedure | top-k10 | 91.1978% | 89.6231% | -0.1330 pp |

Equal test results for power, softmax and joint mean that these families selected
the uniform fallback. They are not measurements of those families' unselected
nonuniform weights. See [metrics.json](../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/metrics.json) for exact values and error
IDs, [selection.json](../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/selection.json) for weights and solver diagnostics, and
[candidate_comparison.csv](../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/candidate_comparison.csv) for all C candidates.

| Group | Test count | Uniform correct | Top-10 correct | Uniform accuracy | Top-10 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00: landbird / land | 2,255 | 2,091 | 2,086 | 92.7273% | 92.5055% |
| 01: landbird / water | 2,255 | 2,024 | 2,021 | 89.7561% | 89.6231% |
| 10: waterbird / land | 642 | 585 | 588 | 91.1215% | 91.5888% |
| 11: waterbird / water | 642 | 586 | 589 | 91.2773% | 91.7445% |
| Overall | 5,794 | 5,286 | 5,284 | 91.2323% | 91.1978% |

Top-10 changes 56 test predictions: 27 previously incorrect predictions become
correct, while 29 previously correct predictions become errors. Net correct
changes are -5/-3/+3/+3 across groups 00/01/10/11. Both waterbird groups improve,
but landbird groups worsen; group 01 remains the worst group. Overall accuracy
falls by 0.0345 percentage points.

## Diagnostics and interpretation

Top-10 wins C by correcting exactly **one additional image** in group 01:
83/93 becomes 84/93. C WGA rises from 89.2473% to 90.3226%, and C overall
accuracy from 216/238 (90.7563%) to 217/238 (91.1765%). That one-image gain does
not transfer to test. The outcome is consistent with selection sensitivity on a
small C set, but does not prove that mechanism or a population-level difference.

The original power candidates and softmax beta 10 change no C decisions.
Softmax beta 30 changes three C decisions and loses one net correct prediction.
Top-5 changes two C decisions with no net accuracy gain, so it loses the tie to
uniform. Thus stronger weights are reaching the probabilities and, for some
rules, changing predictions; the outcome is not simply an unchanged-weight run.

Joint optimization reports success at all three penalties (5, 9 and 17
iterations). Lambda 1 and 0.1 change no C class decisions, so uniform wins their
accuracy ties. Lambda 0.01 concentrates approximately 64.97%/35.03% on heads
00/01 (effective heads 1.8355), changes five C decisions and lowers C WGA to
88.1720%. Strong concentration did not help C here. The archive's B loss
summaries do not independently establish optimizer optimality without B caches.

Top-10 retains heads 00–07, 13 and 14. B WGA ties at the cutoff are resolved by
the predeclared source-head order. This is one fixed subset, not a search over
all possible ten-head ensembles.

The supported conclusion is that **none of the C-selected new procedures improves
test WGA in this seed-0 comparison**. Preserve top-10 as the recorded overall
selected procedure; do not replace it retrospectively using test performance.
Uniform remains the matched reference. This does not rule out other weight
families or improvements on independent runs.

## Review limits and next experiments

An aggregate consistency review checked all 20 candidate records, CSV/JSON
agreement, weight formulas, simplex/effective-head constraints, joint objective
arithmetic, independent application of C tie breakers, test integer counts,
paired deltas and fixed/new error-ID counts. All 11 recorded code/notebook
identities match the executed repository source. Selection/result hashes and
original-run identities also agree.

The new ZIP deliberately excludes B/C/test prediction caches and trained heads.
Therefore C/test metrics, actual probability changes and joint fits could not
be independently recomputed from sample-level predictions in this review. The
scope is aggregate consistency, not another full artifact or image-inference
audit. Retain both the original H2 ZIP and this suite ZIP outside Git.

Next, complete seeds 1 and 2 with the **same fixed grid and C selection rule**,
using their corresponding original H2 runs. Report every seed's selected rule
and paired WGA/accuracy difference before drawing a multi-seed conclusion.
Do not expand the candidate grid in response to this test result. A separately
labeled H2 study can later use repeated partitions or group-stratified B
bootstraps to assess ranking and selection stability. No follow-up measurement
is claimed here.
