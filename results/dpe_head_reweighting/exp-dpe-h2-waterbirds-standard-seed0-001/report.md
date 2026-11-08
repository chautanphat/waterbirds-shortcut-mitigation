# DPE head reweighting: standard ERM, seed 0

## Status and traceability

This is a completed, exploratory Waterbirds adaptation with one seed. The record was prepared on 2026-11-08 from the supplied `download.txt` log and `h2-standard-seed0.zip` archive. Their calendar execution date is not established: the log records elapsed times only. This is not a verified reproduction of the paper or a final multi-seed comparison.

- **Executed source:** `5ebb1128fe84200938e73c1fe85e317ecb39f09d`, recorded clean working tree.
- **Method:** [implementation and mechanism](../../../src/methods/dpe_head_reweighting/README.md).
- **Settings:** [H2 configuration](../../../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/config.yaml), [backbone configuration](../../../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/stage0_config.yaml).
- **Provenance:** [run metadata and source checksums](../../../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/run.json).
- **Execution interface:** `notebooks/dpe_head_reweighting.ipynb` at the executed commit; `TRACK="standard"`, `SEED=0`, `WORKERS=2`, `RUN_NAME="h2-standard-seed0"`.
- **Environment:** Tesla T4 GPU, Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.1.3, CUDA 13.0. The hosting service was Kaggle; the procedure is not specific to that service. End-to-end elapsed time was approximately 104 minutes, including setup and archive export.

The tracked configurations transcribe the archived settings using repository-relative paths. They do not replace or share the hash of the original configuration, which contained an absolute checkout prefix. Original file identities are retained in the run metadata. The archived `run.json` command describes the notebook kernel launcher; it is not evidence of a shell training invocation.

## Question and controlled comparison

Does validation-derived weighting improve worst-group accuracy (WGA) over equal probability averaging of the same frozen DPE heads? Both conditions use the identical backbone, raw features, preprocessing, 15 trained prototype heads and test examples. Only the ensemble weights may differ.

The comparison is group-aware. Group annotations are used for partitioning, balanced head fitting, head epoch selection, reliability scoring and aggregation selection. Standard ERM versus ERM* describes the backbone recipe, independently of annotation availability.

## Data roles and procedure

Waterbirds uses its official 4,795 training / 1,199 validation / 5,794 test examples. Groups are `00` landbird/land, `01` landbird/water, `10` waterbird/land and `11` waterbird/water. The official validation split is partitioned with split seed 0 and group-stratified largest-remainder allocation:

| Role | 00 | 01 | 10 | 11 | Total | Use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 280 | 280 | 80 | 80 | 720 | Head fitting and epoch selection |
| B | 94 | 93 | 27 | 27 | 241 | Per-head WGA reliability scores |
| C | 93 | 93 | 26 | 26 | 238 | Aggregation selection |
| Test | 2,255 | 2,255 | 642 | 642 | 5,794 | Evaluation after selection |

Stage 0 fine-tunes ImageNet `IMAGENET1K_V2` ResNet-50 using cross entropy and class-balanced replacement sampling: 100 epochs, batch size 128, SGD learning rate 0.001, momentum 0.9, weight decay 0, no scheduler, and updated BatchNorm statistics. All splits use square resize to 256, center crop to 224 and ImageNet normalization. No training augmentation is enabled. The final checkpoint is used regardless of validation scores; recording a separate best-WGA checkpoint does not change that choice.

Raw 2048-dimensional features are extracted in metadata order. DPE standardizes coordinates within each example with `ddof=0`, then applies L2 normalization in its IsoMax+ head. Each of 15 sequential heads trains for 20 epochs on a fixed independently drawn balanced subset of A: 80 examples per group, 320 total. Batch size 256 and `drop_last=True` yield one update per epoch. Settings are SGD learning rate 0.001, momentum 0.9, training entropic scale 30, prototype penalty coefficient 10 and diversity coefficient starting at 500,000 with the documented decay. Epoch selection maximizes WGA on all of A, choosing the latest epoch on ties. B and C do not participate in head fitting or epoch selection. Inference temperature is 1.

For each head, its B WGA is the reliability score. Candidates interpolate normalized score powers with uniform weights: `w_h=(1-alpha)/15 + alpha*s_h^gamma/sum(s^gamma)`. The fixed grid is uniform `(alpha=0, gamma=1)` plus `alpha in {0.25, 0.5, 0.75, 1}` crossed with `gamma in {1, 2, 4}`. Selection maximizes C WGA, then C overall accuracy, then prefers smaller alpha and gamma. Test labels do not enter that selection rule.

## Recorded results

### Backbone validation

All 100 epochs and 3,700 updates completed. Training accuracy first reached 100% at epoch 35. At the final epoch, training loss was 0.000956 and validation loss 0.515452. [Epoch records](backbone_epochs.csv) are extracted from the supplied log.

| Checkpoint | Validation overall accuracy | Validation WGA |
| --- | ---: | ---: |
| Final epoch 100, used by H2 | 86.4887% | 60.1504% |
| Best validation WGA, epoch 50, not used | 85.5713% | 62.4060% |

Final validation group accuracies were 99.3576%, 78.3262%, 60.1504% and 96.2406%, respectively. No ERM-only test accuracy or WGA is established by this run's evidence. Backbone validation and DPE test metrics must not be subtracted as a matched test improvement.

### Aggregation selection

B reliability scores range from 0.740741 to 0.888889. All 13 candidates classify the same 238 C examples identically: 216 correct (90.7563% overall), with WGA 83/93 = 89.2473%. Thus selection retains `alpha=0`, `gamma=1`, with each head weighted `1/15`. This is a normal tie-breaking decision, not the all-zero-score fallback.

| Candidate | Head-weight range | Effective heads | Maximum C probability change from uniform | Changed C classifications |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 6.6667% each | 15.0000 | 0 | 0 |
| alpha=0.5, gamma=4 | 5.6945–8.2295% | 14.8379 | 0.1438 percentage points | 0 |
| alpha=1, gamma=4 | 4.7224–9.7923% | 14.3720 | 0.2875 percentage points | 0 |

The [full candidate comparison](candidate_comparison.csv) was independently recomputed from saved C probabilities after execution; it is not a new training run or a new test-driven selection. CSV probability differences use fractions, whereas this table expresses them as percentage points. Complete original candidate and per-head B/C metrics are in [selection.json](selection.json).

### Held-out test

| Metric or group | Count | Uniform correct | Uniform accuracy | Selected correct | Selected accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 5,794 | 5,286 | 91.2323% | 5,286 | 91.2323% |
| 00: landbird / land | 2,255 | 2,091 | 92.7273% | 2,091 | 92.7273% |
| 01: landbird / water | 2,255 | 2,024 | 89.7561% | 2,024 | 89.7561% |
| 10: waterbird / land | 642 | 585 | 91.1215% | 585 | 91.1215% |
| 11: waterbird / water | 642 | 586 | 91.2773% | 586 | 91.2773% |
| Worst-group accuracy | — | — | **89.7561%** | — | **89.7561%** |

The paired selected-minus-uniform difference is **0.0000 percentage points** for overall accuracy and WGA. There are zero fixed errors, zero newly introduced errors and zero changed test classifications. Identical test results follow from selecting exactly uniform weights. Weighted candidates were compared on C; they were not selected using test performance. See [metrics.json](metrics.json) and [per-head group metrics](per_head.csv).

## Artifact audit and interpretation

Independent recomputation checked all head B scores, candidate weights, C metrics, the selected decision and final test metrics. It also reconciled every one of the 45 per-head CSV rows with prediction caches. Parameter/prediction checks found 15 distinct prototype heads and 15 distinct probability arrays per role. On C, pairs of heads disagree on 5.46–21.01% of examples. Identical candidate decisions therefore do not imply duplicated heads.

The audit verified disjoint validation subsets, no validation/test ID overlap, exact agreement with the declared partition algorithm, source-file hashes against the executed commit, and consistency of stored manifest, ensemble and input identities. The original artifact format predates the later selection-hash fix; recomputation of its candidates and metrics found no inconsistency. [Independent audit](independent_audit.json) and [additional artifact checks](artifact_checks.json) retain the evidence.

The supported conclusion is narrow: this power-weight grid did not change C class predictions for standard ERM, seed 0, and the selected method provided no improvement over its matched uniform control. This does not establish that all weighting rules are ineffective. C has only 26 examples in each waterbird group, so a single changed prediction moves either group's accuracy by approximately 3.85 percentage points.

This is not a direct reproduction of the DPE paper's reported mean: H2 reserves B/C and trains heads using A, unlike the historical full-validation procedure, and only one seed is available. For context, the paper's annotated standard ERM + DPE reference is 91.0 ± 0.4% WGA ([Table 2](https://arxiv.org/html/2505.23027v1)); this difference is descriptive and does not isolate an implementation or weighting effect.

## Retention and verification limits

The original log, archive, backbone, features and sample-level prediction caches are kept outside Git. This directory contains only lightweight statistics and summaries. The archive contains trained prototype heads and B/C/test probabilities, but excludes raw images, the backbone checkpoint and raw feature arrays. Their recorded identities are consistent; original image-to-feature inference could not be independently rerun from the supplied archive alone.

Subsequent code audits/fixes are not new measurements of this run. In particular, the corrected raw-feature ERM evaluation in the historical reproduction notebook does not supply an ERM-only test result here.

For read-only verification of the original extracted archive with the current checker:

```bash
python scripts/audit_dpe_head_reweighting.py --run-dir path/to/extracted/h2-standard-seed0
```

The historical implementation and notebook are at commit `5ebb112`. A replay uses the recorded notebook settings and the portable configuration copies linked above; normal execution under newer implementation hashes requires a fresh run directory. Those copies describe a replay and must not be presented as the exact original kernel command or as new execution evidence.

## Additional summary diagnostics — 2026-11-08

These are new calculations from the tracked summaries, not another training run.
The current checkout has no original ZIP or sample-level B/C/test prediction
caches, so the earlier archive audit was reviewed but not rerun here. Calculations
use per-head B/C WGA in [selection.json](selection.json), taking Spearman
correlation with average ranks for tied scores. Candidate weight concentration
and probability changes come from [candidate_comparison.csv](candidate_comparison.csv).
The B/C per-head tables, candidate weights and metrics, and selected/uniform
agreement were cross-checked against the retained records. No one-off diagnostic
script or duplicate generated summary is required to run the experiments.

| Diagnostic | Finding | Implication |
| --- | --- | --- |
| B-to-C per-head WGA Spearman rank correlation, tied ranks averaged | -0.0276 across 15 heads | B's reliability ordering does not transfer to C in this run. This is descriptive; shared heads/data do not support an independent-sample significance claim. |
| Highest B-scoring head, `head-14` | B WGA 88.8889%; C WGA 76.9231% | Making the B-derived weights more concentrated can amplify a score that fails to generalize. |
| Strongest tested weights, alpha=1 / gamma=4 | 4.7224–9.7923%; effective heads 14.3720 | The grid still averages almost all heads. It is a mild perturbation of uniform, rather than a broad test of possible weights. |
| Largest C probability change under that candidate | 0.002875 in probability units; 0 changed classifications | Changed weights do reach the aggregate probabilities, but none cross a decision boundary. |
| B worst-group identities, counting ties | 12 heads include `10`; 2 include `01`; 2 include `11`; none include `00` | Scores are dominated by the 27-example waterbird/land group. Counts total 16 because `head-14` ties on `10` and `11`. |

One correct prediction changes a 27-example B group accuracy by 3.7037 percentage
points; on either 26-example waterbird C group the step is 3.8462 points. For
scale only, an accuracy of 0.8 on 27 independent examples has plug-in binomial
standard error 7.70 points, while the observed standard deviation of the 15 B
WGA scores is 4.28 points. This is not a confidence interval for minimum-group
accuracy or proof that all head differences are noise. It motivates checking
score stability before sharpening the weights.

The CSV's `minority_accuracy` is the minimum of groups `01` and `10`, whereas the
weighting score is the minimum of **all four** groups. For example, `head-06` has
B minority accuracy 81.4815% but B WGA 77.7778%, determined by group `11`.
Diagnostics must use `selection.json`'s WGA, rather than substituting that CSV
column. All selected weights are equal, so a correlation between selected
weight and minority accuracy is undefined; this run cannot support the proposed
low-weight/background-reliance association. Minority-group failure alone also
does not establish which visual features a head uses.

Code inspection found explicit head-axis weighted aggregation and the declared
uniform-first tie breaking; the targeted existing H2/audit regression tests
passed (13 tests). These synthetic tests check behavior such as nonuniform
winners and stale-result rejection; they do not independently reproduce this
run's image inference or accuracy.

## Head-level statistical verification — 2026-11-08

This analysis concerns the 15 A-trained heads in this seed-0 H2 source run. It
uses the original DPE probability-averaging rule; it is not an additional
measurement of the paper's heads or the historical full-validation reproduction.
The preserved reference implementation averages each head's two-class softmax
probabilities before class argmax (`baselines/dpe4subpop/notebooks/src/utils.py`,
`evaluate_ensemble`). Groups are evaluation strata, not four predicted classes.

The 15 test rows in [per_head.csv](per_head.csv) have distinct group-accuracy
vectors. Multiplying each accuracy by its group count recovers integer correct
counts within numerical precision. The following statistics were recalculated
from those rows. SD is the sample standard deviation across heads, descriptive
only: these heads share data and are not independent experimental seeds.

| Test group | Count | Head mean accuracy ± SD | Head accuracy range | Uniform ensemble |
| --- | ---: | ---: | ---: | ---: |
| 00: landbird / land | 2,255 | 89.24% ± 3.06 pp | 83.50–93.35% | 92.73% |
| 01: landbird / water | 2,255 | 86.76% ± 4.17 pp | 79.51–92.24% | 89.76% |
| 10: waterbird / land | 642 | 88.73% ± 2.59 pp | 82.71–92.37% | 91.12% |
| 11: waterbird / water | 642 | 88.22% ± 3.05 pp | 82.40–93.46% | 91.28% |

Individual-head WGA ranges from 79.5122% to 88.2927%, with mean 84.7046% and
SD 2.9453 percentage points. The highest observed single-head test WGA is head
00's 88.2927%; uniform ensemble WGA is 89.7561%, 1.4634 points higher.
The highest observed single-head overall test accuracy is head 06's
5,243/5,794=90.4902%; the ensemble gets 5,286/5,794=91.2323%, 0.7421 points
higher. These are retrospective descriptive comparisons, not test-based head
selection recommendations.

| Predictor | Correct 00 | Correct 01 | Correct 10 | Correct 11 |
| --- | ---: | ---: | ---: | ---: |
| Head 00 | 2,049 | 1,991 | 593 | 589 |
| Head 03 | 2,007 | 1,793 | 579 | 600 |
| Uniform ensemble | 2,091 | 2,024 | 585 | 586 |

These count differences alone prove prediction differences. In group 01, head
03 and the ensemble must disagree on at least `abs(2024-1793)=231` examples.
Summing absolute correct-count differences over disjoint groups gives a lower
bound on total head-versus-ensemble disagreement: at least 86 examples for head
00 and 335 for head 03. Every head has a positive such bound (67–452 across the
15 heads). These are lower bounds, not exact per-head disagreement counts;
opposing errors can cancel in aggregate accuracies.

The earlier [artifact audit](artifact_checks.json) records 15 unique test
probability arrays and 2,216/5,794=38.2465% test images with at least one head
disagreeing with another. Across the 105 head pairs, mean label disagreement
is 11.4910%; the range corresponds to 341–1,025 images per pair
(5.8854–17.6907%). These recorded rates are arithmetically consistent with the
sample sizes and an integer total of 69,908 pair-image disagreements. They
measure head-to-head disagreement, not disagreement with the ensemble.

The recorded waterbird-probability range across all test head/image combinations
is 0.404443–0.646538. This establishes varying probabilities globally, together
with the distinct recorded probability arrays; it does not provide mean
probability by head and group. The original caches are not present in the
current review inputs, so exact head-versus-ensemble disagreement matrices,
groupwise mean probabilities and probability correlations were not recomputed.

## Follow-up work, not yet measured

1. **Complete the original comparison:** run standard-track seeds 1 and 2 with
   the existing partition, score-power grid and matched uniform control. The
   current partition seed is fixed at 0 independently of the training seed, so
   these runs measure training-seed variation, not partition uncertainty. Report
   paired WGA/overall differences and how often selection is nonuniform. Keep
   the original seed-0 result intact.
2. **Check the reliability signal cheaply:** with retained B/C probabilities,
   bootstrap examples within each B group and recompute head scores, rankings
   and weights. Inspect per-group B-to-C performance, C probability margins and
   leave-one-head-out changes. Do not use test predictions to choose heads or
   tune the rule. Bootstrap rank stability and error-overlap analysis cannot be
   recovered from the current aggregate tables alone.
3. **Next H2 method experiment: regularized joint weights.** On the same frozen
   heads, fit nonnegative weights summing to one by minimizing the maximum
   B-group mean negative log likelihood of the *combined probabilities*, plus
   `lambda * sum((w - 1/15)**2)`. Unlike independent WGA scores, this objective
   accounts for how heads' predictions combine. Predeclare a small lambda grid,
   retain exact uniform and the original power rule as controls, and select
   using C WGA then C overall accuracy, preferring uniform on a complete tie.
   Record effective head count, changed C predictions and score/weight
   stability. This is a proposal, not an implemented or measured improvement.

Reusing the saved backbone/features/heads makes the second and third steps
feature-only work; training the backbone again is unnecessary merely to test
another aggregation rule. Repeated use of C can overfit method design, so limit
the candidate family in advance and assess partition sensitivity in separately
labeled experiments. Since seed-0 test results have already been inspected,
these follow-ups are exploratory; lock settings before any subsequent test
evaluation and report all planned seeds, including failures to improve.
