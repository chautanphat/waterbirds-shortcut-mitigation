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

## Follow-up work, not yet measured

Complete seeds 1 and 2 with the original protocol before reporting a mean or sample standard deviation. Separate exploratory experiments may compare softmax weighting and constrained regularized linear stacking against uniform and the existing power rule. Fit on B, select on C, and freeze settings before test evaluation. Since seed-0 test results have already been inspected, revised experiments must be identified explicitly rather than folded into the original result. No new weighting method or improvement is reported in this record.
