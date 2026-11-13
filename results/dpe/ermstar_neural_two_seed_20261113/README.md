# Neural aggregation on ERM* + DPE: two-seed exploratory study

Report date: 2026-11-13. Waterbirds; ImageNet-initialized ResNet-50; source seeds 0 and 1.
This is an **H2 aggregation experiment**, not an H1 or H3 experiment and not formal confirmation.

## Claim supported by this experiment

On the fixed primary validation allocation and these two trained ERM* sources,
plain and prediction-dropout MLP aggregation improve test worst-group accuracy
(WGA) over their matched original-uniform baseline on **both seeds**. At weight
decay .01, their mean improvements are **+0.3440 and +0.4105 percentage points**,
respectively. Logistic aggregation has the highest observed neural mean WGA,
**94.0031%**, but loses on seed 0 and gains on seed 1. These results motivate
further exploration; they do not establish expected improvement across new
sources, alternative partitions, or datasets.

No tested neural configuration's two-seed mean exceeds the **94.0810%** mean of
the full-validation paper-configuration DPE reference. That reference uses a
different head-training allocation, so this comparison measures the complete
procedures rather than the causal effect of aggregation alone.

## Experiment and evidence

Public run: [noerula/dpe-exploration, saved version 3](https://www.kaggle.com/code/noerula/dpe-exploration?scriptVersionId=368838235).
Kaggle's public API returned both compact seed archives and the saved log. An
explicit `v3` log request matched the retrieved latest-output log. The notebook
source-download endpoint omits executed outputs; it must not be used to conclude
that this run has no outputs.

Both seeds completed with no recorded worker errors. The saved backbone summaries
report **14,800 optimizer updates per source**, consistent with 37 updates per epoch
for 400 epochs, and `ckpt_last.pt` as the handoff. Logs identify two Tesla T4 GPUs;
one independent seed process used each GPU. Logged execution through final notebook
conversion was about **7.87 hours**, including preparation and both seeds running
concurrently. This is one observed session, not a general runtime guarantee.
Saved runtime: Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130.

The saved checkout's experiment wrapper, copied upstream head routines and neural
combiner match the corresponding local files by direct text comparison. Runtime
metadata records a clean checkout. These checks connect the artifacts to the
inspected implementation; they do not certify every dependency or prove historical
execution order.

Evidence retained beside this report:

- [Per-seed metrics](per_seed.csv): every procedure, all four groups, overall
  accuracy, NLL, group-balanced NLL/Brier, paired decision changes and error counts.
- [Summary](summary.csv): all configurations independently, without selecting one.
- [Audit](audit.json), [run/configuration metadata](sources.json), and
  [log excerpt](log_excerpt.txt).
- [Replay script](../../../scripts/analysis/audit_dpe_ermstar_neural.py).

Large model/data artifacts remain in Kaggle output. Retain the full run for recovery;
the compact archives alone omit backbone optimizer states and raw frozen features.

## Methods

### Backbone and original DPE components

The upstream [Waterbirds launcher](../../../baselines/dpe4subpop/scripts/train_all.sh)
and [prototype launcher](../../../baselines/dpe4subpop/scripts/train_pe.sh) supply
the paper-configuration targets. The shared ERM trainer implements the 400-epoch
recipe with ResNet-50, ImageNet V2 weights, class-balanced replacement sampling,
training augmentation, batch size 128, SGD learning rate .001, momentum .9,
optimizer weight decay 0, and no scheduler. Validation/test preprocessing is
deterministic; evaluation batch size is 512. The final epoch is handed off even
though a best-validation checkpoint is also saved.

Backbones are frozen before head training. Each arm trains 15 sequential binary
heads, each with one prototype per class. The original IsoMax+ loss, prototype
trainer and group subsampler are preserved in the attributed
[upstream module](../../../src/methods/dpe_ermstar/upstream.py). Each head trains
for 20 epochs, batch size 256 with `drop_last=True`, learning rate .001, momentum
.9, training entropic scale 30, constant diversity coefficient 500,000 and prototype
penalty coefficient 10. Features are standardized per example before normalized
prototype-distance inference. Training's entropic multiplier is not applied again
to the inference softmax. The upstream trainer selects the head by WGA on the
arm's head pool; the selected head epoch itself is not exported.

This is a maintained implementation with checkpoint recovery and a feature-only
evaluation callback, not an assertion of identical upstream random trajectories
or independently verified numerical reproduction of the paper. Exact settings and
code-reuse limitations are documented in the
[method README](../../../src/methods/dpe_ermstar/README.md) and
[configuration](../../../configs/dpe_ermstar/study.json).

### Two arms and the matched baseline

The standard Waterbirds training/validation/test sizes are 4,795/1,199/5,794.
Within each source the two arms share its frozen backbone but train separate heads:

| Arm | Head training and head selection | Aggregation fitting | Evaluation |
| --- | --- | --- | --- |
| Full-validation annotated DPE reference | All 1,199 validation examples | None | Original uniform |
| Neural comparison | Group-stratified 720-example pool | Disjoint 479-example pool | Matched uniform plus ten neural configurations |

The 720/479 allocation uses fixed partition seed 0 and is identical across sources.
Each training subset contains all four groups. Balanced head-training subsamples
contain 532 examples in the full-validation arm and 320 in the neural arm, producing
two versus one optimizer batches per head epoch. Thus allocation and head-training
budget differ across arms. The correct control for aggregation is **uniform from
the same neural-arm heads**, not the full-validation reference.

The inspected implementation freezes both head ensembles and all neural fits
before scoring their source's test set. No additional candidate-selection split
is required or used; every declared configuration is reported independently. All
configurations use the same test images. The artifact audit verifies disjoint
head/combiner/test IDs, but cannot prove execution history from saved states alone.

### Aggregation procedures

Let `q_h(x)` be the waterbird probability from original head `h`. Uniform averaging
uses their arithmetic mean. Neural models receive the centered vector `2q−1`:

- **Logistic:** an affine map followed by sigmoid; 16 parameters. Coefficients
  are unrestricted, so this is not constrained convex weighting.
- **MLP:** 15 → 8 → 1, tanh hidden layer and sigmoid output; 137 parameters.
- **Gate:** 15 → 8 → 15, tanh hidden layer and softmax output weights, followed
  by a convex mixture of the original head probabilities; 263 parameters.
- **Prediction-dropout MLP/gate:** during fitting, drop each head prediction
  independently with probability .25, retaining at least one per example.
  Dropped inputs become neutral; gates exclude dropped heads from their mixture.
  Inference uses every head without dropout.

All learned procedures use 400 fixed full-batch CPU float64 AdamW steps, learning
rate .01, and group-balanced binary cross-entropy (mean of the four group-average
losses). Each family is evaluated at weight decay .01 and .001, with initialization
seed `source_seed + 10000`. There is one fit per source/configuration, not repeated
initialization sampling. No model is selected by test WGA. Inspecting the best
observed mean now is exploratory comparison, not a validation-selected winner.

WGA is the minimum test accuracy across groups `00`, `01`, `10`, `11`, where the
first digit denotes bird class (0 landbird, 1 waterbird) and the second background
(0 land, 1 water). Test counts are 2,255/2,255/642/642. Reported means average each
source's WGA, rather than pooling predictions across sources. Deltas are paired
percentage-point differences within source; ties predict class 0.

## Results

WGA is in percent. Δ compares with uniform in the **same arm**; the reference's
zero is its own baseline. Wins/losses count source-level differences, not independent
hypothesis tests. Equal displayed WGA does not imply identical predictions.

| Procedure | Seed 0 | Seed 1 | Mean WGA | Mean Δ vs matched uniform | Wins/losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-validation DPE uniform | 94.3925 | 93.7695 | 94.0810 | +0.0000 | 0/0 |
| uniform | 94.5483 | 92.2838 | 93.4161 | +0.0000 | 0/0 |
| logistic-wd0.01 | 94.0810 | 93.9252 | 94.0031 | +0.5871 | 1/1 |
| logistic-wd0.001 | 94.0810 | 93.9252 | 94.0031 | +0.5871 | 1/1 |
| mlp-wd0.01 | 94.7040 | 92.8160 | 93.7600 | +0.3440 | 2/0 |
| mlp-wd0.001 | 94.7040 | 92.7716 | 93.7378 | +0.3218 | 2/0 |
| mlp_dropout-wd0.01 | 94.7040 | 92.9490 | 93.8265 | +0.4105 | 2/0 |
| mlp_dropout-wd0.001 | 94.7040 | 92.9047 | 93.8044 | +0.3883 | 2/0 |
| gate-wd0.01 | 91.2639 | 90.5100 | 90.8869 | -2.5291 | 0/2 |
| gate-wd0.001 | 91.2639 | 90.5100 | 90.8869 | -2.5291 | 0/2 |
| gate_dropout-wd0.01 | 91.2639 | 90.5543 | 90.9091 | -2.5070 | 0/2 |
| gate_dropout-wd0.001 | 91.2639 | 90.5100 | 90.8869 | -2.5291 | 0/2 |

### Findings and interpretation

1. **MLPs show a small positive signal against matched uniform.** Plain MLP WD .01
   changes WGA by +0.1558 pp on seed 0 and +0.5322 pp on seed 1. Dropout MLP WD .01
   changes it by +0.1558 and +0.6652 pp. Both corresponding WD .001 fits also improve
   both seeds. Two positive source observations are encouraging but insufficient
   to establish a reliable population-level gain.
2. **Logistic's larger mean is driven by a seed-dependent tradeoff.** Its paired
   changes are −0.4673 pp and +1.6414 pp. On seed 1 it raises the original worst
   group (`01`) enough that `10` becomes the new worst group; on seed 0 the minimum
   shifts to `11`. Averaging obscures this heterogeneity. The two weight decays
   have equal WGA, although other group accuracies can differ.
3. **Dropout superiority is not established.** At either matched weight decay,
   dropout MLP ties plain MLP's WGA on seed 0 and improves seed 1 by +0.1330 pp.
   The mean advantage is only +0.0665 pp and comes from one source. This neither
   establishes an advantage over plain MLP nor overrides the earlier five-source
   evidence on other backbones/validation partitions.
4. **The tested gates hurt WGA on both sources.** Ordinary gates lose −2.5291 pp
   on average; dropout changes this little. This rejects these exact trained gate
   configurations as improvements in this run, not input-dependent weighting as
   a general idea. For example, ordinary gate WD .01 reduces mean group-balanced
   test NLL from uniform's 0.6296 to 0.5610 while worsening WGA: a better fitting
   objective does not guarantee a better worst-group decision boundary.
5. **No neural mean beats the full-validation reference.** Logistic is −0.0779 pp
   below it; plain/dropout MLP WD .01 are −0.3210/−0.2545 pp below it. This does not
   negate their within-arm gains. It shows that recovering an aggregation gain
   after reserving combiner data does not automatically improve the complete
   full-validation DPE procedure. Neural aggregation on full-validation-trained
   heads with a separately available fitting pool was not tested here.
6. **A nonlinear-ensemble mechanism is not isolated.** Logistic is competitive
   with the MLPs; stackers can change confidence and thresholds as well as relative
   head influence, whereas gates remain convex mixtures. There is no matched
   calibration-only or learned-threshold baseline in this run. Attributing the
   positive signal specifically to nonlinear head interactions would be premature.

## Audit and limits on inference

A separate NumPy implementation applies the saved affine/tanh/sigmoid/softmax
parameters to the cached head probabilities without importing the experiment's
prediction code or loading pickled checkpoints. It reproduces **all 24 test WGA,
overall accuracies, four-group accuracies and matched-baseline deltas** within
1e−12, and all 22 saved combiner-training accuracy reports. It checks identical
ordered test IDs/labels/groups across both seeds and arms, all-group coverage,
valid finite probabilities and disjoint fitting/test identities. Logged WGA values
agree with these artifact results. Replay uses saved head outputs; it does not
recompute image features, prototype distances, original head training or neural
optimization. Neither a correct replay nor a fit-complete marker proves the absence
of every possible historical data leak.

Reproduce the audit with the two compact archives in one directory:

```bash
python scripts/analysis/audit_dpe_ermstar_neural.py \
  --artifacts /path/to/seed-archives \
  --output /path/to/audit-output
```

There are **two independent trained sources**, not 24 independent experiments.
The shared test set, shared partition, shared source within arm comparisons, and
related weight decays create dependence. We report paired source values rather
than a significance claim; two sources do not provide a stable estimate of source
variability. Example-level resampling alone would not address it. No alternative
validation allocation or independent confirmation set is evaluated here. These
400-epoch ERM* sources must not be pooled with the older five sources as though
all seven followed one training protocol. Larger test WGA than those earlier runs
cannot be attributed to neural aggregation because the backbone recipe changed.

## Relation to the hypotheses and next experiments

This run addresses **H2** by changing aggregation of the 15 original binary heads.
It does not re-pair prototypes (H1) or compute probabilities jointly from larger
prototype subsets (H3). The report's H3 requires class-balanced larger subsets,
softmax over their negative-distance scores, and class-wise summation; all-30
prototype global pooling is a special case. Neither that experiment on these ERM*
sources nor H3+H2 combinations were executed here.

For continued exploration, retain logistic and both MLP variants as independently
reported candidates; add confidence/threshold controls to interpret the gains;
reuse these frozen ERM* artifacts for H1/H3 after checking geometry and scale
semantics. Additional independently trained ERM* sources and alternative validation
allocations would test stability. Formal confirmation should follow a fixed
protocol later; the present results support a promising H2 signal, not a confirmed
new state of the art, dropout advantage, partition-independent improvement, H3
success, or verified reproduction of the original paper's numerical results.
