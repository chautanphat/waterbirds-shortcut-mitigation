# Should we continue weighting and neural aggregation?

Research decision note, 2026-11-13. This updates the
[H2 literature note](h2_aggregation_literature.md) and complements the
[H1/H3 follow-up](h1_h3_followup_literature.md). No new predictive model was trained.
All suggested methods below are exploratory adaptations, not reproductions or
established DPE improvements. Waterbirds, ResNet-50 and saved backbones stay fixed.

## Decision

Continue a **small, mechanistically distinct H2 exploration**, and prioritize
prototype-level inference alongside it. Deprioritize a broad sweep of larger MLPs,
dropout rates and the existing gate architecture. The evidence rejects neither
learned aggregation generally nor prototype-level weighting, but does not support
a confirmed neural or dropout advance.

The [normal-ERM report](five_seed_verification_20261112/README.md) gives positive
primary-partition MLP means, source-level uncertainty including zero, and sensitivity
to the head/combiner allocation. Its calibration and threshold controls do not
explain the full mean gain, but cannot eliminate all calibration explanations.
The [ERM* report](ermstar_neural_two_seed_20261113/README.md) has only two sources:
MLPs improve matched split-pool uniform on both, logistic has mixed paired changes,
and all neural means remain below our full-validation DPE reproduction. Keep these
reproduction comparisons distinct from the original paper's published results.

## What the literature adds

- [Regularized Linear Models in Stacked Generalization, 2009](https://scispace.com/papers/regularized-linear-models-in-stacked-generalization-4s8n1jnsxl): regularization matters even for linear stackers. SciSpace method extraction describes least-squares stacking with ridge/lasso/elastic-net; it does not establish a DPE or WGA result. This motivates small nested controls, not another claim that generic regularization is new.
- [Minimax Group Fairness, 2021](https://scispace.com/papers/minimax-group-fairness-algorithms-and-experiments-4ges6e9bs3): minimizes maximum group loss rather than equalizing outcomes. Extracted context distinguishes theory/oracle assumptions from practical heuristics. Our minimax frozen-weight proposal is an adaptation; no Waterbirds DPE improvement follows from the paper.
- [Bayesian Stacking via Proper Scoring Rule Optimization using a Gibbs Posterior, 2025](https://scispace.com/papers/bayesian-stacking-via-proper-scoring-rule-optimization-using-cbohfbbu0klj): models uncertainty in weights. Available extraction is predominantly abstract-level and does not verify its exact score, prior or final predictive estimator. Keep it a reserve rather than call an implementation a reproduction.
- [K-NN boosting prototype learning, 2010](https://scispace.com/papers/k-nn-boosting-prototype-learning-for-object-classification-14eisxketz): motivates learned local prototype contributions instead of uniform nearest-neighbor votes. Exact full procedure was not recovered by extraction. Our frozen-distance construction below is specified independently.
- [Multi-prototype SVM, 2005](https://scispace.com/papers/multiclass-classification-with-multi-prototype-support-1tmdqqkgty): supports classification with multiple representatives rather than one per class. Its training/kernel formulation differs from fixed DPE probability pooling. Only partial method context was inspected.

These searches and method extractions used SciSpace. Their incomplete contexts are
explicit limitations; they do not justify claims of full independent method review,
novelty or transferability to our benchmark. The original SoftTriple and group-DRO
formulas checked in the linked follow-up remain the firmer methodological anchors.

## Distinguish head aggregation from prototype aggregation

Original DPE lets each original prototype pair compete, then averages its binary
probabilities. Our MLPs, gates and previous head-weight rules operate **after** that
competition. They cannot change which prototypes originally competed.

Completed normal-ERM exploration did change prototype competition:

- H1 re-paired landbird and waterbird prototypes; WGA did not improve.
- H3 globally normalized scores across all 30 prototypes and summed probability
  mass by class; changes were small and inconsistent across sources.
- Both used equal contributions within their specified construction. Neither
  learned prototype importance or tested 15 distinct larger H3 subsets.

No H1/H3 experiment on the new ERM* sources or real H1/H3+H2 combination has yet run.
Averaging prototype **vectors** is another operation, but it collapses the class
back to one representative and can destroy modes; it is not the leading proposal.

## Next prototype-level candidates

### A. Distinct multi-prototype heads: first priority

Implement the report's H3 literally: give each receiving head a different balanced
subset of existing prototypes, compute softmax within it, sum probabilities by class,
then compare uniform with H2 aggregation of these new outputs. k=1 is the original
pair control; k=15 is identical global pooling. Use exposure-balanced memberships
and explicit temperature/scale controls as described in the linked follow-up.
No new prototypes or backbone optimization is needed.

### B. Learned prototype importance: a genuinely new fitted candidate

For each class c, define nonnegative prototype weights alpha_{cj} summing to one
**within that class**, and fixed scores z_{cj}(x) from its saved prototype distance.
Use

`A_c(x) = sum_j alpha_{cj} exp(z_{cj}(x)/T)`

`P(c|x) = A_c(x) / sum_d A_d(x)`.

Compute this stably in log space. Equal alpha recovers class-balanced global
joint pooling. For 15 prototypes per class there are 28 independent weight degrees
of freedom. Class-wise normalization prevents an arbitrary total weight difference
from silently becoming a class prior; a separate fitted class-bias control can
measure that effect. Source-head scales and score signs must be audited first.

Fit only the alpha parameters on the existing 479 labels, shrinking toward uniform.
Compare average group loss and a separately specified worst-group loss. This is
prototype-level weighting, unlike weighting the 15 original pair probabilities.
Do not assume its optimization is convex: a ratio of separately weighted class
scores has different geometry from a convex mixture of fixed head probabilities.

Essential controls are global equal-alpha pooling, fitted scalar temperature,
fitted class bias, and alpha fitting at a fixed temperature, all with the same
label access. Jointly learning every scale, bias, prototype weight and subset at
once would obscure what helped. Monitor whether a few prototypes dominate and
whether any gain survives source/partition variation.

### C. Local top-k prototype evidence: cheap, label-free probe

For each image, retain the k highest-scoring prototypes **per class** under the
fixed, explicitly scaled negative-distance scores, then aggregate their
negative-distance scores by a predeclared log-mean-exp rule and normalize classes.
Use k=1,3,5,15; k=15 recovers the corresponding equal global construction. Do not
confuse this with selecting top-k original heads by validation accuracy, which was
already tested. Equal class cardinality avoids changing class mass through count.
This could preserve relevant modes while excluding distant prototypes, but nearest
prototypes may themselves encode backgrounds. No improvement is assumed.

## Next H2 candidates

### D. Maximum-group loss instead of average-group loss

The previous group-balanced objectives average four group losses. Test a regularized
simplex mixture minimizing their **maximum**, with epigraph/feasibility checks, and
a separately reported logistic combiner with the same loss choice. This changes the
optimization target, not merely another weight-decay value. Use the existing 479
fitting labels; small groups have only about 53 examples, so overfitting is a serious
empirical question. Better worst-group fitting loss is not guaranteed better WGA.

### E. A small correction to calibrated uniform, with nested controls

A proposed neural parameterization is

`logit P(1|x) = a logit(mean_h q_h(x)) + b + f(q(x) - mean_h q_h(x)) - f(0)`.

Fit (i) scalar calibration only, (ii) a linear disagreement correction and (iii) a
small nonlinear correction, using identical labels, loss and optimization budgets.
Zero correction exactly recovers the calibrated-uniform control. Initialize there
and penalize the correction separately. This cleanly asks whether head differences
add information beyond the uniform confidence; it does not guarantee improved
optimization or generalization, and is not claimed as a literature reproduction.
Try the linear correction before a larger MLP. No inference-time group labels.

### F. Bayesian averaging: reserve, with an algebraic limitation

For a linear probability pool with fixed heads,

`E_w[sum_h w_h p_h(x)] = sum_h E_w[w_h] p_h(x)`.

Thus posterior predictive averaging is exactly a pool with posterior-mean weights;
it does not create a richer prediction family. It may stabilize estimation relative
to a fitted mode or other point estimator, but is not automatically superior to
previous shrinkage. Uncertainty in weights can still be informative. Exact prior,
score and posterior computation require full-method verification before implementation.

## Diagnose gates before expanding them

A CPU-only check of saved version 3 of `noerula/dpe-exploration`, using the
`neural_split/test_heads.npz` files in both compact seed archives, found:

| Source | Uniform errors | Uniform errors where some head is correct | Errors where all heads are wrong | Label-aware oracle WGA |
| --- | ---: | ---: | ---: | ---: |
| 0 | 243 | 219 | 24 | 98.4424% |
| 1 | 307 | 265 | 42 | 98.5981% |

The oracle chooses a correct head using the true label. It is **not deployable**, not
an achieved score, and not a training signal. It bounds what arbitrary per-example
convex weighting could classify correctly on these caches; it says nothing about
whether a small gate can identify those cases from 479 examples. Fixed global
weights are more restricted still. The gate failures therefore do not prove lack
of complementary predictions, but neither does this oracle justify a bigger gate.

Use fitting-set diagnostics to design any new gate. Test labels may describe a
fixed oracle diagnostic, never choose routes, head subsets or hyperparameters.
Bootstrapping the fitting set can diagnose weight instability, but repeated fits
are not independent backbone sources and must not inflate the evidence count.

## Recommendation and stopping rule

Prioritize A and B, run D and the nested controls in E as a small H2 track, and use
C as a cheap inference probe. Retain the original logistic/MLP/dropout configurations
as anchors; deprioritize unchanged Brier/joint-NLL rules, the failed gate family,
a broad dropout sweep and additional equal-generalized-mean orders.

Evaluate geometry-only variants on the full-validation heads against our original
full-validation DPE reproduction. Evaluate fitted variants on split-pool heads with
identical fitting labels and report the full-validation reference separately.
Keep normal ERM and ERM* independent. Report every declared configuration, including
failures, without an additional candidate-selection split or test-selected winner.
All substantial computation belongs in a fresh, tested Kaggle launcher using normal
repository imports, readable modules/settings and robust recovery.

If these distinct, controlled H2 variants do not show a useful gain across the
available sources and validation allocations, pause H2 expansion and concentrate
on prototype geometry. Even a positive exploratory grid would require later formal
confirmation; no result here establishes improvement over paper-reported performance.
