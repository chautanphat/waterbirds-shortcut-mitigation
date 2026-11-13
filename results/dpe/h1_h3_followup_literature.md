# DPE follow-up: multi-prototype inference and robust aggregation

Research note, 2026-11-13. Waterbirds and ResNet-50 remain fixed. All proposals below
reuse trained backbones and existing prototypes; no new benchmark result is reported.
H1/H2/H3 refer to [the project report](../../report/main.tex).

## Evidence that constrains the next experiments

The [normal-ERM three-track exploration](three_track_exploration_356d73809a1222d1/README.md)
already evaluated re-pairing and global all-30-prototype pooling on five sources.
Native re-pairing changed no decisions; global pooling's highest observed mean
was 89.2960% versus matched uniform 89.2337%, with two wins, two losses and one tie.
This is not a confirmed improvement. Combinations were disabled. Distinct larger
prototype subsets per receiving head were not implemented in that experiment.

The [two-source ERM* study](ermstar_neural_two_seed_20261113/README.md) evaluated H2
only. Plain/dropout MLPs improve their 720-head-pool uniform baseline descriptively;
neither exceeds our full-validation DPE reproduction in mean. Normal-ERM and ERM*
results must remain separate. Paper-reported DPE results, our full-validation DPE
reproductions, and our split-pool baselines are three distinct comparators.

The [earlier H2 literature shortlist](h2_aggregation_literature.md) remains relevant,
but equal generalized means and average group-balanced Brier weighting have already
been explored on normal ERM. Repeating those unchanged is low priority.

## Literature reviewed and its limits

Searches used SciSpace for multi-prototype classification, ensemble complementarity,
last-layer retraining and distributionally robust aggregation. Method/limitation
extraction was requested for five papers. Extracted contexts frequently omitted
equations or full protocols; these omissions are not proof that a method or
benchmark is absent from the original paper. Original SoftTriple and group-DRO
method sections were subsequently read to verify their central formulas.

| Source | Established mechanism relevant here | Inspection and transfer boundary |
| --- | --- | --- |
| [SoftTriple, Qian et al., ICCV 2019](https://scispace.com/papers/softtriple-loss-deep-metric-learning-without-triplet-2tfrxagser) | Multiple learned centers per class; within-class soft assignments produce a weighted similarity, followed by a class-level loss. | Original section 3.1/equation 8 checked in [full text](https://arxiv.org/html/1909.05235). It learns centers/embeddings for metric learning. Frozen DPE subset inference is our adaptation, not SoftTriple reproduction. |
| [Multi-Proxy Learning from an Entropy Optimization Perspective, 2022](https://scispace.com/papers/multi-proxy-learning-from-an-entropy-optimization-19s71120) | Multiple learned proxies describe intra-class structure; entropy objectives control assignments. | SciSpace excerpt/extraction only; exact objective not verified. Conceptual motivation, not implementation specification or Waterbirds evidence. |
| [Group DRO, Sagawa et al., 2019/2020](https://scispace.com/papers/distributionally-robust-neural-networks-for-group-shifts-on-560k9rc9z1) | Optimize maximum group loss, with regularization to control generalization. | Original equations 4–5 and Waterbirds setup checked in [full text](https://arxiv.org/html/1911.08731). Applying the objective only to frozen DPE weights differs from its original model training. |
| [Complementary prioritized ensemble selection, 2016](https://scispace.com/papers/complementary-prioritized-ensemble-selection-1dbynscqud) | Select classifiers that cover ensemble weaknesses rather than rewarding generic diversity alone. | Abstract/extraction only; exact priority-queue procedure not verified. Our group-aware pairing proposal is inspired by the idea, not a reproduction. |
| [On the Unreasonable Effectiveness of Last-layer Retraining, 2025](https://scispace.com/papers/on-the-unreasonable-effectiveness-of-last-layer-retraining-vrrd3aom5ezy) | Reports evidence that held-out group balance explains benefits previously attributed to representation geometry. | Abstract/extraction only. Supports careful equal-label-budget controls, not a causal explanation of our own gains. |

None supplies direct evidence that these adaptations improve our DPE. Search
metadata can misclassify publication type; publication labels alone are not used
to certify peer review. The 2025 LLR paper is treated as a preprint here.

## Priority 1: the report's untested H3, distinct subsets

For receiving head i, define a class-balanced subset S_i of the existing prototypes.
With fixed prototype score z_j(x), compute

`P_i(c|x) = sum_{j in S_i, class(j)=c} exp(z_j(x)/T) / sum_{j in S_i} exp(z_j(x)/T)`.

Average the 15 resulting binary probability vectors first. This is the report's
H3 operation, equivalent to class-level softmax of log-sum-exp scores. It is not
SoftTriple's soft-assignment-weighted similarity; those operations differ.

Initial design, before seeing new test results:

- Prototype pairs indexed 0–14; head i receives original pairs with indices
  `(i+r) mod 15` for r=0,…,k−1. Use k=1,2,3,5,15 prototypes **per class**.
- k=1 recovers original pairs. k=15 is the global-pooling special case.
- Cyclic exposure uses each original prototype exactly k times across the 15
  subsets, preventing a prototype-frequency change from masquerading as subset gain.
- Add one fixed random permutation of pair ordering as a membership control;
  do not choose it after test inspection. Pair order is otherwise arbitrary.
- Initial temperatures T=1,.1,.05, independently reported. These sharper settings
  are proposals, not fitted optima. Include identically tempered original pairs
  to distinguish subset effects from ordinary sharpening. Widen later only with
  a documented reason, rather than silently replacing the original grid.
- Native scores use each prototype's source-head distance scale; a shared-scale
  control uses one predeclared within-source common scale. Audit native scale
  semantics and original-pair reconstruction first. Do not assume a scale learned
  for one original pair transfers perfectly across pairs.

**Why explore it:** subsets can allow local within-class alternatives while
retaining differences between receiving heads. Whether trained DPE prototypes
actually encode useful modes is unknown; multiple centers alone do not prove it.

**Why it can fail:** subsets can share nearly identical decisions, add background
modes, or remain in the same weakly nonlinear score regime as the earlier grid.
Measure per-group changes, prediction disagreement, margins and prototype usage.

## Priority 2: H1 under explicit score-sharpness controls

Keep the previously tested permutations, exhaustive pairs and original pairs as
controls. Apply identical predeclared temperature settings to all of them.
With `sigmoid(u) ≈ 1/2 + u/4`, averaging balanced pairings with identical prototype
marginals cancels the pairing at first order. This is a mathematical explanation
for possible insensitivity, not a proof of exact equivalence or a causal diagnosis
of the observed run. Sharpening may reveal a difference and may also worsen WGA.

A second, later adaptation is **complementarity-aware matching**: prefer pairs
whose errors help groups where the current ensemble is weak, while limiting
repeated prototype use. First inspect whether the candidate pair pool has usable
complementarity at all. An oracle based on test labels is diagnostic only and
must never select a deployed ensemble. Any actual learned matching uses only the
479 fitting examples and is labeled a fitted H1 procedure; it no longer has the
same label consumption as a predetermined permutation.

Do not describe this as the cited complementary-selection algorithm until its
full procedure is independently read. Avoid searching huge pairing grids on the
small fitting set without reporting that search and its overfitting risk.

## Priority 3: worst-group H2, then combinations

Our current neural objective is the **average of four group losses**. This gives
groups equal contribution but does not minimize the maximum group loss. Likewise,
the previous Brier experiment optimized average group-balanced loss, not minimax.

For fixed probability outputs q_h(x) and mixture weights w on the simplex, propose

`min_w max_g mean_{x in g}(sum_h w_h q_h(x) - y)^2 + lambda ||w-u||^2`,

where u is uniform. Implement via an epigraph variable bounding each group's
quadratic loss. This is a convex frozen-prediction adaptation; it is distinct from
previous joint-NLL or average-Brier weighting. Check feasibility and optimality
numerically. Initial shrinkages .01 and .1 reuse earlier scales while avoiding
another unconstrained concentration-only experiment. Report both independently.

A separately reported extension replaces Brier with group NLL, or fits the existing
small logistic aggregator to maximum group loss. Convex-mixture and unrestricted
logistic fits must remain distinguished. Small groups can dominate through noise;
regularization is essential to explore, and a better worst-group fit loss still
does not guarantee better test WGA. Do not import original group-DRO regularization
strengths as though their scale and model capacity were identical.

Apply H2 to (a) original pairs, (b) H1's changed pair predictions, and (c) distinct
H3 subset heads. Refit it on the corresponding 479-example predictions each time;
original neural weights are not automatically valid for changed head inputs.

**Degenerate combination to avoid:** if all receiving heads produce the same
all-30-prototype probability p(x), any nonnegative normalized H2 mixture returns
`sum_i w_i(x) p(x) = p(x)`. This holds for input-dependent convex gates too.
An MLP on duplicated p(x) can recalibrate it, but cannot exploit diversity between
identical heads. Compare such a fit to scalar calibration instead of attributing
its gain to ensemble weighting.

## Fair comparisons and decision criteria

Run normal ERM and ERM* as independent studies. Use both available arms when
artifacts permit; do not retrain a backbone merely to build this grid.

1. Full-validation heads: original uniform versus **label-free predetermined**
   H1/H3 variants, using the same backbone and head tensors. This directly compares
   inference rules without changing head data allocation. Fitting labels must not
   be presented as held out if those heads already used all validation examples.
2. Split-pool heads: original uniform, H1-uniform, H3-uniform, H2 alone,
   H1+H2 and H3+H2. Learned variants share the same 479 fitting examples. Fixed
   geometry variants do not consume fitting labels unless a score is fitted.
3. Fit simple controls on those same 479 labels: scalar temperature, affine
   calibration/threshold and logistic aggregation. Evaluate the same objective
   variants when attributing gains to an architecture rather than loss choice.
   Earlier controls on normal ERM do not substitute for ERM* controls.
4. Retain our full-validation original-DPE reproduction separately as the complete
   procedure reference. Keep the original paper's published performance separately
   labeled; do not relabel our reproduction as a paper-reported number.
5. Report every declared configuration, group metrics, decision/error changes,
   fitting diagnostics and paired source deltas. No extra candidate-selection split
   is required. The test set cannot select membership, temperatures or a winner.
   Descriptive inspection of all results is exploratory, not untouched confirmation.

Before any Kaggle computation, prepare a fresh short repository-import notebook,
not a recycled recovery notebook. Keep all implementation in readable modules and
settings in a plain configuration file. Test original-pair recovery, equal-score
and extreme-temperature cases, class balance, duplicated-head degeneracy, train/test
ID separation, cache invalidation and interrupted recovery. Use readable settings
and row identities, with no generated source strings, encodings or file checksums.

Formal confirmation remains later. The immediate objective is to identify a useful
inference/aggregation change under transparent data budgets, not to claim that a
literature-inspired construction is novel or guaranteed to beat the paper.
