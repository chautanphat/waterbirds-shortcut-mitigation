# Recorded implementation decisions

The handoff authorizes bounded grids and independent engineering decisions.
The executed recipe is an exploratory extension with these explicit choices:

- Three-fold OOF predictions are split 80/20 by class. Repeated source seeds
  rerun folds, heads and gates; the frozen backbone is shared. Each OOF row is
  excluded from fitting its own predicting ensemble. This is the cross-fitting
  design specified in the handoff, not a completely independent outer test.
- Heads use class balancing, the final 20th epoch, batches up to 256 including
  partial batches, SGD .001/momentum .9, entropic scale 30, prototype penalty
  10 and inverse-decayed IPS coefficient starting at 500,000. These choices
  deliberately remove group balancing and WGA checkpoint selection from the
  new adaptation. Comparing it to the old annotated DPE changes protocol as
  well as aggregation; matched U0 isolates aggregation more directly.
- Neural screening uses full-batch AdamW .001, at most 100 epochs, patience 10,
  gradient clipping 5 and KL coefficients 0/.01/.1. Static optimization uses
  up to 150 full-batch steps; S2 independently uses constrained SLSQP.
- Temperature fitting searches 25 logarithmically spaced values from .25 to
  4 on gate-training NLL, separately per expert. It never uses selection/test
  labels. D13 uses teacher beta 5 and coefficient .1. D14 adds .2 class-tail
  CVaR. D15 screens gamma 2; gamma 1 is implemented but not in the bounded
  screening grid. D17 drops experts with probability .2 during fitting only.
- D11 is a joint MLP with 64/32 hidden units and all concatenated per-expert
  inference features (probabilities plus confidence, margin, entropy and
  mean-relative probabilities). This augments the handoff's probabilities-only
  D11 input; interpret it as that declared joint-gate variant. D12 uses a fixed
  random 16-dimensional embedding projection, fitted by neither labels nor
  test statistics. D9 uses standardized raw frozen embeddings.
- Correctness trees/logistic models pool expert rows but group the inner
  calibration folds by original sample, so expert rows for one image never
  cross those fold boundaries. The final correctness calibrator uses OOF
  predictions. Forests have 32 trees; histogram boosting has 40 iterations.
- H5 uses five train-confidence quantile bins and Laplace-smoothed correctness.
  R5 maps estimated reliability below .5/.65/.8 to 0/.25/.5, otherwise 1.
  R7 switches at the training disagreement 75th percentile. No test correctness
  determines any weight. Empty reliability sums fall back to uniform.
- S4 tunes a binary threshold in three inner training folds over .3–.7.
  Threshold .5 and class-0 ties are used otherwise. This is a separate decision
  rule. S6 renormalizes class-dependent scores and S7 is logistic stacking;
  both depart from a pure convex probability mixture. R4/R8 use hard votes;
  their soft outputs include a tiny mean-probability tie breaker.
- Feature removal ablations retain full probability vectors, which contain
  redundant confidence/margin information. Top-k acts on already-computed
  member outputs and does not demonstrate CNN/head inference savings.
- Main A18 initially reserves the additional-head budget. A separate locked
  plan adds a five-fold diagnostic using the same winner and final full-val
  heads, without selecting configurations using test outcomes.

All actual configurations and source snapshots are retained in the output
directory. None of these choices establishes improvement before measurement.
