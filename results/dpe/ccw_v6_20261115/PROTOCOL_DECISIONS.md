# Frozen CCW v6 protocol

{
  "model": {
    "schema_version": 1,
    "rho": 0.5,
    "beta": 1.0,
    "gamma": 50.0,
    "alpha": 0.25,
    "kappa": 25.0,
    "sigma_floor": 0.02,
    "std_ddof": 0,
    "z_quantiles": [
      0.3333333333333333,
      0.6666666666666666
    ],
    "boundary_quantile": 0.2
  },
  "methods": [
    "U0",
    "G1_frozen",
    "G2_z",
    "CCW_A",
    "CCW_FULL"
  ],
  "inner_folds": 4,
  "inner_seed": "61000+100*seed+outer_fold",
  "normalizer": "B-inner-train per actual head; ddof0; sigma floor1e-6; reference medians",
  "bins": "2 unweighted pooled z quantiles across heads within each B-inner-train actual ensemble; non-strict edges -> single-bin backoff",
  "parent": "predicted-class weighted marginal shrunk to global with kappa25; global raw weighted mean then0; no self-reference",
  "selection": "Mean worst-class NLL over9 outer B-OOF evaluations, then mean worst-class accuracy, seed-pooled AA, lower concentration, stable method order. Strict per-seed AA delta_U0>-1pp.",
  "seed_pool": "For repeated B image IDs across outer folds, average predictions per image before seed metrics/AA floor; no independent-row assumption.",
  "proposal_gate": "Primary FULL must be AA-eligible and outrank G1. If best eligible overall is CCW_A or CCW_FULL, lock that one; otherwise NO_GO. No G1/G2/U0 promotion and no CCW_A rescue if FULL loses to G1.",
  "retrospective": "Replay only after a GO lock; if NO_GO, retain only Gate0 U0/G1 diagnostic metrics and skip new CCW C replay.",
  "optional_random_controls": false,
  "head_retraining": false,
  "historical_test": "Never read",
  "fresh_required_for_integration": true
}

No retraining or new architecture. Only matching v5 B caches may fit/select. All B-inner statistics, labels, bins, boundary quantile and table support use inner-training only. B group fields are never exported or read by the fitter/selector.
Quantile edges are pooled across normalized head observations within each actual ensemble, not pooled head identities across outer folds. kappa25 support is effective image-weight units: each image weight divided by15.
Seed B pooling averages duplicate image predictions across outer ensembles before AA floor/seed metrics. Mean proxy ranks nine outer B-OOF evaluations; uncertainty clusters on unique IDs. Exactly the five prescribed methods; optional random permutation controls omitted.
A CCW_A best-proxy result can be locked only if primary FULL itself outranks G1 and satisfies AA floor; otherwise NO_GO. This prevents rescuing a failed primary by changing beta/rho. No G1/G2/U0 control is a new promoted method.
If NO_GO, no new CCW C replay or full-B deployment refit; retain Gate0 U0/G1 retrospective results and all B failures. If GO, full-B refit and all frozen control C diagnostics remain exploratory. No fresh data means no integration regardless of C metrics.
Original official labels are retained, including the documented species-label caveat. C and historical test labels never enter fit/selection. A new filename or different backbone on the same previously examined images is not fresh confirmation.

## Revision r2 - report serialization only
After all B fits, selection and NO_GO retrospective skip were committed, missing-fresh-data decision serialization failed on a duplicate reason keyword. This keyword-only dictionary construction was fixed and a regression test added. Nine tests pass. r1 source/manifest/failure state retained; all completed scientific artifacts unchanged. Resume repeats only the uncommitted confirmation report and final reporting. No model, target, normalization, data, split, selector or method parameter changed.
