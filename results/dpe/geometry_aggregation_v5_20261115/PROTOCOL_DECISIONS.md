# Preregistered decisions — DPE geometry v5

{
  "candidates": [
    {
      "id": "U0",
      "method": "U0",
      "alpha": 0.0
    },
    {
      "id": "E1_g1",
      "method": "E1",
      "gamma": 1.0,
      "alpha": 0.25
    },
    {
      "id": "E1_g5",
      "method": "E1",
      "gamma": 5.0,
      "alpha": 0.25
    },
    {
      "id": "G1_g5",
      "method": "G1",
      "gamma": 5.0,
      "alpha": 0.25
    },
    {
      "id": "G1_g20",
      "method": "G1",
      "gamma": 20.0,
      "alpha": 0.25
    },
    {
      "id": "G1_g50",
      "method": "G1",
      "gamma": 50.0,
      "alpha": 0.25
    },
    {
      "id": "G2_g0.5",
      "method": "G2",
      "gamma": 0.5,
      "alpha": 0.25
    },
    {
      "id": "G2_g1",
      "method": "G2",
      "gamma": 1.0,
      "alpha": 0.25
    },
    {
      "id": "G2_g2",
      "method": "G2",
      "gamma": 2.0,
      "alpha": 0.25
    },
    {
      "id": "G3_m3",
      "method": "G3",
      "M": 3,
      "alpha": 0.25
    },
    {
      "id": "G3_m5",
      "method": "G3",
      "M": 5,
      "alpha": 0.25
    },
    {
      "id": "G3_m10",
      "method": "G3",
      "M": 10,
      "alpha": 0.25
    },
    {
      "id": "G3_m1_DIAGNOSTIC",
      "method": "G3",
      "M": 1,
      "alpha": 1.0,
      "diagnostic_only": true
    },
    {
      "id": "S1_t0.025",
      "method": "S1",
      "tau": 0.025,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S1_t0.1",
      "method": "S1",
      "tau": 0.1,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S1_t0.5",
      "method": "S1",
      "tau": 0.5,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S2_t0.025",
      "method": "S2",
      "tau": 0.025,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S2_t0.1",
      "method": "S2",
      "tau": 0.1,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S2_t0.5",
      "method": "S2",
      "tau": 0.5,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S3_t0.025",
      "method": "S3",
      "tau": 0.025,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S3_t0.1",
      "method": "S3",
      "tau": 0.1,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S3_t0.5",
      "method": "S3",
      "tau": 0.5,
      "T_class": 1.0,
      "alpha": 1.0
    },
    {
      "id": "S4_a0.1",
      "method": "S4",
      "tau": 0.1,
      "T_class": 1.0,
      "alpha": 0.1
    },
    {
      "id": "S4_a0.25",
      "method": "S4",
      "tau": 0.1,
      "T_class": 1.0,
      "alpha": 0.25
    },
    {
      "id": "S4_a0.5",
      "method": "S4",
      "tau": 0.1,
      "T_class": 1.0,
      "alpha": 0.5
    }
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "outer_folds": 3,
  "head_fraction": 0.7,
  "head_seed": "1000+100*seed+fold",
  "AB_split_seed": "7000+31*seed+fold",
  "head_training": {
    "members": 15,
    "epochs": 20,
    "batch_size": 256,
    "lr": 0.001,
    "cov_reg": 500000.0,
    "wd_weight": 10.0
  },
  "selection": "Exclude any seed delta_AA<=-1pp; mean worst-class NLL ascending, mean worst-class accuracy descending, mean AA descending, AA SD ascending, stable config id.",
  "locking": "Primary best eligible non-U0/non-diagnostic; secondary best eligible of opposite shared-weight/class-wise family, or next distinct config if opposite family has none. Maximum2.",
  "screening_budget": "Specific narrow grid from repository detailed v5: shared methods alpha=.25; S4 tau=.1 and alpha .1/.25/.5; T_class1. No expansion.",
  "promotion": {
    "mean_delta_WGA_gt0": true,
    "positive_seeds_at_least2": true,
    "every_delta_AA_gt_minus1": true,
    "provenance": true
  },
  "integration": "Newest pasted request governs: integrate first locked passing candidate, if any; never select an outer-WGA maximum.",
  "historical_test": "Never read",
  "backbone": "one frozen verified augmented400ep ERMstar",
  "head_identity": "fold plus actual checkpoint hashes, never slot alone",
  "bootstrap": {
    "audit": 2000,
    "screen": 500,
    "fold": 500,
    "pooled": 2000,
    "gate": 2000
  }
}

The user-pasted v5 and repository v5 are both preserved/hashed. The repository gives a more specific, smaller screening grid; it is used to interpret the pasted “where specified” alpha values. The pasted latest instruction governs promotion of any passing candidate rather than requiring both to pass. No group WGA enters ranking or choice between passing candidates.

All Phase0 score definitions and audit bins are fixed before group diagnostics. Group-blind fitting/selection code never receives group arrays. G2 uses a separate scaler for each actual OOF fold ensemble; within a fixed new nested ensemble fit on B only. A minimum30 fit observations per ensemble is required; no shared OOF-slot scaling.

Train135 new nested A-only heads with the exact prior no-group20-epoch schedule; do not copy v3/v4 heads. The forced epoch1-kill/recovery head is reused as v5 seed0/fold0/head00. Seeds/splits match prior verification for comparability, so prior development inspection remains a limitation; no claim of fresh untouched confirmation.

The two-family shortlist is fixed to assess both shared expert mixtures and class-wise aggregation in nesting when eligible, without selecting by subgroup results. No class-score temperature fitting or extra sensitivity is permitted. S4 residual base is S1. Top1 diagnostic cannot be shortlisted.

Bootstrap intervals condition on fixed fitted pipelines; image IDs, not expert rows, are resampled. Gate resamples align the same IDs across all seeds. WGA is computed on pooled unique C images within each seed, never the average of fold minima.

## Revision r2 - filesystem recovery only
Transient Windows/OneDrive rename locks interrupted status writing and seed2/fold2/head01 epoch checkpoint replacement. Completed tasks and valid epochs were retained. Bounded rename retries preserve atomic fsync/replace semantics; the v5-local epoch adapter has AST-identical functions to the historical adapter, differing only in persistence imports. Nine tests pass including simulated lock/retry and adapter AST equality. Old source/manifest/failure logs remain preserved. No algorithm/grid/split/selection change.

## Revision r3 - conditional production integration
G1_g50 passed; S2_t0.025 failed seed consistency. Added opt-in distance inference and versioned fixed metadata; uniform remains default. Existing 412 committed tasks and experimental results retained. New acceptance covers exact cached G1 probabilities/weights, actual DPE API, metadata and CLI. The legacy Torch mean and NumPy mean differ by floating-point reduction order at <=2e-7; full nested baseline labels are checked identical, while the legacy API itself remains bit-for-bit unchanged. This numerical check does not relax any metric gate.
