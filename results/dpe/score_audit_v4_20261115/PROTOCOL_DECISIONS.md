# Preregistered v4 decisions

{
  "maps": [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6"
  ],
  "knots": {
    "R1": [
      [
        0.5,
        0
      ],
      [
        0.515,
        0.25
      ],
      [
        0.53,
        0.5
      ],
      [
        0.54,
        0.75
      ],
      [
        0.55,
        1
      ]
    ],
    "R2": [
      [
        0.5,
        0
      ],
      [
        0.52,
        0.5
      ],
      [
        0.54,
        1
      ]
    ],
    "R3": [
      [
        0.5,
        0
      ],
      [
        0.6,
        0.5
      ],
      [
        0.75,
        1
      ]
    ],
    "R4": [
      [
        0.5,
        0
      ],
      [
        0.75,
        0.5
      ],
      [
        0.95,
        1
      ]
    ],
    "R6": [
      [
        0.5,
        1
      ],
      [
        1.0,
        1
      ]
    ]
  },
  "alphas": [
    0.1,
    0.25
  ],
  "betas": [
    0.5,
    1.0,
    2.0
  ],
  "kappa": 50,
  "screen": {
    "alpha": 0.1,
    "beta": 1.0
  },
  "selection": "mean across seeds of worst-class NLL, then mean NLL, then map ID/alpha/beta; each seed delta AA strictly > -1pp",
  "shortlist": "top two eligible maps, full six alpha/beta grid, lock ONE globally best recipe before groups/outer evaluation",
  "bootstrap": {
    "per_output": 500,
    "pooled_gate": 2000,
    "seed": 24019
  },
  "outer_seeds": [
    0,
    1,
    2
  ],
  "outer_folds": 3,
  "promotion": {
    "mean_delta_WGA_positive": true,
    "positive_full_seeds_at_least": 2,
    "each_delta_AA_gt_minus1": true
  },
  "R6_representation": "raw",
  "optional_ablations": false,
  "outer_reuse": "identical frozen v3 heads and A/B/C splits; previously inspected development, not untouched confirmation"
}

1. Reuse all 135 actual v3 heads only after checking per-head fit IDs, hashes, final epoch 20, independent deterministic splits and prediction identity. No LCMSW weights or models are reused. Epoch checkpoints remain in their original location. This saves training and preserves evidence.
2. Original nested C outcomes were inspected for v3. These are nested engineering comparisons on reused development data, not an untouched confirmation. Method maps come verbatim from user v4; no adaptation to C outcomes.
3. Fit R4 shared NLL temperature by deterministic bounded minimization in [0.001,10]; R6 uses raw probabilities. Lookup pools expert slots across folds and does not identify heads. No optional ablations.
4. Select exactly one primary recipe by the group-blind proxy, with per-seed AA floor. No outer WGA selection between recipes. Diagnostic oracle never ranks. Test data only enters Phase 0 module.
5. Fixed raw/calibrated Phase 0 edges and 15-bin ensemble ECE; uncertainty resamples unique images within true group after lock. Across seeds use identical image resamples, not 3597 independent rows.
6. Phase 0 uses exact saved gate forward logits, verified against cached direct-softmax weights; missing OOF fold uses -1, meaning final development ensemble. No D19 retraining.
7. Every task has a checksum commit; model/scaler/table and each inference export are separate tasks. Kill/restart of a v4 task and the prior real epoch-1 head recovery are verified before research evaluation.

## Revision r2
Descriptive Spearman boolean-range bug fixed before any Phase 0 task commit; regression test added. D19 external checkpoints/source caches added to input guard. Both acceptance and input tasks retained byte-for-byte. r1 manifest/state/source snapshot and failure log preserved. No scoring math, split, grid, selection or gate change.

## Revision r3
Initialization now supports an absent lineage history on a fresh run. All completed artifacts and algorithms are unchanged; r2 manifest/source remain preserved. A reporting addendum corrects the descriptive broken-group sequence to 2,3,2 from the saved CSVs; all original reports/metrics remain intact.
