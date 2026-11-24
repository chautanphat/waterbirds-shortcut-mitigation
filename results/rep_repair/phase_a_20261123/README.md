# Direction 2 · Phase A Results (Validation Only)

**Pre-Registered Gate 1 Decision:** **NO-GO** (0/4 qualifying standard-ERM seeds).

## Frozen Post-GAP Readout Ladder (5-Fold OOF, Relabelled Validation = 1,199)

| Cohort | Method | OOF WGA (%) | Overall (%) | G01 (%) | G10 (%) | Persistent Failure Rescue (%) |
|---|---|---:|---:|---:|---:|---:|
| standard_erm | `dfr_l2` | 80.98 ± 1.07 | 94.10 | 94.50 | 80.98 | 4.6% |
| standard_erm | `dfr_l1` | 88.03 ± 1.49 | 93.68 | 91.59 | 88.03 | 38.5% |
| standard_erm | `dpe_oof` | 91.54 ± 0.54 | 93.28 | 91.54 | 94.10 | 43.9% |
| standard_erm | `frozen_rbf_svm` | 78.03 ± 1.07 | 95.03 | 96.19 | 78.03 | 33.9% |
| standard_erm | `frozen_mlp` | 91.02 ± 1.04 | 93.31 | 92.30 | 91.15 | 36.9% |
| standard_erm | `frozen_cosine_knn` | 86.39 ± 0.93 | 93.48 | 91.04 | 86.39 | 34.6% |
| erm_star | `dfr_l2` | 92.21 ± 1.74 | 97.12 | 97.25 | 92.21 | 98.1% |
| erm_star | `dfr_l1` | 91.39 ± 0.58 | 95.75 | 94.08 | 91.39 | 78.8% |
| erm_star | `dpe_oof` | 95.00 ± 0.11 | 96.62 | 95.77 | 95.90 | 92.3% |
| erm_star | `frozen_rbf_svm` | 87.70 ± 0.00 | 97.00 | 97.04 | 87.70 | 86.5% |
| erm_star | `frozen_mlp` | 94.82 ± 1.05 | 95.91 | 94.82 | 96.72 | 92.3% |
| erm_star | `frozen_cosine_knn` | 91.39 ± 1.74 | 96.54 | 94.50 | 91.39 | 88.5% |

## Gate 1 Per-Seed Breakdown (Standard ERM)

| Seed | Best Nonlinear Readout | Nonlinear WGA (%) | ERM* Ref WGA (%) | Gap (pp) | Unresolved by Both MLP & SVM | Qualifies |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | `frozen_mlp` | 89.34 | 95.00 | 5.66 | 10/26 (38.5%) | NO |
| 1 | `frozen_mlp` | 91.80 | 95.00 | 3.20 | 9/26 (34.6%) | NO |
| 2 | `frozen_mlp` | 90.98 | 95.00 | 4.02 | 10/26 (38.5%) | NO |
| 3 | `frozen_mlp` | 90.98 | 95.00 | 4.02 | 8/26 (30.8%) | NO |
| 4 | `frozen_mlp` | 91.97 | 95.00 | 3.04 | 7/26 (26.9%) | NO |
