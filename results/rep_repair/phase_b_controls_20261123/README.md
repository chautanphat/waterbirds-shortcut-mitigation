# Direction 2 · Phase B · Slot 1 Control Ladder Results (Validation Only)

**Controls Survive DAR Gate ($\ge 1.5\text{ pp}$ ERM* $-$ std-DAR gap):** **YES** (gap = `12.2951` pp; vs. Oracle-Mask-Pool gap = `5.082` pp).

| Cohort | Method | OOF WGA (%) | Overall (%) | G01 (%) | G10 (%) | Persistent Failure Rescue (%) |
|---|---|---:|---:|---:|---:|---:|
| standard_erm | `dfr_l2` | 81.64 ± 0.93 | 94.20 | 94.59 | 81.64 | 4.6% |
| standard_erm | `dar` | 80.33 ± 1.16 | 94.58 | 94.97 | 80.33 | 36.1% |
| standard_erm | `oracle_mask_pool` | 87.54 ± 0.37 | 95.38 | 94.84 | 87.54 | 40.0% |
| standard_erm | `h2t_dfr` | 82.13 ± 0.90 | 94.31 | 94.55 | 82.13 | 12.3% |
| standard_erm | `bn_stat_only` | 80.82 ± 0.45 | 94.26 | 94.88 | 80.82 | 11.5% |
| standard_erm | `affine_dfr` | 89.34 ± 1.53 | 95.46 | 95.10 | 89.34 | 63.1% |
| standard_erm | `layer4_ft` | 86.07 ± 2.66 | 94.68 | 95.14 | 86.07 | 55.4% |
| erm_star | `dfr_l2` | 92.62 ± 2.32 | 97.16 | 97.36 | 92.62 | 100.0% |
| erm_star | `dar` | 90.57 ± 0.58 | 97.21 | 96.93 | 90.57 | 94.2% |
| erm_star | `oracle_mask_pool` | 92.21 ± 0.58 | 97.54 | 96.83 | 92.21 | 94.2% |
| erm_star | `h2t_dfr` | 91.80 ± 1.16 | 97.29 | 97.04 | 91.80 | 92.3% |
| erm_star | `bn_stat_only` | 90.98 | 96.83 | 96.83 | 90.98 | 84.6% |
| erm_star | `affine_dfr` | 92.62 | 96.91 | 96.62 | 92.62 | 96.2% |
| erm_star | `layer4_ft` | 90.98 | 97.00 | 97.04 | 90.98 | 92.3% |
