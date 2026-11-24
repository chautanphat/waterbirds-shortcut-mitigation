# Direction 2 · Phase B · Slot 2 Counterfactual Causality Probe (Validation Only)

**Causal BG-Flip Gate ($\ge 50\%$ recovery on matched or neutral BG):** **PASS**

## Persistent ERM-Only Failures (26 Images): Accuracy Across Counterfactual Backgrounds (%)

| Cohort | Head | Orig (Minority BG) | Matched Real BG | Flipped Real BG | Neutral Gray BG | OOD Synth BG |
|---|---|---:|---:|---:|---:|---:|
| standard_erm | `dfr_l2` | 6.9% | 96.2% | 69.2% | 76.2% | 86.9% |
| standard_erm | `oracle_mask` | 40.8% | 98.5% | 73.8% | 81.5% | 85.4% |
| erm_star | `dfr_l2` | 100.0% | 100.0% | 94.2% | 98.1% | 98.1% |
| erm_star | `oracle_mask` | 100.0% | 100.0% | 94.2% | 94.2% | 94.2% |

## Layer-Wise Foreground-Cell Feature Drift (`orig -> neutral_gray`, Inside CUB Bird Mask)

| Subset | Cohort | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|---|---|---:|---:|---:|---:|
| persistent_erm_only | standard_erm | 0.0643 | 0.1501 | 0.3726 | 0.8512 |
| persistent_erm_only | erm_star | 0.0648 | 0.1359 | 0.3129 | 0.6862 |
| passing_control | standard_erm | 0.0590 | 0.1428 | 0.3465 | 0.6622 |
| passing_control | erm_star | 0.0594 | 0.1256 | 0.2827 | 0.5642 |
