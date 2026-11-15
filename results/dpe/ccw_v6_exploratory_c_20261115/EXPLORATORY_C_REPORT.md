# CCW v6: requested exploratory held-out C evaluation

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

The original B-selection NO_GO is unchanged. No production promotion or default inference change is authorized by these results. This is the requested measurement after the failed preregistered B proxy gate; it is not a replacement gate.

Nine matching v5 prototype ensembles (3 seeds × 3 outer folds); all 240 B images per ensemble fit the frozen recipe. C has 1,199 distinct images per seed, reused across all three seeds (3,597 prediction rows). A, B and C are disjoint within each ensemble. All normalization, pooled distance-z quantiles, B-only boundary-confidence quantile, shrinkage tables and advantage scale use B only. C labels and groups are used only for baseline verification and post-inference evaluation. No retraining, new configurations or parameter optimization.

Frozen parameters: rho=0.5, beta=1, gamma=50, alpha=0.25, kappa=25. The exact frozen G2 control uses z-score geometry (gamma=1); CCW-A sets beta=0. Uniform and G1 reproduce v5 probabilities, predictions and AA/WGA within 1e-12 (observed probability error zero). All 135 ordered head hashes and all B/C arrays match provenance.

| Method | Mean AA (%) | Mean WGA (%) | ΔAA vs Uniform (pp) | ΔWGA vs Uniform (pp) | ΔWGA vs G1 (pp) |
|---|---:|---:|---:|---:|---:|
| U0 | 95.746455 | 92.481203 | +0.000000 | +0.000000 | -0.501253 |
| G1_frozen | 95.857659 | 92.982456 | +0.111204 | +0.501253 | +0.000000 |
| G2_z | 95.829858 | 92.982456 | +0.083403 | +0.501253 | +0.000000 |
| CCW_A | 95.802057 | 92.982456 | +0.055602 | +0.501253 | +0.000000 |
| CCW_FULL | 95.829858 | 92.982456 | +0.083403 | +0.501253 | +0.000000 |

| Seed | Method | AA (%) | WGA (%) | Worst-class accuracy (%) | ΔAA vs Uniform (pp) | ΔAA vs G1 (pp) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | U0 | 95.829858 | 90.977444 | 93.609023 | +0.000000 | +0.000000 |
| 0 | G1_frozen | 95.829858 | 90.977444 | 93.984962 | +0.000000 | +0.000000 |
| 0 | G2_z | 95.829858 | 90.977444 | 93.233083 | +0.000000 | +0.000000 |
| 0 | CCW_A | 95.913261 | 90.977444 | 93.609023 | +0.083403 | +0.083403 |
| 0 | CCW_FULL | 95.996664 | 90.977444 | 93.609023 | +0.166806 | +0.166806 |
| 1 | U0 | 95.913261 | 93.233083 | 94.736842 | +0.000000 | -0.166806 |
| 1 | G1_frozen | 96.080067 | 93.984962 | 95.112782 | +0.166806 | +0.000000 |
| 1 | G2_z | 96.080067 | 93.984962 | 95.112782 | +0.166806 | +0.000000 |
| 1 | CCW_A | 95.913261 | 93.984962 | 95.112782 | +0.000000 | -0.166806 |
| 1 | CCW_FULL | 95.913261 | 93.984962 | 95.112782 | +0.000000 | -0.166806 |
| 2 | U0 | 95.496247 | 93.233083 | 94.360902 | +0.000000 | -0.166806 |
| 2 | G1_frozen | 95.663053 | 93.984962 | 95.488722 | +0.166806 | +0.000000 |
| 2 | G2_z | 95.579650 | 93.984962 | 95.112782 | +0.083403 | -0.083403 |
| 2 | CCW_A | 95.579650 | 93.984962 | 95.112782 | +0.083403 | -0.083403 |
| 2 | CCW_FULL | 95.579650 | 93.984962 | 95.112782 | +0.083403 | -0.083403 |

Requested questions:

1. Does CCW-Full outperform Uniform in observed WGA? Yes: mean difference +0.501253 pp; per seed [0.0, 0.7518796992481258, 0.7518796992481258]. Paired 95% image-bootstrap CI [-0.286123, +1.253133] pp. This is descriptive exploratory evidence only.
2. Does CCW-Full outperform G1 in observed WGA? No: mean difference +0.000000 pp; per seed [0.0, 0.0, 0.0]. Paired 95% image-bootstrap CI [-0.501253, +0.033359] pp. This is descriptive exploratory evidence only.
3. AA decrease strictly below 1 pp per seed? True versus both references; AA differences (pp): {"U0": [0.16680567139282232, 0.0, 0.08340283569641116], "G1_frozen": [0.16680567139282232, -0.16680567139282232, -0.08340283569641116]}.
4. Actual corrections on the baseline worst group are reported below. WGA depends on the minimum across all four groups, so a net improvement in one group alone need not raise WGA.

| Seed | Baseline | Baseline worst group | n | Baseline acc (%) | Full acc (%) | Corrected | Broken | Net Δ (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | U0 | 2 | 133 | 90.977444 | 90.977444 | 1 | 1 | +0.000000 |
| 0 | G1_frozen | 2 | 133 | 90.977444 | 90.977444 | 0 | 0 | +0.000000 |
| 1 | U0 | 2 | 133 | 93.233083 | 93.984962 | 1 | 0 | +0.751880 |
| 1 | G1_frozen | 2 | 133 | 93.984962 | 93.984962 | 0 | 0 | +0.000000 |
| 2 | U0 | 2 | 133 | 93.233083 | 93.984962 | 1 | 0 | +0.751880 |
| 2 | G1_frozen | 2 | 133 | 93.984962 | 93.984962 | 0 | 0 | +0.000000 |

5. Advantage term versus CCW-A: mean weight total variation 0.00688089, maximum 0.04008197; largest individual expert-weight change 0.04008197. Top expert changes on 134/3597 seed-image rows; prediction changes on 1 rows (1 unique images). Mean absolute probability shift 0.00011109; maximum 0.00143982. Mean AA change +0.027801 pp; mean WGA change +0.000000 pp. These quantities distinguish weight movement from actual classification effects; no post-hoc tuning follows.

Group definitions: 0=landbird/land, 1=landbird/water, 2=waterbird/land, 3=waterbird/water. Per-seed counts are 467, 466, 133, 133. All per-group accuracies, corrections and breakages are in EXPLORATORY_C_PER_GROUP.csv; individual changes versus Uniform, G1 and CCW-A are in EXPLORATORY_C_FLIPS.csv. All full probability shifts and all 15 expert weights are retained in the NPZ, including unchanged predictions.

Uncertainty uses 2,000 paired, group-stratified draws of unique image IDs; each draw is shared across all methods and all three seeds. Mean CIs average seed metrics within each draw. Treating the 3,597 repeated image rows as independent would be incorrect. Intervals are conditional on these fixed fitted pipelines; three head seeds on one shared frozen ERM* backbone do not establish variability across newly trained backbones or new cohorts.

Original official labels are retained, including the documented species-label caveat; no relabeling was performed. These are previously inspected development images, not a fresh untouched test cohort. Neither positive nor negative results confirm generalization. WGA remains primary; AA floor checks use strict > −1 pp per seed. Mean WGA is the average of three per-seed minima, not the minimum of seed-averaged group accuracies.

Crash safety: every full-B fit, fold inference, model evaluation and output table is committed atomically with source/input/config hashes. Real full-B forced termination verified reuse of the committed first fit and replay only of the uncommitted second fit, equal to an uninterrupted control. Acceptance fits are reused in the main run. Fits are deterministic; bootstrap RNG states are saved. All original completion artifacts and the original NO_GO are audited unchanged.

Resume / verify:

```powershell
.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase exploratory_c --resume
.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase exploratory_c --resume --verify
```

Main outputs: EXPLORATORY_C_RESULTS.csv (fold, seed and mean metrics, paired deltas and CIs); EXPLORATORY_C_PER_GROUP.csv; EXPLORATORY_C_FLIPS.csv; EXPLORATORY_C_PREDICTIONS.npz; PREDICTION_SCHEMA.json; ADVANTAGE_AND_WORST_GROUP.json. Fits and complete inference internals remain under fits/ and predictions/. Recovery receipts and RNG state remain under acceptance/ and uncertainty/.
