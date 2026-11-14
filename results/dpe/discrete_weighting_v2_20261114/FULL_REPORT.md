# Discrete reliability v2 complete research report

All test results are exploratory on a repeatedly inspected test set. Three refitted-head seeds share one backbone. No test labels or group annotations enter fitting, calibration or recipe selection. The 29 configurations were fixed before results; the single family comparison uses fixed OOF selection IDs. Calibration, bins, thresholds and tables use meta_train only. No CNN or head retraining was performed.

Winner selected by mean held-out accuracy, then BA, then NLL: `{"agreement": true, "alpha": 0.5, "beta": 5, "family": "D5", "head": true}`. Selection accuracy: 96.80555555555556%.

## Winner and matched controls

| Method | Accuracy % | BA % | WGA % |
|---|---:|---:|---:|
| winner | 96.076 ± 0.339 | 94.843 ± 0.308 | 90.343 ± 1.685 |
| raw_uniform | 95.708 ± 0.117 | 95.182 ± 0.053 | 92.761 ± 0.552 |
| representation_uniform | 95.708 ± 0.117 | 95.182 ± 0.053 | 92.761 ± 0.552 |

Accuracy delta vs raw uniform: +0.3682 pp; WGA delta -2.4185 pp. Conditional paired-example bootstrap 95% interval for the mean across these fixed models: [0.17834541479691637, 0.575307789667472] pp. This does not measure independent-backbone uncertainty.

A positive micro-accuracy difference alone is not a robust win; inspect WGA, per-class confusion/recall, flips and seed variation in the full task CSVs. No winner was changed after test evaluation.

## Phase 1 full configuration selection results

| Family | Config | Accuracy % | BA % | NLL |
|---|---|---:|---:|---:|
| D3 | {"family": "D3", "head": true, "kappa": 10} | 96.528 | 94.617 | 0.63610 |
| D12 | {"alpha": 0.5, "calibration": "head", "family": "D12", "head": true} | 95.972 | 94.486 | 0.13857 |
| D2 | {"family": "D2"} | 96.528 | 94.617 | 0.63583 |
| C0 | {"alpha": 0.5, "family": "C0"} | 95.139 | 95.078 | 0.63862 |
| D3 | {"family": "D3", "head": true, "kappa": 200} | 96.389 | 94.528 | 0.63597 |
| D12 | {"alpha": 0.5, "calibration": "raw", "family": "D12", "head": true} | 95.833 | 94.622 | 0.63757 |
| D0b | {"calibration": "head", "family": "D0b"} | 95.000 | 94.989 | 0.15299 |
| D8 | {"alpha": 0.5, "family": "D8", "head": true, "ordinal": false} | 95.833 | 94.622 | 0.63757 |
| D0a | {"calibration": "raw", "family": "D0a"} | 95.000 | 94.989 | 0.63909 |
| D5 | {"agreement": true, "alpha": 0.25, "beta": 3, "family": "D5", "head": true} | 95.556 | 95.120 | 0.63845 |
| D5 | {"agreement": true, "alpha": 0.5, "beta": 5, "family": "D5", "head": true} | 96.806 | 95.922 | 0.63751 |
| D9 | {"alpha": 0.5, "balance": "predicted", "family": "D9", "head": true} | 95.833 | 94.622 | 0.63757 |
| D5 | {"agreement": true, "alpha": 0.75, "beta": 3, "family": "D5", "head": true} | 96.667 | 95.157 | 0.63719 |
| D11 | {"alpha": 0.5, "family": "D11", "head": true, "rank": true} | 95.278 | 94.716 | 0.63805 |
| D9 | {"alpha": 0.5, "balance": "true", "family": "D9", "head": true} | 95.278 | 94.716 | 0.63722 |
| D3 | {"family": "D3", "head": true, "kappa": 50} | 96.389 | 94.528 | 0.63605 |
| D7 | {"alpha": 0.5, "family": "D7", "head": true, "ordinal": true} | 96.250 | 94.664 | 0.63742 |
| D4 | {"alpha": 0.25, "beta": 3, "family": "D4", "head": true} | 95.417 | 95.031 | 0.63833 |
| D10 | {"alpha": 0.5, "family": "D10", "guard": true, "head": true, "top_k": 15} | 95.833 | 94.622 | 0.63799 |
| D5 | {"agreement": true, "alpha": 0.1, "beta": 1, "family": "D5", "head": true} | 95.278 | 95.167 | 0.63895 |
| D10 | {"alpha": 0.5, "family": "D10", "guard": true, "head": true, "top_k": 3} | 96.111 | 94.350 | 0.63449 |
| D4 | {"alpha": 0.1, "beta": 1, "family": "D4", "head": true} | 95.139 | 95.078 | 0.63894 |
| D10 | {"alpha": 0.5, "family": "D10", "guard": true, "head": true, "top_k": 5} | 96.111 | 94.350 | 0.63513 |
| D4 | {"alpha": 0.75, "beta": 3, "family": "D4", "head": true} | 96.111 | 94.350 | 0.63681 |
| D1 | {"calibration": "raw", "family": "D1"} | 95.694 | 91.829 | 0.63909 |
| D4 | {"alpha": 0.5, "beta": 5, "family": "D4", "head": true} | 96.111 | 94.800 | 0.63716 |
| D6 | {"agreement": true, "alpha": 0.5, "family": "D6", "guard": true, "head": true} | 96.250 | 95.340 | 0.63812 |
| D10 | {"alpha": 0.5, "family": "D10", "guard": true, "head": true, "top_k": 10} | 95.972 | 94.711 | 0.63682 |
| D12 | {"alpha": 0.5, "calibration": "shared", "family": "D12", "head": true} | 96.111 | 94.575 | 0.13614 |

## Phase 2 ablations

| ID | Variant | Accuracy % | Delta raw pp | WGA % |
|---|---|---:|---:|---:|
| A0 | full | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A1 | raw | 95.708 ± 0.117 | +0.000 | 92.761 ± 0.552 |
| A1 | head | 95.702 ± 0.147 | -0.006 | 92.969 ± 0.375 |
| A2 | no_class | 95.449 ± 0.111 | -0.259 | 92.302 ± 0.566 |
| A3 | no_confidence | 96.473 ± 0.085 | +0.765 | 88.577 ± 0.590 |
| A4 | no_agreement | 96.278 ± 0.112 | +0.570 | 90.498 ± 0.679 |
| A5 | no_guard | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A6 | pooled | 96.065 ± 0.331 | +0.357 | 90.343 ± 1.685 |
| A6 | head | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A7 | unsmoothed | 95.910 ± 0.392 | +0.201 | 90.914 ± 1.105 |
| A9 | ordinal | 96.059 ± 0.236 | +0.351 | 89.720 ± 1.358 |
| A9 | continuous | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A13 | balanced | 95.323 ± 0.718 | -0.385 | 90.948 ± 1.479 |
| A13 | empirical | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A15 | static | 95.702 ± 0.108 | -0.006 | 92.731 ± 0.501 |
| A20 | random | 95.484 ± 0.095 | -0.224 | 92.160 ± 0.238 |
| A20 | shuffled_bins | 95.271 ± 0.340 | -0.437 | 92.815 ± 0.386 |
| A8 | 10 | 96.122 ± 0.321 | +0.414 | 90.395 ± 1.709 |
| A8 | 50 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A8 | 200 | 96.065 ± 0.331 | +0.357 | 90.343 ± 1.685 |
| A10 | 0.1 | 95.875 ± 0.192 | +0.167 | 92.316 ± 0.648 |
| A10 | 0.25 | 96.094 ± 0.356 | +0.385 | 91.589 ± 0.809 |
| A10 | 0.5 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A10 | 1.0 | 96.025 ± 0.265 | +0.316 | 89.564 ± 1.889 |
| A11 | 0.5 | 95.956 ± 0.259 | +0.247 | 92.316 ± 0.392 |
| A11 | 1 | 96.082 ± 0.311 | +0.374 | 91.589 ± 0.867 |
| A11 | 2 | 96.145 ± 0.346 | +0.437 | 90.862 ± 1.537 |
| A11 | 5 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A11 | 10 | 95.904 ± 0.375 | +0.196 | 90.083 ± 1.858 |
| A12 | raw | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A12 | head | 96.312 ± 0.234 | +0.604 | 89.979 ± 1.413 |
| A12 | shared | 96.243 ± 0.319 | +0.535 | 90.187 ± 2.043 |
| A14 | 3 | 95.887 ± 0.366 | +0.178 | 90.187 ± 1.641 |
| A14 | 5 | 96.002 ± 0.326 | +0.293 | 90.239 ± 1.792 |
| A14 | 10 | 96.048 ± 0.345 | +0.339 | 90.395 ± 1.709 |
| A14 | 15 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A16 | 3 | 95.990 ± 0.168 | +0.334 | 89.045 ± 1.922 |
| A16 | 5 | 95.950 ± 0.095 | +0.506 | 91.225 ± 0.648 |
| A16 | 10 | 95.990 ± 0.280 | +0.242 | 90.187 ± 1.945 |
| A16 | 15 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A17 | 10 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A17 | 12 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A17 | 14 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A18 | 3 | 96.076 ± 0.339 | +0.368 | 90.343 ± 1.685 |
| A18 | 5 | 95.846 ± 0.294 | +0.138 | 91.848 ± 1.269 |
| A19 | 200 | 96.266 ± 0.055 | +0.558 | 89.823 ± 0.702 |
| A19 | 201 | 96.025 ± 0.339 | +0.316 | 90.758 ± 1.513 |

## Interpretation and limitations

Reliability conditions on predicted class, never unknown true class at inference. Pooled head observations are correlated; counts are descriptive, not independent samples. Confidence tertiles are estimated by predicted class on training only. Agreement is leave-one-out with fixed 1/3 and 2/3 boundaries. Sparse cells back off to confidence/class/global priors; head tables shrink toward pooled correctness. Ordinal bands .5/.65/.8/.95 are predeclared. True-class-balanced fitting is a separately flagged training-only contrast. Predicted-class-balanced fitting changes observation weights; it can coincide with empirical estimates within a conditioned cell.

The threshold control is fitted on meta_train directly and assessed once on select. The search is bounded and predeclared, not exhaustive; select pools of 240 images create selection optimism. The expanded temperature grid .03–8 investigates the earlier .25 boundary without using test NLL for tuning. Empty cells use robust priors. A20 shuffled-bin control rotates expert lookup scores per image; random reliability is label-free and deterministic. D0/D1 controls and X0 diagnostics are not learned neural gates. C1 saved D19 was reused for three seeds; the saved linear control is available only for the original two screening seeds (seed 2 missing, not fabricated).

Final full-validation heads differ from fold-trained OOF heads, so distribution mismatch remains. Historical backbone training IDs/recipe are not embedded in its bare checkpoint. There is no truly untouched external holdout. Prior source attempt and verified migration are preserved; its already-completed reliability work was reused after a comparison-model lookup fix. All fresh training here is pure table/calibrator fitting, persisted per atomic task.

Per-task records in completed/ are authoritative. Results, weights, IDs, corrected/broken case CSVs and class confusion are retained. Resume regenerates tables from valid markers; corrupt/missing outputs and dependents are rebuilt. Tests include separate-process budgeted resumes, abandoned temporary files and output corruption. On abrupt termination at most one incomplete task is retried. `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md` and `state.json` identify the next task. Running --status performs integrity verification.

Main resume: `python -m experiments.discrete_weighting.run --config experiments/discrete_weighting/config.yaml --phase all --resume`. Detailed report regeneration: `python -m scripts.report_discrete_weighting_v2`.
