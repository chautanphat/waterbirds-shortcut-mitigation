# DPE weighting research report

Exploratory empirical study. All percentages below are mean ± sample SD across three refitted head/gate seeds sharing one frozen backbone.

## Protocol and leakage audit

Original validation: 1199 samples. 3-fold class-stratified cross-fitting; 15 independently resampled class-balanced prototype heads fitted on other folds only. Each head uses the fixed final epoch (20), with no group-based checkpoint selection. OOF predictions are split 80/20 by class into gate train/select. All scaler/calibrator/reliability fitting uses gate train only. Early stopping and configuration selection use select accuracy, then balanced accuracy, then NLL. Group annotations are absent from every gate fit and used only after selection. Final heads are refitted on the whole validation pool after both locks. Test is evaluated in one frozen reporting stage; no test-driven changes.

The old checkpoint records all 1,199 validation IDs as its head-training pool. It cannot supply held-out validation gate features. It remains unchanged and is scored only as a separate group-aware full-validation reference. The pure aggregation effect uses matched uniform with the same new heads. Differences from original DPE also include allocation/training protocol.

Baseline parity: U0 is exactly mean(member probabilities). Actual upstream predict_proba(avg=False) shape is [members,samples,classes], as shown by core.py:_predict; new caches explicitly transpose to [samples,members,classes]. Training entropic scale 30 is excluded at inference. Distances are real normalized prototype distances.

Backbone limitation: supplied bare state_dict has no embedded training IDs/epoch record. Its training-only origin is assumed from the handoff, not independently proved. No CNN retraining or validation-based backbone selection was performed here. The old checkpoint uses group annotations, whereas the new heads/gates do not. The same test set has a history of earlier exploration in this repository; this is not a fresh confirmatory test.

## Phase 1

Locked winner: **D19**, config `{"kl": 0.0, "lr": 0.001}`. Mean select accuracy 96.389%. All 42 ranks are represented in phase1_all.csv, including the oracle diagnostic excluded from selection. All declared configurations receive two screening seeds; top five families plus U0 receive three seeds. Large candidate search on a small select pool produces optimistic selection bias; no adaptive extension of grids was used.

Full tables: phase1_all.csv, phase1_ranked.csv (two-seed screen), phase1_finalists.csv (three-seed confirmation).

## Final test comparison

| Predictor | Accuracy % | Balanced accuracy % | WGA % |
|---|---:|---:|---:|
| Matched U0 | 95.708 ± 0.117 | 95.182 ± 0.053 | 92.761 ± 0.552 |
| D19 | 95.961 ± 0.422 | 92.828 ± 1.335 | 83.645 ± 4.083 |

Mean accuracy delta vs matched U0: +0.253 pp. Mean WGA delta: -9.116 pp. Mean accuracy delta vs original group-aware DPE: -0.086 pp.

Per-seed 95% paired bootstrap CIs condition on fitted models and test examples; they do not establish independence across backbones. The 3 seeds rerun splits, heads and gates, not the backbone. Small seed count limits inference.

## Phase 2 and failure analysis

phase2_ablation.csv records every requested A0–A20 category, including explicit N/A reasons. final_test.csv contains every frozen finalist and feasible ablation with per-class/per-group accuracy/counts, NLL, Brier, ECE, accuracy and WGA deltas, paired intervals, corrected/broken counts, mean weights, gate entropy, effective member count, weight mass on correct/incorrect experts, and post-hoc group weight means. Member-count ablations truncate the same ordered heads and refit weighting only. Feature removal can leave redundant information in full probability vectors; this limits causal interpretation.

Latency measures cached-output weighting only, excluding CNN and head inference. Param counts cover trainable weighting coefficients; tree/neighbor model sizes are not represented as neural parameter counts. Head final refitting induces a possible OOF/final-head distribution gap. CVaR uses class tails, never groups.

## Checkpoints and rerun

`python -m experiments.weighting.run --config experiments/weighting/config.yaml --phase all --resume`

See HANDOFF.md and progress.json. Each feature batch, completed head (with RNG), neural epoch (optimizer/RNG), candidate and evaluation is saved. Source/config/input hashes reject incompatible resume. An interruption can replay the current unfinished head; completed heads and candidate outputs are reused. Native winner: gate_best.pt plus fits/<key>/model.joblib and seed*/heads_final.pt. Joblib models are trusted local artifacts and must not be loaded from untrusted sources.

**Robustness trade-off:** WGA decreases for at least one seed; an accuracy gain must not be described as a universal robustness gain.

## All ranked families: best screening configuration

| Rank | Family | Select accuracy % | Delta vs U0 pp | BA % | Post-hoc WGA % |
|---:|---|---:|---:|---:|---:|
| 1 | U0 | 94.583 ± 1.179 | +0.000 | 94.158 | 90.064 |
| 2 | S1 | 95.417 ± 0.589 | +0.833 | 93.679 | 86.058 |
| 3 | S2 | 95.417 ± 1.179 | +0.833 | 93.341 | 84.135 |
| 4 | H1 | 95.000 ± 1.179 | +0.417 | 94.425 | 90.064 |
| 5 | H2 | 95.000 ± 1.179 | +0.417 | 94.425 | 90.064 |
| 6 | H3 | 94.792 ± 1.473 | +0.208 | 94.292 | 90.064 |
| 7 | S3 | 95.417 ± 1.179 | +0.833 | 93.341 | 84.135 |
| 8 | D1 | 95.417 ± 1.768 | +0.833 | 94.693 | 90.064 |
| 9 | D2 | 96.250 ± 1.179 | +1.667 | 95.228 | 90.064 |
| 10 | D3 | 96.042 ± 1.473 | +1.458 | 95.094 | 90.064 |
| 11 | D4 | 96.250 ± 1.179 | +1.667 | 95.228 | 90.064 |
| 12 | S4 | 95.417 ± 1.768 | +0.833 | 90.637 | 78.205 |
| 13 | H4 | 95.000 ± 1.179 | +0.417 | 94.425 | 90.064 |
| 14 | R1 | 94.583 ± 1.768 | +0.000 | 94.496 | 91.422 |
| 15 | R2 | 94.792 ± 2.062 | +0.208 | 94.630 | 91.984 |
| 16 | R3 | 95.000 ± 1.768 | +0.417 | 94.763 | 91.984 |
| 17 | R4 | 93.958 ± 0.295 | -0.625 | 93.757 | 90.064 |
| 18 | S5 | 95.417 ± 1.179 | +0.833 | 93.341 | 84.135 |
| 19 | H5 | 94.792 ± 1.473 | +0.208 | 94.630 | 91.422 |
| 20 | R5 | 95.208 ± 1.473 | +0.625 | 95.235 | 92.352 |
| 21 | R6 | 95.000 ± 1.768 | +0.417 | 94.763 | 91.422 |
| 22 | D5 | 95.625 ± 2.652 | +1.042 | 93.812 | 86.218 |
| 23 | D6 | 96.042 ± 1.473 | +1.458 | 93.742 | 84.135 |
| 24 | D7 | 96.042 ± 1.473 | +1.458 | 93.742 | 84.135 |
| 25 | D8 | 95.417 ± 1.768 | +0.833 | 94.017 | 88.141 |
| 26 | D9 | 94.792 ± 1.473 | +0.208 | 94.292 | 90.064 |
| 27 | D10 | 94.583 ± 1.179 | +0.000 | 94.158 | 90.064 |
| 28 | S6 | 95.417 ± 0.589 | +0.833 | 93.679 | 86.058 |
| 29 | D11 | 95.625 ± 1.473 | +1.042 | 94.826 | 90.064 |
| 30 | D12 | 96.042 ± 1.473 | +1.458 | 95.094 | 90.064 |
| 31 | D13 | 95.208 ± 1.473 | +0.625 | 94.559 | 90.064 |
| 32 | D14 | 96.042 ± 1.473 | +1.458 | 95.094 | 90.064 |
| 33 | D15 | 96.250 ± 1.179 | +1.667 | 95.228 | 90.064 |
| 34 | D16 | 96.250 ± 1.179 | +1.667 | 95.228 | 90.064 |
| 35 | D17 | 96.250 ± 1.179 | +1.667 | 95.228 | 90.064 |
| 36 | D18 | 96.250 ± 2.357 | +1.667 | 94.552 | 87.981 |
| 37 | H6 | 95.000 ± 1.768 | +0.417 | 94.425 | 90.064 |
| 38 | R7 | 94.583 ± 1.179 | +0.000 | 94.158 | 90.064 |
| 39 | D19 | 96.250 ± 1.179 | +1.667 | 95.228 | 90.064 |
| 40 | S7 | 95.833 ± 2.357 | +1.250 | 91.918 | 78.205 |
| 41 | R8 | 93.958 ± 0.295 | -0.625 | 93.757 | 90.064 |
| 42 | X0 | Diagnostic only / skipped | — | — | — |

## Ablation test summary

| Ablation | Variant | Accuracy % | Delta vs U0 pp | WGA % |
|---|---|---:|---:|---:|
| A0 | full_winner | 95.961 ± 0.422 | +0.253 | 83.645 ± 4.083 |
| A1 | remove_calibration | 96.025 ± 0.329 | +0.316 | 84.631 ± 3.331 |
| A2 | uniform | 95.708 ± 0.117 | +0.000 | 92.761 ± 0.552 |
| A5 | remove_confidence | 96.266 ± 0.163 | +0.558 | 86.656 ± 3.163 |
| A6 | remove_margin | 96.266 ± 0.163 | +0.558 | 86.656 ± 3.163 |
| A10 | average_dynamic_weights | 95.754 ± 0.714 | +0.046 | 91.984 ± 0.792 |
| A11 | discrete_weight_mapping | 95.927 ± 0.434 | +0.219 | 83.541 ± 4.104 |
| A12 | members_3 | 95.628 ± 0.466 | -0.081 | 87.407 ± 4.605 |
| A12 | members_5 | 95.996 ± 0.237 | +0.288 | 89.979 ± 1.179 |
| A12 | members_10 | 95.933 ± 0.597 | +0.224 | 87.072 ± 6.492 |
| A12 | members_15 | 95.961 ± 0.422 | +0.253 | 83.645 ± 4.083 |
| A13 | capacity_linear | 96.347 ± 0.085 | +0.639 | 88.577 ± 0.590 |
| A13 | capacity_16 | 96.145 ± 0.225 | +0.437 | 85.929 ± 2.594 |
| A13 | capacity_32x16 | 95.961 ± 0.422 | +0.253 | 83.645 ± 4.083 |
| A13 | capacity_64x32 | 96.145 ± 0.300 | +0.437 | 85.670 ± 3.742 |
| A14 | topk_1 | 95.093 ± 1.192 | -0.616 | 77.207 ± 7.542 |
| A14 | topk_3 | 95.795 ± 0.597 | +0.086 | 82.243 ± 5.241 |
| A14 | topk_5 | 95.921 ± 0.461 | +0.213 | 83.074 ± 4.255 |
| A14 | topk_all | 95.961 ± 0.422 | +0.253 | 83.645 ± 4.083 |
| A15 | loss_brier | 95.967 ± 0.428 | +0.259 | 83.593 ± 4.018 |
| A15 | loss_balanced | 95.662 ± 0.578 | -0.046 | 90.694 ± 1.400 |
| A16 | toggle_CVaR | 95.973 ± 0.400 | +0.265 | 84.268 ± 3.931 |
| A17 | toggle_temperature | 96.025 ± 0.329 | +0.316 | 84.631 ± 3.331 |
| A19 | random_weight_control | 95.864 ± 0.140 | +0.155 | 92.056 ± 0.467 |
| A3 | remove_KL: Winner KL coefficient already zero | N/A | — | — |
| A4 | remove_teacher: No label-conditioned teacher in winner | N/A | — | — |
| A7 | remove_distance: Feature not used by winning family | N/A | — | — |
| A8 | remove_context: Feature not used by winning family | N/A | — | — |
| A9 | remove_embedding: Feature not used by winning family | N/A | — | — |
| A18 | alternative_crossfit_count: Not run: repeats would require a separate refitted head/cache protocol; bounded compute budget | N/A | — | — |
| A20 | diverse_seeds: Covered by the three source/head/gate seed reruns in A0; not an additional independent experiment | N/A | — | — |

## Winner per-group accuracy and failure changes

| Group (2*y+background) | Count | U0 accuracy % | Winner accuracy % |
|---|---:|---:|---:|
| 0 | 2255 | 98.714 | 99.586 |
| 1 | 2255 | 93.540 | 97.324 |
| 2 | 642 | 92.783 | 83.645 |
| 3 | 642 | 95.691 | 90.758 |

Corrected/broken per seed: [(0, 121, 132), (1, 94, 76), (2, 100, 63)].
Paired 95% accuracy-delta CIs (pp): [(0, [-0.7248878149810148, 0.3451846738004833]), (1, [-0.10398688298239556, 0.742147048671039]), (2, [0.20667932343803938, 1.070072488781498])].

## Additional locked A18 diagnostic: 5-fold cross-fitting

The original budget-reserved N/A entry is superseded by `phase2_crossfit5.csv`. This diagnostic was declared before test metrics were read, uses the exact locked winner/configuration, refits the OOF heads/gate for all three seeds, and reuses identical final full-validation heads. It does not select a new winner. Its separate plan/source hashes and per-head/epoch checkpoints are retained.

Accuracy: 95.898 ± 0.320%; WGA: 86.085 ± 5.997%; accuracy delta vs matched U0: +0.190 pp; WGA delta: -6.676 pp.

## Research conclusion and detailed sample audit

The locked winner does not improve mean accuracy over the supplied original DPE reference. Its small accuracy increase over matched U0 comes with a substantial WGA reduction, especially for waterbirds on land (evaluation group 2 in the internal 0-based mapping). Two of three per-seed paired accuracy-delta intervals versus U0 include zero. There is no evidence here to replace the original DPE checkpoint. It has been preserved byte-for-byte.

See `summary.csv`, `failure_cases.csv` (all corrected/broken sample IDs), `failure_changes_by_group.csv`, and `member_ablation_matched_baselines.csv` (same-member-count uniform controls in addition to the full-15 reference). All three canonical winner prediction exports were validated against the full reference metadata. The independent audit recomputed 84 final predictor outputs and checked nine OOF folds. Additional implementation choices and bounded-grid deviations are recorded in `experiments/weighting/PROTOCOL_DECISIONS.md`; in particular D11 uses augmented joint features, and focal screening uses gamma 2. These choices must not be described as an exhaustive Cartesian search.
