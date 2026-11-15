# G1 residual v7: continuous precision weighting results

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

**INDEPENDENT_CONFIRMATION_NOT_PERFORMED**

Frozen primary: **G1_RH**. B verdict: **B_DIRECTION_NO_GO** (DIRECTIONAL_IMAGE_BOOTSTRAP_INTERVAL). C verdict versus G1: **EXPLORATORY_C_WGA_EQUAL**. No method is promoted. Original v6 NO_GO and Uniform/G1 production inference remain unchanged.

The study implements the requested optimization, not just baseline confirmation: six-column continuous signed-delta ridge (explicit penalty 10), exact-head/predicted-class shrunk residuals (kappa 25, clip 0.1), nested predicted-OOF scale matching, and the separately registered margin-gated RHM ablation. Raw G1 geometry, gamma 50, alpha 0.25, every prototype head, backbone, member prediction and IPS training are unchanged. Lambda 0.5 is fixed; no lambda1/grid or C-driven selection was run.

Nine matching ensembles have 240 B images each. Four B crossfit partitions use seed 71000+100*seed+fold; each model uses three secondary class-stratified OOF partitions entirely inside its owning B-training set (seed 72000+1000*seed+100*fold+10*part). There are 36 B-OOF model fits plus nine full-B fits, and 135 secondary ridge/head fits. Every feature RMS, per-head mu/sigma, q20, head effect and residual scale is B-training-only. Pooled and head-aware residual scales are estimated separately. The only numerical signal threshold is the preregistered s_R<0.002 guard. Finite/support failures also block unsafe fits; no post-hoc OOF quality cutoff was introduced.

B targets use (2y−1)*(p_j(1)−raw_Uniform(1)); raw float32 Uniform reduction is retained for exact baselines. Target/true-class-shift identities hold within float32 roundoff. No group field enters B fitting or evaluation. Repeated B images are collapsed by seed before seed metrics: probabilities and scalar gain/ranking diagnostics are averaged over their owning ensembles, never head-slot coefficients across ensembles.

## B-OOF expert-selection evidence

| Method | Seed | Unique images | Uniform-wrong | Mean gain on Uniform-wrong | Δgain vs G1 | Pair rank on Uniform-wrong | AA (%) | ΔAA vs U0 (pp) | ΔAA vs G1 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 | 0 | 613 | 27 | -0.0000000 | +0.0153909 | 0.50000 | 95.595432 | +0.000000 | +0.326264 |
| G1_frozen | 0 | 613 | 27 | -0.0153909 | +0.0000000 | 0.26279 | 95.269168 | -0.326264 | +0.000000 |
| G1_R | 0 | 613 | 27 | -0.0175272 | -0.0021363 | 0.17231 | 95.432300 | -0.163132 | +0.163132 |
| G1_RH | 0 | 613 | 27 | -0.0175979 | -0.0022070 | 0.22028 | 95.432300 | -0.163132 | +0.163132 |
| G1_RHM | 0 | 613 | 27 | -0.0166608 | -0.0012700 | 0.22028 | 95.432300 | -0.163132 | +0.163132 |
| U0 | 1 | 619 | 24 | +0.0000000 | +0.0127782 | 0.50000 | 96.122779 | +0.000000 | +0.161551 |
| G1_frozen | 1 | 619 | 24 | -0.0127782 | +0.0000000 | 0.26111 | 95.961228 | -0.161551 | +0.000000 |
| G1_R | 1 | 619 | 24 | -0.0150407 | -0.0022625 | 0.21488 | 95.799677 | -0.323102 | -0.161551 |
| G1_RH | 1 | 619 | 24 | -0.0146380 | -0.0018598 | 0.24742 | 95.799677 | -0.323102 | -0.161551 |
| G1_RHM | 1 | 619 | 24 | -0.0137404 | -0.0009622 | 0.24742 | 95.799677 | -0.323102 | -0.161551 |
| U0 | 2 | 615 | 23 | -0.0000000 | +0.0155691 | 0.50000 | 96.260163 | +0.000000 | +0.162602 |
| G1_frozen | 2 | 615 | 23 | -0.0155691 | +0.0000000 | 0.29110 | 96.097561 | -0.162602 | +0.000000 |
| G1_R | 2 | 615 | 23 | -0.0176851 | -0.0021160 | 0.23395 | 96.097561 | -0.162602 | +0.000000 |
| G1_RH | 2 | 615 | 23 | -0.0170981 | -0.0015290 | 0.28406 | 96.422764 | +0.162602 | +0.325203 |
| G1_RHM | 2 | 615 | 23 | -0.0165344 | -0.0009653 | 0.28406 | 96.422764 | +0.162602 | +0.325203 |

1. **Continuous ranking/direction:** RH mean signed-gain difference on seed-pooled Uniform-wrong B images is -0.00186525; per seed [-0.002206993463940641, -0.0018597604226813864, -0.001528997779298282]; 0/3 positive. Shared-ID paired 95% CI [-0.002486722185398428, -0.0012001044465132853]. Direction gate passes: False. Distinct Uniform-wrong B counts per seed are in the table; they are not independent cohorts. Per-image ranks (with valid-count handling), geometry comparison and all gains are in B_OOF_MECHANISM.csv and B_OOF_SIGNED_GAIN.csv.

2. **Head effects versus pooled:** head-aware B gain differences must be assessed against R in the table, not inferred from coefficient size. Across 1080 fitted head/class effects, median effective image support 90.000; 0 have <30 distinct images; 443 positive and 637 negative. Within the same actual ensemble, mean four-fold sign agreement is 0.8880; median b SD 0.000558. Full head SHA/support/denominator/clipping/shrinkage and same-ensemble stability are retained. A stable slot is not a causal specialist; no head identity is transferred across folds.

3. **Scale and actual score influence:**

| Method | Split | Mean weight TV vs G1 | Top-head changes | Mean residual/geometry centered RMS | Mean clipping fraction |
|---|---|---:|---:|---:|---:|
| G1_R | B-OOF | 0.02455115 | 77/2160 | 0.361271 | 0.050556 |
| G1_RH | B-OOF | 0.02572769 | 108/2160 | 0.366935 | 0.049321 |
| G1_RHM | B-OOF | 0.00810767 | 51/2160 | 0.079387 | 0.049321 |
| G1_R | C exploratory | 0.02387650 | 110/3597 | 0.359483 | 0.049282 |
| G1_RH | C exploratory | 0.02463705 | 138/3597 | 0.364203 | 0.047484 |
| G1_RHM | C exploratory | 0.00781206 | 65/3597 | 0.077566 | 0.047484 |

Enabled scales: B-inner pooled 36/36, head 36/36; full-B pooled 9/9, head 9/9. All measured s_R/s_G, OOF-vs-null MSE and off reasons are in FITTED_SCALE_SUMMARY.csv. An off-switch is an explicit registered G1 fallback, not an accidental implementation alias.

4. **Direction versus actual threshold crossings:** seed-pooled corrected/broken counts versus both references are in B_SEED_METRICS.csv. The following are distinct per-seed image counts, not expert rows:

| Seed | Method | Corrected U0 | Broken U0 | Corrected G1 | Broken G1 | G1-negative → positive gain on Uniform-wrong |
|---:|---|---:|---:|---:|---:|---:|
| 0 | G1_R | 2 | 3 | 2 | 1 | 0 |
| 0 | G1_RH | 2 | 3 | 2 | 1 | 0 |
| 0 | G1_RHM | 2 | 3 | 2 | 1 | 0 |
| 1 | G1_R | 1 | 3 | 0 | 1 | 0 |
| 1 | G1_RH | 1 | 3 | 0 | 1 | 0 |
| 1 | G1_RHM | 1 | 3 | 0 | 1 | 0 |
| 2 | G1_R | 2 | 3 | 0 | 0 | 0 |
| 2 | G1_RH | 2 | 1 | 2 | 0 | 0 |
| 2 | G1_RHM | 2 | 1 | 2 | 0 | 0 |

5. **Margin gate:** this is a fixed ablation, never a replacement primary selected by C. It is not an AA guarantee. Strata use each B-training q20/q80. Counts below are B validation rows with repeated IDs across outer ensembles; seed-pooled accuracy is reported separately.

| Method | Margin stratum | Rows | Mean gate | Corrected U0 | Broken U0 | Corrected G1 | Broken G1 |
|---|---|---:|---:|---:|---:|---:|---:|
| G1_RH | high | 439 | 1.000000 | 0 | 0 | 0 | 0 |
| G1_RH | low | 432 | 1.000000 | 4 | 7 | 4 | 3 |
| G1_RH | mid | 1289 | 1.000000 | 0 | 0 | 0 | 0 |
| G1_RHM | high | 439 | 0.077746 | 0 | 0 | 0 | 0 |
| G1_RHM | low | 432 | 0.610042 | 4 | 7 | 4 | 3 |
| G1_RHM | mid | 1289 | 0.200672 | 0 | 0 | 0 | 0 |

## C exploratory downstream evidence

6. **All locked methods, all outcomes:** 1,199 unique images per seed (same IDs across three seeds; 3,597 seed-image rows). Group counts per seed are [467,466,133,133]. One shared frozen ERM* backbone; this is not an ERM-only run or independent test. WGA is computed as each seed minimum first, then averaged.

| Method | Mean AA (%) | Mean WGA (%) | Seed SD WGA (pp) | ΔAA vs U0 (pp) | ΔWGA vs U0 (pp) | ΔWGA vs G1 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| U0 | 95.746455 | 92.481203 | 1.302294 | +0.000000 | +0.000000 | -0.501253 |
| G1_frozen | 95.857659 | 92.982456 | 1.736392 | +0.111204 | +0.501253 | +0.000000 |
| G1_R | 95.829858 | 93.233083 | 1.302294 | +0.083403 | +0.751880 | +0.250627 |
| G1_RH | 95.829858 | 92.982456 | 1.148515 | +0.083403 | +0.501253 | -0.000000 |
| G1_RHM | 95.857659 | 92.982456 | 1.148515 | +0.111204 | +0.501253 | -0.000000 |

| Seed | Method | AA (%) | WGA (%) | Worst class acc (%) | Worst group(s) |
|---:|---|---:|---:|---:|---|
| 0 | U0 | 95.829858 | 90.977444 | 93.609023 | [2] |
| 0 | G1_frozen | 95.829858 | 90.977444 | 93.984962 | [2] |
| 0 | G1_R | 95.913261 | 91.729323 | 94.360902 | [2] |
| 0 | G1_RH | 95.913261 | 91.729323 | 94.360902 | [2] |
| 0 | G1_RHM | 95.913261 | 91.729323 | 94.360902 | [2] |
| 1 | U0 | 95.913261 | 93.233083 | 94.736842 | [2] |
| 1 | G1_frozen | 96.080067 | 93.984962 | 95.112782 | [2] |
| 1 | G1_R | 95.913261 | 93.984962 | 95.112782 | [2] |
| 1 | G1_RH | 95.996664 | 93.984962 | 95.112782 | [2] |
| 1 | G1_RHM | 96.080067 | 93.984962 | 95.112782 | [2] |
| 2 | U0 | 95.496247 | 93.233083 | 94.360902 | [2] |
| 2 | G1_frozen | 95.663053 | 93.984962 | 95.488722 | [2] |
| 2 | G1_R | 95.663053 | 93.984962 | 95.488722 | [2] |
| 2 | G1_RH | 95.579650 | 93.233083 | 94.736842 | [2] |
| 2 | G1_RHM | 95.579650 | 93.233083 | 94.736842 | [2] |

7. **Uncertainty and nulls:** 2,000 group-stratified unique-image draws share image IDs across all seeds/methods; intervals are conditional on these fixed pipelines. No independent-row bootstrap over 3,597 repeated entries. B bootstrap likewise shares union-image draws and uses seed-specific availability/subset masks. These intervals do not estimate new-cohort/backbone uncertainty.

| Method | Reference | Mean ΔAA 95% CI (pp) | Mean ΔWGA 95% CI (pp) |
|---|---|---|---|
| U0 | U0 | [0.0, 0.0] | [0.0, 0.0] |
| U0 | G1_frozen | [-0.305810397553515, 0.08340283569641116] | [-1.2531328320802024, 0.17909580818999968] |
| G1_frozen | U0 | [-0.08340283569641116, 0.305810397553515] | [-0.17909580818999968, 1.2531328320802024] |
| G1_frozen | G1_frozen | [0.0, 0.0] | [0.0, 0.0] |
| G1_R | U0 | [-0.11120378092854821, 0.27800945232138163] | [-0.06995761937031636, 1.6492949110974835] |
| G1_R | G1_frozen | [-0.11120378092855192, 0.055601890464277805] | [-0.1430615164520736, 0.751879699248122] |
| G1_RH | U0 | [-0.11120378092854821, 0.2780094523213853] | [0.0, 1.2531328320802024] |
| G1_RH | G1_frozen | [-0.13900472616069265, 0.08340283569641485] | [-0.7518796992481185, 0.751879699248122] |
| G1_RHM | U0 | [-0.055601890464277805, 0.3058103975535187] | [0.0, 1.2531328320802024] |
| G1_RHM | G1_frozen | [-0.11120378092855192, 0.11120378092855192] | [-0.7518796992481185, 0.751879699248122] |

Control RH_residual_slot_permutation: 3000 fixed-seed repeats; mean WGA 93.015622%; fraction at least observed RH WGA 0.928333, strictly above 0.204000, ties 0.724333. These are descriptive null frequencies, not independent significance tests. Residual permutations preserve each image's residual multiset/strength but can change final concentration; G1 weight permutations preserve final concentration.

Control G1_weight_slot_permutation: 3000 fixed-seed repeats; mean WGA 92.524025%; fraction at least observed RH WGA 0.172333, strictly above 0.053333, ties 0.119000. These are descriptive null frequencies, not independent significance tests. Residual permutations preserve each image's residual multiset/strength but can change final concentration; G1 weight permutations preserve final concentration.

8. **Strict AA floor:** B primary passes every seed versus U0: True; differences [-0.16313213703099683, -0.3231017770597755, 0.1626016260162566] pp. C eligibility for every preregistered method:

| Method | Per-seed ΔAA vs U0 (pp) | Every seed > −1 pp | Per-seed ΔAA vs G1 (pp) |
|---|---|---|---|
| U0 | [0.0, 0.0, 0.0] | True | [0.0, -0.16680567139282232, -0.16680567139282232] |
| G1_frozen | [0.0, 0.16680567139282232, 0.16680567139282232] | True | [0.0, 0.0, 0.0] |
| G1_R | [0.08340283569641116, 0.0, 0.16680567139282232] | True | [0.08340283569641116, -0.16680567139282232, 0.0] |
| G1_RH | [0.08340283569641116, 0.08340283569641116, 0.08340283569641116] | True | [0.08340283569641116, -0.08340283569641116, -0.08340283569641116] |
| G1_RHM | [0.08340283569641116, 0.16680567139282232, 0.08340283569641116] | True | [0.08340283569641116, 0.0, -0.08340283569641116] |

9. **Failure/success decomposition:**
B proxy RH−G1 worst-class NLL = -0.00144227; tolerance 0.003 passes: True. Direction passes: False; AA passes: True. Thus the exact B decision is B_DIRECTION_NO_GO. Poor/uncertain OOF advantage ranking, guard-disabled scale, head-support instability, and actual correction/breakage counts are separate diagnostics above. Coefficient/weight movement alone does not establish expert-selection improvement. Limited 180-image B-inner-training support is a limitation, not a proven sole cause. No guard threshold or residual strength was changed after observing outcomes.

Retrospective oracle: 12/30 Uniform-wrong group-2 seed-image rows are correctable at fixed alpha using label-aware best-head routing; 13 distinct hard image IDs are involved. Per-seed counts, current-method corrections and whether another group becomes worst are in ORACLE_FEASIBILITY.json. Oracle labels never enter fitting, selection or deployed inference; this is feasibility, not learnability or a future WGA target.

10. **Persistence and unchanged production:** every secondary fit, B model, validation inference/evaluation, full-B fit, C inference/evaluation and uncertainty batch has an atomic completion receipt with source/config/input/head/split fingerprints. Real workers are force-killed after a committed secondary fit and after a committed C inference; recovered artifacts must match uninterrupted byte hashes and completed mtimes. Validated acceptance outputs are reused. CHECKSUMS.sha256 and DELIVERY_VALIDATION.json verify numerical invariants and protected v1-v6 artifacts. state.json, HANDOFF_STATUS.md and NEXT_AGENT_PROMPT.md are refreshed after every task. A completed --resume --verify reports the exact count; a separate-process zero-work resume is recorded at delivery.

## Limitations and continuation

C is reused development, with original official labels and the documented species-label caveat. No silent relabeling. Shared backbone/head seeds are not independent cohorts. B image support and Uniform-wrong counts are small; class/boundary weighting and in-sample head-residual fitting can be optimistic, hence nested B-OOF evaluation. A common-offset/zero-delta/scale guard has automated coverage. Inference has no y/group/ID lookup. All negative ablations are published. Historical v6 results are loaded as context only; no G2/CCW rerun occurred.

**NO_FRESH_DATA — independent confirmation is blocked.** Do not rename/repartition old C as fresh. If B is NO_GO, a future fresh primary comparison remains G1 versus Uniform; v7 stays diagnostic. If B passes, only the locked RH recipe could be preregistered for a truly independent cohort. Nothing here authorizes integration or C-driven tuning.

```powershell
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase all --resume
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --verify
.venv/Scripts/python.exe -m pytest experiments/g1_residual_v7/tests -q
```

Required files are under metrics/: MEAN_AA_WGA_COMPARISON.csv, PER_SEED_AND_GROUP_METRICS.csv, B_OOF_MECHANISM.csv, B_OOF_SIGNED_GAIN.csv, WEIGHT_AND_SCORE_DECOMPOSITION.npz (label-free), C_EVALUATION_LABELS.npz (evaluator-only), FLIP_AUDIT.csv (every image, including unchanged), PERMUTATION_CONTROLS.json and BOOTSTRAP_CI.json. Exact fitted models, nested OOF estimates and per-fold outputs remain under checkpoints/.
