# CODEX HANDOFF v5 — DPE Geometry-Based Aggregation (WGA First)

## 0. Mission, scope, and non-negotiables

**Research question:** Can the geometry of DPE's class prototypes support better inference-time aggregation than the standard uniform ensemble, **improving worst-group accuracy (WGA)** without reducing average accuracy (AA) by **1.00 percentage point or more in any evaluation seed**?

**This is one focused experiment, not another broad search.** First diagnose whether prototype distance ranks the correct experts better than confidence; then compare a small, frozen set of *training-free geometry-based aggregation* rules; only if there is development evidence run **genuinely held-out nested head-and-aggregation verification**. No new gating MLP, no new specialist head, no retraining the backbone, no pseudo-group optimization in this task.

**Priority:** correctness/provenance > group-blind experimental design > WGA > AA constraint > efficient computation. Original DPE must remain unchanged. Never optimize or select parameters using the repeatedly inspected historical Waterbirds test set.

**Scientific caveat:** More accurate ORACLE expert selection does **not** show a practical router can identify that expert without the label. Smaller prototype distance, lower entropy, and larger confidence can all be misleading on spurious-correlation samples. No claimed WGA improvement until independently verified.

## 1. Available inputs and exact provenance

Search the existing repository recursively for `development_oof_bundle.zip`, DPE source/head-training adapter and saved checkpoints. Extract the bundle read-only. Optional `prediction_bundle.zip` is **historical test** and must **not** be opened by any fitting, tuning, or candidate-selection code. It may only be used for a separately labeled retrospective diagnostic after the recipe is locked; it is not fresh confirmation. Do not request or fabricate missing inputs.

`development_oof_bundle.zip` has 3 seeds; each has `meta_train.npz` (959 images) and `meta_select.npz` (240 images) from the same 1,199 Waterbirds development images. Arrays include:

```
probs      [M,15,2]  # raw temperature-1 member probabilities
logits     [M,15,2]  # actual member logits
# distances are normalized prototype distances returned by the old heads
distances  [M,15,2]
embedding  [M,2048]
y          [M]       # 0 Landbird, 1 Waterbird
sample_id  [M]
group      [M]       # EVALUATION ONLY; mapping 2*y+background
fold       [M]
seed       [M]
split      [M]
head_id    [15]      # slots only; head identity differs across folds
```

The 3 seeds share **one augmented 400-epoch ERM* backbone**, not three independently trained backbones. The OOF heads for one fold are different real models from heads in another fold. Do not pretend that `head-00` is a single expert across folds. `meta_select` was used in earlier searches, so it is exploratory, **not** a fresh confirmatory holdout. `group` is permitted **only after a candidate/configuration is frozen**, for post-hoc WGA; group labels are forbidden for training or hyperparameter selection in this *strict group-blind* track.

**First action:** read the archive README and manifest, verify SHA-256 for the listed NPZ files, verify sample/ID/provenance alignment, per-seed meta_train/meta_select nonoverlap, and exact `p_uniform=probs.mean(axis=1)`. In a nested run, groups may only be read from evaluation folds **after** prediction artifacts are written and committed. Do not aggregate repeated image IDs across seeds as independent examples.

Before using `distances` to score *different heads*, audit whether the exported distances reflect normalized feature/prototype vectors, head-specific learned scale, and/or a shared transformation. Do not assume inter-head distance scale comparability just because both heads output two distance values. Reconstruct `probs` from `logits` (softmax) and test any proposed `-distance` relation numerically. The DPE implementation's learnable distance scale must be respected, not silently discarded. Obtain formulas from installed source if present; the official reference is https://github.com/minhto2802/dpe4subpop.

## 2. Phase 0 — Geometry-to-reliability audit (NO fitting on test)

Use *only* development OOF outputs. On `meta_train`, define:

- `pred_j = argmax_k probs[j,k]`, `correct_j = 1[pred_j==y]` (analysis only);
- `conf_j=max_k probs[j,k]`;
- `H_j = -sum_k probs[j,k]*log(probs[j,k])`;
- `dmin_j=min_k distances[j,k]`;
- `dmargin_j = second-smallest-distance - smallest-distance` (binary: absolute gap);
- `agreement_j = (count of OTHER experts with same predicted class)/14`;
- `uniform_conf = max_k mean_j probs[j,k]`; `uniform_wrong=1[argmax mean_j probs[j,k] != y]`.

For ranking experts by correctness, use scores `-dmin`, `-H`, `conf`, and `dmargin`. Within each **image that has ≥1 correct and ≥1 wrong expert**, compare all correct-vs-wrong pairs, count ties as 0.5, **average within the image first**, and then across distinct image IDs. Report sample-level bootstrap confidence intervals, not treating 15 expert rows as independent.

Mandatory diagnostic questions and outputs:

1. Overall correct-expert ranking accuracy for each signal; same accuracy **restricted to Uniform-wrong images** (this subset is crucial); per-seed results. If signals reverse rank on Uniform-wrong images, warn clearly that routing may lower WGA.
2. Compare rank performance between `-dmin` and confidence with paired per-image differences and uncertainty; error detection versus correctness **must not** be conflated with ability to choose the right expert.
3. Report `dmin`, margin, confidence and entropy distributions by predicted class and by correctness; note inter-head scale/offset heterogeneity.
4. Post-hoc only after scores are fixed: per-group expert ranking, Uniform AA/WGA, subgroup count, group-specific oracle headroom, disagreement distribution and minority-correct-head counts.
5. Heatmaps: predicted class × distance-quantile for empirical correctness, counts, and average score; confidence × distance-quantile for empirical correctness, separate predicted-class panels; `-dmin` versus confidence scatter with correct/incorrect overlay; per-group comparisons **labeled diagnostic only**.
6. Test whether `min distance` is itself informative beyond the difference between the distances to competing classes. A head may have a small min distance but be confidently wrong; nearest distance is **not** a calibrated likelihood of being right.

Use quantiles/bins derived from meta_train only, display distinct-image counts, and suppress strong interpretations for sparse cells. Save `phase0_geometry_audit.md`, numerical CSV sources, PNGs, and `phase0_audit.json` (`supports_distance_routing`: yes/no/undetermined, with evidence). **Do not use group WGA to choose a candidate.**

**Decision gate 0:** If `-dmin` fails to discriminate correct from wrong experts on Uniform-wrong OOF examples across most seeds, continue with the preregistered small baselines for completeness but do **not** claim minimum distance is a robust reliability signal. Do not invent another network or larger sweep.

## 3. Phase 1 — Frozen training-free aggregation candidates

### A. Strict baseline and computational definitions

For each image `x`, let `P[j,k]` be the raw probabilities from the **same 15 ordered heads** and `D[j,k]` the exported distances. `N=15`, `K=2`.

**U0 (mandatory):** `P_uniform[k] = mean_j P[j,k]`. Preserve exact original class ordering and check numerical parity against existing DPE calculations.

**E1 — Low-entropy expert weighting:**

```
r_j = softmax_j(-gamma * H(P_j))
w_j = (1-alpha)/N + alpha*r_j
P_E1[k] = sum_j w_j * P[j,k]
```

**G1 — Minimum-distance expert weighting:**

```
dmin_j = min_k D[j,k]
r_j = softmax_j(-gamma*dmin_j)
w_j = (1-alpha)/N + alpha*r_j
P_G1[k] = sum_j w_j * P[j,k]
```

**G2 — Head-scale/offset-normalized minimum-distance weighting:** calculate an allowed standardized distance using **only the fit split**; for a truly fixed ensemble each head's location/scale can be fitted from B predictions. E.g. `z_j = (dmin_j - mean_B(dmin_j)) / max(std_B(dmin_j),eps)`, then `r=softmax(-gamma*z)`. For OOF developmental data, the same head-slot IDs are **not persistent across folds**. Do not fit a single fictitious head-specific scaler by pooling corresponding slots across OOF folds; if per-fold fit observations are insufficient, mark this candidate unsupported in exploratory OOF and implement it only when a fixed head ensemble is available in nested verification. Never let evaluation-fold statistics set a transform.

**G3 — Distance Top-M routing:** retain `M` experts with smallest `dmin`, give each `1/M`, others zero; optionally a conservative uniform residual `w=(1-alpha)/N + alpha*topM_weight`. Top-M by **distance**, not the D19 learned weights used in prior top-k ablations. Handle ties deterministically.

**S1 — Class-wise Soft-min / LogSumExp on distances:**

```
S_k(x) = tau * ( logsumexp_j(-D[j,k]/tau) - log(N) )
P_softmin = softmax_k(S_k / T_class)
```

Use stable logsumexp; `tau>0`; `T_class>0` is a **class-score scaling** parameter, not the original DPE training temperature. At `tau->0`, the score approaches `-min_j D[j,k]`, while at large `tau` it tends to `-mean_j D[j,k]`. The normalizing `-log N` term must be inside parentheses with `tau` multiplying both terms. LogSumExp is **NOT algebraically the same as a shared per-expert mixture weight**; it gives each class its own effective prototype weights. Do not describe it as just `w_j(x)`.

**S2 — Class-wise LogSumExp on original logits:** same as S1 but replace `-D[j,k]` with the actual `logits[j,k]` and use a separate logged temperature; compare as a distinct control. This tests whether using the source head's true score scale matters, rather than assuming `logits=-distances`.

**S3 — Centered-distance class-wise Soft-min (scale/offset control):** define `Dc[j,k]=D[j,k] - mean_c D[j,c]` per sample/head, then apply S1. This removes per-head common offsets but changes the geometry interpretation; report it separately, not as distance to the closest absolute prototype.

**S4 — Conservative residual of class-wise softmin:** `P_final=(1-alpha)*P_uniform + alpha*P_softmin`, with α fixed/tuned on permissible development labels. This controls class-score scaling and the AA trade-off; benchmark S1 without a residual too. Group labels are never needed at inference.

**All methods are inference-only transformations of already frozen heads.** No labels, correctness, or subgroup IDs may enter any inference function. All confidence/geometry/calibration transformations and optional `T_class` selected or fitted on development training only.

### B. Preregister a deliberately small search budget

**Phase 1 screening:** run these predeclared settings on 3 seed-specific `meta_select` splits, with all preprocessing and any fitting on `meta_train` only:

- E1: `gamma in {1,5}`, `alpha=0.25`.
- G1: `gamma in {5,20,50}`, `alpha=0.25`.
- G2: `gamma in {0.5,1,2}`, `alpha=0.25`, **only when valid per-head training-only normalization is possible**.
- G3: `M in {3,5,10}`, `alpha=0.25` (one `M=1` diagnostic only, not promoted).
- S1: `tau in {0.025,0.1,0.5}`, `T_class=1` as a fixed screening reference.
- S2: `tau in {0.025,0.1,0.5}`, `T_class=1`.
- S3: `tau in {0.025,0.1,0.5}`, `T_class=1`.
- S4: take S1 with `alpha in {0.10,0.25,0.50}` and `tau=0.1`, `T_class=1`.

Do **not** add or alter candidate grids in response to observed WGA. If selected softmin methods are near 0.5 probabilities, this is a known effect of logit scale: record it rather than quietly fitting a new scaling grid. A further *separately declared* small `T_class` sensitivity may be proposed, but only after locking Phase 1 and on new development folds, never on the historical test. Avoid treating a calibration improvement as an aggregation improvement; always report matched uniform using the identical input probability representation.

**Selection rule in strict group-blind track:** before reading any group ID, enforce `Delta_AA > -1.00 pp` versus matched U0 in **each** meta_select seed (strict inequality; no rounding). Then rank eligible recipes by the **worst-class negative log-likelihood** averaged over seeds; tie-break by worst-class accuracy, then mean AA, then simplicity/stability. This is a *proxy for robustness*, **not WGA itself**. Freeze at most **two** candidates with exact configurations in `locked_candidates.json`, then unlock group IDs only for descriptive, post-hoc WGA. If the user explicitly authorizes using subgroup labels for **validation model selection**, make a separate clearly labeled group-aware study, never present it as group-blind. Do not select methods based on the previous historical test or its WGA.

Record per candidate: AA, balanced accuracy, worst-class accuracy, WGA (only after lock, exploratory), NLL, Brier, ECE (with representation), per-group accuracy/counts, corrected/broken flips, flip direction, weight entropy/effective member count for shared-weight methods, support for closest-prototype attribution, runtime, and per-image prediction exports. Use image-level paired intervals, and report variation across seeds while acknowledging shared backbone.

**Phase 1 outcome is NOT promotion evidence.** It is a bounded screen for Phase 2.

## 4. Phase 2 — True held-out verification (nested head/aggregation protocol)

Only run if Phase 0/1 are complete, candidate recipe(s) are locked, source/provenance checks pass, and **existing DPE head training and checkpoints are available**. No synthetic replacement or in-sample substitute.

For seeds 0/1/2 and 3 class-stratified outer folds:

1. Split the outer training pool into disjoint `A` (head training) and `B` (parameter/normalizer/calibration fitting); let outer `C` be held out. Use the exact prior Gate-B 70/30 A/B division of outer training data unless its manifest says otherwise. Save all IDs.
2. With the *same frozen backbone*, train **15 heads from scratch on A** according to the frozen historical no-group DPE adapter (record complete schedule and hashes; never tune on C). Persist model, optimizer and RNG state **each head epoch**.
3. Make predictions for B and C using those **exact same ordered heads**; save arrays once and reuse them for every candidate and matched uniform.
4. On B alone, fit any permitted distance normalization, probabilities/calibration and `T_class` where a configuration **explicitly calls for it**; otherwise use the locked fixed candidate values. Never fit on C, and do not use C group IDs until its predictions have been committed.
5. For each fold and seed, compare candidate with `P_uniform=mean P_j` from the exact same heads on the same C samples; do not compare to the historical group-aware original checkpoint as a pure aggregation effect.
6. Pool **unique held-out C predictions across folds within each seed**, then calculate WGA (minimum of the four true groups, evaluative only), AA, balanced accuracy and per-group metrics. Group annotations must not enter A/B fitting or config selection. No fold averaging of a fold's min-group accuracy in lieu of full-seed pooled WGA.
7. Paired image-level/group-stratified bootstrap confidence intervals and corrected/broken by group; specify that the 3 seeds share one backbone and do not provide independent-backbone uncertainty.

### Fixed promotion rules (no changes post hoc)

A candidate passes engineering verification only if:

- positive mean `Delta_WGA` vs matched uniform across 3 seeds;
- positive full-seed pooled `Delta_WGA` in at least **2 of 3** seeds;
- **strictly** `Delta_AA > -1.00 pp` for **every** seed;
- all leak/provenance/hash checks pass.

These are minimum *engineering* criteria, not evidence of statistical significance or guaranteed new-distribution improvement. Report WGA intervals and any group-switching instability. **NO production integration** if either locked candidate fails. Do not retry a wider grid or tune using outer C outcomes; that would invalidate the gate.

If a candidate passes, add an **opt-in** inference aggregator with method selection (`uniform`, `distance`, `entropy`, `distance_top_m`, `class_softmin` as actually verified), serialized calibration/scale metadata, deterministic numeric-stability tests, and unchanged `uniform` default. Do not change prototype training or IPS. For a scientific improvement claim, still request a fresh untouched confirmatory evaluation on a new backbone/dataset.

## 5. Specific math and evaluation hazards to test

- Validate `softmax(logits)` reproduces `probs`, and verify if `logits` equals `-distances` up to documented learned scale/offset; don't assume.
- In `S1/S2`, softmax is **over classes**, not experts; logsumexp is **over experts**, not classes. Test axes with synthetic exact arrays.
- `S_k=tau*logmeanexp(-D/tau)` is a *class score*; the proposal is **not** a class-independent weighted average, and performance comparisons must say so.
- Inter-head absolute distances may be incomparable because of head-specific scaling or common offsets. Include the centered-distance control and log meaningful transformations. Normalization must never use held-out C observations.
- `dmin` measures closeness to **some** class prototype and can be tiny for a confidently wrong class; a low-entropy expert may also be confidently wrong. Explicitly test these failure modes on Uniform-wrong images.
- A static train/test **class-prior change** can affect AA/WGA independently of method; report class recall and confusion matrices.
- If Group labels are unavailable, WGA **cannot be computed or optimized directly** in strict group-blind selection. Report that clearly, never invent pseudo-group WGA.
- For binary class score temperatures, do not mix original DPE entropic scale `30` used during training with `tau` in logsumexp or `T_class` during aggregation.
- Avoid 'test-set peeking' at historical Waterbirds observations. Existing exploratory group findings can motivate the hypothesis but must not be reused to select settings.
- Candidate ordering by best WGA on meta_select is prohibited in the group-blind track even if group labels happen to exist in NPZ.

## 6. Checkpoints: MANDATORY, BEFORE any expensive work

Create `experiments/geometry_aggregation_v5/` and make these commands work:

```
python -m experiments.geometry_aggregation_v5.run --phase all --resume
python -m experiments.geometry_aggregation_v5.run --phase audit --resume
python -m experiments.geometry_aggregation_v5.run --phase screen --resume
python -m experiments.geometry_aggregation_v5.run --phase nested --resume
python -m experiments.geometry_aggregation_v5.run --status
```

**Run crash-safety tests first**. Persist atomically after verification, preprocessing, each audit table/plot, each screened candidate/seed, each head epoch, each head-final hash, each B/C prediction cache, each nested fold, each pooled-seed evaluation, and each decision gate. Use task IDs and stable input/source/config hashes; incompatible resume must halt rather than silently mix artifacts. One incomplete task may replay after interruption; completed tasks must be reused exactly.

Always update:

- `state.json` — DAG statuses, active task, hashes, seeds/folds and block reasons;
- `HANDOFF_STATUS.md` — concise last validated results and next exact task;
- `NEXT_AGENT_PROMPT.md` — one-copy prompt for another Codex session to continue with `--resume`;
- `completed/<task_id>.json` — output paths and SHA-256, source/config/input hashes, end timestamp;
- `PROTOCOL_DECISIONS.md` — deviations and implementation choices;
- `results/` — append per task immediately, not at run end.

Verify resume by forced process interruption and independent restart; compare resumed results with uninterrupted results. Report exact blockers if data/head code missing; **do not substitute test data or skip nested verification silently**. If quota is exhausted, leave complete handoff files and all completed results, not a promise of autonomous background processing.

## 7. Deliverables and decision-oriented report

1. `phase0_geometry_audit.md`, expert-ranking tables (overall and Uniform-wrong), per-group diagnostic, plots and CSV sources, head-distance comparability audit;
2. `phase1_screen.csv`, `phase1_locked_candidates.json`, per-method per-seed AA/WGA diagnostics, matched baselines, all configs and individual prediction exports;
3. `phase2_nested_results.csv`, group- and class-level tables, paired corrected/broken cases, confidence intervals, A/B/C manifests, trained-head hashes and promotion decisions;
4. `FINAL_REPORT.md` answering: **(a)** Is minimum distance more reliable than confidence precisely when Uniform fails? **(b)** Does class-wise Soft-min (not simply weighted voting) improve WGA? **(c)** Does it meet the <1 pp AA-loss floor for each seed? **(d)** Does it survive genuine nested verification? **(e)** What failed if not?; include the *single best evidence-supported next step*, not an open-ended wish list;
5. All runnable source code/tests/configs plus `state.json`, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`.

**Never claim an improvement purely from development OOF sensitivity results; preserve all failed runs. Do not change the research question mid-experiment.**
