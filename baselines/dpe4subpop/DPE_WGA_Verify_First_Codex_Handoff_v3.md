# CODEX HANDOFF v3 — VERIFY WGA GAINS FIRST, THEN IMPLEMENT

**Goal:** Verify the previously reported WGA/AA improvements for a DPE weighting extension **before** integrating it into the repository. **Primary KPI = WGA**; AA must remain at least **matched-uniform AA − 1.00 percentage point (pp)**. Never promote a method because of AA alone.

**Execution policy:** Run autonomously in order, save results incrementally, never fabricate numbers, never retune using inspected test results. If a hard verification gate fails or inputs are missing, **STOP before production integration**, preserve all artifacts, produce a precise blocker/reproduction report and next-agent handoff. No confirmation prompts for routine engineering choices.

## 0. Inputs, known facts, and limits

Find these inputs in the current workspace, or unpack the accompanying package:

- `development_oof_bundle.zip` — official Waterbirds development/validation examples; 3 seeds, each 959 `meta_train` + 240 `meta_select`, **1,199 unique IDs**. NPZ: `probs [M,15,2]`, `logits`, `distances`, `embedding [M,2048]`, `y`, `group`, `sample_id`, `fold`, `seed`, etc. See bundle `README.md`, `manifest.json`, `fold_provenance.json`.
- `DPE_WGA_Deep_Analysis.md` — original exploratory analysis / exact recipe and qualifications.
- `DPE_WGA_Audit_Repro.zip` — source scripts (`cv_sensitivity.py`, `epigraph_verify.py` when available separately, `bootstrap.py`, etc.) and original CSVs. Note some scripts contain absolute `/mnt/data/...` paths; make paths configurable.
- Existing DPE repository/backbone/head trainer and checkpoints, when available. Locate actual APIs; don't assume modules. Original feature backbone here is **ERM-star, 400 epochs with augmentation**; three head seeds share that same backbone. Do not present it as standard ERM.

**Critical limitations:** Previously reported 5-fold experiment repartitioned *existing 3-fold OOF predictions*, but **did not retrain prototype heads inside every new outer fold**; it is a **sensitivity analysis, not fully nested CV**. Also, the method was nominated after inspecting development group metrics, and the historical Waterbirds test has been repeatedly examined. Do **not** label existing numbers as independently confirmed, purely group-blind model selection, or final/test performance. `group` is **evaluation-only** for all fresh fits and any newly fixed protocol.

## 1. Frozen algorithm specification (do not change to chase WGA)

Name: **Conservative Latent-Cluster Minimax Static Weighting** (LCMSW).

For each weight-fitting pool with DPE member probabilities `P` of shape `[M,15,2]`:

1. Freeze backbone and the 15 already trained prototype classifiers. Fit **PCA(24, svd_solver='randomized', random_state=0)** on *only this pool's* 2048-D embeddings.
2. Fit **KMeans(n_clusters=2, n_init=6, random_state=1)** to the PCA representation on that same pool. This is global clustering, **not** within-class clustering. The clusters are used **only to form the loss groups**; they are not a routing model at inference. Do not use `group` or background labels.
3. Learn one **global, static** vector `w ∈ R^15`, `w_j ≥ 0`, `Σ_j w_j=1`, `w_j≤0.25`, initialized at `u_j=1/15`.
4. For each example, `q_i(w)=Σ_j w_j * P[i,j,y_i]`; `ℓ_i(w)=−log(clamp(q_i(w), 1e−8, 1))`. Let `L_c(w)=mean_{i in cluster c} ℓ_i(w)`. Optimize

   `min_w max(L_0(w), L_1(w)) + 0.02 * Σ_j(w_j−1/15)^2`.

5. **Production solver:** epigraph SLSQP, with auxiliary `t`: minimize `t + 0.02||w-u||²`, subject to `L_0(w)≤t`, `L_1(w)≤t`, simplex and cap constraints. Reproduce source behavior before switching solver. Validate convergence, bounds, feasibility, objective against direct-max optimization; never silently use failed/invalid weights.
6. **Inference:** for every sample and class `k`, `P_final(k|x)=Σ_j w_j P_j(k|x)`, then `argmax_k`. No PCA, KMeans, labels or gate network needed at inference. Compare with the *same heads* averaged uniformly. Preserve member order and head checkpoint identities.

No MLP, dynamic gate, class threshold shift, temperature tuning, group-aware loss, additional search grid or test-driven optimizer changes are part of the frozen primary recipe.

## 2. MANDATORY GATE A — Reproduce the reported numbers exactly

### A1. Audit the source

- Check original SHA256s from manifest (or explain any missing hashes), `probs` shape/finite/normalization, class order `0=Landbird,1=Waterbird`, group mapping `g=2*y+background`, IDs and labels aligned, exact OOF head-fit/held exclusion using fold provenance, per-seed `meta_train ∩ meta_select = ∅`, 1,199 unique IDs, no test examples. Flag any issue; never substitute in-sample final-head predictions.
- Explicitly document head-index mismatch across OOF folds: `head-00` in fold 0 and fold 1 are **different parameter sets**. This threatens transfer of static per-head weights to final refitted heads.
- Lock source/config hashes and library versions (`numpy`, `scipy`, `sklearn`, `torch`) into a manifest before runs.

### A2. Reproduce legacy 80/20 split results

For every seed, fit PCA/KMeans/minimax weights on `meta_train` only, predict `meta_select` only. Use the original **raw temperature-1** member probabilities. WGA uses the minimum accuracy across **all 4 real groups**, calculated *after* predictions (evaluation only). Also report AA, BA, worst-class acc, full per-group counts and weight min/max, top members, effective number of members.

**Expected mean across 3 seeds (percent):** matched-uniform **AA 95.00, WGA 90.72**; candidate cap=.25 **AA 95.97, WGA 92.76**. Expected WGA changes by seed approximately `[0.00, +3.85, +2.27] pp`. This is an exploratory reproduction target, NOT independent confirmation.

### A3. Reproduce the original 5-fold OOF sensitivity analysis

On each seed concatenate the 959/240 OOF rows, retaining sample identity. Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=3001)`, split by `y` only. For each held-out fold: fit PCA, KMeans, `w` on its other four folds only; predict that held-out fold. Compare to uniform on **exactly the same** 15-head OOF probabilities. Pool all five fold predictions **per seed**, then calculate four-group WGA; **do not average fold-level minima**.

**Expected:**

| Seed | Uniform AA% | Candidate AA% | Uniform WGA% | Candidate WGA% |
|---|---:|---:|---:|---:|
| 0 | 94.7456 | 95.0792 | 91.7293 | 93.9850 |
| 1 | 94.8290 | 95.2460 | 92.2747 | 93.1330 |
| 2 | 95.1626 | 95.7465 | 92.4812 | 94.7368 |
| **Mean** | **94.91** | **95.36** | **92.16** | **93.95** |

Thus the previously reported **ΔAA ≈ +0.45 pp; ΔWGA ≈ +1.79 pp**, computed as mean paired deltas across seeds (not percent relative changes). Expected flips across seed×sample rows: **48 total, 32 corrected, 16 broken**; IDs repeated between seeds are NOT independent images.

Independently reimplement the calculation instead of solely copying a precomputed CSV. Also execute/compare the original reproduction script, use the epigraph formulation, and inspect solver constraints. Verify results within **0.03 pp** of rounded reference numbers and document any library/solver-related deviations; metrics at full precision should agree more closely when environment matches. Treat a mismatch >0.03 pp, missing group, leakage, wrong member axis, failed optimizer or an unexplained head ordering issue as **GATE A FAIL**.

Artifacts: `verification/seed*/metaselect_metrics.json`, `verification/seed*/cv5_metrics.json`, `verification/seed*/cv5_predictions.npz`, `verification/seed*/weights_by_fold.csv`, `verification/reference_comparison.csv`, `verification/integrity_report.md`, `verification/solver_audit.csv`, and `verification/GATE_A.json` (`PASS|FAIL`, reasons).

## 3. MANDATORY GATE B — True held-out head+weight verification

**Only after Gate A PASS**, perform a new **group-blind, nested head+weight evaluation** against matched uniform. This stage is needed because Gate A merely reproduces the non-nested sensitivity result. No hyperparameter search: algorithm above is frozen. No WGA-driven selection. Original train backbone remains frozen.

Recommended feasible primary design, using the existing original repository trainer and all 1,199 development examples:

- For each of **3 independent experiment seeds** and **3 outer class-stratified folds**, define `C=outer_holdout`, and split `outer_train` by class into `A=head_fit` (~70% of outer_train) and `B=weight_fit` (~30%). Use deterministic saved manifests. **All three sets must be disjoint per outer fold.** Record ID hashes.
- Fit 15 prototype heads from scratch **only on A** using the original DPE class-balanced resampling, IPS loss, head training schedule and frozen backbone. Preserve the precise per-member order. A may contain overlap among the 15 bootstrap draws; it must not overlap B or C.
- Generate predictions of those *same frozen 15 heads* on B and C. Fit PCA/KMeans/w on **B only** (labels from B allowed). Never refit or reorder heads after learning weights. Predict C by weighted vs uniform aggregation using identical heads, feature extractor and preprocessing.
- Fit no component on C; use C labels **only for final metric computation**. Use `group` **only when reporting held-out WGA**, never for training, selection, clustering, stopping or feature construction. If group labels are needed for confidence intervals, apply them only after predictions are irrevocably saved.
- Aggregate predictions across **all outer folds within a seed** before computing seed-level WGA. Do not interpret 9 folds × 3 seeds as 9 independent backbones; the backbone is shared. Use class-balanced outer splits, not group-stratified splits. Check every group has held-out examples before computing WGA.
- Don't silently replace this with another 5-fold partition of the supplied OOF cache; that would repeat Gate A's limitation. If raw images, official split, frozen backbone, code or legitimate head retraining resources are unavailable, mark **GATE B BLOCKED**, preserve verification, and do not claim independent verification or begin deployment.

**Predeclared decision criteria (no change after seeing groups):** Candidate must have (i) positive mean ΔWGA vs matched uniform, (ii) WGA improvement in at least **2/3 seeds**, (iii) mean ΔAA > **−1.00 pp**, and (iv) no individual seed ΔAA ≤ **−1.00 pp**. All are *engineering promotion criteria on the new nested development experiment*, not proofs about unseen distributions. Report paired uncertainty (bootstrap resample by unique image ID, align repeats across seeds), group counts, corrected/broken, both class recalls, and potential training/refit variability. A CIs crossing zero means evidence is uncertain even if point-estimate gate passes. **Only mark GATE B PASS** if the predeclared criteria pass AND provenance/leakage audit is clean. Otherwise stop production integration and write a failure-analysis report, including whether head-fit sample size or head identity explains a difference.

**Research caution:** Even nested CV on this development dataset is not a brand-new independent test after prior researcher inspection. No claim of a confirmed scientific WGA gain until a **fresh, untouched held-out dataset / external shift benchmark and independent backbone(s)** confirm it. Do not use the historically inspected Waterbirds test to tune or select.

Artifacts: `nested/seed*/fold*/head_fit_ids.csv`, `weight_fit_ids.csv`, `outer_holdout_ids.csv`, per-head checkpoints, B/C prediction caches, fit preprocessing/weights, deterministic run config, per-fold scores, pooled per-seed predictions, paired metrics, `nested/verification_report.md`, and `verification/GATE_B.json`.

## 4. IMPLEMENTATION PHASE — only if Gates A and B PASS

Integrate as an **opt-in** method in the existing DPE repository, without changing default uniform behavior or overwriting old DPE/D19/D5 files.

Required interfaces (adapt names to actual repository):

- `fit_static_minimax_weights(member_probs, labels, embeddings, config) -> weight_artifact`
- `predict_weighted(member_probs, weight_artifact) -> [M,K]` with shape checks and member-order identity validation.
- Save `weights`, `member_ids`, exact head checkpoint hashes, PCA/KMeans *training metadata* for traceability (not required at inference), source split IDs, optimizer status/residuals, objective values and config/hash.
- Expose e.g. `--ensemble-weighting uniform|latent_minimax`; default is `uniform`. Support inference with fixed paired heads+weights; **refuse** a weight artifact against mismatched/reordered heads. Never reuse OOF-fold slot weights against newly refitted final heads without a validated compatibility protocol.
- Include a minimal retrain/deployment recipe with **disjoint head-fit and weight-fit samples** and a documented AA/WGA evaluation command. No group annotations needed for method training.
- Add unit tests: probability normalization/shape, uniform parity, simplex and cap, direct vs epigraph objective, PCA/KMeans train-only, stable ordering, reproducibility, no train/test ID overlap, corrupt artifact fail-closed, label/group inference-time absence, frozen backbone+heads, CLI backward compatibility.
- Provide benchmark and short documentation: WGA first, AA constraint, statistical uncertainty, performance and memory vs uniform. No unverified performance promise.

Suggested commands (implement with repo-appropriate paths):

```bash
python -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --phase audit --resume
python -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --phase nested --resume
python -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --phase integrate --resume
python -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --status
```

The `all` entrypoint should run audit → check GATE A → nested → check GATE B → integration, **fail closed** on failures or blockers. Implement status-only mode without starting expensive work.

## 5. NON-NEGOTIABLE QUOTA / INTERRUPTION SAFETY

**Before expensive work**, implement and test an **idempotent, crash-safe checkpoint system**. No finished experiment should be lost if Codex context, quota, process or machine stops:

- Persist **each integrity check, seed, outer fold, fitted head, epoch (including optimizer/RNG state where training allows), PCA/KMeans fit, optimizer solution, prediction cache, and evaluation** immediately; do not wait for a whole phase.
- Use task IDs + dependency graph and `source/config/code SHA256`, atomic write (`tmp` then rename), durable `completed/*.json` with hashes/size/validation, and a state file. Never label unfinished files completed.
- On `--resume`, verify checksums, skip valid tasks, rerun only missing/corrupt tasks and their dependents. If cancellation occurs mid-head, resume from latest valid epoch checkpoint where supported; otherwise replay only the unfinished head. No inconsistent reuse after source/config changes.
- Maintain after **every completed task**: `state.json`, `HANDOFF_STATUS.md` (achievements, exact metrics, blockers, next pending task), `NEXT_AGENT_PROMPT.md` (one concise command to continue), `results/partial_metrics.csv` and `results/events.jsonl`. Keep existing artifacts immutable.
- Run a **separate-process interruption test** before nested fitting: kill/restart during a synthetic task and during a real head stage, verify no completed task repeats, all checksums pass and no partial output is mistaken for a valid result.
- Minimize Codex token/computation use: cache backbone embeddings and heads, save detailed logs to files not chat, output terse one-line milestones, do not run unexplained sweeps. If quota is low, checkpoint + update the handoff files + stop gracefully rather than starting a long uncheckpointed operation.

## 6. Deliverables and terminal report

1. `verification/REFERENCE_REPRODUCTION.md` — exact measured AA/WGA per seed, paired deltas, max numerical discrepancies and Gate A decision.
2. `nested/NESTED_EVALUATION.md` — matched uniform/candidate per seed and group, CIs, AA budget, provenance, limitations and Gate B decision; if blocked, clear evidence of why.
3. `src`/`experiments` integration + tests **only if** Gate B PASS, plus demo inference and measured regression checks.
4. `state.json`, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`, task checkpoint manifest, reproducibility environment, `FINAL_SUMMARY.md` (what was actually completed, not what was planned).
5. Keep all original checkpoints, past reports and raw data unchanged.

**Compact Codex kickoff:** "Read this file and execute it in order. FIRST independently reproduce +1.79 pp WGA/+0.45 pp AA; SECOND perform genuine held-out head+weight confirmation; ONLY THEN integrate if both gates pass. Checkpoint every task/epoch, maintain live handoff files, and never use previously inspected test labels for model selection. No silent fallbacks or invented results."
