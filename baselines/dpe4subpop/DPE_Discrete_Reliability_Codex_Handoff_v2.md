# CODEX HANDOFF — DPE v2: Discrete Reliability Weighting (crash-safe / quota-safe)

> **Execute autonomously in one run. Most important engineering requirement: SAVE DURABLE CHECKPOINTS AFTER EVERY ATOMIC TASK; RESTART MUST RESUME WITHOUT LOSING VALID COMPLETED WORK. If the Codex context/quota/process ends unexpectedly, the next Codex session must be able to continue from files alone.** Do not wait for the entire experiment suite to finish before writing any results.

## 0. Goal and non-negotiable constraints

Research extension of *Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift* (DPE; official repo `https://github.com/minhto2802/dpe4subpop`). Existing DPE uses `N=15` frozen prototype classifiers with equal probability weights `1/15`. Develop **interpretable discrete, class-conditional reliability-based weighting** and compare against matched uniform aggregation; main goal **held-out top-1 accuracy improvement**, secondary balanced accuracy and worst-group accuracy (WGA). Do not claim gains until measured. This is a NEW extension, not part of original paper.

- Do not modify/delete original DPE checkpoint, prior weighting experiments or reports. Work in a new versioned directory, e.g. `experiments/discrete_weighting/` and `outputs/discrete_weighting_v2/<run_id>/`.
- **Never fit/tune/select with test labels, test group labels, or information derived from the supplied test prediction bundle.** The bundle and prior test diagnostics motivate hypotheses only; test is already repeatedly inspected, so any further evaluation on it is **exploratory, not confirmatory**. If possible, identify a truly untouched confirmatory dataset/holdout, but do not fabricate one.
- No subgroup labels in training/tuning/selection for the `no-group` protocol. Group labels only for post-hoc reporting, strictly after selection lock.
- Never use true label, prediction correctness, `P_j(true class)`, oracle correctness or evaluation group at **inference**. Such quantities are used only to fit reliability tables on labeled out-of-fold (OOF) gate-training data.
- Freeze backbone and all prototype heads. Reuse existing **leak-free OOF** caches / compatible head checkpoints wherever possible. No unnecessary CNN/head retraining. Every result is paired with an equal-weight baseline using the EXACT same heads, sample IDs, class order, and raw/calibrated probability representation.
- Use 3 existing head seeds `{0,1,2}` when possible; avoid exploding search space. Main objective is accuracy, but always expose accuracy–WGA trade-offs and per-class recall. Maintain a strict train/select/test boundary.

## 1. Existing results and known failure modes (diagnostic context ONLY)

Prior report `README(3).md`, `summary.csv`, `phase2_crossfit5.csv`, `DPE_Prediction_Audit.md`, and `prediction_bundle.zip` may exist in the working tree; discover actual paths. The test bundle is **NOT a training dataset**.

- Matched equal-weight `U0`: accuracy `95.708%`, balanced accuracy `95.182%`, WGA `92.761%` (three seeds).
- Prior neural gate `D19`: accuracy `95.961%`, balanced accuracy `92.828%`, WGA `83.645%`; small +0.253 percentage-point accuracy gain, large -9.116 pp WGA drop.
- Across three test seeds, D19 made **586 prediction flips, ALL from Waterbird (label 1) to Landbird (label 0)**; 315 corrections of Landbirds, 271 broken Waterbirds. For 264/271 broken examples, a majority (>=8/15) of heads were actually correct.
- Per-seed any-correct-expert **label-aware oracle** accuracy approximately `99.684%` on average: large diversity/headroom, **NOT an achievable inference result**. Do not optimize towards the observed oracle labels on test.
- Raw mean confidence per head ~0.53 despite individual head accuracies >0.90. All stored member calibration temperatures in supplied test caches were `0.25` (possibly optimizer floor; investigate **without** test tuning). Raw uniform NLL ~0.632, calibrated uniform NLL ~0.481; calibration alone barely alters accuracy. Do not assume raw `confidence > 0.8` or treat temperature=0.25 as a proven bug.
- Test set class distribution is skewed: Landbird `4510/5794`, Waterbird `1284/5794`; optimizing micro accuracy may amplify class-0 preference. Report this explicitly. Earlier 3-fold gate OOF came from validation pool of `1199` samples, 80/20 gate train/select split; selection set ~240 samples is noisy. Previous original group-aware DPE checkpoint is not a clean, matched comparison.

**Current hypothesis:** The gate badly misuses confidence / overweights isolated experts and learns a one-direction class bias. First test transparent, capped, class-conditional reliability rules. A discrete method is an interpretable control, not automatically more accurate or robust.

## 2. Required data protocol (audit BEFORE any experiment)

1. Inspect repository, prior `experiments/weighting` run, metadata, caches, model/class order, code revision, data splits, hardware, and packages. Write `DATA_AUDIT.md` and `manifest.json`. Find OOF predictions from DPE heads **not trained on those OOF samples**. Shapes `[samples, 15, 2]`; labels `[samples]`; stable IDs. Verify no leakage, duplicates across train/select, finite probabilities, class order `0=Landbird, 1=Waterbird`.
2. **Preferred:** reuse 3-fold/5-fold head-refitted OOF predictions from prior implementation. If not available, generate OOF via class-stratified K-fold cross-fitting on original DPE validation pool: each fold refits all 15 heads only on other folds and predicts held-out fold; freeze upstream feature extractor; store each completed fold/head immediately. NO in-sample head predictions to fit/evaluate new gate.
3. Gate fitting: partition OOF predictions into a reproducible, class-stratified `meta_train` and `meta_select` (e.g. 80/20), independent of subgroup labels. Keep fixed split IDs for all methods. Fit quantile bins, confidence calibration, reliability priors, shrinkage estimates, optional models and feature scaling on **meta_train only**. Fit/tune method-specific thresholds using inner folds of meta_train where practical, compare shortlists once on meta_select. Record number of examples per class and bin; small select pool means high variance and selection optimism.
4. After Phase-1 selection, lock the winning recipe/config/seeds/split in an immutable `SELECTION_LOCK.json` with hash **BEFORE reading test results**. May refit the winning reliability tables on all permissible OOF development data according to the locked protocol; document this separately. For test use final full-validation DPE heads as in previous run; compare same heads to raw U0 and calibrated U0. Record OOF/final-head distribution mismatch.
5. If only `prediction_bundle.zip` with TEST predictions exists but **no leak-free OOF data**, do **not** build/tune on it. Implement/test code with synthetic tests, run label-free descriptive checks only, save exact blockers and commands. Never present in-sample/test-chosen gains as validation evidence.
6. In bundled test NPZ: `probs [5794,15,2]` means raw member predictions, `calibrated_probs` calibrated member predictions, `weights` existing D19 weights, `gate_probs` existing mixture. `gate_probs == (calibrated_probs * weights[...,None]).sum(1)`; test labels and `group=2*y+background` exist **for final evaluation only**. Check IDs/axis/transposition (`DPE.predict_proba(avg=False)` originally outputs `[members,samples,classes]`). Do not mix raw probs with calibrated gate weights.

## 3. Proposed method: interpretable reliability table

For new image `x`, each head `j` yields probs `p_j(k|x)`; compute **label-free**:
- `c_j = max_k p_j(k|x)` (confidence); `yhat_j = argmax_k p_j(k|x)` (PREDICTED class, not ground truth).
- `agreement_j = (1/(N-1)) * sum_{h!=j} 1[yhat_h==yhat_j]` (leave-one-out vote agreement; alternatively use vote share; do not include head itself without documenting).
- optional margin `|p_j(1)-p_j(0)|`, rank/confidence relative to others; NO labels or true-class probability.

On labeled **meta_train OOF only**, learn smoothed class-conditional correctness/reliability for bins. Recommended simple base:
- Bin confidence into quantile tertiles **within predicted class**, thresholds estimated on meta_train only. Avoid absolute thresholds (raw expert confidence ~0.53).
- Pool observations across all 15 heads for robust global table `r_pool[predicted_class, confidence_bin]` while acknowledging correlated heads. Optionally compute head-specific `r_head[j, predicted_class, confidence_bin]` with hierarchical shrinkage toward pooled prior:
  `r=(correct_count + kappa*r_pool)/(count+kappa)`; `kappa in {10,50,200}`, predeclared, optionally inner-CV selection. For empty bins fall back to pooled/class/global prior.
- **Class-conditional** means conditioning on `yhat_j`, the *predicted class* available at inference. Never condition on unknown `y_true`. Record class imbalance bias: optimizing correctness may still prefer label 0; evaluate class-balanced fitting/weighting variants.
- Convert reliability to score in `[-1,1]`: `s_j(x)=2*r_j(x)-1`. Then `v_j=softmax_j(beta*s_j)`; `beta in {0.5,1,2,5,10}`. Final weights use **uniform residual** `w_j=(1-alpha)/15 + alpha*v_j`, `alpha in {0.1,0.25,0.5,0.75,1}`; thus weights sum 1 and are nonnegative. If bins missing, NaNs, or no positive evidence, deterministically fall back to uniform. Preserve both continuous softmax mapping and hard/discrete mapping as controlled contrasts.
- Stronger discrete version: bin **predicted class × confidence tertile × agreement tertile**; keep pooled reliability + hierarchical smoothing. As 2×3×3×15 cells may be sparse, prefer shared pooled table first; require `min_count` and backoff to coarser bins. Agreement thresholds based on vote fractions (e.g. low <=1/3, medium <=2/3, high >2/3) or meta_train quantiles, predeclared.
- Truly discrete scoring (not just discrete input bins): map smoothed reliabilities to ordinal `score ∈ {-1,-0.5,0,0.5,1}` using **train-fitted quantile cutpoints** or predeclared correctness-probability bands, then normalize with softmax/uniform residual; compare with continuous `2r-1` output using the *same fitted bins*. Never hand-tune cutpoints from test.
- Consensus guard as an **optional ablation**, not a hard universal theorem: if `votes_supporting_uniform_prediction>=12/15` AND uniform margin >= threshold learned with meta_train/selection-only process, use uniform probabilities; otherwise use discrete reliability weights. This may hurt minority subgroups, so compare carefully.
- Optional reliability-aware **class-balanced** fit: estimate/calibrate correctness with inverse class weights or balanced target loss, but define exactly whether balancing by true class (allowed during training) or predicted class; **at inference no true class**. Keep unbiased empirical correctness as primary baseline and contrast class-balanced variants. Any method involving external model must only see OOF meta_train.

### Critical conceptual caveat
Correctness probability `P(yhat_j==Y|features)` does **not mathematically imply** Bayes-optimal mixture weight. Treat reliability weighting as heuristic to test; also run direct validation accuracy baselines (threshold + uniform and convex residual). If all members are wrong for an example, positive convex weighting cannot necessarily fix it. Oracle is diagnostics only.

## 4. Experiments — Phase 1: find best combination, ranked priority

**No giant grid:** fixed small config set per family, sharing cached OOF predictions. Stage cheap methods first, then only extend when resources permit. Every experiment/seed/config is a durable independent work item.

| ID | Priority | Recipe | Hypothesis / constraint |
|---|---:|---|---|
| D0a | 0 | Raw uniform `mean(raw_probs)` | Exact matched reference |
| D0b | 0 | Calibrated uniform `mean(calibrated_probs)` | Calibration control; fit/calibrator from meta_train only |
| D1 | 1 | Tuned final decision threshold on **uniform** `P(class1)` | Tests whether gain is mainly class-prior/threshold correction; cannot select on test |
| D2 | 2 | Shared 2×3 predicted-class × confidence bins + pooled reliability smoothing | Simplest discrete reliability |
| D3 | 3 | D2 with head-specific smoothed reliability, shrink to pooled | Head reliability differences |
| D4 | 4 | D2/3 + uniform residual (`alpha`) and softmax beta | Prevent isolated expert dominance |
| D5 | 5 | D4 + 3 agreement bins; sparse-cell backoff | Consensus helps detect wrong dissenters |
| D6 | 6 | D5 + consensus guard (vote/margin); include unguarded control | Protect strong correct majorities |
| D7 | 7 | D2/3 ordinal/hard discrete scores `{-1,-.5,0,.5,1}` | Does hard discreteness beat continuous score? |
| D8 | 8 | D2/3 score = continuous `2r-1` with identical bins | Is continuity valuable AFTER discretization of features? |
| D9 | 9 | Predicted-class-balanced reliability/weighted criterion | Reduce one-way class bias |
| D10 | 10 | Reliability top-k (`k∈{3,5,10,15}`) with residual/guard; never oracle | Suppress noisy experts |
| D11 | 11 | Reliability from confidence **rank** rather than raw scale | Robust to miscalibration |
| D12 | 12 | Calibration variants: raw vs per-head temperature vs shared temperature; only train-split fitting | Investigate `T=0.25` floor and calibration confound |
| C0 | Control | Fixed static weights from meta_train reliability, with residual | Dynamic benefit beyond global head ranking |
| C1 | Control | Existing matched D19 and regularized linear gate IF saved from prior run | Fair comparison; no test-based re-selection |
| X0 | Diagnostic | Any-head-correct oracle, overlap, unanimous/all-wrong counts | **LABEL-AWARE, NEVER SELECT / DEPLOY** |

**Budget:** run all cheap D0–D5 and X0 first; then D6–D12/C0/C1 if dependencies/compute allow. Screen e.g. `alpha={0.1,0.25,0.5,0.75}` `beta={1,3,5}` `kappa={10,50,200}` using a budgeted, predeclared staged or coordinate search (not exhaustive 4×3×3×all variants); document exact list before results. Use identical split/seeds for candidates. Prefer inner CV on meta_train, keep meta_select for one family-level selection, tie-break by balanced accuracy then NLL. Ranking must be decided before test is opened. No model or threshold choice from WGA if subgroup annotations are evaluation-only.

**Metric table for EACH completed candidate, seed and config:** select top-1 accuracy; delta to both uniform controls in pp; balanced accuracy; class-0/1 recall; NLL; Brier; ECE; mean effective members exp(H(weights)); average/max largest-member weight; weights on wrong experts (select diagnostics only); prediction flips 0→1 and 1→0; corrections/breaks; wall time; sample/bin counts; config and provenance. If group annotations available, report WGA only after selection lock (post hoc).

## 5. Phase 2: controlled ablation of locked winning family

Run automatically after Phase 1 **using exact locked winner**, across seeds 0/1/2, matched final DPE heads. Predeclare ablation matrix now (skip N/A with reason):

- A0 full selected method; A1 uniform raw/calibrated; A2 remove predicted-class conditioning; A3 remove confidence binning; A4 remove consensus features; A5 disable consensus guard; A6 pooled vs head-specific reliability; A7 remove smoothing (`kappa=0`, with robust empty-bin fallback); A8 vary `kappa`; A9 ordinal hard vs continuous score with fixed bins; A10 `alpha={0.1,0.25,0.5,1}`; A11 vary `beta`; A12 raw vs calibrated vs shared-temperature inputs; A13 class-balanced vs empirical reliability; A14 top-k=3/5/10/15; A15 replace gate with training-average/static weights; A16 3 vs 5 vs 10 vs 15 members **with count-matched uniform control**; A17 disagreement / consensus threshold sensitivity; A18 3-fold vs 5-fold OOF (reuse prior caches, only if valid); A19 repeat seed/split sensitivity; A20 random reliability and shuffled-bin negative controls.
- After lock, evaluate on test **once per frozen family/ablation**; don't change winner based on ablation test results. Note all results on current previously-explored test are exploratory. If a clean new holdout exists, final confirmatory stage MUST remain untouched until all method decisions locked.
- Report mean±sample SD across seeds, per-group WGA/accuracy (post-hoc), per-class confusion, corrected/broken IDs, confidence bins, expert weight distribution, majority-overturn counts, paired sample bootstrap CI for accuracy delta where feasible; explicitly note shared backbone and only 3 seeds.

## 6. MANDATORY crash-safe, quota-safe incremental checkpointing — HIGHEST PRIORITY

### Persistent filesystem + directory layout

Create a **new run directory** with immutable run_id from config/source/input fingerprints; **never overwrite old runs**:

```
outputs/discrete_weighting_v2/<run_id>/
  config.lock.yaml                 # full resolved config, immutable
  manifest.json                    # code commit, file hashes, schema, split hashes
  DATA_AUDIT.md
  state.json                       # authoritative task state (atomically replaced)
  progress.ndjson                  # append-only events, flushed
  HANDOFF_STATUS.md                # ALWAYS CURRENT: next task and exact resume cmd
  NEXT_AGENT_PROMPT.md             # concise self-contained continuation message
  cache/oof/seed<k>/fold<f>/...    # valid per-fold/per-head OOF cache
  fits/<candidate>/<seed>/<cfg_hash>/...
  completed/<task_id>.json         # commit marker ONLY after artifacts verified
  results/phase1_incremental.csv   # one completed result row per task, dedup by task ID
  results/phase2_incremental.csv
  results/phase1_ranked.csv
  SELECTION_LOCK.json              # written before any test evaluation
  test_results/<candidate>/<seed>/... # write ONLY after lock
  REPORT.md                        # progressive factual report
  BLOCKERS.md
  run.log
```

### Exactly-once durable work units

1. Decompose run into tasks: `environment_audit`, `oof:seed:fold:head`, `oof_verify:seed`, `calibrate:seed`, `fit:phase:method:seed:cfg`, `score_select:...`, `lock_selection`, `test_eval:...`, `ablate:...`, `report`. Each has stable `task_id`, dependency IDs, deterministic seed/config hash and required output artifacts.
2. **After EACH completed head/fold, fitted reliability table, tested candidate/config/seed, ablation, or final report stage:** write artifacts to a temp path; `flush()`/`fsync()` where practical; verify shapes/checksums/finite values/reconstruction; atomic `os.replace(temp, final)`; then atomically publish `completed/<task_id>.json` (hashes+metrics+paths); then update `state.json`, incremental CSVs, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`. Never mark a task complete before its artifacts are valid. On Windows use same-directory temp files and closed handles before `os.replace`.
3. **Save a row immediately when each experiment/seed/config finishes.** Never accumulate Phase-1 results only in memory or wait until all experiments complete. Incremental CSV updates must be atomic (`temp→replace`), stable schema, no duplicate task IDs; treat completed task records as source of truth and regenerate CSV idempotently at resume. Logs can be appended with flush. Store at least a per-task JSON even if full report incomplete.
4. If any ML training is involved, checkpoint **each epoch** (model, optimizer, scheduler, RNG state, epoch, best epoch, feature scaler/calibrator, config fingerprint). If pure table/heuristic fitting, checkpoint each finished bin table/candidate (not only at phase end). Save per-head/fold outputs immediately; do not repeat expensive backbone inference unnecessarily.
5. `--resume` must: verify manifest/config/source/input hashes; scan `completed/` artifacts and checksums; **skip all valid completed tasks**, rebuild missing/corrupt only; reconstruct ranked results/status without duplication; resume incomplete epoch (or restart at most one unfinished atomic task); never blindly rerun full sweeps. On incompatible source/config do **not** silently reuse: start a new run ID or exit with explicit guidance. Support `--status` and `--dry-run`.
6. **Quota/interrupt behavior:** On SIGINT/SIGTERM or `--max-minutes`/`--max-tasks` limit, stop accepting new tasks, finish/commit current small unit where possible, update state + handoff files, exit cleanly. If abrupt quota kill prevents final callback, latest committed task markers and incremental results are sufficient to resume. No network/session reliance. Do not pretend to detect remaining Codex account quota; instead use small task granularity and persistent checkpoints.
7. Keep `HANDOFF_STATUS.md` automatically refreshed after every task, **not just on graceful shutdown**. Include run_id, timestamps, overall status (RUNNING/INTERRUPTED/BLOCKED/COMPLETE), code commit, completed/total counts by phase and seed, best **selection-only** candidate so far (never peek test), last completed task, next pending task, known issues, exact commands, artifact paths, and short summary of decisions. `NEXT_AGENT_PROMPT.md` must instruct a fresh Codex session to read status+manifest+config then run `--resume`, not repeat completed work. Update `REPORT.md` progressively without fabricating incomplete metrics.
8. Failure isolation: catch candidate-level exceptions; write `failed/<task_id>.json` with traceback, reason, retry count, and continue independent tasks. Block only required dependencies. Validate consistency and test leakage checks before final reporting. Keep logs concise, print each task completion with committed output path.
9. **Interruption acceptance test (required):** run with `--max-tasks 2`, inspect persisted state/results; re-run `--resume --max-tasks 2`; verify no rerun/no duplicates, then run to completion. Simulate truncated `.tmp` artifact and missing/corrupted completion output; resume must recover safely. Test resume across a different process. Include pytest for stable hashing, deterministic task IDs, atomic updates, integrity checks, idempotence, no use of labels/groups in inference, and class/probability normalization.

### Suggested CLI (adapt to real repo; maintain one-command complete run)

```bash
python -m experiments.discrete_weighting.run \
  --config experiments/discrete_weighting/config.yaml \
  --phase all --resume

# Progress / new-session continuation
python -m experiments.discrete_weighting.run \
  --config experiments/discrete_weighting/config.yaml \
  --status

# To limit work before a quota/context boundary
python -m experiments.discrete_weighting.run \
  --config experiments/discrete_weighting/config.yaml \
  --phase all --resume --max-tasks 5
```

Do not require interactive questions to resume. On a blocked environment, write `BLOCKERS.md` with **exact** missing paths, expected shapes, prior tasks completed, and next executable command; preserve partial results.

## 7. Final deliverables and success criteria

Files: executable code+tests; resolved config; OOF/parity/leakage audit; per-task checkpoints; `state.json`, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`, `progress.ndjson`; Phase-1 full incremental + ranked results; Phase-2 ablation results; matching per-seed test predictions; corrected/broken case CSVs; `REPORT.md`, `RESULTS_FOR_USER.md` (Vietnamese, <=1 page), blockers if any. Include small SHA-based provenance. One command must execute all valid remaining work with `--resume`.

A method is **not a win** merely because it increases confidence/NLL/calibration. Compare held-out accuracy vs matched DPE, report seed variability, class-specific effects and WGA. Do not bury a substantial WGA decrease. If none beats uniform reliably, conclude that honestly and present the most promising control+diagnostic. Never present the label-aware oracle as deployable performance.

**End-of-run console summary:** finished/pending/skipped tasks; best **selection-locked** method/config; matched baseline, accuracy delta, balanced accuracy, WGA tradeoff if reportable; output directory; **exact resume command**. If quota stops execution early, the crucial deliverable is the **most recent durable checkpoint and self-contained handoff status**, not a verbal promise to finish later.
