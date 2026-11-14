# CODEX HANDOFF v4 — DPE Score-Mechanism Audit + Interpretable Piecewise Reliability

**Single research question:** Can an *inference-available* estimate of prototype-classifier reliability yield **higher worst-group accuracy (WGA)** than matched uniform DPE, while the decrease in average accuracy (AA) is **strictly < 1.00 percentage point per evaluation seed**?

**Two connected tasks, not an unrestricted model search:** (0) explain the failure of the existing continuous D19 gate using score/weight/reliability visualizations; (1) test a small, preregistered family of transparent, piecewise-linear *teacher* scores followed by an inference-safe smoothed reliability lookup; (2) verify any promising recipe using genuinely held-out head-and-weight evaluation. **Do NOT immediately create another neural architecture, tweak feature extractor, add pseudo-clusters, or retrain all prior experiments.**

**Priority order:** data/provenance correctness > reproducible mechanism audit > WGA > per-seed AA floor > compute/token efficiency. Preserve the original DPE and all historical artifacts. Treat previously inspected test results as diagnostic only, never model-selection evidence.

---

## A. Existing evidence and explicit constraints

* DPE has **N=15 frozen prototype heads**, 2 classes (`0=Landbird`, `1=Waterbird`). Uniform reference: `p_uniform(x) = mean_j p_j(x)` (use identical ordered heads and the proper probability representation for each comparison).
* Prior continuous D19 gate: ~**AA 95.961%, WGA 83.645%** vs matched uniform **AA 95.708%, WGA 92.761%**. It frequently reallocated excessive weight to a wrong expert and its changed labels were overwhelmingly Waterbird→Landbird. Historical test has been inspected repeatedly. These values describe prior experiments only; **not** fresh confirmatory evidence.
* Prior discrete D5: **AA 96.076%, WGA 90.343%**, still below uniform WGA. It conditions empirical reliability on predicted class, confidence and leave-one-out agreement; this is an important *control*, not a verified improvement.
* Prior LCMSW Gate B failed (nested head/weight verification): three-seed mean WGA +0.2915 pp, mean AA −0.8340 pp, only 1/3 seeds improved WGA; no production promotion. **Do not resurrect or retune LCMSW in this task.**
* `development_oof_bundle.zip` contains per seed 959 meta_train and 240 meta_select samples, with OOF predictions, frozen embeddings, labels and groups. The 3 seeds share **one ERM* backbone** and the image IDs can repeat across seeds. Fold-specific head slots represent **different actual heads**, so do **not** assume `head-03` on fold 0 equals `head-03` on fold 2. Use provenance/hashes. This OOF dataset is *exploratory*, not fully nested and not an untouched test.
* `prediction_bundle.zip` contains D19 historical **test** probabilities and `weights`, with labels/groups only for *post-hoc diagnosis*. D19 combines **calibrated_probs × weights**; reconstruct using those, not raw probabilities. **NEVER fit bins, calibration, thresholds, teacher scores, weights or select a candidate on these test rows.**

### Data and source lookup (no silent fallbacks)

Search repository and this handoff's `inputs/` recursively for:

- `development_oof_bundle.zip` (development data; mandatory to train/evaluate new scorer);
- `prediction_bundle.zip` (optional for *diagnostic-only* D19 score/weight heatmaps);
- D19 saved gate checkpoint/forward implementation, prior result CSVs/READMEs, original DPE head training code.

If original D19 raw gate logits or compatible checkpoint are unavailable, recover only **centered logits** from positive weights `center(log(weights), axis=experts)` when weights were made by a direct softmax. They are equivalent to the original logits up to common offset and any gate temperature; **do not report them as absolute raw logits**. If weights themselves came from residual mixing or another transform, reconstruction is invalid; report this and use final weights alone. Never invent missing arrays or silently re-run D19 with different heads.

### Development NPZ arrays

`probs [M,15,2]`, `logits [M,15,2]`, `distances [M,15,2]`, `embedding [M,2048]`, `y [M]`, `sample_id`, `group` (evaluation only, mapping `2*y + background`), `fold`, `seed`, `split`, `head_id`. Read/export manifest, class ordering, backing state hashes, train/held IDs. Validate shapes, sample alignment, `probs.sum(-1)≈1`, IDs disjoint **within each seed's** meta_train/meta_select, OOF provenance disjoint with each predicting head's training IDs. Do not pool same-image rows across seeds as independent observations.

---

## B. Phase 0 — Audit existing continuous D19 gate BEFORE training any new score model

**Purpose:** distinguish (H1) predicted-class bias, (H2) score/correctness misalignment, (H3) softmax over-concentration. No D19 retraining. Use old test bundle *solely to describe mechanisms*, label outputs clearly **HISTORICAL TEST / DIAGNOSTIC ONLY**. Where a D19 gate checkpoint can be scored on development OOF, explicitly identify predictions that were in-sample for that gate; they are not held-out validation.

For each image `i` and expert `j`, export one row:

`seed, split, sample_id, fold, expert_id, y_true[AUDIT_ONLY], group[AUDIT_ONLY], predicted_class, confidence, signed_margin, normalized_distances[2], loo_agreement, gate_raw_logit(if exact), gate_centered_logit(if valid), gate_final_weight, expert_correct[AUDIT_ONLY], uniform_prediction, gate_prediction, flip_type(corrected/broken/unchanged)`.

Definitions:

- confidence `c_ij = max_k p_ijk` for the **explicitly named** probability representation; separate raw and calibrated views;
- `loo_agreement = #{l != j : argmax p_il == argmax p_ij}/14`;
- `a_centered_ij = a_ij - mean_l a_il`; use this instead of uncentered gate logits for cross-sample comparisons;
- weight on wrong experts for analysis only: `sum_j(w_ij * 1[expert_prediction_ij != y_i])`;
- per-image effective number of experts: `1/sum_j(w_ij^2)`; also entropy `-sum w log w`.

### Required visualizations (PNG + source CSV, labeled bins and counts)

1. **Predicted class × confidence bin**: mean centered gate score.
2. Same: empirical expert correctness.
3. Same: mean final gate weight.
4. Same: cell counts; mark low-support cells (e.g. <30 *distinct images*).
5. **Confidence bin × leave-one-out agreement bin**, separate panel for each predicted class: mean centered score and empirical correctness, side by side; add count panels.
6. Correctness versus score deciles (calibration/ranking plot), separately by predicted class and optionally group, with uncertainty and raw counts.
7. Weight mass on **incorrect** experts, effective expert count and corrected/broken flips **by true subgroup** (post-hoc only).
8. Scatter/histogram `max_j w_j` versus expert consensus, highlighting cases where a minority wrong head overturns a correct majority.

Confidence bin edges must be trained/defined from **development meta_train** when the figure will be compared with new candidates. For historical test D19 visualization, fixed **predeclared** edges are acceptable, but do not look at test labels to pick edges. Always show whether plots use raw or calibrated probabilities.

**Quantitative checks** (per seed, predicted class, subgroup and overall): pairwise within-image expert ranking accuracy `P(a_correct > a_wrong)` over samples with both correct and wrong experts (ties count 0.5; average at image level), score-vs-correctness Spearman / AUROC as applicable, `mean(w | expert correct)` vs `mean(w | wrong)`, gate flip-direction table, score-vs-reliability misalignment, and ECE/NLL with correct denominators. Note that the gate score is *relative preference*, not necessarily a calibrated probability of correctness; nonmonotonic confidence alone is not proof of a bug.

**Output** `phase0_mechanism_report.md` with direct conclusions for H1/H2/H3, plots and reproducible source tables. **Do not change model on basis of the historical test diagnostic.**

---

## C. Phase 1 — Test interpretable signed piecewise scores (no new CNN/head training)

### C1. Core teacher idea and nonnegotiable inference distinction

User hypothesis: high-confidence **correct** expert → large positive score; high-confidence **wrong** expert → large negative score, with linear interpolation through a gray zone.

On **labeled meta_train only**, define an *oracle/teacher training target*:

`h(c) = interpolation of a finite knot table, clipped to [0,1]`

`t_ij = +h(c_ij) if expert_j predicts y_i else -h(c_ij)`.

**Never apply the correctness sign using real y at inference.** Treat direct use of `t_ij` on meta_select/outer held-out/test as **ORACLE, NON-DEPLOYABLE** and exclude it from ranking. An inference-time scorer must infer expected signed score from observable features only.

### C2. Audit of the proposed 0.75 confidence threshold

Development raw-confidence findings from prior direct OOF audit (revalidate before use): mean ~0.530, raw 95th percentile ~0.559–0.561, observed raw max ~0.604–0.609. **No raw member confidence reaches 0.75** on meta_train. Hence `c>=0.75` is an empty rule on the raw scale. A fixed `T=0.25` re-softmax of **original member logits** yields only ~2.2% high-confidence observations; a shared NLL-fitted `T~0.03` yields ~89–90% >0.75. The numeric cutoff is therefore **representation-dependent**, not universally meaningful.

* First validate that `softmax(logits/T)` matches the intended probability definition; do not naively apply temperature twice or convert already-softmaxed probabilities as if logits. Head calibration, if fitted, must use only meta_train and be frozen for select.
* Report coverage and empirical accuracy for each proposed interval by predicted class, count **distinct image IDs** and warn on sparse confident-wrong cells.
* 0.75 is kept as a **hypothesis/control**, not the primary raw threshold.

### C3. Preregister these six scoring maps (do not add more after seeing held-out WGA)

Each row lists `(confidence, h(confidence))` knots; `h` is linearly interpolated between adjacent knots, saturates outside; signed target `t = +/-h(c)`.

| ID | Probability representation | Knots `(c,h)` | Role |
|---|---|---|---|
| R1 | **raw** member probs | (0.500,0), (0.515,0.25), (0.530,0.50), (0.540,0.75), (0.550,1) | **Main raw scale**, realistic coverage |
| R2 | raw | (0.500,0), (0.520,0.50), (0.540,1) | Broad sensitivity control |
| R3 | original member logits, **T=0.25 fixed** | (0.500,0), (0.600,0.50), (0.750,1) | Direct user 0.60/0.75 hypothesis; sparse 0.75 tail |
| R4 | logits, shared temperature fitted on meta_train NLL only | (0.500,0), (0.750,0.50), (0.950,1) | Calibration sensitivity |
| R5 | raw, **predicted-class-specific quantiles from meta_train** | (0.500,0), (50th percentile,0.50), (75th,0.75), (90th,1) | Scale-robust; handle ties, invalid ordered knots with logged fallback |
| R6 | any, h(c)=1 | constant 1 | Correct/incorrect-only teacher; isolates confidence magnitude |

Also R0: uniform DPE without any learned scorer. R3 implements user's stated idea using linear interpolation: at 0.60 the **correct** score is +0.50, at 0.75 it is +1; **wrong** scores are the negatives. On the raw scale, no examples reach these points; no artificial rule should force the desired outcomes.

### C4. Infer signed scores without true labels

For each image/expert, calculate only inference-observable features: `(predicted_class, confidence_bin, loo_agreement_bin)`; bins constructed **from meta_train only**. Simple initial scheme: 3 confidence quantile bins *within predicted class*, 3 LOO agreement bins (<1/3, [1/3,2/3), >=2/3). Optional per-expert table via shrinkage; default pooled table due to only 959 images per seed and correlated heads. Head ID is interpretable only within a fixed head ensemble; it does not transfer naively across OOF folds.

For each cell, estimate expected signed teacher score:

`score_hat(cell) = (sum_train t_ij + kappa * score_parent) / (n_cell + kappa)`, with **kappa=50** initial, parent=predicted-class marginal (then global fallback). Keep within [-1,1], log raw counts, unique-image counts, support and backoff. `score_hat` rather than `t_ij` is used for **every** meta_select/outer held-out prediction. Optional single ablation: `score_hat = h(c)*(2*r_hat(cell)-1)` with separately smoothed empirical correctness `r_hat`; distinguish it from average signed h in bins.

Weights:

`q_ij = softmax_over_experts(beta * score_hat_ij)`

`w_ij = (1-alpha)/15 + alpha*q_ij`

`p_mix_ik = sum_j w_ij*p_ijk`

**Preregister** `alpha in {0.10,0.25}`, `beta in {0.5,1,2}`, default `kappa=50`, plus `alpha=0` matched uniform. Optionally test `kappa={10,200}` *only as labeled ablations after initial results*; no hidden adaptive hyperparameter search. Keep positive weights summing to 1; record mean/max weight, entropy and effective member count. Use each candidate's explicit probability representation consistently for both its uniform control and its weighted output; separately report original raw-uniform to expose representation effects.

### C5. Avoid excessive selection on 240 images

**Two-stage bounded grid** on the pre-existing seed-specific `meta_select` (already explored; exploratory only):

1. Screen R1–R6 at fixed `alpha=0.10, beta=1, kappa=50`; record all outcomes (do not discard losing data).
2. Using **a group-blind, prespecified** selection proxy (e.g. class-balanced worst-class log loss, with AA floor), take the top **two** non-oracle scoring maps, run the 6 alpha/beta combinations each, freeze 1–2 recipes for Phase 2. Do not sort by group WGA unless running a **separately labeled group-aware selection track**.
3. Include historical D5/D19 for diagnostic comparison only when source heads/representations match; do not treat differences in head training as pure weighting effect.

**Primary goal WGA**; however, for a **group-blind** method, labels `group` may be accessed **only after** candidate config is frozen to compute descriptive WGA. Selection proxy is not WGA itself. Report all groups and AA. Enforce AA loss **strictly <1.00 pp** versus matched uniform on model-selection data; if none meets the rule, report no eligible winner. No test-driven tuning.

### C6. Required Phase 1 outputs

Per candidate/seed: AA, balanced accuracy, worst-class accuracy, **WGA(post-selection diagnostic)**, per-group count/accuracy, NLL, ECE/Brier (where meaningful), corrected/broken by true class/group, flip directions, score ranking accuracy, mean wrong-head weight, 95% intervals computed resampling **distinct image IDs**, all cell counts and confidence coverage. Save `phase1_all.csv`, `phase1_ranked_by_predeclared_proxy.csv`, `phase1_locked_candidates.json`, `phase1_report.md`, calibration/score lookup tables and plot sources. Do not claim independent 15×more samples simply because each image has 15 head predictions.

---

## D. Phase 2 — Genuinely held-out head-and-weight verification

**Prerequisite:** Phase 0 diagnostics finished, Phase 1 candidate list locked, provenance valid. Follow the previously documented **Gate B protocol**, not OOF-slot transfer:

1. For each of 3 seeds × 3 class-stratified outer folds, use disjoint `A`=head fitting, `B`=score-table fitting (approximately 70/30 of outer training pool), `C`=outer held-out evaluation. Never let C labels, correctness or group labels influence A/B, preprocessing, calibration, score bins, hyperparams, early stopping or candidate promotion.
2. Train 15 real DPE heads from scratch **on A** according to frozen prior adapter schedule. Keep backbone frozen. For B and C, use predictions from the exact **same ordered heads**. Do not assume pre-existing OOF `head-03` can be reused as that head. Cache heads and predictions *once* per fold and reuse across all scoring maps.
3. Fit all scalers/temperatures/bin thresholds/reliability tables on B. Evaluate uniform and the locked candidate(s) with the **same heads**, **same representation**, **same C IDs**. Include diagnostic oracle on C only as non-deployable upper bound, clearly isolated.
4. Report AA, WGA, worst-class accuracy, BA, per-group counts, paired corrected/broken flips, and uncertainty across **unique image IDs**, separately per seed/fold; 3 seeds share backbone, do not claim 3 independently trained backbone replicates.
5. **Promotion gate (frozen before evaluation):** `mean(Delta_WGA)>0`, **Delta_WGA positive in at least 2/3 full-seed pooled outputs**, `Delta_AA > -1.00 pp` **for every seed**; provenance/leakage check passes. Include confidence interval and fragility disclosure even if passed. Do not round a failing `-1.0008` into a pass. If gate fails, **NO production integration** and do not change thresholds after seeing outcomes.
6. Group labels remain *evaluation only* in the group-blind track. If robustness cannot be selected without annotations, state the limitation instead of covertly selecting by outer-fold WGA. The previously inspected Waterbirds test set is **never** a new confirmation set.

If Phase 2 passes, integrate the winning score lookup as a clearly opt-in DPE inference mode (default uniform unchanged) with versioned metadata, score-table/calibrator serialization, all weight invariants and CLI flag; otherwise stop at report + preserved checkpoints. Integration is conditional, **not guaranteed**.

---

## E. Mandatory checkpoint / quota-safe recovery contract — implement BEFORE expensive runs

Keep all work under `experiments/score_audit_v4/` or equivalent, separate from earlier experiment output. Implement a single `python -m experiments.score_audit_v4.run --phase all --resume` command and `--status`, `--phase 0|1|2`. It must be idempotent: re-running completed tasks with matching input/source/config hashes **does not recompute them**. On incompatible hashes, fail transparently instead of silently mixing runs.

**Persist atomically** (temp + fsync/replace where practical; per-task success marker only after checksum validation):

- After input verification/unzip/manifest verification; after every expert-logit export and heatmap aggregate;
- After each confidence transformation/calibration, binning fit, score table, candidate evaluation, seed and fold;
- During head training **after every epoch**, including model/optimizer/RNG state, then head-final hash; after each cached prediction stage;
- After every plot/report and at each phase gate; never delete or overwrite historical artifacts.

Files that must always reflect latest committed work:

- `state.json`: task DAG/status, hashes, active phase/task, seeds/folds/configs done, failures/blockers;
- `HANDOFF_STATUS.md`: human-readable concise status, validated results and next step;
- `NEXT_AGENT_PROMPT.md`: executable 5–10 line prompt for a new Codex session to resume;
- `completed/<task_id>.json`: artifact path(s), checksums, input/code/config hashes, completed timestamp;
- `results/` incrementally saved per task, not only at final run;
- `PROTOCOL_DECISIONS.md`: any justified departures, no silent parameter changes.

Before real training, **simulate interruption** mid-task, resume in a new process, verify existing completed artifacts were reused and all checksums match. A quota/context cutoff must leave at most one unfinished task to repeat. No background promises; execute the full feasible workflow in the current Codex run, resuming from saved states if needed.

### Acceptance tests

- Softmax/weight normalization and mixture probability mass, class order, sign rules at knots, interpolation continuity, clipping beyond knots, boundary at 0.50, agreement self-exclusion;
- Inference scorer API MUST NOT accept `y`, `correct`, `group` (unit-test this); training-only teacher API explicitly needs `y`;
- Detect zero samples >=0.75 in raw meta_train and report it; not classify as failure;
- Reconstruct D19 using its **calibrated_probs**, not raw; validate saved gate prediction if present;
- `meta_train`/`meta_select` no ID overlap *within seed*, proper OOF/head provenance, A/B/C IDs disjoint, group-label access audit;
- Same-head comparison reproduces uniform exactly; no test-data loading in Phase 1/2 selection code;
- Restart from checkpoints after forced kill, CRC/hash checks on serialized arrays, write no fake measurements.

---

## F. Final deliverables (in this order)

1. `phase0_mechanism_report.md`, every requested heatmap PNG and their numeric source CSV, audit table with hypotheses H1/H2/H3 supported/refuted/undetermined;
2. `phase1_all.csv`, locked candidate manifest, class-dependent reliability tables, confidence threshold coverage, comparison of actual scores versus empirical correctness, per-seed score/WGA/AA results;
3. `phase2_nested_verification.csv`, full A/B/C manifests and checkpoint hashes, promotion gate JSON, WGA/AA deltas and uncertainty, corrected/broken by subgroup;
4. `FINAL_REPORT.md` (WGA first; AA constraint second; original DPE and D19/D5/LCMSW history; whether 0.75 was meaningful; whether correct-expert ranking was possible; whether failure arose in scoring versus softmax concentration; limitations and exactly one evidence-driven next step);
5. `state.json`, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`, all reproducible scripts/tests/configs and a compact rerun command; **integration only if Phase 2 passes**.

**No fabricated numbers. No selecting on historical test. No promoting a development-only result. No broad unplanned model sweep. If artifacts are missing, finish all feasible phases and report exact missing paths, do not silently substitute test data.**