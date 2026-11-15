# CODEX HANDOFF v6 — Calibrated Counterfactual Weighting (CCW) for DPE

**Research objective:** Improve **worst-group accuracy (WGA)** of the existing 15-head DPE by optimizing **input-dependent ensemble weights only**. Frozen backbone, existing prototypes/head parameters, member probability definitions, original IPS training algorithm, and default Uniform DPE inference must remain unchanged. **AA may decrease by strictly less than 1.00 percentage point per seed** against matched Uniform DPE.

**Research priority:** (1) provenance/no leakage, (2) reproducible fit and verification, (3) WGA, (4) AA constraint, (5) minimal compute. Implement **crash-safe checkpoint/resume BEFORE expensive work**. Run autonomously as far as inputs allow. Do not invent results, missing files, success, or future confirmations.

**The single proposed algorithm:** **Calibrated Counterfactual Weighting (CCW)** = residual G1 geometry prior + *partial* per-head distance normalization + a *small smoothed conditional relative-expert-advantage table*. This is **weight optimization**, not new head training, another MLP, pseudo-group DRO, class-wise Soft-min, or alpha-only routing. Run only specified controls/ablations, without post-hoc WGA-tuned grids.

---

## 0. Context and known results — don't conflate datasets

- 15 binary prototype experts; class order `0=Landbird`, `1=Waterbird`. True subgroup mapping `group = 2*y + background`, so group 2 is **Waterbird on land**.
- Historical DPE v5 nested development: G1 (`gamma=50`, `alpha=.25`) mean **ΔWGA +0.501253 pp**, mean **ΔAA +0.111204 pp**, positive WGA in 2/3 head seeds. Its paired ΔWGA interval **[−0.2147, +1.2531] pp crosses zero**. Three seeds share one augmented 400-epoch ERM* backbone and 1,199 image IDs, so this is **not independent multi-backbone validation**.
- 9 cached held-out-C evaluations = 3 seeds × 3 outer folds, 3,597 seed/image rows (1,199 unique image IDs). All old C results have been inspected. Exact G1 reproduction succeeded. Historical test is repeatedly inspected. **Neither old C nor historical test is eligible for selecting CCW or declaring fresh confirmation.**
- Retrospective mechanism audit: of 30 group-2 Uniform-wrong seed/image rows, fixed G1 geometry adjustment points the wrong direction in 25; 3 are corrected already; 2 point correct but cannot cross even at `alpha=1`. **Alpha-only tuning cannot fix the main direction problem.**
- Head slot 0 has raw minimum distance smallest in 2,763/3,597 observations, but only 6/30 group-2 Uniform-wrong observations. Head-0 dominance alone is **not proof** it causes most minority errors; **do not ban or penalize head 0 by index**. IPS-based causal explanations are unverified.
- Random-permutation controls: ~18% of matched-concentration random assignments reached G1's observed ΔWGA. Evidence for geometry's robustness advantage is still weak.
- Prior approaches D19 continuous gate, D5 discrete correctness reliability, LCMSW and SAGW hard-min failed to reliably improve WGA. CCW must distinguish itself from them: **target marginal predictive contribution versus Uniform, not just expert correctness**.

### Canonical inputs already known from the project

`DPE_v5_Analysis_Inputs.zip` (or its extracted `dpe_v6_analysis_inputs/` folder) supplies:

```
CONTENTS.md
manifest.json
verify_package.py
caches/seed{0,1,2}/fold{0,1,2}/
  C_predictions.npz
  ABC_manifest.json
  head_manifest.json
  export_provenance.json
source/geometry.py
source/v5_aggregation.py
source/verified_distance_v1.json
provenance/development_index.csv
provenance/source_run_manifest.json
...
```

Per C cache: `probs [n,15,2]`, `distances [n,15,2]`, `logits [n,15,2]`, `sample_id [n]`, `y [n]`, `group [n]`, `seed`, `fold`, `head_id`, `head_sha256`, `head_abs_scale`. `logits = -abs(scale_j)*distances`, so **raw distances differ from logits**. Head slots from different folds represent **different fitted heads**. Group/labels are **evaluation-only for C**.

**Mandatory missing-or-to-be-located assets:** the **matching B-partition member predictions** for every one of those nine exact ordered head ensembles (`probs`, `distances`, `sample_id`, `y` plus manifests), and/or the actual trained head checkpoints + frozen backbone/feature cache + correct inference source sufficient to compute B predictions *without retraining*. Existing v5 per-head G2 B normalizers are useful when available, but must be validated as fitted on the **same B and same ordered head hashes**. Never reuse unrelated OOF heads or their normalizers. The old `development_oof_bundle.zip` belongs to a **different cross-fitted head setup** and is not a substitute for these B caches. Never fit CCW on C.

**Fresh confirmation** requires a cohort/images not previously exposed to researcher selection and the same prescribed head inference, with subgroup labels strictly held until evaluation. If it does not exist, finish implementation and permitted development diagnostics, record `CONFIRMATION BLOCKED — NO FRESH DATA`, and do not claim CCW beats G1 or integrate into production.

---

## 1. Gate 0 — Inventory, integrity, and exact G1 reproduction

Before model fitting:

1. Discover existing repository artifacts recursively; create an inventory with exact paths, schema, sizes and SHA-256 hashes. Do not download or fabricate files.
2. Run package verifier (`python verify_package.py --reproduce-g1` within extracted v5 analysis package). Verify all nine C caches and their manifests: aligned labels/IDs/head hashes, 3× full 1,199-image outer coverage, A/B/C disjoint within fold, 15 heads and exact class order. Record verification version and output.
3. Reproduce matched Uniform and frozen G1 predictions on C without reading labels/groups in aggregation code. Confirm per-seed metrics against v5 report to numerical precision; the reproduction is **diagnostic only**. `U0 = mean_j(probs_j)`, `G1: r=softmax_j(-50*min_k distance_jk), w=.75/15+.25*r`.
4. Locate or generate exact **B predictions** for the nine fold ensembles. Generation is inference from the original saved **same** heads, not training. Check B sample IDs match original ABC manifests, are disjoint from A/C, and logits/probabilities/distance scales align. Save B caches and inference metadata atomically. **If B cannot be obtained, stop before any CCW fit**. Write `INPUT_REQUEST.md` with exact missing assets, folder identifiers and instructions; keep previously finished checks and handoff files.
5. Treat C and historical test labels as forbidden in every training, binning, calibration, model-selection, config-search, and fit module. Put C-evaluation logic in separate, read-only evaluators loaded only after configuration locking.

**Gate-0 outputs:** `INPUT_INVENTORY.md`, `PROVENANCE_AUDIT.json`, `B_INPUT_STATUS.json`, `g1_reproduction.csv`, `INPUT_REQUEST.md` if blocked.

---

## 2. CCW mathematics — do not change formulas silently

For image `x`, expert `j∈{0,...,14}`, class `k∈{0,1}`:

- Fixed original head probability `p_j(k|x)`; `u_k(x)=(1/15) Σ_j p_j(k|x)`.
- Distance `d_j(x)=min_k D(f(x),p_j^{(k)})` using the **actual saved normalized Euclidean `distances`**, not scaled logits.
- On **B fitting data for that exact fold/head ensemble**, estimate `mu_j=mean_B(d_j)`, `sigma_j=std_B(d_j)` (document ddof). Compute global references `mu_ref=median_j(mu_j)` and `sigma_ref=median_j(sigma_j)` (fixed across images for that ensemble). Set:

```text
z_j(x)       = (d_j(x) - mu_j) / max(sigma_j, 1e-6)
d_cal_j(x)   = mu_ref + sigma_ref*z_j(x)
d_new_j(x)   = (1-rho)*d_j(x) + rho*d_cal_j(x)
```

- Fix `rho=0.50` for the proposed primary method; `rho=0` and `rho=1` are **controls** (not a post-hoc sweep).

### 2.1 Training-only signed *relative advantage* target

For **labeled B training image** `i` and its expert `j`, define:

```text
t_ij = p_j(y_i | x_i) - u_(y_i)(x_i)
```

`t_ij∈[-1,1]`. Unlike binary correctness, this target measures whether expert `j` increases or decreases **true-class probability relative to the same Uniform ensemble**. It is also the per-expert component of the **first derivative of the mixture's CE** at uniform interpolation. It is **not a direct surrogate or guarantee for WGA**.

At inference **no y, correctness, group or true subgroup is available**. Predict expected advantage from a tiny transparent cell table conditioned only on:

```text
phi_ij = (
   predicted_class_j,                         # argmax_k p_j(k)
   z_distance_bin_j,                          # 3 within-head-ensemble B-train-derived quantile bins
   disagree_with_uniform_j                    # argmax p_j != argmax u
)
```

Use `2 × 3 × 2 = 12` pooled cells. Quantiles/bins are fit **only on the training portion of B** for each internal CV fold, not on its validation portion. If ties yield non-strict quantile edges, use deterministic documented backoff. `head_id` must **not** be pooled as a persistent global expert identity across folds. Initial default is **head-pooled** table, not 15 head-specific sparse tables.

Fit the expected signed advantage with class-balanced, boundary-emphasized observation weights:

```text
m_i     = abs(u_i(class1) - 0.5)
q20     = 20th percentile of m on B-training (NOT B-validation/C)
omega_i = [n_Btrain / (2*class_count[y_i])] * (1 + indicator(m_i <= q20))
omega_i = omega_i / mean_Btrain(omega)
```

To avoid treating `15` correlated expert rows as `15` independent images, allocate each image's weight across its heads: `omega_ij=omega_i/15`. For each cell `c`, compute weighted sum `T_c = Σ_(i,j in c) omega_ij*t_ij` and effective weighted support `n_c=Σ_(i,j in c) omega_ij`. Define hierarchical shrinkage with frozen `kappa=25` **in effective image-weight units**:

```text
parent(c) = smoothed predicted-class marginal (then global mean; then 0)
A_hat(c) = (T_c + kappa*parent(c)) / (n_c+kappa)
```

Document the parent computation without self-referential shrinkage. Clamp `A_hat∈[-1,1]`. Record `n_c`, **distinct-image support**, prediction-class coverage, per-cell target mean, and parent. Implement deterministic out-of-range/empty-cell fallbacks. `sigma_A = max(weighted_std_Btrain(t_ij), 0.02)` is training-only (fixed 0.02 numerical floor). Define `g_j(x)=clip(A_hat(phi_j)/sigma_A,-2,+2)` to cap the effect of scarce-cell estimates. This clipped standardized advantage is for the score, not a calibrated probability.

### 2.2 Final weights — fixed alpha and geometry scale

```text
a_j(x) = -50*d_new_j(x) + beta*g_j(x)
r_j(x) = softmax_over_15_experts(a_j(x))
w_j(x) = (1-0.25)/15 + 0.25*r_j(x)
p_CCW(k|x) = sum_j w_j(x)*p_j(k|x)
```

**Primary candidate:** `rho=0.50`, `beta=1.0`, `gamma=50`, `alpha=0.25`, `kappa=25`, z-bins=3, no trainable MLP. The values are **preregistered research choices**, not tuned-for-WGA conclusions. `rho=0,beta=0` must reproduce G1 exactly. Do not use actual group to set weights. `beta` and `rho` are fixed for the primary candidate — if it fails, report failure; don't browse old C WGA for a rescue grid.

**Numerical tests:** stable logsumexp softmax, all weights finite/nonnegative/sum=1, mixture probabilities sum=1, class order, handling ties/zero variance, monotonic effect of signed advantage on relative score, correct training-only fit boundary, no label-bearing arguments in inference API. Instrument score dynamic ranges so `beta*g` cannot silently dominate `-50*d_new`; if it does, **report rather than retune on C**.

---

## 3. Preregistered comparisons (one proposal + diagnostic ablations)

All variants use **identical ordered 15 heads and probabilities**, with default residual `alpha=0.25` except Uniform. Each fold's normalizers come from the matching B. Keep all failing results.

| ID | Definition | Scientific question |
|---|---|---|
| `U0` | Original uniform averaging | Fixed baseline / AA floor |
| `G1_frozen` | Original raw-distance `gamma=50,alpha=.25` | Current opt-in winner |
| `G2_z` | Full z-distance, `gamma=1,alpha=.25`, **control only** | Is full normalizing useful? |
| `CCW_A` | Partial calibrated distance `rho=.5,beta=0,gamma=50,alpha=.25` | Does partial calibration alone help? |
| `CCW_FULL` | Partial distance + smoothed advantage `rho=.5,beta=1,gamma=50,alpha=.25` | Main mechanism hypothesis |

Historical D5, D19, LCMSW and class-wise S2 are background only; avoid comparing them as pure weight ablations if their heads differ. Do not explore arbitrary beta/rho/alpha grids, automatically add learned gates, or change IPS/head training as an escape route.

### Counterfactual target verification

On B cross-validation, compute (no subgroup labels):
- image-level correlation/AUROC or pairwise ranking of `A_hat(phi)` versus **true `t_ij`** on validation images, separately overall, by predicted class, low-Uniform-margin stratum, and where experts disagree with Uniform;
- weighted signed-advantage gain `Σ_j r_j*t_ij` and whether it is positive when Uniform misclassifies an image, with caution about few such samples;
- contributions of distance versus advantage logits, top-head frequencies, per-image weights, entropy/effective expert count and corrected/broken versus Uniform and G1 **using B-validation class labels only**;
- worst-class NLL, worst-class accuracy, AA, balanced accuracy and loss by true class; 95% image-cluster bootstrap intervals (do not bootstrap expert rows as iid).

**Do NOT interpret AUROC of raw t versus rank as guaranteed group robustness.** Log class-specific effects: a class-balanced proxy may still miss subgroup shortcuts.

---

## 4. Fit, select and lock exclusively within B; old C is already explored

For each seed×outer-fold, the historical v5 split is `A`=head fitting, `B`=weight/calibration fitting, `C`=held-out v5 evaluation. **B is approximately 240 images; heads have already been trained on A**. Every B row must come from the exact same head ensemble used on C. Neither B normalization nor B scoring may use C labels/probs in its fit.

1. Use **4-fold stratified-by-class cross-fitting within B**, deterministic seeds fixed in config. In each inner fold, fit `mu/sigma`, bin edges, class counts, q20 and the table **on B-inner-train only**, then score B-inner-validation. Cache each inner fit and its predictions. Do not use B-inner-validation labels for fitting its own lookup.
2. Aggregate out-of-fold B predictions **by image** across four inner folds for each of five methods. Report objective and AA for all, not only CCW winner. Use the same B images for all comparisons and evaluate all seeds/folds without presenting them as independent backbone replicates.
3. The **prespecified group-blind selector** ranks by mean *worst-class NLL* across B-OOF evaluations; exclude any proposed candidate with `ΔAA <= -1.00 pp` vs matched U0 in any seed (pool B-OOF across the outer folds per seed). Tie-break by worst-class accuracy, then AA, then lowest score/weight concentration. **Never select using WGA/group IDs**. Primary CCW is a hypothesis: if CCW_FULL loses to G1 under proxy, report NO-GO rather than changing hyperparameters. Lock at most one *new* method. G1 remains a fixed control, not a selectable 'novel gain'.
4. Refit the locked CCW variant and its control normalizers/tables on **all B** in each outer fold. Save serializable, versioned metadata: head SHA/order; sample IDs; mu/sigma; reference stats; quantiles; score-table values/support; sigma_A; selected formula/config; code/data hashes. **No C labels**.
5. The existing v5 C predictions may then be replayed **once as transparently labeled retrospective/exploratory diagnostic only**, only after candidate is locked from B. They were already examined and influenced method design: metrics on them **cannot be called independent verification**, even if CCW fitting didn't touch their labels. Do not promote on these metrics; report C results with this warning.
6. Independent confirmation requires an **untouched new cohort** (and preferably independently trained backbone(s)); freeze all code/config/selector/heads where appropriate *before* accessing its labels/groups. If unavailable: deliver code, B crossfit, exploratory C report and `CONFIRMATION BLOCKED` without production promotion.

**Critical protocol caveat:** B-OOF cross-fitting validates a **lookup fitted on approximately 3/4 of B**; actual deployed table uses full B. Small B and sparse bins can make validation noisy. No false claim that this is a high-powered statistical test. Do not infer success on a new test distribution without fresh evaluation.

---

## 5. Evaluation and scientific promotion

### Required metrics

For each method, seed and fold (and seed-pooled): `AA`, `WGA`, `ΔAA_pp` vs matched U0, `ΔWGA_pp` vs U0 and **vs G1**, balanced accuracy, worst-class NLL, per-group support and accuracy (group labels only at final evaluation), corrected/broken by group, uniform-vs-weighted direction `q_y-u_y` for retrospective audit only, weight entropy/effective # experts, head-00 top-weight share (descriptive, no index penalty), and paired bootstrap intervals over distinct images. Report group counts, not just rates. Same image IDs across head seeds: use clustered uncertainty and note shared backbone; do not interpret seed replicates as independent datasets.

### Prespecified acceptance

**Engineering checks (necessary, not sufficient):** verified matched-head reproducibility, inference-only inputs (`probs`,`distances`,versioned B fitted params), unit tests pass, no group leakage, unchanged legacy uniform and raw/member APIs.

**Fresh confirmatory WGA gate (no exceptions):** `mean ΔWGA(CCW − U0)>0`, CCW improves WGA over U0 in **≥2/3 head seeds**, `ΔAA(CCW−U0)>−1.00 pp` **in every seed**, and record CCW versus frozen G1. For a claim of **optimizing over G1**, require `mean ΔWGA(CCW−G1)>0` and improvement versus G1 in ≥2/3 seeds, in addition to the U0 checks. Give CIs; if CI includes zero, describe the evidence as tentative even if a point-estimate engineering gate passes. A passing gate does not itself make previously inspected development data fresh.

If fresh data do not exist or gate fails: **NO production integration of CCW**, retain opt-in G1 v5, do not revise configurations after seeing results. Preserve all negative-result diagnostics.

### Matched-concentration random controls

Optionally (and only as *diagnostic*) preserve exact per-image CCW weight multiset and randomly permute the association to expert IDs with fixed predeclared RNG seeds; use independent-per-image and fixed-per-actual-fold permutations. Do not choose favorable permutations or treat a retrospective tail fraction as fresh confirmatory significance. Preserve all draws/histograms and identical-row matching. This is a test of mapping quality, not a replacement for validation.

---

## 6. CRITICAL: quota-safe checkpoint, atomic persistence and resume — implement first

Use new isolated workspace `experiments/ccw_v6/`; **never overwrite v1–v5 outputs or production defaults**. Implement a single resumable runner and status view:

```bash
python -m experiments.ccw_v6.run --phase all --resume
python -m experiments.ccw_v6.run --phase 0 --resume
python -m experiments.ccw_v6.run --phase 1 --resume
python -m experiments.ccw_v6.run --phase 2 --resume
python -m experiments.ccw_v6.run --status
python -m experiments.ccw_v6.run --verify
```

Use `--phase 0` for input/G1 integrity; `1` for B fitting/crossfit/selection/locking; `2` for frozen retrospective diagnostics, fresh confirmation **only if new cohort is provided**. Command syntax can differ if repository conventions require it, but document exact commands and preserve this functionality.

### Persistence contract

**Before any expensive task:** write a task DAG with prerequisites, config version, code hash, data hashes, RNG seeds, active task, and explicit atomic commit markers. Resume must be **idempotent**: skip completed tasks whose input/code/config hashes match and whose artifacts pass checksum validation; fail on incompatible provenance. No silent reuse when model/heads/data versions change.

Save immediately after **each** of the following:

- extraction/manifest verification and provenance audit;
- every B fold prediction/cache export; every `(seed, outer_fold, inner_fold, method)` fitted normalizer/table and validation predictions;
- every scoring method/seed/fold evaluation, before the next task;
- every B-fit full table, selection-lock JSON, C diagnostic replay, plot and report;
- every independent-cohort fold/seed evaluation and promotion decision;
- if **any** head must be retrained under a separately justified regeneration path (not default), checkpoint **after every epoch** with head/model, optimizer, scheduler, RNG, batch cursor if mid-epoch resume is supported, training data hashes and run config. Never claim an epoch checkpoint allows mid-epoch exact resume if it only resumes from epoch end.

Use atomic temp-write + fsync where practical + `os.replace`, bounded Windows/OneDrive rename-lock retries, SHA256 verification and success receipts **only after** artifacts committed. Resume after forced interruption (kill process mid-inner-fold and after one completed fold); verify equivalence to uninterrupted run and no duplicated work. Safe to redo **at most one uncommitted atomic task**; no partially written CSV misleadingly marked complete. Avoid too-frequent large-array copies: cache expensive head probabilities once per fold and reuse for every method.

These files are **required and updated continuously**, not only at the end:

```text
experiments/ccw_v6/
  state.json                    # DAG, timestamps, active/complete/blocked tasks, hashes
  HANDOFF_STATUS.md            # current truthful results, blockers and exact next step
  NEXT_AGENT_PROMPT.md         # 5–15 lines new-agent continuation prompt
  INPUT_REQUEST.md             # exact missing B/fresh inputs; only if blocked
  PROTOCOL_DECISIONS.md        # preregistered settings, deviations (none silent)
  LOCKED_CANDIDATE.json        # immutable method/config/provenance when selection done
  completed/<task_id>.json     # source/input/config hashes, outputs + checksum
  checkpoints/                # granular, atomic; per inner fit/fold/epoch if needed
  inputs/manifest.json
  results/                     # incremental CSV/NPZ; never final-only
  figures/                     # every plot with source CSV
  reports/                     # phase0, phase1, phase2 and final
```

At quota/context interruption, leave state and handoff files complete **before** any resource-intensive task. A new Codex session must be able to read `NEXT_AGENT_PROMPT.md`, run `--resume` and continue exactly. Do not promise asynchronous execution or silently start fresh from scratch. **If inputs are missing, finish all feasible verification/code/tests now and stop with an exact input request.**

### Acceptance/unit tests (automate)

1. Input hashes, ID joins, 9 C cache shapes and distinct A/B/C manifests; every B fold's correct head SHA/order; no original test array in training imports.
2. `rho=0,beta=0` reproduces G1 weights and probabilities (numerical equality with documented tolerance); `alpha=0` reproduces exact Uniform; all simplex invariants.
3. Inference function accepts **no** true labels, correctness, group or C-specific flags. Separate teacher-table fitter requires B labels; test that scoring the same image without labels is identical.
4. No class leakage into B inner-fold normalizers, quantiles, q20, table support, standardization or selection; fit/transform train/validation separation; group-label access audit.
5. Tests for equal/zero-distance heads, ties in bins, all-experts-agree, no experts-agree, zero support, near-0/near-1 probabilities, class swaps, overflow/underflow, fixed schema and head-order mismatch.
6. Forced-kill/restart after a committed inner-fold; verify checksums, no rescore, no changed metrics, no duplicated CSV lines.
7. Strict AA threshold `−1.0000 pp` **fails** (not rounded), consistent WGA group mapping, correct pooled counts and image-level bootstrap.
8. If fresh cohort absent, test that promotion **cannot** run and production default remains Uniform; old verified G1 opt-in remains intact.

---

## 7. Final deliverables and minimum report structure

1. `INPUT_INVENTORY.md`, validated data/head provenance and `B_INPUT_STATUS.json`, missing-asset request if any.
2. Exact U0/G1 reproduction receipts and all nine C source checksums.
3. `B_crossfit_per_image.csv` (B only), `B_crossfit_all_methods.csv`, `B_cell_tables.csv` with support, `B_selection_proxy.csv`, `LOCKED_CANDIDATE.json`, and B-fit serialized normalizers/reliability tables for each fold.
4. `CCW_MECHANISM_AUDIT.md`: advantage signal vs actual t on **B OOF**, low-margin cases, loss/proxy tradeoffs, head top-weight rates, and diagnosed failures; clearly separate raw distance, full calibrated and partial calibrated ablations.
5. `C_EXPLORATORY_ONLY.csv`, corrected/broken-per-group, visualization sources if retrospective C run is possible. Banner: **REUSED DEVELOPMENT; NOT CONFIRMATORY; NO C-BASED TUNING**.
6. `FRESH_CONFIRMATION.csv`, group-level WGA/AA and paired confidence intervals **only if truly fresh inputs are supplied**; otherwise `CONFIRMATION_BLOCKED.md`.
7. `FINAL_REPORT.md`: first-page decision (`CCW not yet confirmed` / `passed on independent data` / `failed`), WGA vs G1 and Uniform, AA floor per seed, estimated advantage reliability on B, reason for improvements/failures, limitations and **exactly one justified next step**.
8. Complete runnable code/tests, `state.json`, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`, checkpoint manifests, hashes, reproduction command. Integrate CCW only as an **opt-in mode** if the fresh confirmation gate passes, default uniform unchanged.

**Stop conditions:** B input not verifiable ⇒ no CCW fit. Advantage ranking/proxy not meaningfully better on B ⇒ report failure; do not invent a new model. No fresh cohort ⇒ no confirmatory improvement claim or CCW promotion. Any leak/head mismatch/source hash inconsistency ⇒ fail loudly with saved artifacts. Remember: primary objective **WGA**, AA drop **strictly <1 pp per seed**, **checkpoint/resume is non-negotiable**.
