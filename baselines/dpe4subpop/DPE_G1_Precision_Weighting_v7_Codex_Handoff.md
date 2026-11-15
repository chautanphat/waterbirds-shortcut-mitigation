# CODEX HANDOFF — DPE v7: Fix G1 Weighting's Demonstrated Blind Spots

**Working name:** G1-R / G1+ — *Geometry-Preserved, Head-Aware Continuous Advantage Residual*  
**Version:** `v7.0-research-spec`  
**Status:** implementation request; **NOT** an observed WGA improvement.  
**Execution philosophy:** optimize the specific G1/CCW blind spots identified in v5–v6, while retaining G1 as a bit-exact baseline and the only existing opt-in production candidate. Implement, fit, evaluate, and checkpoint the proposed corrections; do **not** replace the task with a G1-only confirmation exercise.  
**Science boundary:** 9 known C caches are *EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION*. No new method may be tuned against their labels/WGA or promoted based on their results.

> **Mission for Codex:** Use existing DPE v5/v6 caches, source and checkpoints to investigate **why geometry-based G1 weighting fails to identify genuinely useful experts on Uniform-wrong inputs**, and implement a small, auditable **head-aware, continuous, correctly scaled score residual** to improve that selection. Test whether each correction changes the **direction of the expert mixture**, not merely the numerical weights; quantify WGA/AA and preserve every checkpoint. Keep the frozen backbone, all 15 prototype heads, IPS training, member logits/probabilities and G1 geometry exactly unchanged.

---

## 0. Objectives, non-negotiable constraints, and existing evidence

### 0.1 Actual optimization goals

1. **Primary scientific goal:** show that a *learnable input-dependent weighting rule* can select useful expert predictions better than G1 on **held-out B**. The mechanistic endpoint is a higher *signed true-class gain* on the images where **Uniform is wrong**, accompanied by stable all-image behavior; this is a necessary diagnostic, **not** a proof of WGA improvement.
2. **Primary downstream performance endpoint:** **Worst-Group Accuracy (WGA)**, preferably greater than **frozen G1** on a genuinely independent cohort; compare with matched Uniform, not a historical run from a different protocol.
3. **Hard AA constraint:** for **every** seed, `AA_new - AA_Uniform > -1.0000 percentage point` (strict inequality; equality fails). Also report AA difference against G1, even if the Uniform floor passes.
4. **No changes to representation:** no backbone fine-tuning, no retraining/replacing 135 prototype heads, no change to prototype distances/member probabilities, no revised IPS schedules, no synthetic labels, no group labels during fitting.
5. **Small model:** ridge/partial pooling; no MLP gating, mixture-of-experts retraining, hidden high-dimensional feature sweep, hard veto, group oracle or majority-disagreement reward by fiat.
6. **Crash-safe, idempotent checkpoint/resume is mandatory and built before expensive work**; never rerun completed artifacts because of quotas/interruptions.

### 0.2 Immutable matched reference (v5 nested, reused C)

| Method | Mean AA | Mean WGA | Mean ΔWGA vs Uniform |
|:--|--:|--:|--:|
| Uniform / U0 | 95.746455% | 92.481203% | 0.000000 pp |
| **G1_frozen** (`gamma=50`, `alpha=.25`) | **95.857659%** | **92.982456%** | **+0.501253 pp** |
| G2_z (v6 exploratory) | 95.829858% | 92.982456% | +0.501253 pp |
| CCW_A (v6 exploratory) | 95.802057% | 92.982456% | +0.501253 pp |
| CCW_FULL (v6 exploratory) | 95.829858% | 92.982456% | +0.501253 pp |

- Three seeds (`0,1,2`), three outer folds per seed; each seed's C concatenation has **1,199 unique images**, and the **same 1,199 IDs are reused across seeds**. A, B, C are disjoint **within an ensemble**. Every C ensemble has its own ordered 15 head checkpoints. Total 3,597 seed/image prediction rows, **not** 3,597 independent images.
- Subgroups: `0=landbird/land`, `1=landbird/water`, `2=waterbird/land`, `3=waterbird/water`; fixed per-seed subgroup sizes `[467,466,133,133]`. Always compute each seed's worst group first, then average the three per-seed WGA values.
- G1 WGA increment comes from just a few decision flips: in group 2 each seed has one correction vs Uniform; seed 0 also has one broken image so net zero, seeds 1/2 net +1/133 each. v5 CI includes zero. A permutation of G1's weights matched/exceeded G1 gain in approximately 18% of retrospective controls. **No generalization claim**.
- **Head 0 raw-distance bias:** top-1 geometry head on 2,763/3,597 image/seed rows (~76.81%), yet **only 6/30** group-2 Uniform-wrong rows. **Never hard-code a penalty for head 0 or call it a proven spurious head.** Head indices are fold-local, and any head-specific offsets must be fitted per actual head/hash.
- **CCW v6 failure:** partial distance calibration (`rho=.5`) made B worst-class NLL worse than G1 (`0.625208` vs `0.620634`); CCW_FULL only recovered ~`0.000025` of NLL, with average residual/advantage score ranges about `0.17–0.27` vs geometry `3.4–3.9` on Uniform-wrong B images. The 12-cell pooled tables had sparse cells. On C, CCW_FULL changed expert weights but **did not change any group-2 prediction versus G1**.
- A *label-aware oracle* can correct up to **12/30 Uniform-wrong group-2 seed/image rows** using the same heads and fixed `alpha=.25`, while G1 corrects 3/30. This proves **representational feasibility for those rows**, not learnability, not 12/30 future WGA, and not a ceiling accounting for other groups becoming worst. **Oracle labels can be used only in an explicitly retrospective diagnostic, never in fitting/selection/inference.**

**Crucial distinction:** Four methods with the same WGA do **not** mathematically prove an absolute weighting ceiling. Conversely, an oracle gap does **not** prove its identity can be inferred from 240 B images. v7 must test that *information gap* honestly.

### 0.3 Exactly which blind spots v7 must optimize

| Blind spot | v7 correction | Required falsification / audit |
|:--|:--|:--|
| G1 minimum distance is not a reliable correctness router on Uniform-wrong images | **Preserve G1 geometry**, add conditional *relative-advantage residual* | True-class signed gain on B-OOF Uniform-wrong images vs G1; expert-ranking diagnostics |
| Head-specific specialists can be washed out by a pooled 12-cell table | **Head-specific, predicted-class partial pooling** fit on B for exact heads | Pooled vs head-aware ablation; support and shrinkage; no memorized slot-0 bonus |
| CCW advantage was numerically swamped by geometry | **Train-only scale matching** based on *predicted*, between-head advantage spread, bounded | RMS/range and total variation of weights; off-switch for near-zero/noisy signal |
| Binning lost continuous information and direction | Use **continuous signed** `delta_j=p_j(1)-u(1)` and a small, explicit linear interaction model | Conditional OOF accuracy/signed gain; compare to coarse/discrete mechanism without grid search |
| Concern that corrections break clear Uniform predictions | **Optional margin gate**, separately ablated, not a magical AA guarantee | Margin-stratified prediction flips / AA, especially high-confidence images |
| Apparent G1 gain could be chance/cohort-specific | Matched random-weight and residual-permutation controls; fresh confirmation | Paired subgroup-aware CI, seeded controls, no claims from old C alone |

**Explicitly not optimizing:** `rho`/G2 normalizers, softmax temperature `gamma`, residual mixing `alpha`, individual prototypes, IPS loss, 15-head count, class-specific oracle weights or C group labels.

---

## 1. Locate existing assets and reproduce G1 exactly

### 1.1 Search actual repo; do not guess paths

Existing known materials (names are examples; discover exact locations):

- `DPE_v5_Analysis_Inputs.zip` and `source/geometry.py`, `source/v5_aggregation.py`, `verify_package.py`.
- `DPE_CCW_v6_Exploratory_C.zip`, `EXPLORATORY_C_PREDICTIONS.npz`, `EXPLORATORY_C_REPORT.md`.
- `experiments/ccw_v6/model.py`, `experiments/ccw_v6/run.py`, `experiments/ccw_v6/exploration/run.py`, `experiments/ccw_v6/store.py`, and their unit tests (present in the v6 exploratory package's `source_snapshot/` if local repo is incomplete).
- Previously fitted **matching** B prediction caches and CCW full-B fit/provenance files for all `(seed, outer_fold)` pairs. B prediction inputs were already verified in v6. **Reuse them**; do not demand a new head train or confuse the unrelated older `development_oof_bundle.zip` with matching B predictions.
- B labels may be used for fitting, **B groups must never be used**. C labels/groups are evaluation-only. No old official test set for model selection or retrospective tuning.

Before proceeding, persist `INPUT_INVENTORY.json` (absolute/relative paths, sizes, SHA-256, source commit, schema, access status), `DATA_PROVENANCE.md`, and `BASELINE_REPRODUCTION.csv`.

### 1.2 Exact baseline equations

For `j=0..14`, `k∈{0,1}` and member probabilities `p_j(k|x)`:

\[
 u_k(x)=\frac1{15}\sum_{j=0}^{14}p_j(k\mid x),\qquad
 d_j(x)=\min_{k\in\{0,1\}}D_j(x,k).
\]

**Use saved normalized Euclidean `distances`, not `logits` or learned abs scales**, and preserve the original v5 numerical order/dtype when reproducing bitwise G1:

\[
 g_j(x)=-50d_j(x),\quad
 r_j^G(x)=\operatorname{softmax}_j(g_j(x)),\quad
 w_j^G(x)=\frac{0.75}{15}+0.25r_j^G(x).
\]

\[
 p_G(k\mid x)=\sum_{j=0}^{14}w_j^G(x)p_j(k\mid x).
\]

**Validation requirements:** 9/9 exact ordered head hashes; aligned B and C split sample IDs; 15 heads; finite probability simplex; class `0=Landbird`, `1=Waterbird`; all seeds' Uniform/G1 probabilities and AA/WGA identical to previously saved baselines within the original precision/tolerance. If any mismatch: STOP v7 fits, write `BLOCKERS.md`, save state/checksums, do not fabricate comparison metrics.

---

## 2. The v7 proposed algorithm — G1 geometry + conditional reliability *residual*

### 2.1 Preserve G1 and modify only its **expert-score direction**

\[
 a_j^{(v7)}(x)=g_j(x)+\lambda\,\widetilde A_j(x),\qquad
 r_j^{(v7)}(x)=\operatorname{softmax}_j(a_j^{(v7)}(x)).
\]

\[
 \boxed{w_j^{(v7)}(x)=0.75/15+0.25r_j^{(v7)}(x)}\,,\qquad
 p_{v7}(k\mid x)=\sum_jw_j^{(v7)}(x)p_j(k\mid x).
\]

**Invariants:** `lambda=0` must return **exact** frozen G1; no positive reward for agreement/disagreement by default; neither `group` nor `y` exists in inference API; `gamma=50`, `alpha=.25` stay frozen. Score offsets shared equally across heads are irrelevant to softmax; center learned residual per image for interpretability.

**Proposed frozen v7 primary:** `G1_RH`, `lambda=0.50`, continuous pooled ridge + strongly shrunk head/class residuals; raw geometry unchanged. `lambda=0.50` is a **predeclared research choice**, not a claimed optimum. An optional `lambda=1.0` stress-test is allowed **on B-only** if declared before any new C result; never select lambda from old C. Simplicity preferred: ship the locked primary + ablations instead of a broad hyperparameter sweep.

### 2.2 Why the target is relative contribution, not expert correctness

For **labeled B-training image** `i` (never for inference), define

\[
 t_{ij}=p_j(y_i\mid x_i)-u_{y_i}(x_i)
          =(2y_i-1)\underbrace{\left[p_j(1\mid x_i)-u_1(x_i)\right]}_{\delta_{ij}}.
\]

Here `delta_ij` is **continuous signed probability displacement**, available without labels at inference. `t_ij` uses B training labels only. For each image, `sum_j t_ij = 0` (up to numeric tolerance). Positive signed gain indicates the combination increases probability of the true class vs Uniform; this **does not ensure an accuracy flip**.

We want a small predictor

\[
 \widehat A_j(x)\approx\mathbb E[t_{ij}\mid\text{label-free inputs},j],
\]

**not** an unrestricted selector that requires knowing which group is hard. If the predictor does not generalize on B-OOF, a good oracle bound does not rescue it.

### 2.3 Exact continuous *pooled* ridge feature basis

Use existing `u_1`, member probabilities, and optional training-only standardized *relative-distance* features. **The normalization of distances is a feature only; it must not replace G1's raw score.**

For the exact seed/fold ensemble on a B-training partition:

```text
delta_ij = p_j(class1|x_i) - u_i(class1)
mu_j     = mean_Btrain[ min_k distances_ijk ]
sigma_j  = std_Btrain[ min_k distances_ijk ], ddof=0; floor 1e-6
z_ij     = clip((d_ij - mu_j) / sigma_j, -3, 3)           # feature only
c_i      = 2*u_i(class1) - 1                               # signed Uniform margin
h_i      = abs(c_i)                                         # unsigned confidence
a_ij     = 1[argmax(p_j) != argmax(u_i)]                   # disagreement marker
v_ij     = 1[argmax(p_j) == 1]
q_ij     = delta_ij / max(RMS_Btrain(delta), 1e-4)

base_ij = [1, z_ij, c_i, h_i, a_ij, v_ij]
X_ij    = q_ij * base_ij                                  # 6 columns
```

- `X_ij=0` whenever `delta_ij=0`, reflecting the exact target's zero contribution. Feature expansion uses only continuous quantities and two small categorical flags; **no 12-cell coarse binning**.
- For numerical conditioning, divide each `X` column by its **training-only RMS** (`floor=1e-3`), persist all normalizers, do **not** subtract per-column means that would make `delta=0` yield nonzero features.
- Fit no-intercept weighted ridge regression on `t` using **image-cluster-respecting** sample weights `omega_ij=omega_i/15`. Fixed initial ridge penalty `10` applied to the standardized six coefficient vector; log its effective regularization scale. If an installed sklearn version's `alpha` semantics differ, implement and test the explicit normal equations `theta=(X^T W X+10 I)^{-1}X^T W t`. Never include a per-head row as an independent image for CV/bootstrap.
- Weight `omega_i` follows the existing v6 *class-balanced, boundary-upweighted* recipe derived from B-training only:

```text
m_i      = abs(u_i(1) - 0.5)
q20      = quantile_Btrain(m_i, .20)
omega_i  = [len(Btrain)/(2*count_Btrain(y_i))] * [1 + indicator(m_i <= q20)]
omega_i /= mean_Btrain(omega_i)
```

The v6 class/boundary weights are a **training design choice** and not proven optimal for WGA. Check minority-class support and label noise; do not use subgroup labels.

**Pooled estimate:** `A_pooled_ij = X_ij @ theta` in *probability units*. Save six coefficients, all column scales, B-training IDs, class counts, q20, feature/RMS statistics, ridge condition number, weighted MSE, and train-target diagnostics.

### 2.4 Head-specific specialist residual by partial pooling

The 15 heads are **not exchangeable within a fitted ensemble** and may be different across seeds/folds. Avoid both extremes: a single pooled correctness lookup (washout) **and** 15 independent high-dimensional predictors (overfit).

For head `j`, predicted class `c`, training-only ridge residual `e_ij=t_ij-A_pooled_ij`, and continuous `q_ij` defined above, compute a one-dimensional effect:

\[
 b_{j,c}^{raw}=\frac{\sum_{i:\hat y_{ij}=c}\omega_i q_{ij}\,e_{ij}}
                     {\sum_{i:\hat y_{ij}=c}\omega_i q_{ij}^{2}+10^{-8}},
\quad n_{j,c}=\sum_{i:\hat y_{ij}=c}\omega_i.
\]

\[
 \boxed{b_{j,c}=\operatorname{clip}(b_{j,c}^{raw},-0.1,0.1)
             \frac{n_{j,c}}{n_{j,c}+25}}.
\]

**Head-aware estimate:**

\[
 \widehat A_{j}^{RH}(x)=\widehat A_{j}^{pooled}(x)+q_j(x)\,b_{j,\hat y_j(x)}.
\]

The fixed `kappa=25` is an **effective image-weight shrinkage** parameter. A head/class cell with no support has exactly `b=0`. Log distinct-image support and denominators; low-support correction must shrink heavily. `b` belongs to the **actual head SHA-256** inside the given outer fold; never transfer `b_6` to another ensemble's slot 6. Do not assume heads 1, 6, 10 are persistent causal specialists.

This is **not** a proof that the residual targets are unbiased: in-sample residual effects are optimistic. Check all gains exclusively with proper B-OOF predictions.

### 2.5 Fix the actual *scale mismatch* without amplifying noise blindly

The earlier CCW divided estimated advantage by `std(t_raw)` and residual logits barely moved geometry; that is precisely the failure v7 targets. Scale using **predicted conditional advantage**, not raw targets, but impose a null-signal guard.

For a model trained on some B-training subset, estimate `A_OOF` on **held-out subfolds within that subset**, using a deterministic 3-fold image-stratified (by class only) inner-inner CV. Compute

\[
 s_R=\operatorname{RMS}_{i,j}\left(\widehat A_{ij}^{OOF}-\frac1{15}\sum_l\widehat A_{il}^{OOF}\right),
\quad s_G=\operatorname{RMS}_{i,j}(g_{ij}-\bar g_i).
\]

**Never use C labels or C scores to fit these scales.** `s_G` comes from the same B-training samples, `s_R` from genuinely cross-fitted predictions on those samples; training-only inputs. Treat each image as one unit when computing both RMS values. Save all secondary OOF predictions and their train/validation ID hashes.

At inference, for head `j`:

\[
 A_{j}^{c}(x)=\widehat A_j(x)-\overline{\widehat A}(x),
\]

\[
 \boxed{\widetilde A_j(x)=
 \begin{cases}
 0 &\text{if }s_R<0.002\text{ or nonfinite scores}\,,\\
 \frac{s_G}{s_R}\operatorname{clip}(A_j^c(x),-2s_R,2s_R)&\text{otherwise.}
 \end{cases}}
\]

- **Frozen primary `lambda=0.50`:** typical RMS residual score on B is half the centered geometry-score RMS; unlike old CCW's arbitrary `beta=1`, it has a measurable interpretation. `lambda=1.0` is **not** assumed superior.
- Clipping caps each head's correction; it does not guarantee AA safety. Record residual score RMS/range, fraction clipped, per-image max `|w-w_G|`, total variation of weight vectors, changed top head, `q_1-u_1`, and margin crossings.
- **Do not turn a near-zero `s_R` into enormous residual weights** via a tiny denominator. If effective support, ridge validation quality or fit integrity fails, fall back to G1 for that fold with a logged reason. Do not silently tune guard thresholds after seeing old C labels.
- Add a unit test that a global common advantage offset cancels exactly after centering and the resulting `r,w` obey simplex constraints.

### 2.6 Optional *margin gate* as one isolated ablation

Only after building `G1_R` and `G1_RH`, add one predeclared ablation `G1_RHM`:

\[
 \psi(x)=\exp\!\left(-\frac{|u_1(x)-0.5|}{\tau_B}\right),
\qquad \tau_B=\max(Q_{0.2,Btrain}(|u_1-0.5|),10^{-4}),
\]

\[
 a_j^{RHM}(x)=g_j(x)+0.5\psi(x)\widetilde A_j^{RH}(x).
\]

**Important:** exponential gating is never exactly zero on confident images and may also suppress signal on Uniform-wrong ones. It **does not** mathematically ensure `broken=0` or protect 80% of majority images. Diagnose high-margin harm, low-margin corrections, and gate values separately. **Do not change the frozen primary to RHM because RHM looks better on old C.**

### 2.7 A required counterexample: do not mistake margin or disagreement for a label

No static formula using `sign(delta)`, `argmax disagreement`, or small Uniform margin can know true labels by itself. An expert correctly disagreeing with Uniform on a minority sample and incorrectly disagreeing on a majority sample can have identical observables. If B-OOF cannot discriminate those cases, a residual is not learnable from the proposed features; **report `NO_GO_INFORMATION_LIMIT`** rather than adding a bigger model to fit already inspected errors.

---

## 3. Frozen experiment matrix — only the identified fixes

| ID | Score | Role | What it isolates |
|:--|:--|:--|:--|
| `U0` | `w=1/15` | invariant baseline | effect of any post-hoc weighting |
| `G1_frozen` | `g=-50d` | **reference / opt-in existing** | existing geometry weighting |
| `G1_R` | `g+0.50*R_pooled_scaled` | ablation 1 | continuous relative-advantage and correct scaling without head effects |
| **`G1_RH`** | `g+0.50*R_head_scaled` | **frozen primary** | head-specific partial pooling on top of continuous signal |
| `G1_RHM` | `g+0.50*psi(m)*R_head_scaled` | ablation 2 | whether low-margin gating protects AA |
| `G1_RH_perm` | same amount of R but **permute R across head slots** | negative control, **not candidate** | is useful head correspondence learned or merely changed concentration? |

**No G2/CCW-A/CCW-FULL rerun is required** except loading previously verified reference outputs for context. No parameter search beyond these fixed candidates; a separately registered B-only `lambda=1` sensitivity can be documented **without using C to pick a winner**. Do not conflate a sensitivity check with a new primary. If needed, first deliver the full frozen matrix above.

**Critical anti-leakage:** pre-register every candidate/config/source hash in `EXPERIMENT_LOCK.json` **before opening the new candidate's C labels or calculating any new candidate C WGA**. All C results are archival/exploratory; the code must not branch on C outcomes to spawn new candidates.

---

## 4. The actual B-only workflow: fit and mechanistic verification

### 4.1 B sample ownership and CV

For each `(head_seed, outer_fold)` in 9 ensembles:

1. Load its matching `B_predictions.npz` (`~240` distinct IDs, same 15 ordered head hashes, B labels `y`, no group labels). Check no overlap with that ensemble's A or C.
2. Predeclare 4 **image-level** stratified folds on B with reproducible seed `71000 + 100*head_seed + outer_fold`; stratify by **class only**, not hidden groups. Save exact `B_INNER_SPLITS.json` with image IDs and hashes.
3. For each of four inner validation partitions, fit ALL feature statistics, ridge coefficients, head-specific effects, boundary q20, and scale estimator **using its corresponding B-inner-training images only**. Compute scale `s_R` from 3-fold secondary cross-fit *inside this B-inner-train*. Save one fitted model and held-out validation predictions as atomic checkpoints.
4. Aggregate B-OOF validation predictions for `U0`, `G1`, `G1_R`, `G1_RH`, `G1_RHM`. Each B image is scored once by a model that did not train on its labels. Do not select heads using held-out per-image true labels.
5. For formal comparison across the three outer folds and the repeated B IDs, report per-ensemble metrics **and** collapse repeated images by seed before overall AA/CI when IDs overlap, following v6's documented convention. Clearly distinguish the 9 fits from 9 independent cohorts.
6. After B-OOF audit and config lock, refit each method on all matching B (`240`) for its C inference. Save the model including its secondary B-OOF scale and all fitted statistics. Do not refit on C.

**Be explicit:** The 3-way secondary OOF needed for the score scale is nested entirely inside an outer B-training partition, with no B validation leakage. This is modest compute (small ridge regressions) and is essential for a credible scaling comparison.

### 4.2 Required OOF measures (group-blind fitting, supervised audit on B only)

For each B-OOF image `i` and method `M`:

\[
 t_{ij}=p_j(y_i|x_i)-u_{y_i}(x_i),\qquad
 G_M(i)=\sum_j r_{M,j}(x_i)t_{ij}.
\]

`G_M>0` means its non-uniform component moves true-class probability in a favorable direction; for the actual residual mixture, `p_M(y)-u_y = .25 G_M`.

Save **per image** (not just averages):

- `G_M`, `G_M-G_G1`, sign of `G_M` and the fraction of G1-negative images converted to positive, all images and **Uniform-wrong subset**. `Uniform-wrong` is defined by B validation labels *after* inference; it is **not** an inference-time feature.
- True-class probability shift, B-OOF Cross-Entropy/NLL, **worst-class NLL** (class 0/class 1), AA and class-wise recall; exact AA floor by seed vs U0 and G1.
- Rank association between predicted `A_hat` and true `t_ij` per image (defined only with nondegenerate ranks), and expert ranking *within each image*; report valid counts and comparison with the G1 geometry rank. A global pairwise score correlation pooling 15 correlated rows is **not** a substitute.
- `r,w`, residual score range/SD/RMS and share of geometry-score RMS; entropy/effective number of experts; `top_j`, `rmax`, slot-0 top frequency, true-class gain and margin of corrected/broken images.
- Differences between pooled and head-aware estimated advantage; head/class support counts, positive/negative shrunk `b_{j,c}` values; stability of head effects within the **same head ensemble**; **no transfer of slot IDs between folds**.
- Margin bins from B-training quantiles (include low/mid/high), gate magnitude, classification flips; label-based subgroup WGA not available/allowed for B selection.
- Uncertainty intervals of differences by **image-resampling**, not by expert rows; avoid false independence across seeds sharing images and backbone.

### 4.3 Transparent B-only decision, WITHOUT hiding exploratory WGA

The fixed primary is `G1_RH` regardless of B outcomes; `G1_R` and `G1_RHM` are fully predeclared ablations. Do **not** choose whichever has the best old-C WGA. Record the following B-only criteria against G1:

1. **Mechanistic direction gate:** B-OOF mean `G_RH-G_G1` on Uniform-wrong images is **positive**, and per-seed values are positive in at least **2/3** seeds; save counts/intervals (small samples mean uncertainty may remain huge).
2. **AA floor:** `AA_RH-AA_U0 > -1.000 pp` in **every seed** on B-OOF. Report whether `AA_RH >= AA_G1` separately; do not silently substitute an AA vs-G1 floor for the original vs-Uniform requirement.
3. **No catastrophic proxy deterioration:** B-OOF worst-class NLL not above G1 by more than a predeclared tolerance `0.003`; this is an **engineering gate**, not an equivalence test nor a guarantee on WGA. Record exact class-0 and class-1 NLL.
4. **Actual score influence:** fitted scale is nondegenerate and resulting expert-weight TV is nontrivial (report distribution; do not make an arbitrary >0.01 cutoff a post-hoc selection lever).

Set `B_DIRECTION_GO` only if 1–3 pass. Otherwise `B_DIRECTION_NO_GO` and explain separately if the failure was (a) prediction quality, (b) scale, (c) head-effect sparsity or (d) AA/proxy trade-off. **Both statuses allow the predeclared old-C `EXPLORATORY_ONLY` diagnostics to be run**, because the user's research goal explicitly includes measuring WGA; neither status allows production promotion. This is an explicit correction to v6's overly restrictive proxy-based *measurement* stopping rule, **not** a claim that poor B evidence becomes acceptable.

If B signal is too noisy to support a conclusion, write `INCONCLUSIVE` in addition to the exact gate outcome and refrain from sweeping variants based on C.

---

## 5. Old C exploratory evaluation — measure WGA even when B gate fails

**Run once for the entire predeclared v7 matrix**, with no new candidate added after viewing results. Only fixed B-fitted models may be used. Old C is previously inspected development, not an untouched confirmation set.

1. Inference API inputs: `member_probs`, `member_distances`, `fitted_B_model`, `head_sha256`. **No `y`, `group`, `sample_id`-based specialist lookup or C-based normalization in inference.** IDs may be used solely to align and evaluate after predictions are committed.
2. For each outer fold, write `[n_C,15]` `r`, `w`, `Ahat`, `geom_scores`, `residual_scores`, `[n_C,2]` final `probs`, and `sample_id`. Store hashes and ordered head metadata before any evaluation.
3. Concatenate C folds for each seed with no repeated IDs and exactly 1,199 images. Confirm labels/group mapping and per-group counts. Compare paired predictions against **exactly matched** U0 and G1, not an unrelated test baseline.
4. Compute **AA, four group accuracies, WGA (minimum of four per seed), worst-class accuracy, changes vs U0 and G1**, and within-group corrected/broken, net flips, probability shifts, signed gains, weight TV and concentration. Include per-fold diagnostics only as secondary; do not average per-fold WGA to substitute for per-seed WGA.
5. Compute paired stratified-image bootstrap **over 1,199 unique IDs** (same resampled image IDs across all three seeds and all methods); 2,000 fixed-seed replicates, plus G1-vs-U0, R-vs-G1 and R-vs-U0 95% intervals. A shared backbone and cohort still limits external validity.
6. **Permutation controls:** independently permute new residual assignments across the 15 expert slots (matched within-image concentration/strength; fixed 3,000 seeded repeats on the already scored C caches for descriptive nulls); optionally also a fixed-per-fold residual permutation to check stability. These controls are **diagnostic only** and not a valid excuse to tune C.
7. Precompute an **oracle feasibility diagnostic** without any model selection: for each Uniform-wrong subgroup-2 image, `max_j p_j(y|x)` can cross 0.5 at fixed alpha or cannot; count by seed and by current method. Explain that oracle uses labels and is undeployable. Check if other groups become worst.
8. **No promotion on C** even if WGA rises; a new independent cohort/backbone replication and new preregistered protocol are required for confirmation.

**Mandatory tables:**

- `MEAN_AA_WGA_COMPARISON.csv`: U0, frozen G1, R, RH, RHM (historical v6 values separately for context), mean±seed spread, differences, AA eligibility.
- `PER_SEED_AND_GROUP_METRICS.csv`: seed, method, group, `n`, correct, accuracy, seed AA/WGA, seed worst-group identity.
- `B_OOF_MECHANISM.csv` and `B_OOF_SIGNED_GAIN.csv`: all B-OOF diagnostics, with IDs, crossfit predictions and no group columns.
- `WEIGHT_AND_SCORE_DECOMPOSITION.npz`: exact scores, weights, probabilities, distance, delta, head hashes, B model hashes, split/seed labels; C labels/groups restricted to separate evaluator output.
- `FLIP_AUDIT.csv`: full per-image Uniform/G1/R/RH/RHM comparisons and corrected/broken flags. **No selective reporting of only improved images**.
- `PERMUTATION_CONTROLS.json`, `BOOTSTRAP_CI.json`, `EXPLORATORY_C_REPORT.md`, `LIMITATIONS_AND_NEXT_STEPS.md`.

---

## 6. Fresh confirmation protocol — a separate, later gate

If and only if genuinely unseen images/independent backbone runs exist, pre-register a **single** v7 candidate **from B-only evidence** before opening their labels. If B evidence fails, the scientifically correct primary fresh comparison is **frozen G1 vs Uniform**, while v7 remains a negative/diagnostic method. If B evidence supports `G1_RH`, use that frozen recipe as the fresh v7 candidate and compare with G1 and Uniform.

If no new cohort exists, create `CONFIRMATION_BLOCKED.md` containing `NO_FRESH_DATA` and do not imply independent confirmation. **Never reuse the inspected 1,199 images as “new” by reshuffling folds/seeds.** A new 15-head seed on the same images/backbone offers additional head randomness, not an independent image cohort or backbone replication. Do not consume official test arrays that were previously viewed as if pristine.

To **attribute** a future WGA improvement to weighting, compare Uniform, frozen G1, and new `G1_RH` on **the exact same fixed heads and images**, with no head retraining. Retain matched residual permutations and a mechanistic B-OOF improvement as evidence about *why* weights helped. **The oracle upper bound is not a performance target or proof of mechanism.**

---

## 7. Checkpointing, automatic resume, and crash tests — highest operational priority

**Before fitting any B model, implement/extend the v6 durable task journal.** Copy/reuse the tested `experiments/ccw_v6/store.py`/`exploration/journal.py`, but place v7 outputs in **a new versioned namespace**. Never mutate old v1–v6 artifacts or original `B_SELECTION_NO_GO`.

### 7.1 Recommended v7 artifact layout

```text
experiments/g1_residual_v7/
  config.lock.json                   # immutable registered candidate/thresholds/splits
  INPUT_INVENTORY.json
  DATA_PROVENANCE.md
  BASELINE_REPRODUCTION.csv
  state.json                         # DAG + task status, hashes, version
  events.jsonl                      # append-only receipt log
  HANDOFF_STATUS.md                 # current progress/results/blockers
  NEXT_AGENT_PROMPT.md              # executable resume context
  INPUT_REQUEST.md                  # only if exact assets missing
  completed/<task_hash>.json        # authoritative committed receipts
  checkpoints/
    seedS/foldF/b_innerK/
      scale_subcv/*.json            # subcrossfit artifacts and RNG metadata
      continuous_ridge.json
      head_effects.json
      feature_stats.json
      B_val_predictions.npz
      diagnostics.json
    seedS/foldF/full_B/
      model.json
      model_provenance.json
      B_scale_OOF.npz
      B_train_manifest.json
    seedS/foldF/C_inference/
      G1_R.npz
      G1_RH.npz
      G1_RHM.npz
      inference_receipts.json
  metrics/
    B_OOF_MECHANISM.csv
    B_OOF_SIGNED_GAIN.csv
    MEAN_AA_WGA_COMPARISON.csv
    PER_SEED_AND_GROUP_METRICS.csv
    FLIP_AUDIT.csv
    BOOTSTRAP_CI.json
    PERMUTATION_CONTROLS.json
  EXPLORATORY_C_REPORT.md
  CONFIRMATION_BLOCKED.md           # if no fresh cohort
  CHECKSUMS.sha256
  tests/
```

### 7.2 Task graph and atomic writes

For each `(seed,outer_fold,inner_fold,secondary_scale_fold,method)`:

- Task ID is a stable hash of `phase`, task name, input file SHA-256s, ordered head SHA-256s, exact train/validation ID lists, method/config hash, source revision, and deterministic RNG seed.
- A task is `COMMITTED` only after all output files have been written to temporary paths, flushed/fsynced, atomically renamed, re-opened and verified, and a receipt containing hashes/shape/provenance has been committed. Do not mark complete before receipt is on disk.
- Use an **OS-level single-writer lock** and bounded retry/backoff for Windows/OneDrive `os.replace` transient file locks; no two active workers may write the same checkpoint.
- `state.json` is a convenient task index, **`completed/*.json` plus verified artifacts are authoritative**. Rebuild state if state file is stale; never silently skip a missing/corrupt checkpoint.
- Save every 3-fold secondary scale fit, every 4-fold B cross-fit, every full-B refit, each seed/fold C inference, each evaluation and every bootstrap/permutation batch. **If training from checkpoints is unexpectedly required, stop and request clarification instead of silently starting a 400-epoch backbone retrain**; this v7 spec is post-hoc weighting only.
- Persist RNG generator state and permutation/bootstrap progress at batch granularity, including sorted resampled ID hashes. Results must be deterministic under `--resume`, source/config/hash match, and Windows-safe paths.
- After **each committed task**, refresh `HANDOFF_STATUS.md` and `NEXT_AGENT_PROMPT.md` with phase, precise finished tasks, next tasks, B/C roles, counts, blockers, best *B-only* observation and the one-line resume command. The status file must never claim fresh confirmation.
- On `--resume`, verify checksums and skip completed tasks **without rewriting timestamps or retraining**. If task inputs/source changed, create a new namespace/revision and compare; never reuse stale output under a new config.

### 7.3 CLI and acceptance tests

Provide (commands below are **interfaces to implement**, not a claim they exist already):

```powershell
.venv\Scripts\python.exe -m experiments.g1_residual_v7.run --phase inventory --resume
.venv\Scripts\python.exe -m experiments.g1_residual_v7.run --phase B_crossfit --resume
.venv\Scripts\python.exe -m experiments.g1_residual_v7.run --phase exploratory_c --resume
.venv\Scripts\python.exe -m experiments.g1_residual_v7.run --phase all --resume
.venv\Scripts\python.exe -m experiments.g1_residual_v7.run --verify
.venv\Scripts\python.exe -m pytest experiments/g1_residual_v7/tests -q
```

**Mandatory tests (all automated, with assertion outputs):**

1. `lambda=0` **bit-exact** G1 predictions/weights across all 9 C folds.
2. Inference function signature rejects/requires no `y`, `group`, or true correctness; any label-conditional route is a leak.
3. B feature normalizers, B-only q20, ridge targets and scale fitted on train subsets only; assert B-OOF validation IDs not in any fit's training IDs.
4. B full model uses only same-ensemble B, not C; ordered head SHA-256 matches each fold's cache.
5. Head indices are not transferred across seeds/folds; input permutation of head order + matching model/head-hash permutation preserves output, or rejects mismatches.
6. `delta=0` implies the continuous contribution is zero; `sum_j t_ij≈0`; common advantage offset does not affect softmax.
7. Predicted residual scale floor/off-switch for degenerate/nonfinite data; finite logits, `r,w≥0`, row sums 1; `w_j≥0.05` for `alpha=.25`.
8. `G1_RH` and `G1_RHM` may not equal G1 by accidental implementation; record spread/TV, yet **do not require a prediction flip** as a correctness test.
9. All methods' C predictions match manifest IDs, group distribution and exact fixed baseline precision; no duplicate C IDs within a seed.
10. Fixed seeded bootstrap uses shared resampled unique IDs across methods/seeds; no independent-row bootstrap over 3,597 entries.
11. Forced process kill after one secondary subfit and after one C fold inference; restart in a new process with `--resume`; verify prediction/model/checkpoint byte hashes equal uninterrupted run, previously finished file mtimes unchanged, no duplicate receipts.
12. Atomic IO on Windows/OneDrive path simulation, single-writer lock, corrupt/truncated receipt detection.
13. An end-to-end run followed by `--resume --verify` produces **zero new scientific tasks** and prints exact completed task count.

### 7.4 Exit-state requirements

- `DONE_EXPLORATORY`: all allowed existing-data work complete; include AA/WGA table with both negative and positive outcomes, bootstrap, score mechanism, checkpoints.
- `NO_GO_DIRECTION`: residual fails B directional/generalization gate; **still publish the locked exploratory C ablation** as negative result, never promote.
- `CONFIRMATION_BLOCKED_NO_FRESH_DATA`: all exploratory work done, no untouched cohort. Can coexist with prior states.
- `BLOCKED_INPUTS`: essential matching B/cache/head provenance missing; stop honestly with exact request.
- `INTEGRITY_FAILURE`: numerical/provenance/checkpoint mismatch; stop without compromised metrics.

---

## 8. What the final report must answer (no hand-waving)

1. Did continuous `delta_j` and pooled regression achieve **positive true signed advantage ranking** on B-OOF, especially Uniform-wrong images? How many such distinct images were available? Give uncertainty and compare with G1.
2. Did head/class partial pooling outperform pooled-only correction? Is any head effect stable *within its own* seed/fold model, or is it sparse/noisy? Do not label a fixed slot as a causal specialist.
3. Did predicted-advantage scale matching **materially shift `r` and `w`** compared with G1? Show score spread and changed top-head frequencies, not just a coefficient.
4. On B-OOF, did the residual change `q(x)-u(x)` in the **correct direction** more often where Uniform was wrong? How many images crossed the true decision boundary; how many correct images were broken?
5. Did the margin gate help AA or merely suppress both useful and harmful corrections? Show gating values by margin stratum.
6. On old C **exploratory only**, what are per-seed and mean **AA/WGA of U0, G1, R, RH, RHM**, and ΔWGA vs G1? Did the worst group switch? How many independent image IDs are involved?
7. Did matched residual-permutation controls achieve similar improvements, and do bootstrap intervals include zero? Distinguish a fixed dataset effect from generalizable expert selection.
8. Does every seed pass the strict AA floor (`> -1.000 pp` vs its matched U0)? If not, clearly mark failure.
9. If R/RH/RHM fail, is that due to insufficient B sample support, poor signed-advantage prediction, overly strong/weak scaling, or simply no probability threshold crossings? Provide decomposition, **not** new unregistered tuning.
10. Were all checkpoints verified/resumable with no re-training, and is the v1–v6 original behavior untouched? Provide exact command, task counts, hash manifests and next-agent prompt.

### Required final verdict vocabulary

Use **one** of: `B_DIRECTION_GO`, `B_DIRECTION_NO_GO`, `B_INCONCLUSIVE`; and **one** of: `EXPLORATORY_C_WGA_UP`, `EXPLORATORY_C_WGA_EQUAL`, `EXPLORATORY_C_WGA_DOWN`, `C_NOT_AVAILABLE`. These are **descriptive labels, not proofs or promotions**. If no fresh data, state **`INDEPENDENT_CONFIRMATION_NOT_PERFORMED`** prominently.

---

## 9. Explicit Codex execution directive

**Implement and execute the exact v7 G1-blind-spot optimization above. This is *not* a request only to verify G1 on new data.**

Priority order:

1. Implement crash-safe journal, `--resume`, and source/provenance inventory.
2. Restore the verified v5/v6 pipeline; reproduce bit-exact U0 and G1.
3. Build a **continuous signed relative-advantage predictor** on matching B only; first pooled (`G1_R`), then head-aware (`G1_RH`). Fit scale using nested B-only OOF; raw G1 distances remain untouched.
4. Add optional margin-gated ablation (`G1_RHM`), no unsanctioned method sweep.
5. Perform complete 4-fold B-OOF diagnostics, preserve signed-gain/AA/proxy result and the honest B gate.
6. Refit using full B, run the **predeclared** old-C exploratory WGA/AA evaluation for all methods **whether B passes or fails**, with **no C-driven changes**.
7. Run matched permutation controls, paired bootstrap, flip audit; verify resume under forced interruption.
8. Deliver comprehensive `EXPLORATORY_C_REPORT.md`, `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md`, manifests and checkpoints. If fresh independent evaluation is missing, clearly mark it as blocked without preventing the rest of the task.

**No promises of WGA improvement. Optimize the defined blind spots; publish exact negative findings if the information is not learnable.**
