# DPE WGA-first audit — development OOF bundle

**Status:** exploratory research audit, 14 November 2026. **Primary objective:** increase worst-group accuracy (WGA); allow average accuracy (AA) to decrease by <1 percentage point at most. The results do **not** establish performance on a fresh test set.

## 1. Data integrity and scope

- Input: `development_oof_bundle.zip` (three seeds; each has `meta_train=959`, `meta_select=240`, 1,199 unique official Waterbirds development images). Per image: 15 head probabilities/logits/distances, frozen 2,048-D embedding, class, image ID, OOF fold, and four-way group (for diagnostic evaluation).
- Head predictions are 3-fold OOF. Checked no `meta_train`/`meta_select` ID overlap for each seed, 1,199 unique IDs per seed, and probabilities sum to one within floating precision. Export manifest independently records held/fit exclusion and SHA256 hashes.
- Labels `y`: 0=Landbird; 1=Waterbird. Post-hoc diagnostic `group`: 0=Landbird/Land, 1=Landbird/Water, 2=Waterbird/Land, 3=Waterbird/Water. Group was never an input to the fitted weighting methods below.
- All seeds share the same 400-epoch augmented ERM* backbone; this is **not** the standard ERM/no-augmentation track. Group labels were inspected in *this research audit* when comparing candidates, so hyperparameter choice informed by those results is exploratory/group-aware at the research-selection level despite group-free fitting.
- The original `meta_select` has been reused for previous searches; it is not untouched. The additional 5-fold sensitivity check below repartitions already-generated OOF predictions **without refitting the underlying prototype heads in each outer fold**; it is not fully nested CV. Both require new confirmation.

## 2. Uniform DPE and oracle headroom

**Original `meta_select` (240 images per seed):**

| Seed | Uniform AA | Uniform WGA | Any-correct oracle WGA | Baseline worst subgroup |
|---|---:|---:|---:|---|
| 0 | 95.42% | 91.67% | 100.00% | group 2: 22/24 |
| 1 | 93.75% | 88.46% | 96.15% | group 2: 23/26 |
| 2 | 95.83% | 92.05% | 97.73% | group 1: 81/88 |

Oracle means choose a correct individual expert using the **true label**. It is diagnostic, not feasible inference.

All 11/11, 13/15, and 8/10 uniform errors, respectively, have at least one correct individual head. However, many are corrected by only a *minority* of the experts; identifying the reliable head is the hard part. WGA on groups with 23–26 observations moves by ~3.8–4.3 pp whenever one sample changes, making selection by a single 240-image split unstable.

## 3. What did not reliably work

- `accuracy`-optimized global threshold gives good AA in some seeds but can sharply lower WGA (class bias). Threshold chosen using worst-class accuracy is also inconsistent across seeds.
- `mean` vs `median`, trimmed/Huber averaging, majority vote: no consistent WGA advantage on the original three `meta_select` pools.
- Weakly regularized static cross-entropy sometimes concentrates weight on a single head (up to ~0.72); worsens WGA substantially in some seeds.
- Within-class KMeans + aggressive max-loss weighting (cap 0.35) also degrades WGA in some seeds. More elaborate grouping is not automatically better.
- Dynamically chosen cluster-specific weights sometimes improve WGA but are less consistent than a single group-robust, static vector. Adding a larger MLP is not justified by these diagnostics.

## 4. Best exploratory candidate: conservative latent-cluster minimax static weights

### Model

1. Freeze all existing feature extractors and prototype classifiers.
2. On **weight-training** examples only, take frozen embeddings `z=f(x)`; fit PCA with `n_components=24`, `svd_solver='randomized'`, `random_state=0`.
3. Fit a **group-blind** KMeans with `K=2`, `random_state=1`, `n_init=6` to PCA features. This creates two *pseudo-clusters* **for loss computation only**; no subgroup labels are needed.
4. For samples in cluster `c`, form true-class mixture probability `q_i(w) = sum_j w_j * P_j(y_i | x_i)`.
5. Fit one 15-dimensional **static** weight vector, constrained to simplex and upper-capped:

\[
\min_{\substack{w_j\ge0,\ \sum_jw_j=1\\w_j\le0.25}}
\quad\max_{c\in\{0,1\}}\frac{1}{|S_c|}\sum_{i\in S_c}\Big[-\log(\sum_{j=1}^{15}w_jP_j(y_i\mid x_i))\Big]
+0.02\sum_{j=1}^{15}(w_j-1/15)^2.
\]

The numerical solution was independently checked with an **epigraph formulation**: minimize `t + 0.02*||w-u||²` subject to each cluster cross-entropy `L_c(w) <= t`, plus the simplex and weight caps. SLSQP converged successfully for all 3 original train/select fits and all 15 (3 seeds × 5) sensitivity fits, reproducing the same AA/WGA. For production, use the epigraph form to handle the nonsmooth `max` robustly.

This is **a group-blind surrogate for WGA**, not direct WGA optimization. During inference, no KMeans, neural gate, true class, or subgroup label is needed: `P_final(k|x) = sum_j w_j * P_j(k|x)`.

**Caution:** Global KMeans `K=2` mostly divides features by bird *class*, not land/water background (cluster sizes about 711 and 248 in meta-train). Hence this method should be called **latent-cluster minimax weighting**, not subgroup discovery or background-aware gating. It nevertheless reduces weight-induced class imbalance and yields better held-out group accuracy in the checks below.

### Original meta_train/meta_select split (group labels used only for audit)

| Method | Mean meta_select AA | Mean meta_select WGA | Mean WGA change vs uniform |
|---|---:|---:|---:|
| Uniform 15-head average | 95.00% | 90.72% | — |
| Ridge 0.1, class-balanced CE static weights | 95.69% | 91.48% | +0.76 pp |
| K=2, cluster-max loss, ridge .02, cap .25 | **95.97%** | **92.76%** | **+2.04 pp** |
| K=2, cluster-max loss, ridge .02, cap .30 | 96.25% | 92.76% | +2.04 pp |

For cap .25, the WGA changes by seed are `[0.00, +3.85, +2.27] pp` (no decline in these seeds); small group denominators make this very noisy. **The candidate family and settings were compared after inspecting development WGA; do not claim an untouched no-group-selected result.**

### Additional 5-fold sensitivity check over all 1,199 OOF examples per seed

For each of three seeds, repartition existing 1,199 OOF rows into 5 class-stratified folds; for each fold fit PCA/KMeans/weights on other four folds, predict the held-out fifth. We **did not refit the underlying DPE heads** to exclude outer-test labels from their training, so this sensitivity check is not independent, fully nested cross-validation.

| Method | Mean AA | Mean WGA | ΔAA vs uniform | ΔWGA vs uniform |
|---|---:|---:|---:|---:|
| Uniform | 94.91% | 92.16% | — | — |
| K2 minimax / ridge .02 / cap .25 | **95.36%** | **93.95%** | **+0.45 pp** | **+1.79 pp** |
| K2 minimax / ridge .02 / cap .30 | 95.36% | 93.77% | +0.45 pp | +1.61 pp |

Per-seed WGA (uniform -> cap .25):
- seed 0: 91.73% -> **93.99%** (+2.26 pp)
- seed 1: 92.27% -> **93.13%** (+0.86 pp)
- seed 2: 92.48% -> **94.74%** (+2.26 pp)

Per-group averages across the same 5-fold sensitivity predictions:

| True subgroup (evaluation only) | Uniform acc | cap .25 acc | Δ pp |
|---|---:|---:|---:|
| Landbird + Land | 96.79% | 96.43% | -0.36 |
| Landbird + Water | 92.99% | **94.21%** | +1.22 |
| Waterbird + Land | 92.73% | **94.49%** | +1.75 |
| Waterbird + Water | 97.24% | 96.49% | -0.75 |

The desired WGA trade-off is visible: small reductions on the easy groups, improvements on two challenging groups, **and AA also increases**.

On 3×1,199 paired sample-seed predictions, the cap .25 method changes **48** class decisions (1.33%): **32 corrected, 16 broken**, for a net gain of 16 correct decisions. Specifically, the minority/background-mismatched subgroups get **19 fixes / 2 breaks** (Landbird+Water) and **7 fixes / 0 breaks** (Waterbird+Land). Changes go **both ways** (20 Landbird->Waterbird and 28 Waterbird->Landbird), unlike the earlier D19 gate which flipped in only one direction. Note duplicated images across seeds: counts are sample-seed events, not 48 distinct images.

A 4,000-replicate stratified, paired *conditional* bootstrap using shared image IDs across seeds gives an approximate 95% interval for cap .25 minus uniform WGA of **+0.25 to +3.19 pp** and AA of **+0.03 to +0.86 pp**. These intervals condition on the fixed OOF ensemble predictions and the methods chosen after development exploration; they exclude selection bias, independent-backbone variation, and uncertainty from retraining heads. **Do not read them as confirmatory evidence**.

## 5. Why this candidate is preferable to the prior MLP or discrete D5

- Objective emphasizes the most difficult **latent cluster**, rather than pooled average CE, and avoids an accuracy-only class shortcut.
- Explicit simplex cap `w_j<=.25` prevents a single expert getting the 86.8% weight seen in problematic prior D19 examples; no per-image highly concentrated gate is allowed.
- Fits only 15 numbers, so less overfit-sensitive than a 2048-D embedding-aware MLP on ~959 development examples.
- Inference cost remains nearly identical to original DPE and entirely deterministic; no extra NN is required.
- Observed sample-level changes are small, bidirectional, and mostly beneficial to groups 1 and 2 in the sensitivity check.
- Still **not** a guarantee of WGA improvement: fixed member indices across independently trained OOF head ensembles may not transfer to final full-validation heads.

## 6. Most serious remaining risks

1. **Head identity / refit mismatch:** head index `j` refers to distinct learned prototypes in the three source OOF folds. A static weight for slot `j` learned over those ensembles may not generalize to independently refitted `heads_final.pt`. Fix by retraining true nested head+weight pipelines or evaluating against an untouched holdout using the actual final heads.
2. **The `K=2` pseudo-clusters are mainly class splits**, not necessarily real background groups; an improvement on Waterbirds does not establish robustness to arbitrary latent shifts.
3. **Sample size:** meta_select worst group has just 23–26 images. A single changed prediction moves its measured accuracy by ~4 pp; the more complete 5-fold sensitivity uses 1,199 rows but not independent heads/backbones.
4. **Multiple comparisons:** many configs were tried and this recommendation is influenced by WGA audits of development data. Treat numeric gains as hypothesis-generating. For a strict "no subgroup labels at any stage" paper, pre-register all parameters and repeat on fresh data without viewing group labels until evaluation.
5. **AA safety:** the observed cap .25 gains satisfy the AA-no-more-than-1pp criterion *in this development audit*, but no weight loss can guarantee this inequality for all future distributions; an external holdout is required.

## 7. Recommended execution plan for Codex

**First, confirm the simple robust static baseline, not a new MLP.**

- Freeze candidate recipe: `PCA(24, randomized, seed=0)`, `KMeans(2, seed=1, n_init=6)`, `max_cluster_classification_CE + 0.02*||w-uniform||²`, `w∈simplex`, `max(w)<=0.25`. Optionally pre-register one conservative control `max(w)<=0.10` and uniform reference, without further test-driven tuning.
- Generate *properly nested* held-out predictions where **both heads and weights** have not used each evaluation example (outer class-stratified folds; within each outer train, train prototype ensembles and weight models on disjoint roles, keeping the original DPE backbone frozen). Check IDs/provenance and preserve exact group-blind settings.
- Measure AA, WGA, worst-class accuracy, per-group confusion, corrected/broken, and flip directions; compare against matched uniform with the same heads. Require AA >= AA_uniform - 1.0 pp. Use group IDs for **evaluation only** in the strict no-group track.
- Test on a **new, previously untouched** split/dataset, preferably also a second shift benchmark, since the historical Waterbirds test is repeatedly inspected. Run independent backbone/head seeds and paired uncertainty estimates.
- Check final-refit transfer: compare slot-weight distributions on different OOF folds and independently trained final heads. If unstable, avoid head-specific static transfer or change the deployment protocol to preserve head identity.
- If static minimax gains replicate, only then investigate conservative local/gated extensions: (a) robust-loss + uniform-residual linear gate; (b) predicted-class-conditional reliability with worst pseudo-cluster loss; (c) pseudo-group DRO/CVaR. Keep the simple static method as the primary baseline.
- Codex should checkpoint **after each new fold, fitted head, weight solve, seed, and evaluation**, write atomic task markers, maintain state/next-agent handoff and verify a real `--resume` process restart. Never delete existing DPE/D19/D5 artifacts.

**Decision:** The strongest *exploratory, test-free candidate in this audit* is `global-KMeans(2) + worst-cluster cross-entropy + uniform ridge (0.02) + max expert weight 0.25` learned as one static 15-head ensemble vector. No tested method can be certified as optimal, and the observed WGA gain must be independently confirmed.

## Reproduction files

Supporting scripts and CSV tables are in the accompanying `DPE_WGA_Audit_Repro.zip`. Scripts expect the original `development_oof_bundle.zip` extracted to `dpe_dev_analysis/development_bundle/` and use NumPy, Pandas, SciPy, scikit-learn and statsmodels. None of these scripts train or evaluate on the historical Waterbirds test set. The `5-fold` sensitivity is intentionally labeled as such, rather than fully nested training of the original prototype ensembles.
