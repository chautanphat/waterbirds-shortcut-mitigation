# Codex one-shot handoff — Adaptive weighting for DPE

**Task:** implement, run, compare, and report two phases (combination search -> winner ablation) in one reproducible invocation. Prioritize **unseen-split top-1 accuracy** over the original equal-weight DPE; also report balanced accuracy and WGA (if group labels exist). Do not guarantee gains or fabricate results. Do not stop for questions; inspect repository/data, make conservative decisions, log any blockers, and execute all feasible jobs.

## Ground truth / references
- Paper: To et al., *Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift*, ICML 2025, https://proceedings.mlr.press/v267/to25a.html ; HTML https://arxiv.org/html/2505.23027 .
- Official repo: https://github.com/minhto2802/dpe4subpop . DPE: frozen ERM backbone -> 15 prototype classifiers trained on class-balanced subsets of held-out **validation** features; each has one learnable prototype per class; IPS encourages same-class prototype diversity. Original aggregation: `mean_j P_j(y=k|x)`; no learned ensemble weights. Group-balanced protocol is distinct and requires annotations; prefer no-group setting.
- In official `src/dpe/core.py`, `DPE.predict_proba(target=...,avg=False)` returns member probabilities (inspect actual dimension; implementation returns `[N,B,C]`), `raw_logits=True` exposes logits; README lists `feats_val.npy`, `feats_test.npy`, metadata; `src/dpe/core.py` and `src/dpe/isomaxplus.py` contain relevant code. **Audit actual repo behavior**, especially stage resampling, checkpoint selection, split leakage, per-expert output calibration, metadata alignment, group usage; do not blindly trust README claims. Record git commit/config.
- This is a *new proposed extension*, NOT part of the paper. All outcomes empirical.

## Data protocol (CRITICAL)
1. First inspect working tree, dataset, checkpoints, feature arrays, labels, IDs, groups, hardware, dependencies. Preserve original source checkpoints; implement a separate `experiments/weighting/` package, minimal hooks only. No new downloads unless essential; never pretend unavailable data are present.
2. **Never train/tune/select on test labels or test-group metrics.** No sample may be in both training of DPE members and the held-out predictions used to train/evaluate the gate. Subgroup labels, if available, are **evaluation-only** in the no-group protocol; do not use for training, selection, or early stopping. Existing DPE checkpoints trained or selected on the *full val pool* cannot serve as unbiased gate features on that same pool.
3. Preferred: **K=3 cross-fitting over original DPE validation pool** (K=5 if cheap). Each fold: fit or refit entire 15-member DPE on other folds, including any checkpoint selection (selection may use ONLY training-fold internal data); predict every expert on the held-out fold. Assemble OOF `[M,N,C]` expert probabilities/logits, labels and optional frozen embeddings/distances, preserving unique IDs. Split OOF into `meta_train` (80%) and `meta_select` (20%) with stratification by class (and groups only if doing a separately flagged group-aware experiment). Choose on meta_select. Once frozen, refit DPE on all permitted DPE validation data for FINAL TEST inference; note any OOF/final-head distribution gap. Test is inspected once, after configuration frozen. Multi-seed repetitions should rerun the relevant folds.
4. Cheaper valid fallback: **one predeclared disjoint split** of original DPE-val pool: 60% `proto_train`, 20% `gate_train`, 20% `gate_select`, stratified by class. Retrain DPE on `proto_train` ONLY, and ensure checkpoint selection uses only its internal split; then predict gate_train/gate_select/test. Fewer samples but much cheaper. Prefer this if full cross-fit cannot finish; explain choice.
5. If immutable existing checkpoints can be proven independent of a separate labeled gate pool, use that pool instead. If no leak-free meta labels exist and no DPE refit is feasible, run diagnostics/synthetic unit tests only and state **blocked**, never report in-sample gains as generalization.
6. Cache per-sample frozen outputs ONCE per fitted DPE: `.npz` containing `sample_id`, `y`, `split`, `probs[M,N,C]`, optionally `logits`, `distances[M,N,C]`, `embedding[M,d]`, `group` (group must be isolated from gate fit). Validate softmax axis, class order, member count, absence of NaNs, exact ID/split disjointness and baseline equality to original mean-probability inference. Cache hash/checkpoint provenance. Missing features simply disable dependent methods; don't silently fake distances.
7. Primary **selection metric**: micro/top-1 accuracy on held-out `meta_select`; tie-break by balanced accuracy then NLL, no test peeking. Report final test accuracy vs equal-mean DPE (percentage-point delta), balanced accuracy, per-class accuracy, NLL, Brier, ECE, WGA/per-group accuracy if groups known, inference latency, training time, seeds, 95% paired bootstrap CI for delta accuracy if feasible. Never tune on test group labels. If WGA falls, highlight tradeoff.

## Math & methodological constraints
- Expert `j` outputs `p_j(k|x)`, N=15 by default. Nonnegative normalized weights: `w_j(x)>=0`, `sum_j w_j(x)=1`; ensemble `p_mix(k|x)=sum_j w_j(x)*p_j(k|x)`. A **static** weighting vector is sample-independent; a **dynamic** gate outputs weights per sample. Initialize static weights and dynamic gate at uniform `1/N`; original equal-mean is an unchanged baseline.
- User's label-conditioned heuristic is for **TRAINING TARGETS ONLY**: binary `s_ij = 2*p_j(y_i|x_i)-1 in [-1,1]` (class-correctness signed confidence); multiclass `s_ij=p_j(y_i|x_i)-max_{k!=y_i}p_j(k|x_i)`. Strongly wrong and confident -> negative; correctly confident -> positive. `q_ij=softmax_j(beta*s_ij)` may supervise a gate. Do **not** input `y`, correctness, `s_ij`, or true-class probability to inference-time gate; that would be leakage. Inference features may contain entire per-expert probability vectors, confidence, entropy, prediction margin, logits, distance gaps, ensemble disagreement, class-agnostic features, and optionally frozen `z`.
- Confidence is not accuracy; test calibration (global/per-expert temperature fit on meta_train only) and protect against miscalibrated high-confidence errors. For 2-class labels, prediction threshold 0.5 by default; optional threshold tuning strictly on meta_select as a *separate reported decision rule* (guard against using same split for endless searches).
- Weight network **trains on OOF meta_train** (or disjoint gate_train), NOT on DPE training samples. Freeze backbone and all 15 DPE heads throughout weighting experiments. Gate inputs `probs/logits/distances` are detached cached tensors. No train/test contamination via scalers, calibration, early stopping or checkpoint selection. Train data statistics only on meta_train.
- Default train objective: `CE(y,p_mix) + lambda*KL(w(x)||Uniform)`, with lambda in `{0, 0.01, 0.1}`. Optional alternatives: class-balanced CE, Brier, focal, CE+top-20%-per-class CVaR, CE+label-conditioned teacher KL; compare on **accuracy**. Use numeric clamp; no NaNs. An overconfident wrong expert must be penalized, not automatically given higher weight.
- Avoid combinatorial explosion: rank methods below by hypothesized payoff, **not proven superiority**. Stage screening/finalists/ablation with one orchestrator command. Use common OOF predictions/seeds and fair tuning budget. Optional libraries (scipy/sklearn) only if installed or installable.

## PHASE 1 — ranked combination search (priority rank, ascending; **all** feasible; methods sharing input use cached predictions)
`U`=uniform; `H`=heuristic; `S`=static learned weights; `D`=dynamic; `R`=discrete/rule. Implement each rank as an experiment ID, no giant Cartesian product.

**Tier A — highest expected value / low cost**
01 U0: original uniform probabilities `mean_j p_j` [required reference].
02 S1: optimize global softmax logits `a[N]` via full-batch CE (`w=softmax(a)`), L2/KL-to-uniform, zero init.
03 S2: constrained simplex optimize N weights for meta_train CE (SLSQP or projected gradient), verify same optimum trend as S1.
04 H1: `score_j=mean_i (2*p_j(y_i)-1)` binary or signed margin multiclass; `w=softmax(beta*score)`; beta {0.5,1,2,5,10}.
05 H2: `score_j=-mean_i[-log(p_j(y_i))]`; softmax beta; report class-macro variant.
06 H3: `score_j=-mean_i (1-p_j(y_i))^2` (proper multiclass Brier optional), softmax beta.
07 S3: static learned weights optimizing **class-balanced CE** (vs micro CE), compare potential accuracy tradeoff.
08 D1: softmax **shared linear gate** over per-expert `[p_1, conf, signed margin, entropy, gap_to_mean]` (for multiclass use top probabilities, entropy, margin); global expert bias, zero-init, CE+KL.
09 D2: softmax shared MLP `F->32->16->1` + expert-specific bias, ReLU, zero-initialize last layer and bias; CE+KL; *default neural architecture*.
10 D3: D2 with per-expert prototype-distance features (normalized min distance, nearest-vs-second distance); skip if unavailable.
11 D4: D2 + ensemble-context statistics (variance of probs, vote agreement, max disagreement, entropy of mean); CE+KL.
12 S4: optimize global expert weights on meta_train CE; tune binary threshold via inner CV within meta_train, evaluate/choose on meta_select, no test tuning.
13 H4: reliability score based on per-expert **accuracy**, with beta and Laplace smoothing; macro-class reliability variant.
14 R1: highest-confidence single expert per sample, **no ground truth at inference**; naive baseline (may be bad).
15 R2: confidence top-k experts (k=3,5,7,10) with uniform normalized weights; tune k using meta_select.
16 R3: confidence top-k experts with continuous softmax weights among selected; tune k/temperature.
17 R4: vote-majority (hard vote) and probabilistic median aggregation as two variants; contrast with equal-probability voting.
18 S5: accuracy-based convex combination penalizing correlated error vectors; `CE + rho*w^T(E^TE/M)w` where `E_ij=1[argmax p_j(x_i)!=y_i]`, fit train only; rho small grid.

**Tier B — promising, more modeling/assumptions**
19 H5: score from *confidence-binned* empirical reliability (e.g. 5 quantile bins per expert, Laplace smoothing, only meta_train labels) -> input-adaptive weights; interpolate bins.
20 R5: discrete correct/error-risk bands estimated on meta_train, thresholds on confidence/margin, assign per-expert `{0, 0.25, 0.5, 1}` reliability, normalize; fallback U if sum zero. Rules must use train-estimated calibration, never test correctness.
21 R6: discrete top-k by calibrated empirical reliability from R5, not raw confidence.
22 D5: shallow decision tree trained to predict each expert's correctness from inference-time features; use calibrated predicted correctness as gating score; depths 2/3/4, grouped crossfit.
23 D6: random forest predicts per-expert correctness from inference-time features, then softmax/normalize reliabilities (small grid).
24 D7: HistGradientBoostingClassifier / gradient-boosted trees for per-expert correctness; normalize outputs; avoid data leakage.
25 D8: ridge/logistic regression for expert correctness -> normalize reliability (compare with shared linear gate D1).
26 D9: small kNN kernel/local-reliability gate: feature embedding `z` (if cached) or class-probability vector; neighbors only from meta_train, Laplace-smoothed expert correctness, normalize.
27 D10: fit per-class expert reliability on meta_train (class-conditional calibration); at inference use soft estimate `sum_c p_mix_base(c|x)*reliability_{j,c}` rather than true label; one refinement max.
28 S6: class-dependent weights `w_{j,k}` optimized with positive normalization and **renormalize final class scores**; document that this departs from pure convex probability averaging and can change class priors.
29 D11: joint MLP that takes `[N*C]` probabilities and outputs N logits (`(N*C)->64->32->N`, zero final layer); small weight decay/dropout.
30 D12: shared MLP with a low-dimensional frozen embedding projection `z->16` concatenated per expert; CE+KL, avoid fitting large backbone.
31 D13: shared MLP trained with label-dependent teacher `q=softmax(beta*s)`, combined `CE + eta*KL(q||w)+KL(w||U)`; teacher never at inference.
32 D14: D2 trained with CE+per-class top-20% CVaR of loss (hard-sample proxy, **not** group knowledge).
33 D15: D2 trained with focal loss + KL (gamma {1,2}); test average accuracy.
34 D16: D2 trained with Brier + KL; check calibration and accuracy.
35 D17: D2 trained with expert dropout during meta-training; always restore all experts at inference.
36 D18: D2 with top-k learned **sparse gate** (train soft, then inference k in {3,5,7}); compare speed/accuracy.
37 H6: static weights from pairwise complementarity `score_j=mean correctness_j + rho * mean_i[correct_j & many others wrong]`; softmax.
38 R7: uncertainty-adaptive switch: if expert disagreement below trained threshold use uniform; otherwise D2 gate or H5 reliability rule.
39 D19: per-expert temperature-scaled probabilities + D2 (fit temperatures on meta_train only).
40 S7: stack logistic regression on concatenated OOF expert probabilities (not constrained mixture; clearly label as **nonconvex-combination benchmark**), regularized.
41 R8: per-expert top-1 vote weighted by train reliability, with tie-breaking by averaged probability; compare against soft voting.
42 X0: **oracle upper bound (DIAGNOSTIC ONLY)**: label-aware choose one expert correct per sample if one exists, measure fraction covered; NEVER offer as deployable method or selection target.

**Search control:** Run cheap 01–07 + 13–18 + 37–42 first, and D1/D2. Then Tier B. Each candidate gets small prespecified tune grid; early stop gates, 2 random seeds initial screening, retain top 5 (+ U0) by held-out micro accuracy, then retest finalists with 3 seeds (or repeated OOF splits if affordable) under equal budget. Avoid repeatedly adapting rules to the same tiny holdout; limit trial count, report optimistic selection bias. For ML models, max epochs 100, AdamW lr `1e-3`/`3e-4`, early stop patience 10, batch 64 or full for tiny sets, grad clipping; regularization grid lambda `{0,0.01,0.1}`, dropout `{0,0.1}` only for finalists. Simple heuristics use beta `{0.5,1,2,5,10}`. Test compute once after locked selection.

## PHASE 2 — ablation on winning *family* (one automated continuation)
Run against U0 and the exact winning Phase-1 configuration; fixed data protocol and 3 seeds. Lock Phase-1 winner before any test evaluation. Phase-2 is diagnostic, not an excuse to tune on test. If winner has no neural network, ablate its defining heuristics rather than forcing an MLP. Requested ablations:
A0 full winner; A1 remove calibration; A2 uniform (U0); A3 remove KL/uniform prior; A4 remove label-conditioned score teacher (if used); A5 remove confidence; A6 remove margin; A7 remove distance features; A8 remove ensemble-disagreement/context; A9 remove embedding; A10 replace dynamic weights with their training-set average (static); A11 discrete binning vs continuous mapping; A12 vary number of DPE members `{3,5,10,15}` by truncating same pretrained ordered ensemble, no retraining unless explicitly labeled; A13 gate capacity `{linear,16,32x16,64x32}`; A14 top-k `{1,3,5,all}`; A15 replace CE by Brier and by class-balanced CE; A16 add/remove hard-tail CVaR if feasible; A17 expert temperature calibration on/off; A18 change training split fractions / crossfit count, if cost allows; A19 feature shuffle or random-weight negative control; A20 diverse-seed reruns.
If inapplicable, mark `N/A` with reason; don't report missing feature as tested. Show each metric as `mean ± sd`, accuracy delta in percentage points relative U0, and latency/params. Include failure analysis of samples corrected/broken by weighting, weights by correct/incorrect experts, gate entropy/effective number `exp(H(w))`, weight distribution by group ONLY FOR POST-HOC TEST ANALYSIS if annotations exist (never choose model using test groups).

## Engineering & reproducibility
- Implement `experiments/weighting/{data.py,cache.py,methods.py,gates.py,losses.py,metrics.py,run.py,README.md}`, with pytest for dimensions, normalization, uniform parity, score polarity, leak isolation, deterministic seeds, gradient flow, zero-sum fallback, and oracle prohibition at inference. Reuse existing repo utilities when feasible.
- One CLI entry point `python -m experiments.weighting.run --config experiments/weighting/config.yaml --phase all --resume`. Config auto-detect/accepts dataset paths, DPE checkpoint/feature paths, `proto_source`, crossfit option, seed, CPU/GPU, max wall time/budget, method whitelist, output dir. Resume/cache stage outputs; log skipped methods with reasons. Use bounded parallelism, no giant nested-grid job explosion; graceful CPU fallback.
- Outputs `outputs/weighting/{manifest.json,cache_index.json,phase1_all.csv,phase1_ranked.csv,phase2_ablation.csv,final_test.csv,gate_best.pt,plots/*.png,REPORT.md,run.log}`. Rank table must show rank, method, config, accuracy, delta vs U0, BA, WGA if available, NLL, runtime, params. `REPORT.md` in English or Vietnamese: protocol, split/leak audit, train setup, baseline parity, Phase-1 all-method results, chosen winner, Phase-2 ablation, per-group tradeoffs, uncertainties, known limitations, exact single-line rerun command. Save brief `RESULTS_FOR_USER.md` Vietnamese summary.
- Never overwrite existing checkpoints. No mock/fake test metrics. If files/hardware absent, implement and run unit/synthetic smoke tests, write clear `BLOCKERS.md` with exactly missing paths and commands to finish. If no improvement, say so honestly and preserve best baseline.
- DO NOT spend time repeatedly asking the user; inspect and execute autonomously. Keep final console output <= 20 lines: data situation, leak status, baseline, winner, delta accuracy, WGA delta if available, completed/skipped experiment counts, output locations.

## Intended final conclusion
Determine empirically whether heuristic, discrete, static trainable, or dynamic input-conditioned weighting reliably beats original DPE on **unseen** accuracy, without secretly learning from ground-truth label at inference and without hiding subgroup-robustness losses. The order above is a **hypothesis-led priority**, not a guaranteed ranking by outcome.
