# DPE Geometry-Based Aggregation — Codex Research Handoff v5

## Research objective

Investigate whether replacing uniform DPE aggregation with prototype-geometry-based inference rules improves **Worst-Group Accuracy (WGA)** without reducing Average Accuracy (AA) by 1 percentage point or more.

This study must be **group-blind during training and model selection**. Subgroup annotations are evaluation-only.

Do not introduce additional neural gating networks, specialist classifiers, pseudo-group DRO, or new backbone training.

## Existing context

- Model: Diverse Prototypical Ensembles (DPE), Waterbirds.
- Frozen feature extractor: augmented ERM* backbone.
- Number of prototype classifiers: 15.
- Classes: Landbird (0), Waterbird (1).
- Original aggregation: uniform probability averaging.
- Historical reference: Uniform DPE AA ≈ 95.708%, WGA ≈ 92.761% on the previously inspected test set.
- Previous continuous and discrete weighting methods failed to improve WGA consistently.
- Previous LCMSW method reproduced its exploratory improvement but failed nested verification.

Previous evidence suggests that correct experts are not necessarily the most confident experts on samples the uniform ensemble gets wrong.

The central hypothesis is therefore:

**Prototype geometry may provide better expert-selection signals than confidence, particularly for difficult examples.**

This is a hypothesis, not an established result.

## Input data

Locate `development_oof_bundle.zip` recursively in the repository.

It contains 3 seeds with:

- `meta_train.npz`: 959 samples per seed.
- `meta_select.npz`: 240 samples per seed.
- `probs`: [M, 15, 2].
- `logits`: [M, 15, 2].
- `distances`: [M, 15, 2].
- `embedding`: [M, 2048].
- `y`, `sample_id`, `group`, `fold`, `seed`, `split`, `head_id`.

Verify manifest hashes and OOF provenance before using the arrays.

Important: head IDs are slots within each fold ensemble. They do not identify the same actual classifier across different OOF folds.

`prediction_bundle.zip` contains historical test data. Never use it for training, calibration, configuration selection or threshold tuning.

## Phase 0 — Geometry and reliability audit

Before implementing aggregation, test whether prototype distance can identify reliable classifiers.

For each sample and classifier, calculate:

- Predicted class and correctness.
- Maximum class probability.
- Prediction entropy.
- Minimum prototype distance.
- Prototype distance margin.
- Leave-one-out expert agreement.

Compare these expert-ranking scores:

1. Negative minimum distance.
2. Negative entropy.
3. Maximum confidence.
4. Prototype distance margin.

Compute correct-vs-wrong expert ranking accuracy at the image level.

**Critical requirement:** Report results separately for all samples and for samples where Uniform DPE predicts incorrectly.

For images containing both correct and wrong experts, estimate how often each score ranks a correct expert above a wrong expert. Count ties as 0.5.

Include per-seed metrics and uncertainty.

Post-hoc, report metrics by true subgroup, but never use these group results to select a method.

Generate heatmaps and CSVs showing distance, confidence, agreement and empirical correctness.

Audit distance scaling across classifiers. Do not assume a smaller absolute distance from one head is comparable to a distance from another head without checking head-specific scale and normalization.

Check that `softmax(logits)` reconstructs exported probabilities and determine the exact relationship between logits and prototype distances.

Deliver `phase0_geometry_audit.md`, plots, source tables and a written conclusion about whether distance-based routing is promising.

## Phase 1 — Training-free aggregation experiments

Implement a small, fixed collection of methods.

**U0: Uniform baseline**

\[
P_U(k|x)=\frac1N\sum_j P_j(k|x)
\]

**E1: Entropy weighting**

\[
r_j=\operatorname{softmax}_j(-\gamma H(P_j))
\]

**G1: Minimum-distance weighting**

\[
r_j=\operatorname{softmax}_j(-\gamma \min_k D_{jk})
\]

**G2: Normalized-distance weighting**

Use head-specific distance normalization fitted on training-only observations. Do not pool unrelated OOF head identities.

**G3: Distance Top-M routing**

Select the M classifiers with smallest minimum prototype distance. Give selected classifiers equal weight.

**S1: Class-wise Soft-min / LogSumExp**

\[
S_k=\tau\left[
\log\sum_{j=1}^{N}\exp(-D_{jk}/\tau)
-\log N
\right]
\]

Convert class scores into probabilities using a class softmax.

**S2: Class-wise LogSumExp on actual logits**

Use original classifier logits rather than assuming they equal negative distances.

**S3: Centered-distance Soft-min**

Subtract each head's mean distance across classes before applying class-wise aggregation, to test the impact of head-specific offsets.

**S4: Conservative Soft-min Residual**

\[
P_{\text{final}}=(1-\alpha)P_U+\alpha P_{\text{Softmin}}
\]

For expert-weighting methods, use:

\[
w_j=\frac{1-\alpha}{N}+\alpha r_j
\]

Important: Class-wise Soft-min has class-specific effective prototype contributions. It is not mathematically equivalent to a shared expert-weighted average.

### Fixed screening configurations

Use:

- Entropy gamma: {1, 5}.
- Minimum-distance gamma: {5, 20, 50}.
- Standardized-distance gamma: {0.5, 1, 2}.
- Top-M: {3, 5, 10}; Top-1 diagnostic only.
- Soft-min tau: {0.025, 0.1, 0.5}.
- Residual alpha: {0.10, 0.25, 0.50} where specified.
- Screening class-score temperature: 1.

Do not expand the grid in response to observed WGA.

Train or fit any normalizers only on meta_train.

Screen candidates on meta_select.

Since this is a group-blind study, select configurations with a prespecified robustness proxy:

1. Exclude configurations whose AA decreases by 1 pp or more on any meta_select seed.
2. Rank eligible configurations by worst-class negative log-likelihood.
3. Tie-break using worst-class accuracy, AA and stability.

Freeze at most two configurations before examining subgroup WGA.

Subgroup labels must remain evaluation-only.

Save all candidates, including unsuccessful ones.

## Phase 2 — Nested verification

Use three seeds and three class-stratified outer folds.

Within each outer fold, construct disjoint datasets:

- A: Train 15 prototype classifiers.
- B: Fit aggregation normalization/calibration, if needed.
- C: Held-out evaluation.

Train real DPE heads from scratch on A using the frozen historical no-group head-training protocol.

Cache predictions for B and C from the exact same ordered heads.

Compare every candidate against Uniform DPE using identical heads and samples.

Never substitute historical OOF heads for the newly trained heads.

Pool C predictions across outer folds within each seed before computing WGA.

Report WGA, AA, balanced accuracy, worst-class accuracy, per-group accuracy, prediction flips and paired uncertainty.

### Promotion requirements

A candidate passes only if:

- Mean WGA improvement is positive.
- WGA improves in at least 2/3 seeds.
- AA decrease is strictly less than 1 pp in every seed.
- All data leakage and provenance checks pass.

Do not relax these criteria after examining results.

If neither candidate passes, stop without production integration.

If a candidate passes, implement it as an opt-in DPE inference mode. Preserve Uniform DPE as the default.

Even a passed nested verification is not proof of improvement on new unseen distributions; report uncertainty and the need for a fresh confirmatory dataset.

## Mandatory checkpoint and resume

**Implement crash-safe persistence before performing expensive experiments.**

Save checkpoints:

- After data verification.
- After every audit calculation.
- After each candidate and seed.
- After each preprocessing/calibration fit.
- After every prototype-head training epoch.
- After every completed outer fold.
- After each evaluation and promotion decision.

Persist model parameters, optimizer state, RNG state, configurations and data/source hashes where relevant.

Use atomic writes and integrity checks.

Required files:

- `state.json`
- `HANDOFF_STATUS.md`
- `NEXT_AGENT_PROMPT.md`
- `PROTOCOL_DECISIONS.md`
- `completed/<task_id>.json`
- Incrementally saved CSV/NPZ results and checkpoints.

Implement:

```bash
python -m experiments.geometry_aggregation_v5.run --phase all --resume
python -m experiments.geometry_aggregation_v5.run --status
```

Resume must reuse already completed tasks without recomputation. Test forced interruption and recovery before expensive training.

If quota or context is exhausted, preserve every completed result and write exact instructions for a new Codex session to continue.

## Final outputs

Generate:

- `phase0_geometry_audit.md`
- `phase1_screen.csv`
- `phase1_locked_candidates.json`
- `phase2_nested_results.csv`
- `FINAL_REPORT.md`
- All relevant figures, prediction exports, checkpoint manifests and runnable scripts.

The final report must answer:

1. Does minimum prototype distance rank correct experts better than confidence, especially when Uniform DPE is wrong?
2. Does class-wise Soft-min improve WGA?
3. Can geometry-based aggregation preserve AA within the 1 pp constraint?
4. Does the improvement survive nested verification?
5. If not, what is the main failure mechanism?

Do not optimize against previously inspected historical test data. Do not fabricate results. Do not introduce unrelated algorithms or expand the experiment search opportunistically.

**Primary metric: WGA. Secondary constraint: AA decrease strictly less than 1 pp per seed.**