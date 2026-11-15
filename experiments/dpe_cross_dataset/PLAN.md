# Cross-dataset test of DPE post-hoc procedures (pre-declared plan)

Status: **draft plan, not yet frozen.** Nothing here has been run. Drafted 2026-11-15.
The plan becomes binding when it is committed together with `summarize.py`
(step 0.5); after that commit, the rules in "Test-set rules" apply.

## 1. Purpose

The ACIIDS draft finds that, on Waterbirds, three post-hoc procedures improve
full-validation DPE by 1.1–1.7 pp WGA on correctly labelled test images, and that
the official labels hide this. Two of its own limitations are that the study
covers one dataset and that the procedures were chosen while the Waterbirds test
set was being inspected.

This experiment applies the **same procedures, with every choice fixed in
advance**, to datasets whose test sets nobody on the team has looked at. If the
gains are real and were hidden on Waterbirds only by the "tern" label error, they
should appear on these datasets' **official** labels.

What this experiment does **not** test: the validation-label lever (+2.3 pp). That
comes from a bug specific to the Waterbirds generator and has no counterpart here.

## 2. Datasets

| Priority | Dataset (SubpopBench name) | Task | Classes | Backbone | Status |
| --- | --- | --- | ---: | --- | --- |
| 1 | `CelebA` | blond hair; attribute: male | 2 | ResNet-50 | in scope |
| 2 | `CivilCommentsFine` | toxicity; attribute: identity mentions | 2 | BERT-base | in scope |
| 3 | `MultiNLI` | entailment / neutral / contradiction; attribute: negation words | 3 | BERT-base | **conditional** (Sect. 9) |

- Group definitions, splits and metadata are the SubpopBench ones that DPE uses
  (`baselines/dpe4subpop/utils/datasets.py`). For CivilComments this means the
  SubpopBench "fine" groups, not the overlapping WILDS identity groups, because
  group-balanced head training and stratified folds need disjoint groups.
- Expected CelebA split sizes (train / val / test): 162,770 / 19,867 / 19,962,
  smallest group blond male (val 182, test 180). Step 0.2 verifies these numbers and
  records the actual group sizes for every dataset; if they differ, the
  verified counts are used and the difference is noted.

Reference numbers reported by DPE (ERM* + DPE, validation group labels):
CelebA 90.3, CivilComments 70.8, MultiNLI 75.3 (upstream README). Our backbones
are standard ERM, so these are context, not targets. TODO (Person 4): copy
the paper's standard-ERM + DPE numbers for these datasets, if reported.

## 3. Backbones

- SubpopBench ERM, pinned commit `4d3dbbe21029666ef19d040e110ec22908640c5b`
  (same as the Waterbirds fresh backbones), `--algorithm ERM --train_attr no
  --hparams_seed 0`, default hyper-parameters and default step count.
- **Default steps are 30,001 for all three datasets** (5,001 for Waterbirds).
- Seeds 0–4 per dataset (n = 5 backbones). Same checkpoint rule as
  the Waterbirds fresh-backbone run (`experiments/dpe_label_corrected_backbones`).
- Record per backbone: checkpoint SHA-256, steps completed, wall time, GPU,
  ERM test WGA (official), package versions.
- No ERM* backbones in this experiment.

## 4. Procedures (fixed now)

Everything below is identical to the strictly nested Waterbirds run
(`results/dpe/strict_nested_20261115/protocol.json`, protocol
`strict-nested-postmean-v2`) except for the items marked **[dataset]**.

**Heads.** 15 two-prototype DPE heads trained in sequence on group-balanced
subsets of the validation features, using our `src/methods/dpe_independent`
trainer with the authors' per-dataset head settings **[dataset]**, taken from
`baselines/dpe4subpop/scripts/train_all_pe.sh`:

| Dataset | lr | cov_reg | entropic scale | epochs | batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Waterbirds (reference) | 1e-3 | 5e5 | 30 | 20 | 256 |
| CelebA | 5e-4 | 5e5 | 30 | 20 | 256 |
| CivilCommentsFine | 1e-4 | 1e5 | 30 | 20 | 256 |
| MultiNLI | 1e-4 | 5e6 | 30 | 20 | 256 |

Momentum, weight-decay weight (10), the standard-ERM covariance schedule and the
per-head checkpoint rule stay as in the Waterbirds standard-ERM recipe.

**Reference arm.** Full-validation DPE (`fullval_uniform`), retrained with the
same seeds in every cell, as in the paper.

**Arms.** All 27 arms of the strict nested run, under their existing IDs. The
DPE-only arms are kept; DFR-lite is dropped (it was only needed for the label
lever).

**Holdout partition** **[dataset]**: 60/40 group-stratified split of the validation
set (the Waterbirds 720/479), used by the `pool_*` arms.

**Settings and selection.** The four settings {average, maximum} × λ ∈ {0.01, 0.1};
fixed setting average/0.1; selection by group-balanced NLL (primary) and WGA
(secondary) in a strictly nested inner 5-fold CV; 5 cross-fit folds; 400
full-batch Adam steps.

**Split seeds.** 10 per backbone, as on Waterbirds. If step 0.4 shows this
does not fit the budget, reduce to 5 **before** the plan is frozen, never after.

## 5. Hypotheses and declared contrasts

All contrasts: test WGA difference (pp) against same-seed `fullval_uniform`,
**official test labels**, averaged over split seeds within a backbone, then
mean and pointwise 95% t interval over the 5 backbones, plus the count of
backbones with a positive difference. Analysed per dataset; no pooling across datasets.

| ID | Arm | Role |
| --- | --- | --- |
| **H1 (primary)** | `crossfit_affine_sel_nll` | Cross-fitted recalibration (primary arm on Waterbirds) |
| H2 | `fullval_affine_oof_sel_nll` | Out-of-fold map applied to full-validation DPE |
| H3 | `pool_linear_sel_nll` | Linear residual on head disagreements |
| H4 | `pool_linear_sel_nll` − `pool_affine_sel_nll` | Head information beyond scalar recalibration |

**Decision rule, applied once per dataset and stated in the paper as written here:**

- *Supported*: mean > 0 and the 95% interval excludes 0.
- *Not supported*: anything else. The result is reported either way.

H1 on CelebA is the single primary test. H2–H4 and the other datasets are
secondary and are described as such. Every arm is reported in the supplementary tables.

**Prediction from the paper's account:** H1–H3 positive on official labels. A
result on CelebA (about 17 times more validation images than Waterbirds) may be
smaller because holding images back costs less; Sect. 7 tests this.

## 6. Reporting (per dataset)

- Overall accuracy, accuracy and n for each group, WGA, for `fullval_uniform`
  and the H1–H3 arms (`docs/research_protocol.md`, Sect. 5.3 of `AGENTS.md`).
- Which group is worst for full-validation DPE and for each arm.
- ERM backbone WGA, for context.
- Within-backbone SD of `fullval_uniform` over split seeds.
- Paper: one compact cross-dataset table (H1–H4, official labels), plus one
  sentence each in the abstract, contributions and limitations. The 12-page budget
  does not allow a section per dataset.

## 7. Secondary, declared analysis: validation-set size (CelebA only)

To separate "the procedures do not transfer" from "they help only when the
validation set is small", repeat H1–H3 on CelebA with the validation set
subsampled (group-stratified, fixed seed) to 1,199 images, the size of Waterbirds'.
The test set is unchanged. Run it only if Sect. 4 finishes within budget; it is
labelled secondary in every report.

## 8. Test-set rules

1. Nothing in Sects. 2–7 changes after the freeze commit. Changes needed
   because something breaks (a crash, a wrong path) are logged in Sect. 12 with a
   reason, and must not depend on any test result.
2. `summarize.py` is written, committed and run on a **synthetic** results file
   before any real run is scored.
3. SubpopBench logs ERM test accuracy during backbone training. This cannot change
   any choice, because every choice is fixed before training starts. Do not
   use those logs for any decision.
4. The post-hoc pipeline is first run end to end with test scoring disabled
   (smoke mode); test predictions are scored once, by `summarize.py`, after all
   cells of a dataset finish.
5. Partial results are not inspected dataset by dataset while cells are still running.

## 9. MultiNLI (conditional)

The Waterbirds procedures act on one binary logit. MultiNLI goes ahead only if
this 3-class version is accepted **before the freeze**:

- Offset-only and affine maps: a per-class logit offset vector *b* (and a scalar
  temperature *a* for the affine map) applied to log mean head probabilities,
  fitted by group-balanced multiclass cross-entropy with penalty λ[(a−1)² + ‖b‖²].
- Linear residual: adds a linear function of the per-class head disagreements
  q_jk − q̄_k.
- Exact WGA thresholds are dropped for MultiNLI (no single threshold exists);
  those arms are reported as not applicable.

If the team does not accept this by the freeze date, MultiNLI is left out
of the paper and named as future work.

## 10. Work plan

Step 0 blocks only the scoring runs. Backbone training (step 1) starts as soon as
step 0.1 passes.

| Step | What | Owner | Done when |
| --- | --- | --- | --- |
| 0.1 | Timing smoke test: 300 SubpopBench steps per dataset on T4 x2; check whether SubpopBench can resume from a checkpoint | Persons 1–3 | s/step and resume behaviour recorded in Sect. 12 |
| 0.2 | Data setup and verification of split and group counts | Persons 1–3 | Counts in Sect. 12; matches Sect. 2 or the difference noted |
| 0.3 | Generalise `src/methods/dpe_independent` (dataset config: sizes, groups, classes, head settings) | Person 4 | **Waterbirds regression:** the generalised code reproduces the saved strict nested official-label WGA for every cell and arm, exactly |
| 0.4 | One-cell timing of the nested pipeline on CelebA features (smoke mode) | Person 4 | CPU time per cell recorded; split-seed count confirmed (Sect. 4) |
| 0.5 | Write `summarize.py`, test it on synthetic results; **freeze commit** of this plan + `summarize.py` | Person 4 | Commit hash recorded below |
| 1 | Train 5 backbones per dataset, extract val/test features | Persons 1–3 | Checkpoint hashes + feature caches on Kaggle outputs |
| 2 | Run the nested pipeline (CPU sessions) | Person 4 + whoever is free | All cells complete, self-tests pass |
| 3 | `summarize.py` → results folder `results/dpe/cross_dataset_<date>/` | Person 4 | Report + per-group tables |
| 4 | Paper table and text | all | Numbers traceable to step 3 |

Suggested owners: Person 1 CelebA, Person 2 CivilComments, Person 3 MultiNLI if
it goes ahead, otherwise more CelebA seeds or the Sect. 7 analysis; Person 4
pipeline and analysis.

## 11. Compute budget (estimates, to be replaced by step 0.1 measurements)

- Measured on Waterbirds (T4, fresh-backbone run): one SubpopBench ERM job of 5,001
  steps, including feature extraction and one head fit, took 6,540–6,781 s
  (about 1.8 h); a DPE head-only job took about 390 s.
- CelebA at 30,001 steps on the same ResNet-50 input size: roughly 6× the
  training part, **about 10 h per backbone on one T4.** That is close to Kaggle's
  12 h session limit, so either resume across sessions must work (step 0.1) or each
  session trains one backbone per GPU with a time guard, as in the Waterbirds runner.
  Two GPUs per session → 2 backbones per session → 3 sessions for 5 backbones.
- BERT datasets (batch 32, sequence length 220 for CivilComments): unknown; measure in step 0.1.
- The nested pipeline runs on CPU (`dpe_independent` trains heads on CPU), so it
  does not use GPU quota. The Waterbirds 70-cell run took 3 h 01 min on one Kaggle
  CPU session; larger validation and test sets will increase this (step 0.4).
- Check each account's weekly GPU quota before step 1.

## 12. Log of measurements and deviations

(Filled in during steps 0–3. Each entry: date, who, what, why, and whether any
test result was visible at the time.)

- Freeze commit: _not yet_
- Deadline / go–no-go date for MultiNLI: _to be set by the team_
