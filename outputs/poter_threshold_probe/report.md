# Fast POTER class-wise threshold diagnostic — no retraining

## Main conclusion

**The primary POTER threshold question remains BLOCKED by absence of independent four-group development data.** No POTER threshold was optimized, and fixed0.5 POTER results do not establish whether threshold tuning would fix its asymmetry. DPE is also blocked because all validation images fitted its heads. ERM provides a valid cheap clean-validation-selected threshold baseline, with limitations below. A new VA-POTER training method is **not justified by this run**; neither is a claim that threshold tuning has solved POTER's errors.

## Validation-only lock

ERM seed0 selected threshold **0.13**, using91 candidates0.05…0.95 step0.01; maximize original-clean validation WGA, then AA, then proximity to0.5, then lower threshold. Locked JSON was written **before loading test labels for the analysis/evaluation stage**. Test-source files were integrity-hashed during audit but no test metric or label entered selection. POTER/DPE clean-WGA thresholds are null with explicit BLOCKED reasons; fixed threshold0.5 is predeclared for all models. Optional joint threshold was skipped; no independent development blur data. Thresholds are not reselected during bootstrap or after test inspection.

Validation1199 is not a shared held-out development set for all three methods: POTER used599 as OT reference and DPE fitted all1199. The remaining600 POTER non-reference images lack two groups, so WGA over four groups is undefined there. Splitting them now cannot repair the missing independent groups. Do not select a favorable threshold using existing test labels.

## Main tables

All metrics below use official labels; percentages. Validation POTER/DPE rows are **reference/fitting-exposed descriptive** diagnostics. All test rows are **already-inspected exploratory evaluation**. Sigma4 is supplementary and excluded from selection. Clean original test and clean composites are distinct image populations.

|Model|Strategy|Dataset|Sigma|Threshold|AA %|WGA %|Landbird %|Waterbird %|
|---|---|---|---|---|---|---|---|---|
|POTER|fixed05|canonical_validation1199|0|0.50|91.58|90.13|91.32|92.48|
|POTER|fixed05|original_test5794|0|0.50|91.99|91.04|91.97|92.06|
|POTER|fixed05|counterfactual100|0.5|0.50|92.00|83.00|86.00|98.00|
|POTER|fixed05|counterfactual100|1|0.50|88.75|74.00|78.50|99.00|
|POTER|fixed05|counterfactual100|2|0.50|82.50|58.00|65.00|100.00|
|ERM_enhanced|fixed05|canonical_validation1199|0|0.50|93.16|75.94|94.86|87.22|
|ERM_enhanced|fixed05|original_test5794|0|0.50|92.34|74.45|94.30|85.44|
|ERM_enhanced|fixed05|counterfactual100|0.5|0.50|93.00|86.00|93.00|93.00|
|ERM_enhanced|fixed05|counterfactual100|1|0.50|85.25|69.00|84.00|86.50|
|ERM_enhanced|fixed05|counterfactual100|2|0.50|74.00|39.00|68.00|80.00|
|ERM_enhanced|clean_wga|canonical_validation1199|0|0.13|92.08|84.55|92.07|92.11|
|ERM_enhanced|clean_wga|original_test5794|0|0.13|90.82|82.17|90.93|90.42|
|ERM_enhanced|clean_wga|counterfactual100|0.5|0.13|90.25|69.00|84.50|96.00|
|ERM_enhanced|clean_wga|counterfactual100|1|0.13|83.25|52.00|75.00|91.50|
|ERM_enhanced|clean_wga|counterfactual100|2|0.13|72.75|30.00|61.50|84.00|
|DPE_legacy|fixed05|canonical_validation1199|0|0.50|97.16|96.35|97.11|97.37|
|DPE_legacy|fixed05|original_test5794|0|0.50|96.05|94.39|96.41|94.78|
|DPE_legacy|fixed05|counterfactual100|0.5|0.50|96.25|91.00|93.00|99.50|
|DPE_legacy|fixed05|counterfactual100|1|0.50|90.25|80.00|81.50|99.00|
|DPE_legacy|fixed05|counterfactual100|2|0.50|85.00|64.00|71.00|99.00|

ERM original test WGA changes **74.45%→82.17%**, paired Δ **+7.72 pp**, foreground-bootstrap CI95% **[+4.19, +10.00] pp**. Landbird Δ=-3.37 pp; Waterbird Δ=+4.98 pp; AA Δ=-1.52 pp. These are conditional descriptive contrasts on an already-inspected test set, not an unbiased generalization gain. Read every severity/group in `locked_threshold_evaluation.csv` and `classwise_metrics.csv` rather than optimizing for blurred-test results.

## Class-wise errors and background shifts

POTER fixed0.5, natural composites of100 test foregrounds (200 observations per true class at each sigma):

|Sigma|Land→Water errors|Water→Land errors|FPR %|FNR %|Predicted Water %|CFPS predicted Water %|
|---|---|---|---|---|---|---|
|0|24|3|12.00|1.50|55.25|88.89|
|0.5|28|4|14.00|2.00|56.00|87.50|
|1|43|2|21.50|1.00|60.25|95.56|
|2|70|0|35.00|0.00|67.50|100.00|

On original unmodified5794 test images, POTER has similar class-conditioned error rates (Land→Water8.03%, Water→Land7.94%). Increasing error direction toward Waterbird is therefore a property of these degraded/composited inputs, not evidence that the original clean task has the same class asymmetry. Original-test class counts are imbalanced (4510 Landbird/1284 Waterbird), whereas the100-foreground cohort is balanced; raw error fractions and CFPS depend on that prevalence. Neither CFPS nor these fixed-threshold observations isolate calibration or an internal mechanism.

`classwise_metrics.csv` contains aggregate, true-class and class×background rows: AA/WGA when defined, four group accuracies, full confusion counts, FPR=Landbird→Waterbird rate, FNR=Waterbird→Landbird rate, fraction predicted Waterbird, mean/median P(Waterbird), and CFPS by predicted class. CFPS is wrong-to-class / all wrong; undefined with zero errors. In binary classification it restates error direction and is not an independent mechanism or proposed novel score. Class/group rows retain their own denominators; four-group WGA is undefined on a single class/group and on gray backgrounds.

`paired_gray_natural_replicates.csv` retains all background pairs; `paired_gray_natural_shifts.csv` averages four natural backgrounds within each source foreground. `paired_gray_natural_ci.csv` resamples foregrounds within true class (2000 draws), preserving all background repeats. These fixed0.5 probability/flip/accuracy shifts describe background dependence in the existing test cohort. They do not identify an internal mechanism or justify an OT adjustment. POTER's original goal concerns spurious correlations/label noise; corruption robustness is an additional diagnostic, not a guaranteed objective.

## Statistical interpretation

`paired_bootstrap_ci.csv` compares locked ERM threshold versus fixed0.5 on identical original/blurred/gray images. At least2000 class-stratified foreground draws per condition; all background repeats travel together. WGA is recomputed over all four groups per draw. Undefined group resamples are excluded and counted. Selection-data CIs are conditional/in-sample, not selection-adjusted; no unbiased validation gain is claimed. Test CIs remain exploratory. One checkpoint per method cannot estimate training-seed variability. We report no new multi-seed stability or untouched-test claim.

## Decision and remaining evidence

- **POTER/DPE: decision gate4 — insufficient independent development data for threshold selection.** Cannot distinguish cheap threshold correction from representation/background sensitivity with a tuned threshold yet.
- **ERM: clean-WGA-selected baseline executed**, with class/AA/WGA/blur trade-offs quantified. Its behavior cannot be transferred to POTER's probability scale or taken as evidence of a POTER-specific mechanism.
- Obtain genuinely independent labeled development images spanning all four groups, with foreground masks only if joint blur selection is desired; freeze threshold rules there, then evaluate on a separately untouched final set. Current original/blur test images have already been inspected and cannot become fresh by relabeling their split.
- Do not train a new method, tune OT references, or spend on additional POTER seeds merely to run this diagnostic. Ordinary calibration/cost-sensitive baselines and representation/logit analysis remain future options once the data separation is valid.

## Execution and files

Inference completed using actual checkpoints and1199 validation originals, reusing cached ERM embeddings only after8 predeclared live feature/probability parity checks. No training, OT recomputation or new experimental images. Validation inference elapsed 44.9s; phase counts/errors in `validation_inference.json`. Cached26082 test/control rows reused; background-only1200 have no classification target. No unchanged test CNN inference was rerun. See `files_manifest.csv` and `integrity.json` for exported file hashes.

An initial8-image ERM preflight using default cuDNN TF32 failed cache parity and produced no accepted prediction CSV. Matching the existing FP32 comparison flags resolved it (feature max error1.12e-5, ERM probability max error8.94e-7, DPE reference error8.94e-8). The successful validation run used those fixed flags; `preflight_attempts.json` records the rejected attempt. No tolerance was relaxed to accept it. Total additional ERM diagnostic CNN forwards are24 images across failed/successful checks (8+8+8), separate from the1199 cached embeddings and1199 new POTER forwards.

Required outputs: audit.md, classwise_metrics.csv, threshold_sweep_validation.csv, selected_thresholds.json, locked_threshold_evaluation.csv, paired_bootstrap_ci.csv, five PNG plots and this report. Additional validation probabilities, membership, per-pair shifts and source/code provenance are retained. Run `python scripts/run_threshold_probe.py --stage analysis` to regenerate reports from verified validation predictions; no model inference or training in that stage. The locked JSON is preserved and a changed choice causes an error rather than silent retuning.

Automated checks: 14 passed in 10.98s
