# Fast test-oracle threshold sweep

**TEST ORACLE / IN-SAMPLE UPPER BOUND. Test labels intentionally used to select thresholds. These are not validation-selected methods, deployable policies, or unbiased generalization gains.** No checkpoint loading, model inference, training, images or OT updates.

**Yes, threshold changes substantially recover POTER composite WGA in-sample:** sigma1 recovers +14 pp to88% (grid t=0.64); sigma2 recovers +25 pp to83% (t=0.65). This is a test-label-fitted feasibility result, not evidence of a deployable correction.

A single test-joint threshold0.65 gives composite WGAs91/89/87/83% at sigma0/.5/1/2, versus per-severity93/92/88/83%. Per-severity selection adds at most3.00 pp over that single global oracle on the primary levels. The clean-transfer threshold0.60 reaches 71.00% at sigma2. Thus the transfer table, rather than a claim of one universally suitable threshold, determines consistency.

**Original-clean trade-off:** applying0.65 unchanged to full5794 originals changes WGA 91.04→82.71% (-8.33 pp), while AA changes91.99→94.55% and Waterbird accuracy changes 92.06→83.18%. Composite-clean WGA changes86.00→91.00%; those two clean conditions must not be conflated. At composite sigma2,0.65 changes Landbird65.00→86.00% and Waterbird100.00→89.50%.



## POTER primary results

All values are percentages or percentage-point changes. Natural composites only:400 images/sigma,100 source foregrounds, two Land and two Water backgrounds/foreground; four groups each contain100 rows. Sigma4 is supplementary.

|Sigma|WGA@0.50|WGA@0.65|Oracle threshold|Oracle WGA|Gain vs0.50 pp|AA change pp|Landbird change pp|Waterbird change pp|
|---|---|---|---|---|---|---|---|---|
|0|86.00|91.00|0.60|93.00|+7.00|+1.00|+6.00|-4.00|
|0.5|83.00|89.00|0.58|92.00|+9.00|+1.75|+6.50|-3.00|
|1|74.00|87.00|0.64|88.00|+14.00|+1.75|+12.50|-9.00|
|2|58.00|83.00|0.65|83.00|+25.00|+5.25|+21.00|-10.50|
|4|23.00|77.00|0.65|77.00|+54.00|+13.25|+43.00|-16.50|

## Threshold transfer across severities

Each cell below is threshold/WGA(%). `oracle_per_sigma` fits labels at each severity separately. `oracle_clean_transfer` fits composite test sigma0 only. `oracle_joint` fits the unweighted mean of four test WGAs at sigma0/.5/1/2; tie-break clean WGA, mean primary AA, proximity to0.5, lower threshold. Both global oracle strategies are **test-adapted**, not held-out validation calibration. Threshold0.65 is a user-mentioned reference recomputed here, not validation-selected.

|Strategy|Sigma0|Sigma0.5|Sigma1|Sigma2|Sigma4 supplementary|
|---|---|---|---|---|---|
|fixed_05|0.50/86.00|0.50/83.00|0.50/74.00|0.50/58.00|0.50/23.00|
|fixed_065|0.65/91.00|0.65/89.00|0.65/87.00|0.65/83.00|0.65/77.00|
|oracle_per_sigma|0.60/93.00|0.58/92.00|0.64/88.00|0.65/83.00|0.65/77.00|
|oracle_clean_transfer|0.60/93.00|0.60/92.00|0.60/85.00|0.60/71.00|0.60/53.00|
|oracle_joint|0.65/91.00|0.65/89.00|0.65/87.00|0.65/83.00|0.65/77.00|

POTER clean-transfer threshold=0.60; joint threshold=0.65. `clean_WGA_drop_vs_fixed05` is positive for a loss of clean WGA; >2pp flags are reporting flags, never selection constraints. Per-sigma strategy has no single clean-transfer threshold, so that column is undefined for it.

## Original unmodified full test — no retuning

These5794 cached original images are a different input condition from clean composites. The100 composite foregrounds overlap this full test; this is **not independent confirmation**. Composite-selected global thresholds are applied unchanged using `original_group`. Status: AVAILABLE; full5794 original images.

|Strategy|Threshold|Original WGA %|AA %|Landbird %|Waterbird %|Clean WGA drop vs0.50 pp|Drop >2pp|
|---|---|---|---|---|---|---|---|
|fixed_05|0.50|91.04|91.99|91.97|92.06|+0.00|False|
|fixed_065|0.65|82.71|94.55|97.78|83.18|+8.33|True|
|oracle_clean_transfer|0.60|86.76|94.39|96.39|87.38|+4.28|True|
|oracle_joint|0.65|82.71|94.55|97.78|83.18|+8.33|True|

## Exact-boundary check and other models

Primary strategy comparison uses the declared0.01 grid. Separately, exact finite-score decisions are enumerated at distinct probabilities, float64 `nextafter` points in adjacent open intervals, and endpoints0/1 (plus0.5 for the proximity tie-break). This covers all attainable >=threshold classifications for t in[0,1], including ties. It does not establish an optimum on a new score distribution or meaningful deployment precision.

- POTER, sigma0: grid t=0.60, WGA=93.00%; exact finite-score boundary t=0.590086996555, WGA=93.00% (extra+0.00 pp).
- POTER, sigma0.5: grid t=0.58, WGA=92.00%; exact finite-score boundary t=0.574302256107, WGA=92.00% (extra+0.00 pp).
- POTER, sigma1: grid t=0.64, WGA=88.00%; exact finite-score boundary t=0.637435972691, WGA=88.00% (extra+0.00 pp).
- POTER, sigma2: grid t=0.65, WGA=83.00%; exact finite-score boundary t=0.646659612656, WGA=83.00% (extra+0.00 pp).
- POTER, sigma4: grid t=0.65, WGA=77.00%; exact finite-score boundary t=0.648608267307, WGA=77.00% (extra+0.00 pp).
- ERM_enhanced, sigma0: grid t=0.57, WGA=89.00%; exact finite-score boundary t=0.562889575958, WGA=89.00% (extra+0.00 pp).
- ERM_enhanced, sigma0.5: grid t=0.45, WGA=86.00%; exact finite-score boundary t=0.453511059284, WGA=86.00% (extra+0.00 pp).
- ERM_enhanced, sigma1: grid t=0.69, WGA=71.00%; exact finite-score boundary t=0.680601418018, WGA=71.00% (extra+0.00 pp).
- ERM_enhanced, sigma2: grid t=0.87, WGA=53.00%; exact finite-score boundary t=0.861778497696, WGA=53.00% (extra+0.00 pp).
- ERM_enhanced, sigma4: grid t=0.91, WGA=32.00%; exact finite-score boundary t=0.903910338879, WGA=32.00% (extra+0.00 pp).
- DPE_legacy, sigma0: grid t=0.51, WGA=96.00%; exact finite-score boundary t=0.508833348751, WGA=96.00% (extra+0.00 pp).
- DPE_legacy, sigma0.5: grid t=0.51, WGA=94.00%; exact finite-score boundary t=0.506922960281, WGA=95.00% (extra+1.00 pp).
- DPE_legacy, sigma1: grid t=0.51, WGA=88.00%; exact finite-score boundary t=0.513143718243, WGA=93.00% (extra+5.00 pp).
- DPE_legacy, sigma2: grid t=0.51, WGA=83.00%; exact finite-score boundary t=0.511778414249, WGA=87.00% (extra+4.00 pp).
- DPE_legacy, sigma4: grid t=0.51, WGA=69.00%; exact finite-score boundary t=0.512813985348, WGA=75.00% (extra+6.00 pp).

ERM_enhanced/DPE_legacy are included because their cached probabilities were already present. Their full metrics, ties, global thresholds, clean trade-offs and full-original results are in the same CSVs. DPE's probabilities are narrowly distributed, so a0.01 grid can miss useful boundaries; never conflate the grid maximum with the exact finite-score maximum.

## Audit and scope

Source `C:\Users\ADMIN\OneDrive - KonTum01\Desktop\POTER\outputs\mild_blur_v2\predictions.csv`, SHA256 `44fa3f1a0712987c67166d0af8c8e43bf81048da6eb82fbc81c4634361654fe3`. The requested browser name predictions(5).csv was not supplied; canonical audited9000-row mild-blur CSV is used. Actual cohort values:['replication100']; control values:['background_only', 'foreground_only', 'natural', 'original']. Included natural rows=6000; excluded controls={'foreground_only': 1500, 'background_only': 1200, 'original': 300}. No excluded controls contribute to composite WGA.

Canonical labels0=Landbird/1=Waterbird, source IDs/species/splits and original-group joins checked. Composite WGA uses **2*true_y+natural background type**, not the source photo's original background. Every model×sigma has400 rows,100 foregrounds, all four groups with100 rows each, four fixed background IDs and two replicas/type. Duplicate/missing/invalid rows cause explicit errors. Probabilities are finite/bounded/normalized and checked against logits (DPE uses log mean head probabilities); >=0.50 predictions are recomputed. POTER fixed WGA86/83/74/58/23 reproduced as checks, never used as optimization targets. Seed coverage remains POTER1, ERM0, DPE0; no multi-seed claim. Official labels are unchanged; known source-label issues and legacy DPE head/backbone provenance limitations from previous reports remain.

Grid91 candidates0.05…0.95. Per-sigma tie-break: WGA, AA, proximity to0.5, lower threshold. `best_thresholds.csv` includes all WGA-tied grid thresholds and remaining AA ties. Probabilities are unrounded; percentages are rounded only for Markdown display. `audit.json` retains denominators and hard-label/logit checks. Repeated background images and severities are dependent observations; no naive confidence interval or statistical-significance claim is made after oracle selection.

## Interpretation and next-step decision

Threshold changes can reveal a favorable decision boundary in these cached scores. Even large in-sample WGA recovery would show only that **this cohort's scores contain useful class-discriminative ordering**; it would not identify calibration, learned priors, representation mechanisms, or the cause of POTER training behavior. Inspect the original-clean, composite-clean, AA and Waterbird trade-offs before calling any recovery useful.

The previous independent-development limitation remains: POTER validation overlaps OT reference; DPE validation fitted its heads. This oracle exercise does not repair it and does not replace the validation-only report or its locked JSON. **Prioritize genuinely independent four-group threshold selection/calibration feasibility before expensive VA-POTER retraining.** No new training algorithm or mitigation is justified solely by an oracle gain. A future genuine method needs separate labeled development data (or a separately specified label-free adaptation rule) and untouched final evaluation.

## Reproduction and exports

```powershell
.\.venv\Scripts\python.exe scripts\run_test_oracle_threshold_sweep.py --input outputs/mild_blur_v2/predictions.csv --output outputs/test_oracle_threshold_sweep --original-test results/poter_failure/original_test_predictions.csv
```

Use `--skip-exact` to omit the supplemental exact-boundary pass. `--original-test` may point to a missing file; that condition is marked NOT AVAILABLE, with no substitute inference. The reusable script accepts only the specified audited100-foreground factorial design. Required CSVs/three PNGs are present; additional exact optima, joint candidates, audit/config/status and hashes support review. Existing predictions/checkpoints and earlier threshold locks are untouched.

Automated checks: 12 passed in 2.88s
