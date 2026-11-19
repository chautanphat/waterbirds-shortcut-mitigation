# Constrained threshold and FDAT feasibility

## Executive verdict: **INCONCLUSIVE**

Global thresholds preserving original-clean WGA recover only a minority of the available blur oracle gain. Extra gray-view scores substantially improve retrospective composite performance. **FDAT does not yet add WGA beyond the simpler foreground-only classifier for POTER, and original-clean protection is unmeasurable.** Stop FDAT-specific algorithm/training expansion at this point; the next informative check is genuine original-gray pairs, not more hyperparameter searching or retraining.

|Decision|Result|Reason|
|---|---|---|
|A: constrained global|Not sufficient under declared80% triage|epsilon2 recovers 36.96% of unconstrained blur gain|
|B: FDAT prototype|INCONCLUSIVE|blur gate passes; clean-original gate BLOCKED; no added POTER WGA over gray-only|
|C: deployable/publishable|BLOCKED|independent dev/final test, original gray pairs, realistic masks and checkpoint replication absent|

Every optimized result below is **TEST ORACLE / IN-SAMPLE** or **retrospective species-blocked transfer**. None is an unbiased gain on a fresh test set. POTER's original objective concerns subgroup/spurious-correlation robustness and label noise; foreground blur/composition is an additional conditional shift.

## Reproduced historical anchors

POTER fixed0.50 natural-composite WGA86/83/74/58% at sigma0/.5/1/2; fixed0.65 gives91/89/87/83%. Prior per-severity grid oracles0.60/0.58/0.64/0.65 give93/92/88/83%. On5794 originals,0.50 WGA91.04%/AA91.99%;0.65 WGA82.71%/AA94.55% (WGA−8.33pp). These checks were recomputed from unchanged CSVs, not forced by editing data. **All14 anchors match reported precision.** Exact recomputation and reported-precision comparisons are in anchor_reproduction.csv. Clean composite sigma0 and original benchmark photographs remain distinct distributions with overlapping source birds.

## Stage A: constrained global threshold

Maximize mean composite WGA at sigma0.5/1/2 subject to full-original WGA loss<=epsilon. Tie original WGA, mean blur AA, proximity to0.5, lower threshold. Exact finite-score decisions use sorted boundaries plus adjacent float64 intervals; the0.001 grid is reported separately. POTER exact selections:

|Eps pp|Threshold|Original WGA %|Original loss pp|Mean blur WGA %|Blur gain pp|Fraction unconstrained gain|
|---|---|---|---|---|---|---|
|0|0.506795763969|91.4330|-0.3909|72.6667|+1.0000|6.52%|
|1|0.519467711449|90.4984|0.5437|74.0000|+2.3333|15.22%|
|2|0.558038711548|89.0966|1.9456|77.3333|+5.6667|36.96%|

The baseline mean blur WGA is71.6667%. The unrestricted finite-score global optimum is87.0000%; preserving original-clean WGA within2pp recovers36.96% of that gain. Our predeclared "most" criterion is80%; it is not met. This makes additional cue feasibility worth testing, without establishing a safe/validated constraint. `constrained_global_thresholds.csv` retains every epsilon/candidate and all per-sigma/class/group/AA/error/CFPS metrics. `stage_a_report.md` was written before Stage B.

## Stage B: FDAT candidate, ablations and oracle results

`d=max(0,p_full−p_gray); t(x)=min(t_max,0.5+lambda*d); yhat=int(p_full>=t(x))`.

Inference reads only scores and fixed parameters. True label/species/group/sigma/original/composite/split never enter the formula. A valid mask and one extra gray prediction are real input dependencies. Twenty capped candidates (lambda0/.25/.5/.75/1; cap.55/.60/.65/.70), plus5 optional cap1 ablations, are scored. Identity lambda0 is exact for every row. `fdat_paired_predictions.csv` contains150000 matched candidate decisions; `pair_identity_audit.csv` identifies the6000 actual natural-gray pairs. Shuffled-gray aggregate controls and the foreground-level permutation are retained.

POTER best extended-grid oracle is lambda1/cap1; mean blur WGA89.33%. Main capped-grid oracle is lambda1/cap0.7, with sigma2 WGA83.00%. The selected extended-grid sigma2 WGA is86.00%; using the same parameters with shuffled gray gives60.00%. This supports the value of matched foreground scores in this cohort, while the shuffled rule can itself have threshold effects. Negative controls were never tuning candidates.

## Stage C: retrospective species-blocked transfer and uncertainty

Five deterministic folds,2 species/class/fold,20 foregrounds/fold. Each source's backgrounds/severities remain together. Tune on4 folds by mean WGA at sigma.5/1/2; tie composite-clean WGA, mean blur AA, smaller lambda/cap. Global comparators tune on the same training species and constrain original-clean scores only after removing every held-out species from the original fitting pool. Both use test-derived data retrospectively; hypothesis/design already saw this cohort.

All POTER folds chose lambda1/cap1 **with the optional cap1 ablation allowed**. On the observed POTER scores, this gives the binary AND of full>=0.5 and gray>=0.5. In general cap1 is not literally infinite: if p_full=1 and p_gray<0.5, clipping t to1 and the >=tie keep Waterbird, unlike AND. `and_rule_equivalence.csv` records this boundary check for every model/sigma. It is a simple two-view decision, not established adaptive calibration. The primary capped-only oracle remains separate; no primary-capped-only OOF performance is asserted.

|Sigma|OOF strategy|WGA %|AA %|Landbird %|Waterbird %|WGA gain pp|
|---|---|---|---|---|---|---|
|0|fixed05|86.00|93.25|88.00|98.50|+0.00|
|0|fixed065|91.00|94.00|96.50|91.50|+5.00|
|0|global_constrained_transfer_eps2|90.00|94.25|91.50|97.00|+4.00|
|0|fdat|94.00|96.00|94.50|97.50|+8.00|
|0|foreground_only_classifier|94.00|96.00|94.00|98.00|+8.00|
|0.5|fixed05|83.00|92.00|86.00|98.00|+0.00|
|0.5|fixed065|89.00|93.25|96.00|90.50|+6.00|
|0.5|global_constrained_transfer_eps2|89.00|93.00|90.50|95.50|+6.00|
|0.5|fdat|94.00|95.00|94.50|95.50|+11.00|
|0.5|foreground_only_classifier|94.00|95.00|94.00|96.00|+11.00|
|1|fixed05|74.00|88.75|78.50|99.00|+0.00|
|1|fixed065|87.00|90.25|91.50|89.00|+13.00|
|1|global_constrained_transfer_eps2|78.00|88.75|81.00|96.50|+4.00|
|1|fdat|88.00|93.00|88.50|97.50|+14.00|
|1|foreground_only_classifier|88.00|93.00|88.00|98.00|+14.00|
|2|fixed05|58.00|82.50|65.00|100.00|+0.00|
|2|fixed065|83.00|87.75|86.00|89.50|+25.00|
|2|global_constrained_transfer_eps2|60.00|83.25|68.50|98.00|+2.00|
|2|fdat|86.00|92.00|86.00|98.00|+28.00|
|2|foreground_only_classifier|86.00|92.00|86.00|98.00|+28.00|

At sigma2 the pooled OOF gain is+28.00pp; descriptive species-bootstrap95% interval[+19.00,+36.00]pp. 5/5 held-out folds exceed8pp. Fold WGAs are not averaged to obtain pooled WGA; group counts are pooled first.

|Fold|FDAT WGA sigma2 %|Gain vs0.5 pp|Lambda|Cap|Matched original count|
|---|---|---|---|---|---|
|0|80.00|+25.00|1|1|0.0|
|1|80.00|+35.00|1|1|0.0|
|2|80.00|+25.00|1|1|0.0|
|3|90.00|+25.00|1|1|0.0|
|4|90.00|+20.00|1|1|0.0|

Gain appears in9 of20 species at sigma2; the top2 positive species contribute35.71% of positive net corrections, so gains are not entirely one/two species. Waterbird loss at this severity is concentrated in one species; see per-species table. All uncertainty remains conditional on these folds and the already-inspected test cohort, with no parameter refitting inside bootstrap.

## Does FDAT beat foreground-only or global thresholds?

On POTER, foreground-only and FDAT tie WGA at all four primary sigmas. At sigma2 their predictions are identical on all400 composites; other primary severities differ on only2 rows each. Full per-sigma disagreement counts:{'0.0': 2, '0.5': 2, '1.0': 2, '2.0': 0, '4.0': 1}. Thus a large improvement versus default0.5 does **not** establish incremental value for FDAT over the simpler foreground-only rule. The full image adds virtually no WGA here. OOF global epsilon2 WGA at sigma2 is60%, versus86% for both gray-based rules, but only the global rule has measurable original-clean score constraints; FDAT's clean trade-off is missing, so joint clean/blur dominance cannot be claimed.

Model-agnostic sigma2 OOF comparison:

|Model|Fixed0.5 WGA|FDAT extended-grid WGA|Foreground-only WGA|Global eps2 transfer WGA|
|---|---|---|---|---|
|POTER|58.00|86.00|86.00|60.00|
|ERM_enhanced|39.00|60.00|62.00|39.00|
|DPE_legacy|64.00|75.00|62.00|72.00|

Other models also benefit in some conditions; ERM can lose Waterbird recall and be weaker than gray-only, while DPE gains from combining views. No POTER-specific novelty or isolated OT mechanism is established. One checkpoint/method, unmatched training seeds/recipes and legacy DPE historical binding limitations remain.

Only four fixed backgrounds were tested, including semantic distractors identified in earlier visual QC; compositing/segmentation artifacts and altered gray scene statistics may contribute. Independent human recognizability and realistic-mask robustness are not established. A cue that distinguishes these composites from originals is not evidence of adaptation to a naturally occurring shift.

## Original full5794 clean evaluation: **BLOCKED for FDAT**

Full original score cache is AVAILABLE for global thresholds. All5794 mask files are present and checksum-valid. But only100 composite-gray foreground scores exist, and **0/100** candidate original-gray pairs pass exact opaque-foreground pixel identity. Matching dimensions and source ID do not prove identical foreground RGB: gray used original-CUB pixels while originals use benchmark/HF JPEG pixels. No100-original FDAT WGA and no5794-original FDAT WGA is imputed. Therefore the<=2pp matched-original protection gate cannot be evaluated, regardless of strong blur results.

## Mask/runtime dependencies and contribution assessment

Masks are oracle dataset assets; realistic segmentation and mask-error robustness have not been tested. Serial inference typically adds one gray-view forward, approximately doubling network passes plus segmentation/composition; model weights can be shared. A later5794-gray job has an inference-throughput proxy of3.61GPU-minutes from earlier measured validation inference, plus unmeasured staging/composition/QA/latency overhead. No such job, segmentation or training was started. See deployment_audit.md for assumptions/memory and exact pairing requirements.

The [POTER paper](https://arxiv.org/abs/2610.01028) describes OT reweighting using limited validation-group reference information. [Medi et al.](https://arxiv.org/abs/2411.19853) motivates class-wise robustness/error-direction analysis. FDAT is the candidate supplied by this handoff; no claim is made that either paper introduces it, and no comprehensive novelty/prior-art review was performed. Binary CFPS restates error direction and is not a new objective. These tests manipulate outputs of frozen weights; they do not improve POTER's OT training algorithm.

## One next action / STOP

**STOP FDAT-specific algorithm expansion and expensive retraining.** If continuing the cheap diagnostic, the next gate is to obtain genuinely paired gray scores from the exact original benchmark pixels and compare the already-fixed AND rule and foreground-only against0.5, without more test-label hyperparameter search. Only a favorable original-clean result would justify planning independent development data and realistic-mask evaluation. The present run cannot support deployable, publishable or method-level robustness claims.

## Reproducibility and outputs

Run `.\.venv\Scripts\python.exe scripts\run_threshold_algorithm_feasibility.py`; `--stage A|B|C|report` regenerates individual supported stages. Configuration and all hypotheses/tie rules precede fold evaluation. No hyperparameter was changed after the held-out fold numbers were inspected; no new CNN predictions or generated experimental images were added. Parameter-grid and bootstrap code is NumPy/Pandas; Pillow is used only for existing-pixel identity audit. `files_manifest.csv` lists every generated artifact/hash, including all required CSVs/PNGs and the one-page ChatGPT handoff. Earlier experiment outputs and threshold locks are unchanged.

Automated checks: 15 passed in 4.14s
