# Original Waterbirds gray-only feasibility gate

## 1. Executive verdict: **PASS**

Gray-only clean-protection gate **PASS**, locked FDAT **PASS**. Observed FDAT minus gray-only WGA is-1.25pp; paired95% descriptive CI[-2.18, -0.47]pp. This answers the one missing original-test check without another parameter search. **STOP FDAT expansion: gray-only ties/exceeds FDAT WGA. Next validate foreground isolation with independent development/final data and realistic masks.**

## 2. Data/pixel audit and score parity

Exactly5794 benchmark/HF JPEGs with official masks, one-to-one IDs and all four official original groups; exclusions0. Gray uses the exact benchmark RGB, not the old CUB foreground RGB. Shared square256/bilinear and center224 geometry; neutral gray127; original grayscale mask alpha0..255; original ImageNet normalization. Foreground RGB before blending and every opaque/core model pixel is identical. Soft-alpha pixels blend by design, including broad soft mask interiors; those values are reported, not misrepresented as unchanged opaque foreground. No label/group/species/sigma enters the builder or model forward.

All original model-view pixel hashes match the5794 cached full inputs.24 diverse preflight examples passed source/mask/alignment/pixel checks and visual review; live full-score parity max probability error5.36e-07, with unchanged prespecified tolerances. See audit.md, preflight_probability_parity.csv and three eight-example contact-sheet pages for source details. Mask alignment is dataset-provenance/geometry plus visual QC, not a guarantee that every soft boundary is semantically perfect.

## 3. Original-test results

G0=Landbird/Land, G1=Landbird/Water, G2=Waterbird/Land, G3=Waterbird/Water. Official denominators:{'0': 2255, '1': 2255, '2': 642, '3': 642}. AA uses all5794 images; WGA is minimum of the four group accuracies. CIs are paired source bootstrap within those groups (2000 draws), recomputing WGA every draw. Full class/group/error/AA intervals are in bootstrap_ci.csv.

|Policy|AA %|WGA % [95% descriptive CI]|Landbird %|Waterbird %|G0 %|G1 %|G2 %|G3 %|ΔWGA pp [paired CI]|Gate|
|---|---|---|---|---|---|---|---|---|---|---|
|full_05|91.99|91.04 [+89.40, +91.80]|91.97|92.06|92.90|91.04|92.52|91.59|+0.00 [+0.00, +0.00]|PASS|
|gray_05|94.72|91.59 [+89.10, +92.83]|95.59|91.67|95.79|95.39|91.74|91.59|+0.55 [-1.25, +2.10]|PASS|
|locked_FDAT|95.13|90.34 [+87.85, +91.59]|96.47|90.42|96.72|96.23|90.34|90.50|-0.70 [-2.65, +0.90]|PASS|
|boolean_AND|95.13|90.34 [+87.85, +91.59]|96.47|90.42|96.72|96.23|90.34|90.50|-0.70 [-2.65, +0.90]|PASS|
|historical_full_065|94.55|82.71 historical comparator|97.78|83.18|98.31|97.25|83.64|82.71|-8.33 not a candidate|HISTORICAL_DIAGNOSTIC|

The0.65 row is a historical diagnostic comparator from cached full scores; it was not optimized or introduced as another candidate. All policies in this task were locked before new gray inference.

## 4. Clean protection and group-specific harm

Gate uses the observed WGA change>=−2.0pp. It is a project triage bound, not a statistical safety guarantee; its point decision can differ from whether the descriptive CI crosses−2pp. Gray ΔWGA=+0.55pp, paired CI[-1.25, +2.10]pp; FDAT ΔWGA=-0.70pp, paired CI[-2.65, +0.90]pp.

- gray_05: most harmed group G2 (Waterbird/Land) changes-0.78pp; its worst absolute group is G3.
- locked_FDAT: most harmed group G2 (Waterbird/Land) changes-2.18pp; its worst absolute group is G2.

Read all four group changes and class recalls rather than relying on AA or a PASS alone. paired_flips.csv retains full-correct→policy-wrong and full-wrong→policy-correct counts by class/group; group_score_diagnostics.csv reports score shifts, not a causal mechanism.

## 5. Relation to historical composites

The previous sigma2 natural-composite WGA58% full/86% gray/86% FDAT is fixed exploratory context, not a new independent finding. It involved100 foregrounds, four repeated backgrounds and old CUB-RGB gray views. This run evaluates5794 distinct original images with new benchmark-RGB gray inputs. Source birds overlap prior tests; results cannot be combined as independent replications. Composite-clean sigma0 is also different from unmodified original-clean. Historical blur gray inputs were not regenerated using this exact benchmark-RGB-first builder; this clean gate does not revalidate the historical+28pp under one production image-generation pipeline. No new blur experiment was run.

## 6. FDAT added value and boundary ablation

Locked formula d=max(0,p_full−p_gray),t=min(1,.5+d),predictWater iff p_full>=t. It agrees with gray-only on5738/5794 images and changes56. FDAT/AND disagreements:0; full=1 with gray<.5 cases:0. The capped >=rule is not mathematically AND at every possible saturated input; exact differences are reported rather than assumed.

Relative to gray-only, FDAT corrects40 mistakes and introduces16 new mistakes across its56 changed decisions. AA changes+0.41pp; Landbird accuracy+0.89pp; Waterbird accuracy-1.25pp. Individual group changes areG0 +0.93pp; G1 +0.84pp; G2 -1.40pp; G3 -1.09pp. Thus it helps Landbird groups/AA while harming Waterbird groups and the primary WGA compared with gray-only; no added worst-group value is demonstrated here. Pairwise transitions by group are in paired_flips.csv (reference gray_05, policy locked_FDAT), and species_summary.csv exposes species concentration. No label-selected gates, per-class thresholds, new lambda/cap or model changes were evaluated.

## 7. Practical limitations and cost

Dataset masks are oracle assets. Quantized soft alpha sometimes covers foreground interiors; blending can change contrast there. A realistic segmentation pipeline, mask-error robustness and deployment latency are untested. This diagnostic added one gray forward per source while reusing cached full scores. Deployed gray-only could replace the full classifier with one gray-view forward plus segmentation/composition; FDAT needs both full and gray classifier views, normally two forwards plus segmentation. Added full-data gray inference took155.38s of this run's loop plus preflight/audit/export overhead. ModelAdapter runtime:{'torch': '2.5.1', 'cudnn': 90100, 'cudnn_allow_tf32': False, 'matmul_allow_tf32': False, 'deterministic_algorithms': False, 'cudnn_benchmark': False}. Only one POTER checkpoint was checked; no seed-stability or POTER-specific novelty claim. Official labels and known dataset-label limitations are unchanged.

Source-image bootstrap intervals describe this fixed, repeatedly inspected test set. They do not account for algorithm ideation on prior test analyses, new environments, segmentation-model errors, or independent training randomness; species can remain dependent. No untouched holdout or unbiased deployment gain is asserted.

## 8. One next action

**STOP FDAT expansion: gray-only ties/exceeds FDAT WGA. Next validate foreground isolation with independent development/final data and realistic masks.** Do not start another search, fabricate original performance from old composite scores, invent a new adaptive branch, or automatically retrain OT/POTER. Any later contribution claim needs genuinely independent development/final data and realistic masks.

## Reproduce and artifacts

```powershell
.\.venv\Scripts\python.exe scripts\run_original_gray_feasibility.py --stage preflight
# Inspect the24 saved examples and record preflight_visual_review.json.
.\.venv\Scripts\python.exe scripts\run_original_gray_feasibility.py --stage infer
.\.venv\Scripts\python.exe scripts\run_original_gray_feasibility.py --stage report
```

The infer stage resumes verified gray-cache shards and never overwrites them. The report stage reads saved scores only. Required original_gray_pairs.csv, original_gray_metrics.csv, paired_flips.csv, bootstrap_ci.csv, decision.json, contact/group/tradeoff PNGs and manifests are present. Source/config/checkpoint/runtime/input/output hashes are retained. No earlier report/cache/checkpoint was overwritten.

Automated checks: 12 passed in 1.29s
