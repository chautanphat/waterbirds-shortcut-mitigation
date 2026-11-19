# Failure Case Explorer

## Original benchmark gray feasibility: completed

`outputs/original_gray_feasibility/algorithm_feasibility_report.md` completes the original5794
clean gate with new benchmark-RGB gray inputs, not the old CUB-RGB controls. Fixed POTER
WGA: full91.04%, gray91.59%, locked FDAT90.34%; both gray/FDAT pass the observed−2pp
gate, but gray-only has higher WGA. The recommendation is to stop FDAT expansion and
check foreground isolation on independent data/realistic masks. This remains exploratory,
one-checkpoint evidence. No training or threshold tuning was performed.

## Constrained threshold and FDAT feasibility

`outputs/threshold_algorithm_feasibility/algorithm_feasibility_report.md` records the
executed score-only constrained-oracle, paired-gray FDAT and five species-blocked folds.
Verdict: **INCONCLUSIVE**. POTER sigma2 OOF WGA improves58→86%, but gray-only also yields86%;
full-original FDAT protection is BLOCKED because cached gray foreground pixels do not match originals.
No training/inference/segmentation was started. Read `handoff_to_chatgpt.md` for the one-page transfer summary.

## Separate test-oracle threshold feasibility probe

`outputs/test_oracle_threshold_sweep/report.md` is explicitly **TEST ORACLE / IN-SAMPLE UPPER BOUND**:
its fitted thresholds use already-inspected test labels and are not deployable validation-selected baselines.
The CPU-only script `scripts/run_test_oracle_threshold_sweep.py` reads cached probabilities,
checks per-severity/fixed/global thresholds and full-original clean trade-offs, and loads no model.
It leaves the validation-only threshold report and its selection guards unchanged.

## Fast threshold diagnostic (no retraining)

`outputs/poter_threshold_probe/report.md` contains executed validation inference,
class-wise diagnostics and a validation-selected ERM threshold baseline.
POTER and legacy DPE threshold selection are blocked: canonical validation overlaps
POTER's OT reference or DPE head fitting, leaving no independent four-group development pool.
Their fixed0.5 results remain descriptive. Test comparisons are exploratory.
Regenerate analysis with `.\.venv\Scripts\python.exe scripts\run_threshold_probe.py --stage analysis`;
this stage loads saved predictions and does not train or run model inference.

## Paired background failure verification

The executed analysis and all ten plots are in `outputs/failure_verification/final_report.md`.
Experiment 1 uses cached real predictions, 7,200 paired interventions and 2,000 bootstrap draws.
Experiment 2 has one available training seed per method and remains incomplete; no additional training was started.
Training requires notifying the user and waiting for confirmation. This analysis entrypoint never trains or runs model inference:

```powershell
.\.venv\Scripts\python.exe scripts\run_failure_verification.py
# Refresh reports and plots from verified saved statistical tables only:
.\.venv\Scripts\python.exe scripts\run_failure_verification.py --stage report
```

## Saved POTER / ERM / DPE comparison

```powershell
.\.venv\Scripts\python.exe scripts\failure_case_explorer.py --comparison-root results/poter_failure --port 8788
```

Open [the three-model saved-results gallery](http://127.0.0.1:8788). Choose the 100-foreground Gaussian experiment or the full 5,794-image original test. Filter failures by model, true class, severity, background, source ID/species, model disagreement, or clean-correct/degraded-wrong cases. Selecting a case shows the original source, segmentation overlay, exact shared 224-pixel model input, all three real predictions, and the matched severity/background variants. CSV downloads retain the original saved result fields.

This mode loads no model/checkpoint and performs no inference. Images are reconstructed on demand with the existing compositor; raw and model-view pixel hashes must match the saved inference CSV before an image is served. No new PNG cache is written. ERM is enhanced ERM, seeds are unmatched, and the legacy DPE head/backbone historical hash binding is absent; this remains the exploratory comparison documented in `results/poter_failure/summary.md`. Background-only images have no semantic accuracy.

For the paired mild-blur v2 results, use `--comparison-root outputs/mild_blur_v2 --port 8789` and open [the mild-blur gallery](http://127.0.0.1:8789). This cohort contains 100 original source images, not the complete 5,794-image test. The severity selector uses the actual saved sigma values; sigma4 is marked supplementary.

## Earlier native Lab runs

```powershell
python scripts/failure_case_explorer.py --no-model
```

Open [Failure Case Explorer](http://127.0.0.1:8787), or use **Open in Failure Explorer** in Lab (the same server also serves `/explorer`). Both entrypoints read `experiments/counterfactual_lab/` by default, including runs saved before restart. Browsing saved results requires no inference. To run experiments through its Lab link, omit `--no-model` and supply the checkpoint/device options documented for Lab.

Choose a run, then search IDs/species/checkpoint and filter model, label, original group, background, severity, predicted class, correctness, paired flips, confidence, margin gap and provenance quality. Rank by high-confidence failures, aligned-correct/conflicting-wrong, clean-correct/degraded-wrong, matched margin gap or actual model disagreement. The default gallery shows known-label natural-image failures; confidence alone never defines failure. All rows and all OOD controls remain available through the case-type filter.

Select a row for source, mask overlay, natural Land/Water composites, exact model-seen 224 RGB inputs, four-severity comparisons, probability curves, pair metrics, row provenance, synchronized image zoom/scroll and an explicitly labeled absolute pixel-difference heatmap. The heatmap is not saliency. Missing/not-run results are marked unavailable. If DPE is not configured, the interface states this and does not invent a comparison.

Dataset-level tables always use the complete saved experiment; gallery filtering never changes denominators. Per-severity group counts, confusion matrices, paired margins, flip counts and clean-to-degraded transitions are retained. WGA requires all four target/background groups and is undefined for a one-class demo. Confidence intervals resample source birds after averaging their repeated background pairs. One bird has no CI, and a single fixed model seed has no independent seed dispersion. Test analyses are retrospective exploratory. Neutral/no-bird controls are kept out of all accuracy/failure denominators.

**Export complete run ZIP** includes all assets and config. **Export filtered CSV** identifies a reviewed subset without altering aggregate results. **Export PNG / PDF + CSV / JSON packet** exports every saved row for the selected bird; packet row IDs reconcile exactly to predictions and manifest. Captions list source ID, quality, checkpoint hashes and run ID. Two independent reviewer identities and an adjudicator can append tags and notes; these never change ground truth or model predictions.

Optional existing-prediction import:

```powershell
python scripts/import_counterfactual_assets.py predictions --manifest existing_predictions.csv --mapping column_mapping.json --output experiments/imported_predictions/run_name
```

The mapping explicitly maps standardized keys to source CSV columns and requires `image_path`, `pred_y`, both probabilities, checkpoint SHA and preprocessing fingerprint; optional `true_y` is validated against canonical metadata. Import rejects unknown/duplicate IDs, invalid probabilities, inconsistent labels and forged-format hashes. It preserves source CSV/mapping/metadata hashes. Imported files are currently audited standalone artifacts; Lab's paired gallery only accepts native complete runs with source assets. Absent model-view pixels and paired counterfactuals are not reconstructed from probabilities.
