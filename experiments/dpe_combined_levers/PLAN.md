# Combined levers: post-hoc procedures on corrected validation labels (declared plan)

Drafted and committed 2026-11-15, **before any test result of this run was scored**.
The Waterbirds test set and these backbones were inspected in earlier stages, so the run is
exploratory evidence, as in the paper. The result is reported whatever it shows.

## Question

The ACIIDS draft measures two levers separately: post-hoc procedures on DPE heads
(+1.1 to +1.7 pp WGA on the clean test subset, official validation labels) and refitting DPE on
corrected validation labels (+2.3 pp). Do the post-hoc procedures still add WGA once the 18
validation labels are corrected?

## Run A: strictly nested pipeline on corrected validation labels

- Code: `combined_levers.py`, a wrapper around the frozen pipeline in
  `../dpe_nested_corrected/code` (protocol `strict-nested-postmean-v2`, unchanged). The only
  change is the validation label set: `relabel` = the 18 images of species 021/088/102
  relabelled landbird (group follows). Inner CV, selection and every fit use these labels.
- Sources: the five standard-ERM backbones (`normal-erm-seed0..4`); ERM* (2 backbones) only if
  time allows, reported descriptively.
- Split seeds: 0–9, the same as the official-label strict nested run. If the time budget runs
  out, only splits completed for **all five** backbones are analysed, and the number is stated.
- Official-label counterpart: `results/dpe/strict_nested_20261115` re-scored in
  `results/dpe/corrected_test_rescore_20261115/per_arm.csv` (same seeds, same code).

### Declared contrasts (clean subset, 5,704 test images)

Paired per backbone, averaged over split seeds within a backbone, mean and pointwise 95% t
interval over the five backbones, count of positive backbones.

| ID | Contrast | Role |
| --- | --- | --- |
| **C1 (primary)** | `crossfit_affine_sel_nll` − `fullval_uniform`, both fitted on corrected val labels | Does the paper's primary post-hoc procedure still help? |
| C2 | `fullval_affine_oof_sel_nll` − `fullval_uniform` (corrected) | secondary |
| C3 | `pool_linear_sel_nll` − `fullval_uniform` (corrected) | secondary |
| C4 | `pool_linear_sel_nll` − `pool_affine_sel_nll` (corrected) | head information beyond recalibration |
| I1–I3 | (C1, C2, C3 under corrected labels) − (same contrast under official labels) | interaction: do the gains add or overlap? |
| T1 | `crossfit_postmean_thrwga` − `crossfit_uniform` (corrected) | exact-threshold control |

Decision rule for C1–C4: *supported* if the mean > 0 and the interval excludes 0; otherwise not
supported. Also reported: corrected-label and official-label views, per-group accuracies, and
which group is worst.

**Prediction written before the run.** Correcting the validation labels already raises
landbird-on-water and lowers the waterbird groups (gap to waterbird-on-land 5.2 → 2.2 pp on
standard ERM, `results/dpe/label_correction_20261115`). We therefore expect C1–C3 to be smaller
than under official labels (I1–I3 < 0), i.e. the levers overlap. If the paper's account of
exact thresholds is right (they tune to the wrong worst group), T1 should be less negative
than under official labels (−0.63 pp against cross-fit uniform, official test labels).

## Run B: matched random-drop control

- Code: `scripts/analysis/dpe_label_correction.py run --conditions drop randdrop0 ... randdrop4 --no-dfr`.
- `randdropK`: remove 18 correctly labelled validation images, 11 from waterbird-on-land and
  7 from waterbird-on-water (the official groups of the 18 misannotated images), drawn with
  `RandomState(500+K)`. Five draws. Full-validation DPE, head seeds `source + 1000*s`, s=0..4,
  as in `results/dpe/label_correction_20261115`. `drop` is rerun and must reproduce that folder.
- **C5 (declared):** `drop` − mean over draws of `randdropK`, clean subset, paired per backbone,
  95% t interval over the five standard-ERM backbones. If C5 ≈ the drop gain (+2.27 pp), the
  gain comes from removing the wrong labels, not from having fewer images.

## Outputs

`results/dpe/combined_levers_20261115/` with README, per-cell JSON, summary tables.
