# DPE experiment results

The [five-seed report](five_seed_verification_20261112/README.md) is the current scientific assessment. Core per-seed results, alternative-partition results, controls, combiner repetitions and uncertainty are alongside it.

The [H2 literature note](h2_aggregation_literature.md) summarizes related work without presenting new experiment results. Earlier three-seed reports, historical tables, verification logs, and archived verification notebooks are retained in [supporting_evidence.zip](supporting_evidence.zip). Paths within the ZIP reproduce their former repository locations, and `SHA256SUMS` records each file’s hash.

The supporting archive is **not** the original uploaded experiment bundle: predictions, model checkpoints and raw data remain outside Git. Retain the original verification ZIP and Kaggle output for a complete audit or recovery.

2026-11-15 additions (validation-budget study for the ACIIDS 2027 draft):

* [Nested validation-only selection](nested_selection_20261115/README.md): 7 sources x 10
  split seeds. No cross-fit or recalibration variant beats full-validation DPE once the
  setting is chosen on validation data; DFR-lite is 1.5 pp below DPE.
* [Threshold and offset controls](threshold_controls_20261115/README.md): affine
  recalibration of the fold ensemble equals an offset-only shift; transferred thresholds hurt.
* [Species-label error and validation-label methods](label_correction_20261115/README.md):
  fixing 18 mislabelled validation images raises DPE's corrected-test WGA by +2.48 pp
  ([+1.80, +3.15], 5/5 standard-ERM sources) but lowers its official-label WGA.
* [Upstream reproduction on fresh backbones](upstream_reproduction_20261115/README.md):
  official code, seeds 5 and 6, DPE 89.41 and 86.47 vs the paper's 91.0.

[ERM* + DPE neural aggregation: two-seed report](ermstar_neural_two_seed_20261113/README.md):
all 24 test results independently replayed. Plain/dropout MLPs improve matched
uniform on both sources; logistic has the highest neural mean but mixed paired
effects. No neural mean exceeds the full-validation DPE reference. Exploratory H2
evidence only; H1/H3 on these ERM* sources are not tested.
