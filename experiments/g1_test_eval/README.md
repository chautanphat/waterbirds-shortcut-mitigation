# G1 geometry weighting on full-validation DPE: Waterbirds test set

Status: **built and software-tested; not yet run on Waterbirds.**

## Question

G1 (teammate study, branch `dpe_optimized_version_5`, `results/dpe/geometry_aggregation_v5_20261115`)
reweights DPE's 15 heads per image by closeness to their nearest prototype:
`w_j = 0.75/15 + 0.25 * softmax_j(-50 * min_k d_jk)`. It was evaluated only on held-out folds of the
validation set, with one ERM* backbone (+0.50 pp WGA, 95% CI [-0.21, +1.25]). G1 has no fitted
parameters, so it can be applied directly to **full-validation DPE**. This run asks whether it improves
DPE on the **test set** across backbones, against controls.

## Protocol (declared in `g1_eval.py` before running)

* 7 saved backbones (5 standard ERM, 2 ERM*) x 10 split seeds. Full-validation DPE is retrained with
  exactly the seeds of the strictly nested run; the uniform arm must reproduce its recorded test WGA
  (`g1_expected.json`), which makes G1 paired with our paper's baseline.
* Arms on the same heads: `uniform`, `g1` (locked gamma 50, alpha 0.25), `g1_a0.1` and `g1_a0.5`
  (sensitivity, never selected), `perm` (G1 weights permuted across heads per image, 5 permutations),
  `conf` (confidence routing at the same mixture).
* Primary: `g1 - uniform`, **official test labels**, standard ERM, paired per cell, mean over splits per
  backbone, 95% t interval over 5 backbones. **PASS** if mean >= +0.30 pp, interval above 0, >= 4/5
  backbones positive, and `g1 - perm` interval above 0. ERM* is descriptive. Corrected labels and the
  clean subset are logged as secondary columns; the strict run's validation-selected offset on the same
  heads is reported as context.
* The test set and backbones were inspected in earlier work, so a PASS is exploratory evidence that
  justifies a confirmation (fresh backbones, second dataset), not a final claim.

## How to run (Kaggle, CPU)

1. Kaggle -> Create -> New Notebook -> File -> Import Notebook -> `artifacts/g1_test_eval_kaggle.ipynb`.
2. Settings: **Internet On**, CPU runtime.
3. **Save Version -> Save & Run All.** Expected 20-40 minutes: preflight tests, feature recovery
   (the same public outputs as `noerula/dpe-test`), 70 full-validation ensembles, report.
4. Download `g1_test_eval_results.zip` (REPORT.md, summary.json, per-cell JSON, and per-cell
   `q1`/`dmin` caches that allow further CPU analysis without retraining).

If cells fail, results so far are packaged; re-running resumes and skips finished cells.

## Files

* `g1_eval.py`: the experiment (run / summarize). `g1_selftest.py`: checks that G1 matches the
  teammate's verified recipe to 1e-12, permutation-control properties, and a synthetic end-to-end smoke
  run through both real head trainers.
* `build_notebook.py`: builds `g1_expected.json` from `results/dpe/strict_nested_20261115` and
  `results/dpe/corrected_test_rescore_20261115`, bundles the frozen code of
  `experiments/dpe_nested_corrected/code` with these files, and writes the notebook.
* `artifacts/`: notebook, code ZIP (bundle SHA-256 in `artifact_manifest.json`).

Local checks done: unit tests and synthetic smoke (both trainers); the three test views on real
metadata (5,794 / 5,794 / 5,704 images, 90 affected); notebook bundle extracted, hash-verified and its
tests re-run; summarizer and verdict logic on fabricated cells. Not tested here: real feature recovery,
exact reproduction of the recorded heads, run time.
