# Counterfactual implementation and executed validation

Validated locally on 18 November 2026. Both applications are implemented in this checkout and use real saved POTER inference. No training, model selection using test labels, generative editing, or fabricated application predictions were performed.

## Launch

From the repository root in PowerShell, using the local environment created for this task:

```powershell
.\.venv\Scripts\python.exe scripts/counterfactual_lab.py --device cuda
.\.venv\Scripts\python.exe scripts/failure_case_explorer.py --no-model
```

The applications use [Lab on port 8786](http://127.0.0.1:8786) and [Explorer on port 8787](http://127.0.0.1:8787). Existing POTER Runner on port 8766 was preserved. A temporary Mixer smoke-test server was stopped after verification. CPU inference is available with `--device cpu`.

## Dataset and checkpoint

- Downloaded the pinned `arubique/waterbirds` dataset with the official Hugging Face client. Its twelve Parquet bundles were materialized into the exact original image-folder JPEGs.
- Verified 17,385 source files, including 17,382 images: 5,794 scenes, foreground-only images and backgrounds each. Canonical metadata has 11,788 rows; all supplied images join to the 5,794 official test birds. No train/validation image availability was invented.
- Prepared **5,794 original CUB RGB + official-mask cutouts**, with 2,897 Land and 2,897 Water backgrounds. Full metadata join, no corrupt/quarantined images, no cross-split Places-filename conflicts. Two exact duplicate background sets are excluded from automatic sampling; 63 average-hash collision sets are flagged for review.
- Both official CUB archives passed published checksums. Offline validation preserved selected original metadata, RGBA and mask SHA-256s. A complete materialized snapshot hash inventory permits reuse without network downloads; redundant Parquet bundles were pruned.
- Used `checkpoints/poter_kaggle_recovered/CUB/Clean/POTER_CUB_seed1_kaggle/best_validation.pt`, actual epoch **258**, SHA-256 `4f405e0e9ec34420aeae5fc2452635c6bc700d714fd033c4c289bea9d2434a9e`.
- Actual class order is `[landbird, waterbird]`. Image processing is square 256 resize, center 224 crop, RGB float tensor and ImageNet normalization. Exact model-view pixels and normalized tensor hashes are exported.

Detailed source terms, pinned commits, categories and label-encoding audit: [DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md). Machine-readable audit is `data/waterbirds/dataset_audit.json`.

## Model parity

Three independent checks were executed:

| Check | Observed maximum error |
|---|---:|
| Frozen pre-refactor Runner loading/transforms vs shared adapter, same GPU batch | 0 logits |
| GPU batch size 1 vs 2 | 0.000052572 logits; 0.000013530 probabilities |
| Repeated GPU inference, same batch | 0 probabilities |
| Already-running Runner vs adapter on its CPU device | 0.000000014901 probabilities |
| Existing CPU Runner vs CUDA adapter | 0.000041559 probabilities |

Model-view pixels matched exactly. Same-batch reference tolerance is `atol=2e-5, rtol=2e-5` for logits. Cross-batch GPU tolerance is `atol=1e-4` logits / `2e-5` probabilities. The live same-device comparison uses probability `atol=2e-6`; cross-device comparison uses `1e-4`. Hardware/library rounding is recorded separately from preprocessing parity.

Reports: `experiments/counterfactual_verification/real_verification.json` and `live_runner_parity.json`.

## Saved experiments and exports

| Run | What was verified |
|---|---|
| `20261118T123846Z_1f31a11c5f` | Browser-driven known Waterbird #1; eight natural Gaussian variants, six diagnostic controls and one separate original-scene baseline; persisted reload and direct Explorer link. |
| `20261118T123059Z_48f0272960` | Fixed first eight official test IDs per canonical class, two backgrounds per category, four Gaussian levels. **400 real predictions:** 256 natural variants, 64 background-only controls, 64 neutral controls and 16 originals. |
| `20261118T125909Z_f754cb571a` | Separate real contrast run: factors 1, .75, .4, .1; 15 total rows. |
| `20261118T125922Z_7cc3576177` | Separate real partial-occlusion run: fractions 0, .1, .3, .5; 15 total rows. |

The 16-bird run contains **80 natural-image failures, one original-scene failure and 10 matched-pair flips**. These are engineering/demo observations from the entire saved cohort, not a representative benchmark or independent causal confirmation. Every row, including correct predictions and controls, is retained. Gallery filters do not change summary denominators. Original-scene accuracy is evaluated separately from intervention accuracy; no-bird and neutral controls have no accuracy/WGA.

The case packet exported through the browser for `hf-fg-190` is under:

```text
experiments/counterfactual_lab/20261118T123059Z_48f0272960/exports/case_hf-fg-190_ef892185/
```

Its PNG metadata, CSV and JSON contain the same **25 ordered row IDs and probabilities**. The montage is 1120 × 2510; PDF and caption are also present. `export_reconciliation.json` records this check. Complete run ZIPs retain all raw PNGs, exact model-view PNGs, config, source/mask assets, predictions, pair metrics and summaries. SHA-256 integrity checks reject changed core files and image assets.

Browser screenshots are in `experiments/counterfactual_verification/`: `lab_browser.jpg`, `failure_explorer_browser.jpg` and `mixer_browser.jpg`. The Mixer imported two real local images, produced an enabled PNG export and reported no browser errors. Both new interfaces were checked at the available narrow and desktop layouts; a horizontal overflow was fixed. No browser errors remained.

## Automated validation

- Windows-applicable repository suite: **241 passed, 9 skipped, 5 deselected, 27 subtests passed**. The five exclusions are existing Linux orchestration/notebook fixtures listed below.
- Focused app acceptance suite: **15 passed**, including real checkpoint inference, live Runner HTTP contract, independent frozen preprocessing reference, single/batch repeatability, three separate foreground operators, mask identity, negative controls, unknown external labels, split checks, WGA/paired metrics, immutable exports, and tamper detection.
- Mixer Node checks: **8 passed**. Lab control-serialization checks passed for primary presets, separate operators, custom physical units and balanced quotas. JavaScript syntax and Python compilation checks passed.
- Offline data validation, materialized snapshot hash-cache verification and PNG/CSV/JSON export reconciliation passed. `git diff --check` passed.

The unfiltered repository suite exposed pre-existing Windows incompatibilities: four tests patch nonexistent `os.killpg`, and one notebook fixture inserts Windows paths into Unix Python string literals. Baseline method code was preserved. These exact tests were excluded from the Windows-applicable run:

```text
tests/methods/dpe_ermstar/test_ermstar.py::Tests::test_dual_gpu_isolation
tests/methods/dpe_ermstar/test_ermstar.py::Tests::test_failed_worker_retains_other_seed
tests/methods/dpe_ermstar/test_ermstar.py::Tests::test_interrupt_terminates_both_worker_groups
tests/methods/dpe_ermstar/test_ermstar.py::Tests::test_timeout_preserves_recovery_status
tests/methods/dpe_ermstar/test_ermstar.py::Tests::test_actual_notebook_cells_execute_with_synthetic_workers
```

Commands and complete output are retained in `experiments/counterfactual_verification/final_pytest.log` and `final_app_tests.log`. The tests can be reproduced with:

```powershell
$env:PYTHONUTF8="1"
$env:RUN_REAL_INFERENCE="1"
python -m unittest discover -s tests -p test_counterfactual.py -v
node tests/test_mixer_core.cjs
node tests/test_lab_controls.cjs
```

## Files created or modified

Existing files modified:

```text
.gitignore
README.md
scripts/poter_preview.py
scripts/image_mixer.html  (already present but untracked when work began)
```

New shared services and interfaces:

```text
src/counterfactual/__init__.py
src/counterfactual/common.py
src/counterfactual/compositor.py
src/counterfactual/data.py
src/counterfactual/downloads.py
src/counterfactual/experiment.py
src/counterfactual/importers.py
src/counterfactual/metrics.py
src/counterfactual/model.py
src/counterfactual/server.py
src/counterfactual/web/app.css
src/counterfactual/web/common.js
src/counterfactual/web/lab.html
src/counterfactual/web/lab.js
src/counterfactual/web/explorer.html
src/counterfactual/web/explorer.js
```

New entrypoints, utilities, configurations and tests:

```text
scripts/counterfactual_lab.py
scripts/failure_case_explorer.py
scripts/setup_waterbirds_data.py
scripts/run_counterfactual_experiment.py
scripts/import_counterfactual_assets.py
scripts/verify_counterfactual.py
scripts/mixer_core.js
configs/counterfactual_lab/waterbird_demo.json
configs/counterfactual_lab/balanced_16_birds.json
configs/counterfactual_lab/contrast_demo.json
configs/counterfactual_lab/occlusion_demo.json
tests/test_counterfactual.py
tests/test_mixer_core.cjs
tests/test_lab_controls.cjs
requirements-counterfactual.txt
requirements-counterfactual-dev.txt
```

New documentation:

```text
INTEGRATION_MAP.md
DATASET_PROVENANCE.md
README_COUNTERFACTUAL_LAB.md
README_FAILURE_EXPLORER.md
KNOWN_LIMITATIONS.md
docs/counterfactual_validation.md
```

All large datasets, checkpoints, generated runs and validation artifacts remain local and gitignored. No files under `baselines/poter/` were modified. Other pre-existing uncommitted work was preserved.

## Remaining limitations

No compatible hash-bound original-DPE head/backbone pair is available in this checkout; its explicit adapter and setup errors are implemented, but no DPE predictions are claimed. The derivative dataset supplies only official test images. Content screening for humans/animals is not automated, and near-duplicate candidates need review. Legacy CSV imports are audited standalone artifacts when exact model-view assets/pairing are absent. Resume creates a new immutable run rather than continuing a partial GPU batch. See [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md) for details.
