# Run the fully crossed POTER diagnostic

## Inputs and provenance

The benchmark cache and Waterbirds assets are read from the original `POTER` checkout. The run uses the fixed POTER seed-1 checkpoint at epoch 258, verified by SHA-256 against the recovered-run audit, and the usual Runner inference path. No checkpoint training, OT replay, threshold fitting or generation of new scene images took place. The results are `TEST-REUSED / DESCRIPTIVE ONLY`.

The [historical audit](old_two_pair_audit.md) compares the local `outputs/poter_test_background_swap/full/paired_cases.csv` and `predictions.csv` row-by-row, including their file hashes. Those files are the local equivalents of the handoff inputs. The standalone uploaded copy was not located. The available train CUB RGB subset was already assigned to the earlier train intervention or held back by that small study. The remaining available assets do not form a new verified cohort large enough for this test. The selected CUB test cutouts were never part of POTER training, OT reference or validation selection, but are previously test-exposed. The masks are hashed and sampled using seed `764901` by species round-robin within each of G0–G3, independent of model predictions. Forty were selected per group.

The `configs/poter_fully_crossed_background.json` file fixes the two visually screened background banks and all sampling seeds, tolerances and repeat counts. Candidate backgrounds are native crops from official Waterbirds train donor scenes, with source places absent from the experiment's test and validation metadata and excluded if used by previous scored banks. The experiment uses 10 land + 10 water scenes per bank. Per-scene luminance, saturation, RGB summaries and coarse color histograms are in [scene property audit](scene_property_audit.csv).

## Commands

Run from the current worktree in PowerShell. The published folder is locked against parameter changes. For a reproduction in a new folder, use a fresh output path in all four commands and reproduce the same visual QA record before scoring. The compiler accepts `--device cuda:0` by default on this machine.

```powershell
$source = 'C:/Users/ADMIN/OneDrive - KonTum01/Desktop/POTER'
$python = "$source/.venv/Scripts/python.exe"
$out = 'outputs/poter_fully_crossed_background'
& $python scripts/prepare_incremental_backgrounds.py --archive "$source/outputs/vulnerability_ot_incremental/assets/segmentations.tgz" --cohort "$source/outputs/vulnerability_ot_incremental/cohort_manifest.csv" --data-root "$source/data/waterbirds" --output "$out/assets/candidates" --per-scene 36 --crop-size 224
& $python scripts/run_poter_fully_crossed_background.py preflight --source-root $source --output $out
```

Inspect `mask_contact_sheet.png` and confirm the original parity checks. For the approved sheet, record `status` as `PASS` and its SHA-256 as `contact_sheet_sha256` in `preflight_review.json`. Then run these commands.

```powershell
& $python scripts/run_poter_fully_crossed_background.py pilot --source-root $source --output $out
& $python scripts/run_poter_fully_crossed_background.py full --source-root $source --output $out
& $python scripts/analyze_poter_fully_crossed_background.py --output $out
```

## Runtime and cache

Windows, RTX 4050 Laptop GPU, Python 3.10.22, Torch 2.5.1, CUDA 12.1, torchvision 0.20.1, NumPy 1.26.4, pandas 2.3.3, Pillow 12.3.0, SciPy 1.15.3, and Matplotlib 3.10.9. The pilot took 15.5 seconds for 960 views. The full stage took 60.8 seconds and reused all 960 pilot predictions, adding 5,440 views. Timings include reopening assets and loading the adapter. The 24-original parity rerun is recorded separately. Default argmax was unchanged. The hash-bound run lock covers code, configuration, manifests and checkpoint. The prediction cache is organized by stable foreground ID and validates scene coverage and checkpoint identity on resume.

[Final report](final_report.md) · [Run lock](run_lock.json) · [Full predictions](fully_crossed_predictions.csv) · [Evidence hashes](file_hashes.json)
