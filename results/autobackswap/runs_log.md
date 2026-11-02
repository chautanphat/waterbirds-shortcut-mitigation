# Run log

One entry per run (mirrors `runs_log.csv`). Never delete failed runs.

## Manual changes made during setup
| Date | Change | Reason |
|---|---|---|
| 2026-10-28 | `brew install python@3.12` (3.12.16); venv at `env/venv` | No Python 3.12 on the machine |
| 2026-10-28 | Installed torch==2.11.0 / torchvision==0.26.0 from PyPI (macOS arm64, MPS) instead of +cu126 | Apple M1, no CUDA GPU; user chose Metal |
| 2026-10-28 | Planned: `--device=mps --cuda_optimizations=false` instead of `--device=cuda:0` | README says false disables autocast(bf16) and torch.compile; deviation to record per run |
| 2026-10-28 | Repo cloned at commit b060532 (AOT-GAN submodule 2cd1afd) | Step 2 |

## Runs
<!-- Template per run:
### <run_label> — <status: success | failed | partial>
- Start / end:
- Command:
- Mode / dataset / detector dataset / use_minority / seed:
- Model class / CUDA optimizations:
- Commit / torch / torchvision / GPU:
- Results dir / test_accuracies.txt:
- WGA / Acc:
- Deviations from README:
- Error (if any):
-->

## Data and weights (Kaggle, 2026-10-31)
| Item | Source | Size / checksum | Note |
|---|---|---|---|
| AOT-GAN `G0000000.pt` | Google Drive folder from repo README (only this file) | 60829150 bytes, sha256 `9c30e3b979b69e46ca80482f8a75f37fa2d62e97c678e7c7b0f12775e75d9e8e` (read from notebook output) | placed at `repo/AOT-GAN-for-Inpainting/` |
| Waterbirds main data | WILDS `waterbirds_v1.0.tar.gz`, URL in the WILDS source (`worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/`) | 489289072 bytes, sha256 `395444bbd7ca02036173877d72f9e818031e577d44b79e0fda7794c5e0072cc4` | **Deviation:** the README link (nlp.stanford.edu) refused connections from Kaggle and from the user's Mac. WILDS is the alternative listed in the official group_DRO README. Its certificate had expired, so it was fetched with `curl -k` (user approved; certificate not verified; no official checksum to compare). Archive was only listed and extracted, nothing executed. |
| Waterbirds segmentations | data.caltech.edu record w9d68-gec53, `segmentations.tgz` | 39272883 bytes, sha256 `dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f` | extracted to `segmentations/` |

Layout built under `data/waterbird_complete95_forest2water2/`: `images/<class>/*.jpg` (11788), `segmentations/<class>/*.png` (11788), `metadata.csv`, `RELEASE_v1.0.txt`. No `masks.csv` (only used by the non-segmentation detector dataset).
Checked: metadata splits train/val/test = 4795/1199/5794. Training groups (y, place): (0,0) 3498, (0,1) 184, (1,0) 56, (1,1) 1057, matching the paper.

## Smoke test and environment changes (Kaggle T4, 2026-10-31)
| Change / finding | Reason |
|---|---|
| `MPLBACKEND=Agg` on run.py calls | Kaggle sets `MPLBACKEND=module://matplotlib_inline.backend_inline`, invalid in the venv (first attempt crashed on import) |
| symlinks `images/` and `segmentations/` into `repo/data/waterbird_complete95_forest2water2/` | the repo ships `metadata.csv`, `masks.csv`, `readme.txt` there; WILDS images live outside the repo. Repo `metadata.csv` is byte-identical to the WILDS one |
| deleted `repo/methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0` (twice, explicit path) | the detector ignores `--results_dir` and writes there with `exist_ok=False`, so smoke-test output must be removed before the real run (remove it again before each new seed/setting rerun) |

### smoke — success (2-epoch detector, T4)
- Data loaded: detector train 640, valid 160 (from `detector_n_samples=200` in `config_wb.yaml`; paper text says 287 hand-labeled samples; the README command uses the segmentation-derived dataset, so 200 is what runs).
- About 6.6-7.8 s/epoch, so 200 epochs is about 22-26 min. Likely CPU-bound: Kaggle has 4 cores, the DataLoader asks for 8 workers.
- Only the detector was tested. Generation and classifier not yet run.

### T4 precision measurement (4096x4096 matmul)
| dtype | ms |
|---|---|
| float32 | 34.7 |
| float16 | 4.86 |
| bfloat16 | 63.64 |
`torch.cuda.is_bf16_supported()` is True but bf16 is emulated on the T4, so it is about 1.8x slower than fp32. The repo default (`--cuda_optimizations=true`, bf16 autocast) is therefore the slowest choice on T4. Options: fp32 via `--cuda_optimizations=false` (README switch, no code change) or an fp16 autocast patch (code change, needs GradScaler). Not decided yet.

## Waterbirds w/o minority, seed 0 (Kaggle T4)
- Decision (user, 2026-10-31): follow the README standard settings from now on; speed is not a goal. Earlier fp32 choice (`--cuda_optimizations=false`) dropped.
- Detector: success, 200 epochs, about 19 min, best val_acc 91.09% (detector balanced accuracy, not the paper metrics).
- Generate in fp32: failed, CUDA OOM (see csv). `augment_batch_size` has no effect for Waterbirds: `generate_waterbirds` sends a whole class folder (about 60 images) as one batch, so batch 8 would not help.
- Generate with defaults (bf16 autocast + torch.compile): relaunched. bf16 is emulated on T4, expect it to be slow.
- Note: an input mistake in the Kaggle editor (typed text sent as notebook shortcuts) created a stray Markdown cell and empty cells; the stray cell was deleted, no data or process was affected.
- README quirk: `--run_label="ours_minority_$WB_USE_MINORITY_seed0"` expands to `ours_minority_` in bash; the explicit label `ours_minority_false_seed0` is used.
- Generate with README defaults (bf16 + torch.compile): failed, CUDA OOM on the 15 GB T4 (tried 3.75 GiB, 3.48 GiB free). Same cause as fp32: a whole class folder is one batch. No further code or flag change made; waiting for the user's decision. Empty notebook cells were deleted.

### Deviation: chunked generate (user-approved, 2026-10-31)
- Reason: README-default generate OOMs on the 15 GB T4 because `generate_waterbirds` sends a whole class folder as one batch.
- Change: `repo/methods/ours/augment_data.py` `generate_waterbirds` now calls `separator.separate` in chunks of `augment_batch_size` (16) and concatenates the outputs. Everything else (detector, thresholds, AOT-GAN, bf16 autocast, torch.compile) is unchanged. Each image is processed independently, so outputs should match the unchunked run apart from bf16 numerics. Patch: `patches/augment_waterbirds_chunk.patch`, applied on Kaggle with `git apply`.
- Relaunched as pid 13488 (generate, then classifier), log `logs/ours_false_seed0_std.log`.

### Switch back to fp32 + chunked generate (user decision, 2026-10-31)
- User asked to return to the faster fp32 option because bf16 generate was too slow (~25 s per folder, 15/200 after 6.7 min, about 80 min projected; no OOM at chunk 16, 5 GB used).
- Stopped the bf16 chain (pid 13488, partial, abs_seed0 deleted by explicit path) and relaunched as pid 13552 with `--cuda_optimizations=false` for generate and classifier, plus the chunk-16 patch. Log: `logs/ours_false_seed0_fp32.log`.
- This supersedes the earlier "README standard settings" decision. Deviations for report.md: fp32 instead of bf16 autocast + torch.compile; chunked generate; detector_n_samples 200.
- First fp32 folder: ~9.8 s. GPU memory showed 14.8 GB used at the start, so OOM is still possible.
- fp32 chunked generate finished without OOM (about 32 min). Classifier (ours, fp32) started at 1793414185-ish: epoch 1/50 took 62 s (about 52 min for 50 epochs), best val_acc 89.67% after epoch 1, 3.5 GB GPU. DataLoader warns 8 workers vs 4 CPUs (config default, not changed).

### Kaggle session lost mid-classifier; classifier rerun (2026-10-31)
- The Kaggle draft session shut down (the last readable poll was epoch 2; when I reopened, the session was "off" and had to be restarted). /kaggle/working survived (8.2 GiB). The classifier process died at epoch 42/50 (log `logs/ours_false_seed0_fp32.log` ends at 42/50, no END line, best val_acc 96.67%).
- The repo has no resume: test accuracy is computed only after all 50 epochs, so the partial run produced no WGA/Acc. Not evaluating the epoch-42 checkpoint, since that is not the specified 50-epoch run.
- Partial dir kept (renamed, not deleted): `results/ours_minority_false_seed0_waterbird_wom_resnet_seed0_partial_epoch42` (contains checkpoint.pt, config.yaml).
- Generated data (200 inpainted folders) and detector checkpoint reused. Classifier relaunched alone as pid 137, fp32, log `logs/ours_false_seed0_cls2.log`.
- Classifier relaunch pid 137 failed at once: `env/venv/bin/python: Permission denied`. After the session restart the venv's `bin/python` was a 0-byte regular file (its symlink target, the uv-managed Python 3.12 outside /kaggle/working, was gone; system python is 3.13.15). /kaggle/working itself was intact.
- Fix (environment only, no code change): removed `env/venv` (explicit path), recreated with `uv venv --python 3.12` using `UV_PYTHON_INSTALL_DIR=/kaggle/working/uvpy`, reinstalled the same pins (torch 2.11.0+cu126, torchvision 0.26.0, repo/requirements.txt), then classifier chained after it. Logs: `logs/venv_rebuild.log`, `logs/ours_false_seed0_cls2.log`. Chain pid 166.
- Keep-alive: Mac kept awake via the app, and a session cron (every 7 min) polls the Kaggle notebook read-only.
- Venv rebuilt OK (VENV_DONE; torch 2.11.0+cu126, torchvision 0.26.0, CUDA True). Next failure: `FileNotFoundError .../images/168.Kentucky_Warbler/...`. In `repo/data/waterbird_complete95_forest2water2/` both `images` and `segmentations` links were gone (not link, not file, not dir) after the session restart. Recreated with `os.symlink` to `/kaggle/working/AutoBackSwap-repro/data/waterbird_complete95_forest2water2/...`, checked a sample image exists.
- Next failure: `FileExistsError` because the failed launch had already created the results dir (only config.yaml). Renamed (not deleted) to `results/..._failed_nofiles`.
- Classifier running as pid 367, log `logs/ours_false_seed0_cls2.log`: epoch 1 took 51 s, best val_acc 89.67%, about 42 min for 50 epochs. Keep-alive cron re-created (every 7 min).
- Lesson: after a Kaggle session restart, symlinks (venv python, data links) are not restored; check them before launching.

### RESULT: AutoBackSwap, Waterbirds without minority, seed 0 (Kaggle T4, fp32, chunked generate)
Source: `results/ours_minority_false_seed0_waterbird_wom_resnet_seed0/test_accuracies.txt` (50/50 epochs, END reached, classifier about 46 min, log `logs/ours_false_seed0_cls2.log`).
| group | acc % |
|---|---|
| Landbird on Land | 96.674 |
| Waterbird on Land | 92.195 |
| Landbird on Water | 92.368 |
| Waterbird on Water | 92.835 |
Worst Group Accuracy 92.195, Overall Accuracy 94.028, Majority 95.823, Minority 92.233.
Paper (5 seeds, without minority): WGA 92.2±0.4, Acc 95.1±0.3.
- WGA: 92.195 vs 92.2±0.4, within 2σ.
- Acc: 94.028 vs 95.1±0.3, difference 1.07 = 3.6σ, outside 2σ: does not reproduce (single seed vs paper mean; cause not verified).
- Deviations in this run: fp32 instead of bf16 + torch.compile; chunked generate (16); detector_n_samples 200 (paper: 287 hand-labeled); WILDS data source; Kaggle session restart mid-run (classifier rerun from scratch, same seed).

### Correction to the RESULT table above (2026-10-31)
The repo's `test_accuracies.txt` swaps the labels of the two minority rows. By group size and the per-example CSV: 92.195 % is **Landbird on Water** (group 2, n=2255, 176 errors), 92.368 % is **Waterbird on Land** (group 3, n=642, 49 errors). The worst group is therefore Landbird on Water. WGA 92.195 and Overall 94.028 are unchanged.

### Step 6: misclassified-examples artifact
- `analysis/dump_predictions.py` run on Kaggle with the best-val `checkpoint.pt` (fp32). It reproduced test_accuracies.txt exactly (WGA 92.195, overall 94.028, 346 errors of 5794).
- Output `artifacts/misclassified/ours_minority_false_seed0/` (errors.csv, index.md, index.html, summary.md; no images). Pushed from Kaggle as commit e9bb814 (text files only, staged names checked for images/weights), pulled into this Mac checkout with `git pull --rebase`.
- Local viewing of error images (user request, 2026-10-31): zipped the 346 misclassified test images plus index.html/index.md/errors.csv/summary.md into `/kaggle/working/error_images_view/ours_minority_false_seed0_errors.zip` (13,433,106 bytes) on Kaggle, outside the git folder. Not pushed anywhere; the user downloads it themself from the Kaggle Output panel.

## 2026-10-31: AutoBackSwap Waterbirds with minority, seed 0 (classifier only)
- User chose this run ("Chạy ABS có minority"). Launched `run.py --mode=ours --waterbirds_use_minority=true --random_seed=0 --cuda_optimizations=false --run_label=ours_minority_true_seed0 --results_dir=/kaggle/working/AutoBackSwap-repro/results`, log `logs/ours_true_seed0_cls.log` (new file), expected results dir `results/ours_minority_true_seed0_waterbird_wm_resnet_seed0`.
- Reuses the seed-0 detector (`methods/ours/detector/results_waterbird_waterbirds_from_segmentation_seed0`) and generated `abs_seed0` data from the without-minority run. Basis: `grep use_minority` shows the flag is read only in `ours.py` (training groups), `common_utils.load_data`, `get_run_name_and_path` and the baselines; detector and `augment_data.py` do not read it, and `generate_waterbirds` processes every image. Not verified by regenerating; if the paper's with-minority pipeline retrains the detector differently, this is a deviation.
- Session restarted again: venv `bin/python` was a 0-byte file (uvpy dir also gone). Relinked `env/venv/bin/{python,python3,python3.12}` to `/usr/bin/python3.12` (3.12.3, matches `pyvenv.cfg`), site-packages intact (torch 2.11.0+cu126, torchvision 0.26.0, CUDA True). Recreated `images`/`segmentations` symlinks in `repo/data/waterbird_complete95_forest2water2/`.
- Started about 10:53 UTC (Kaggle clock), pid 164, about 57 s/epoch (first epoch), 50 epochs.

## Result: AutoBackSwap Waterbirds with minority, seed 0 (classifier only)
Source: `results/ours_minority_true_seed0_waterbird_wm_resnet_seed0/test_accuracies.txt` on Kaggle (50/50 epochs, about 49.6 min at 59.5 s/epoch, best val_acc 95.41%, log `logs/ours_true_seed0_cls.log`). Values read from the notebook output; the text file is not yet copied to the Mac or pushed.
- Printed rows (repo labels; rows 2 and 3 are swapped, see earlier note): 98.049, 94.457, 92.835, 92.368. By metadata groups: Landbird on Land 98.049, Landbird on Water 94.457, Waterbird on Land 92.835, Waterbird on Water 92.368.
- Majority 96.790, Minority 94.097, Overall (Acc) 95.444, WGA 92.368 (Waterbird on Water).
- Paper (5 seeds): WGA 93.0 +- 0.9, Acc 95.0 +- 0.5. WGA diff 0.632 (0.7 sigma), Acc diff 0.444 (0.9 sigma): both within 2 sigma, single seed.
- Detector and generated data reused from the without-minority run (assumption stated above, not verified by regenerating).
- User re-ran the launch cell once during the run; its `assert not os.path.exists(L)` stopped it, no second process.

## 2026-10-31 ERM baselines, seed 0 (user said "có chạy tiếp, có push")
- Pushed 19 local commits to origin/main first (user approved).
- Kaggle session had restarted again (kernel "died unexpectedly" on reopen, then a queue wait). This time `repo/env` was gone entirely (not just the python link). Rebuilt with `pip install uv`, `uv venv env/venv --python /usr/bin/python3.12`, torch==2.11.0+cu126 / torchvision==0.26.0+cu126 from the cu126 index plus `requirements.txt` (log `logs/venv_rebuild2.log`, ended with `2.11.0+cu126 True`, VENV_DONE). Recreated `images`/`segmentations` symlinks in `repo/data/waterbird_complete95_forest2water2/` (200 species folders visible).
- Launched two ERM runs in parallel, one per T4 (the session has 2 GPUs now), fp32 like the other runs: `run.py --dataset_type=waterbird --mode=erm --random_seed=0 --cuda_optimizations=false --results_dir=/kaggle/working/AutoBackSwap-repro/results --run_label=erm_minority_<false|true>_seed0 --waterbirds_use_minority=<false|true>` with `--device=cuda:0` (without minority) and `--device=cuda:1` (with minority). Logs `logs/erm_false_seed0.log`, `logs/erm_true_seed0.log`. Launched at unix 1793455167. Both showed 99% GPU utilization a minute later. Not the README's `--run_label="erm"` so labels match the other runs; README command otherwise unchanged except `--cuda_optimizations=false`, `--results_dir`.
- Expected paper reference: ERM without minority 33.4±4.4 / 68.4±1.9; with minority 74.9±3.3 / 93.6±0.4. No result yet.

## 2026-10-31 ERM seed 0 results (without minority) and with-minority restart
- **ERM without minority, seed 0** (`erm_minority_false_seed0`, GPU 0, fp32, 50 epochs, 52.4 s/epoch, about 43.7 min): finished. `test_accuracies.txt` (values read from the Kaggle notebook output, file not yet copied to this repo): WGA 34.900, Overall (Acc) 69.762, majority 99.033, minority 40.490. Repo text labels are swapped for the two minority groups, so by metadata: Landbird on Land 99.601, Landbird on Water 34.900 (worst group), Waterbird on Land 60.125, Waterbird on Water 97.040. Paper ERM without minority 33.4±4.4 / 68.4±1.9: WGA diff 1.5 (0.3σ), Acc diff 1.362 (0.7σ), both within 2σ (single seed).
- **ERM with minority, seed 0 first attempt: failed, no test accuracy.** The log stopped at epoch 42/50 (val_acc best 93.49%, 69 s/epoch) and `run.py` was no longer running when I looked (0 processes, GPU 0 MiB). Cause, as far as the evidence goes: while checking status in the Kaggle notebook I typed a long line into a cell that was not in edit mode, so the characters were read as notebook command-mode shortcuts (cells were created/deleted and the kernel restarted, cell counter went back to [1]). The ERM jobs had been started from that kernel with `nohup ... &` but not `setsid`. That the restart killed them is the most likely explanation but I did not verify it. The partial output is kept (renamed, not deleted) as `results/erm_minority_true_seed0_waterbird_wm_resnet_seed0_partial_epoch42` (checkpoint.pt, config.yaml only). The repo has no resume.
- **Relaunch** (with minority, seed 0, GPU 0 only, fp32, same command and label as before, plus `setsid`): log `logs/erm_true_seed0_rerun.log` (new file; old log `logs/erm_true_seed0.log` kept), launched at unix 1793458272. Same seed, so it should reproduce the first attempt up to GPU nondeterminism; not checked.
- Lesson: in the Kaggle notebook, click the cell and zoom to confirm the caret before typing; launch long jobs with `setsid`.

## 2026-10-31 ERM with minority, seed 0: result (rerun finished)
- The rerun (`logs/erm_true_seed0_rerun.log`, launched unix 1793458272, GPU 0) completed 50/50 epochs (47.3 min, 56.8 s/epoch, best val_acc 93.49%). The Kaggle interactive session was shut down for inactivity while I was not watching it (it showed "Draft Session off"); the run had finished by then, and the results folder was intact after the session restarted.
- `test_accuracies.txt` (values read from the notebook output, file not yet copied to this repo). Printed as in the repo (two minority labels swapped): Landbird on Land 99.734, "Waterbird on Land" 92.683, "Landbird on Water" 77.414, Waterbird on Water 94.081. By metadata: Landbird on Water 92.683, **Waterbird on Land 77.414 (worst group)**. Majority 98.481, minority 89.299, **Overall (Acc) 93.890, WGA 77.414**.
- Paper ERM with minority 74.9±3.3 / 93.6±0.4: WGA diff 2.514 (0.8σ), Acc diff 0.290 (0.7σ), both within 2σ (single seed).
- Output size on Kaggle was 15.3/19.5 GiB; nothing deleted (stale duplicate venv `AutoBackSwap-repro/env`, 6.4 GB, is the main candidate; waiting for the user's decision).

## 2026-11-01 cleanup and results copied back
- User approved deleting the stale duplicate venv. Deleted `/kaggle/working/AutoBackSwap-repro/env` (explicit path; checked first that it had no `DO_NOT_DELETE.txt` and that `repo/env/venv/lib` exists). Kaggle output went from 15.3 to 9 GiB (`AutoBackSwap-repro` 16 G to 8.5 G). Side effect: the tracked file `env/env_info.txt` shows as deleted in the Kaggle clone (not committed; the Mac repo is unaffected).
- Result text files (`config.yaml`, `losses.txt`, `test_accuracies.txt`) of the two ERM runs, the ERM with-minority partial config and AutoBackSwap with-minority were committed on Kaggle (fa24fa4) and pushed to the temporary branch `origin/kaggle-results-tmp` (not main, avoids a non-fast-forward with the Mac commits), then copied into this repo with `git checkout origin/kaggle-results-tmp -- results`. Their `test_accuracies.txt` values equal the numbers read from the notebook earlier. Checkpoints (`*.pt`) stay on Kaggle only.
- Found after the session restart: `repo/env/venv/bin/python` printed "not found" (venv python link broken again in the new container). Not fixed yet; relink to the system python 3.12 (or rebuild) before the next run.

## 2026-11-01 Waterbirds seeds 1 and 2 launched (user asked to run all remaining Waterbirds runs)
- Kaggle kernel was unresponsive (a trivial shell cell hung >1.5 min; cause not verified), so I restarted it from the notebook toolbar. Files under /kaggle/working persisted; `repo/env/venv/bin/python` link and `repo/data/.../{images,segmentations}` symlinks were missing again and were recreated (python -> /usr/bin/python3.12; torch 2.11.0+cu126, torchvision 0.26.0, 2 GPUs confirmed).
- Launched with `setsid nohup`, scripts `chain_abs_seeds12.sh` and `chain_erm_seeds12.sh` in the Kaggle workspace (fp32, `--cuda_optimizations=false`, same flags as seed 0):
  - GPU 0: AutoBackSwap without minority, seed 1 (detector, generate, classifier) then seed 2. Labels `ours_minority_false_seed1/2`. Log `logs/abs_seeds12.log`.
  - GPU 1: ERM seed 1 (without, with minority) then seed 2 (without, with minority). Labels `erm_minority_{false,true}_seed{1,2}`. Log `logs/erm_seeds12.log`.
- Launch time about 2026-11-01 01:13 UTC. Both chains were confirmed running (PIDs present, GPU 0 3.4 GB, GPU 1 3.5 GB at 100%). Kaggle disk 9.1G used of 20G at launch.
- Spawrious and Spurious Vehicles not started: they need dataset downloads and SAM3 segmentation masks; the user chose to finish Waterbirds first.

## 2026-11-01 per-example predictions for the four seed 0 runs
- `analysis/dump_all_predictions.py` (new) writes `predictions.csv` with every one of the 5794 test examples (image_path, true/predicted label, correct, group, spurious attribute, confidence). No image is copied or uploaded. Run on Kaggle (fp32, best-val `checkpoint.pt`, first attempt failed on the matplotlib backend, rerun with MPLBACKEND=Agg).
- Files copied back through `origin/kaggle-results-tmp` (user approved pushing to that branch only): `artifacts/predictions/{ours_minority_false_seed0,ours_minority_true_seed0,erm_minority_false_seed0,erm_minority_true_seed0}/predictions.csv`.
- Check against `test_accuracies.txt`, recomputed from the copied CSVs on the Mac: ours without minority 346 errors, WGA 92.195, overall 94.028; ours with minority 264 errors, 92.368 / 95.444; ERM without minority 1752 errors, 34.900 / 69.762; ERM with minority 354 errors, 77.414 / 93.890. All match.
- Before this, only `ours_minority_false_seed0` had an error list (errors only, no correct examples). The partial ERM with-minority run (epoch 42) has no checkpoint-based predictions and no test score.

## 2026-11-01 ERM without minority, seed 1 finished
- `erm_minority_false_seed1` (GPU 1, chain_erm_seeds12.sh): finished about 47 min after launch. Result read from the Kaggle notebook output at 02:25 UTC (the `test_accuracies.txt` file is not copied to the Mac yet; value to be rechecked once it is).
- Repo text file lines: Landbird on Land 99.690, "Waterbird on Land" 31.885, "Landbird on Water" 52.025, Waterbird on Water 97.508; Majority 99.206, Minority 36.348, Overall 67.777, WGA 31.885. The two middle labels are swapped in the repo file, so by metadata: Landbird on Water 31.885 (worst), Waterbird on Land 52.025.
- Paper ERM without minority 33.4 +- 4.4 / 68.4 +- 1.9: WGA diff 1.515 (0.34 sigma), Acc diff 0.623 (0.33 sigma), both within 2 sigma. Seed 0 gave 34.900 / 69.762.
- No traceback in logs/abs_seeds12.log or logs/erm_seeds12.log at 02:25 UTC. Progress then: ABS seed 1 classifier epoch 12/50 (about 64 s/epoch, best val_acc 95.00%), ERM with minority seed 1 epoch 21/50 (about 65 s/epoch, best val_acc 92.16%). Output 10 GiB of 19.5.

## 2026-11-01 ERM with minority seed 1 and AutoBackSwap without minority seed 1 finished
Both read from the Kaggle notebook output at 03:24 UTC (files not yet copied to the Mac; to be rechecked once they are).
- `erm_minority_true_seed1`: repo text lines LoL 99.468, "Waterbird on Land" 91.175, "Landbird on Water" 80.997, WoW 94.860; Majority 98.447, Minority 88.920, Overall 93.683, WGA 80.997. Labels of the two middle rows are swapped in the repo file: by metadata Landbird on Water 91.175, Waterbird on Land 80.997 (worst). Paper ERM with minority 74.9 +- 3.3 / 93.6 +- 0.4: WGA diff 6.097 (1.85 sigma, inside 2 sigma but near the edge), Acc diff 0.083 (0.2 sigma). Seed 0 gave 77.414 / 93.890.
- `ours_minority_false_seed1` (detector, generate, classifier on GPU 0): repo text lines LoL 97.384, "Waterbird on Land" 92.284, "Landbird on Water" 93.769, WoW 94.860; Majority 96.824, Minority 92.613, Overall 94.719, WGA 92.284. By metadata: Landbird on Water 92.284 (worst), Waterbird on Land 93.769. Paper 92.2 +- 0.4 / 95.1 +- 0.3: WGA diff 0.084 (0.2 sigma), Acc diff 0.381 (1.27 sigma), both within 2 sigma. Seed 0 gave 92.195 / 94.028 (Acc 3.6 sigma off), so the Acc shortfall is not the same size across seeds; with two seeds this still does not establish its cause.
- Progress at 03:24 UTC: ABS chain started seed 2 (detector); ERM chain on seed 2 without minority, epoch 28/50 (about 59 s/epoch, best val_acc 99.50%). Output 10.3 GiB of 19.5. No traceback in either log.

## 2026-11-01 ERM without minority seed 2 finished (check at about 03:55 UTC)
- `erm_minority_false_seed2`, read from the Kaggle notebook output (file not yet copied to the Mac). Repo text lines: LoL 99.601, "Waterbird on Land" 33.348, "Landbird on Water" 53.738, WoW 97.508; Majority 99.137, Minority 37.867, Overall 68.502, WGA 33.348. By metadata: Landbird on Water 33.348 (worst), Waterbird on Land 53.738. Paper ERM without minority 33.4 +- 4.4 / 68.4 +- 1.9: diffs 0.052 and 0.102, both well inside 2 sigma. ERM without minority so far: 34.900 / 31.885 / 33.348 WGA for seeds 0, 1, 2.
- Progress: ABS seed 2 generate step 143/200 (about 10 s/it, about 10 min left), then its classifier; ERM with minority seed 2 at epoch 8/50 (about 63 s/epoch, best val_acc 88.16%). Output 10.9 GiB of 19.5. No traceback in either log.

## 2026-11-02 ERM with minority seed 2 and AutoBackSwap without minority seed 2 finished (found after the Kaggle session ended)
Both read from the Kaggle notebook output saved at 06:28 UTC on 2026-11-01 (the session was off when read on 2026-11-02, so no live check; `test_accuracies.txt` files not yet copied to the Mac, to be rechecked when they are). Logs show `END` stamps at 04:40:01 UTC (ERM) and 04:57:35 UTC (AutoBackSwap), no traceback in either log, and both GPUs idle at 06:28 UTC.
- `erm_minority_true_seed2`: repo text lines LoL 99.512, "Waterbird on Land" 91.885, "Landbird on Water" 76.947, WoW 93.925; Majority 98.274, Minority 88.574, Overall 93.424, WGA 76.947. By metadata (middle labels swapped in the repo file): Landbird on Water 91.885, Waterbird on Land 76.947 (worst). Paper 74.9 +- 3.3 / 93.6 +- 0.4: WGA diff 2.047 (0.62 sigma), Acc diff 0.176 (0.44 sigma), both within 2 sigma. ERM with minority WGA so far: 77.414 / 80.997 / 76.947 for seeds 0, 1, 2.
- `ours_minority_false_seed2`: repo text lines LoL 97.472, "Waterbird on Land" 92.506, "Landbird on Water" 92.835, WoW 94.704; Majority 96.859, Minority 92.579, Overall 94.719, WGA 92.506. By metadata: Landbird on Water 92.506 (worst), Waterbird on Land 92.835. Paper 92.2 +- 0.4 / 95.1 +- 0.3: WGA diff 0.306 (0.77 sigma), Acc diff 0.381 (1.27 sigma), both within 2 sigma. AutoBackSwap without minority Acc so far: 94.028 / 94.719 / 94.719 for seeds 0, 1, 2, all below the paper mean; the cause is not established.
- All three seeds now exist for the four Waterbirds configurations except AutoBackSwap with minority (seed 0 only).
- Not done: `predictions.csv` for seeds 1 and 2 (needs a new Kaggle session and the checkpoints; unverified whether `checkpoint.pt` files are still in /kaggle/working), copying result files to the Mac.

## 2026-11-02 misclassified images collected (Waterbirds downloaded locally)
- User approved downloading `waterbird_complete95_forest2water2.tar.gz` (489,698,023 bytes, from downloads.cs.stanford.edu/nlp/data/dro/) into `repo/data/_dl_waterbirds/` (gitignored). Only the 1896 unique test images misclassified by at least one of the four seed 0 runs were extracted.
- User approved copying them to their Google Drive (this overrides the earlier "never upload images" constraint, for these images and this Drive only): `My Drive/AutoBackSwap-misclassified/` with `images/`, `all_errors.csv` (per-run predicted label and confidence, n_runs_wrong), `per_run/<run>/errors.csv`, `README.md`, and the existing viewer for `ours_minority_false_seed0`. Errors per run: ERM without minority 1752, ERM with minority 354, AutoBackSwap without minority 346, AutoBackSwap with minority 264. Seeds 1 and 2 not included.
