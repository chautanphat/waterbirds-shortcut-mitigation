# Experiment Log

Record every meaningful run here. This is an index, not a replacement for configurations, method READMEs, or untracked output files.

## Entry template

```markdown
## <experiment-id> — YYYY-MM-DD

- **Method / version:**
- **Purpose:** smoke test / original reproduction / Waterbirds adaptation / final comparison
- **Branch or commit:**
- **Configuration:** `configs/...` or `experiments/...`
- **Dataset and split:**
- **Seed:**
- **Environment and hardware:**
- **Command:**
- **Output location:** local, untracked path
- **Results:** overall accuracy; per-group accuracy; worst-group accuracy
- **Paper comparison or expected behavior:**
- **Status:** exploratory / smoke test / partial reproduction / verified reproduction / final
- **Notes and follow-up:**
```

Use one entry per meaningful run or a short grouped entry for repeated seeds with the same configuration. Never present a result without its configuration, command, seed, and evaluation context.

Use repository-relative paths in every entry. Link implementation changes to `baselines/` or `src/methods/`, tracked settings to `configs/`, run metadata to `experiments/`, and lightweight summaries to `results/`; keep large outputs and checkpoints local and untracked.



## autobackswap-wom-resnet50 — 2026-10-31 to 2026-11-01

- **Method / version:** AutoBackSwap, `baselines/autobackswap/` (upstream commit `b060532c2632039ffc2918828780d2b696763af6`), plus `baselines/autobackswap/patches/augment_waterbirds_chunk.patch`
- **Purpose:** partial reproduction
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_autobackswap_resnet50.yaml` (`use_minority: false`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0, 1, 2
- **Environment and hardware:** Kaggle, Tesla T4, Python 3.12, torch 2.11.0+cu126, fp32
- **Command:** see `baselines/autobackswap/README.md`; full per-run lines in `results/autobackswap/runs_log.csv`
- **Output location:** local, untracked (`results/<run_label>_waterbird_wom_resnet_seed<k>/`); copies of `test_accuracies.txt` in `results/autobackswap/test_accuracies/`
- **Results:** worst-group acc 92.195 / 92.284 / 92.506; overall acc 94.028 / 94.719 / 94.719 (per-group values and group sizes in `results/autobackswap/summary.md`)
- **Paper comparison or expected behavior:** paper 92.2 ± 0.4 / 95.1 ± 0.3. Worst-group within 2σ; overall accuracy mean 94.49 is 2.04σ below.
- **Status:** partial reproduction
- **Notes and follow-up:** cause of the overall-accuracy gap not isolated. Seed 0 first attempts failed or were interrupted (out of memory, session shutdown); they are in `results/autobackswap/runs_log.md`.

## autobackswap-wm-resnet50-seed0 — 2026-10-31

- **Method / version:** AutoBackSwap, same as above
- **Purpose:** partial reproduction
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_autobackswap_resnet50.yaml` (`use_minority: true`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** classifier step only (`--mode=ours --waterbirds_use_minority=true`), reusing the detector and generated images of `autobackswap-wom-resnet50` seed 0
- **Output location:** local, untracked
- **Results:** worst-group acc 92.368, overall acc 95.444
- **Paper comparison or expected behavior:** paper 93.0 ± 0.9 / 95.0 ± 0.5, both within 2σ (single seed)
- **Status:** partial reproduction
- **Notes and follow-up:** one seed only; reuse of generated data was inferred from code reading, not verified by regenerating.

## erm-wom-resnet50 — 2026-10-31 to 2026-11-01

- **Method / version:** ERM from the AutoBackSwap codebase (`--mode=erm`)
- **Purpose:** baseline for the AutoBackSwap comparison
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_erm_resnet50.yaml` (`use_minority: false`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0, 1, 2
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** see `baselines/autobackswap/README.md`
- **Output location:** local, untracked; `test_accuracies.txt` copies in `results/autobackswap/test_accuracies/`
- **Results:** worst-group acc 34.900 / 31.885 / 33.348; overall acc 69.762 / 67.777 / 68.502
- **Paper comparison or expected behavior:** paper 33.4 ± 4.4 / 68.4 ± 1.9, both within 2σ
- **Status:** partial reproduction
- **Notes and follow-up:** 3 seeds against the paper's 5.

## erm-wm-resnet50 — 2026-10-31 to 2026-11-01

- **Method / version:** ERM from the AutoBackSwap codebase
- **Purpose:** baseline for the AutoBackSwap comparison
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_erm_resnet50.yaml` (`use_minority: true`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0, 1, 2
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** see `baselines/autobackswap/README.md`
- **Output location:** local, untracked; `test_accuracies.txt` copies in `results/autobackswap/test_accuracies/`
- **Results:** worst-group acc 77.414 / 80.997 / 76.947; overall acc 93.890 / 93.683 / 93.424
- **Paper comparison or expected behavior:** paper 74.9 ± 3.3 / 93.6 ± 0.4, both within 2σ (seed 1 worst-group is 1.85σ above the paper mean)
- **Status:** partial reproduction
- **Notes and follow-up:** the first seed 0 attempt stopped at epoch 42 of 50 (most likely a notebook kernel restart, not verified) and was rerun with the same seed.
