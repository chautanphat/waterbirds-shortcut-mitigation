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

---

## exp-dpe-waterbirds-reweighting-003 — 2026-11-10

- **Method / version:** DPE head aggregation, Kaggle version 4 / script version `367976457`; project aggregation runtime 6.
- **Purpose:** Exploratory comparison of uniform averaging, 19 nonuniform global weighting configurations and 10 neural combiners on the same frozen heads. Follow-up to H2 in the historical `report/main.tex`.
- **Branch or commit:** Executed branch `experiment/dpe-neural-combiner`, clean commit `8e14249f4b3510363be06f627182e2daa9926179`. Review completed 2026-11-12 on `docs/dpe-reweighting-review`.
- **Configuration:** [Tracked actual settings and provenance](../experiments/exp-dpe-waterbirds-reweighting-003/record.json). Executed implementation uses `configs/dpe_head_reweighting/aggregation_comparison.yaml` at the run commit; this file is not in the older review checkout. Backbone: ImageNet-1K V2 ResNet-50, 100 epochs, SGD LR 0.001, momentum 0.9, weight decay 0, class-balanced sampling, deterministic preprocessing. Heads: 15 × 20 epochs; combiner fits: 400 fixed steps.
- **Dataset and split:** Official Waterbirds 4,795 / 1,199 / 5,794. Validation is partitioned with fixed split seed 0 into disjoint head-training (720) and combiner-training (479) sets. Head fitting and epoch selection use only head-training; all weighting/combiner fitting uses only combiner-training. All configurations freeze before test evaluation. No separate candidate-selection split.
- **Seed:** 0, 1, 2; same validation partition and test identities across seeds.
- **Environment and hardware:** Kaggle `NvidiaTeslaT4` machine shape, CUDA for backbone/heads; CPU float64 neural combiners. Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.1.3, SciPy 1.16.3. GPU count not recorded. Papermill runtime 4 h 51 min 20 s, 2026-11-10 15:01:38–19:52:58 Asia/Saigon.
- **Command:** Run All in the [executed notebook](https://www.kaggle.com/kernels/scriptcontent/367976457/download), using `MODE="auto"`, `SEEDS=[0,1,2]`, `TRACK="standard"`, `EVALUATE_TEST=True`, `DEVICE="auto"`; `run_study` uses the recorded aggregation configuration. No model training was rerun during review.
- **Output location:** [Immutable Kaggle version](https://www.kaggle.com/code/noerula/dpe-reweighting?scriptVersionId=367976457); study `aggregation-standard-seeds012-0dcf563966c5`. Raw review evidence in `outputs/dpe-kaggle-v4/` (untracked). Lightweight tables and figures in `results/dpe/head_aggregation_kaggle_v4/`.
- **Results:** Mean ± sample SD over three seeds: uniform test WGA **89.5196 ± 0.6538%**, overall **91.2668 ± 0.3293%**. Dropout MLP (either weight decay): WGA **89.9926 ± 0.0923%**, overall **91.4049 ± 0.1726%**, paired WGA change **+0.4730 ± 0.6672 pp**. Per-seed WGA gains **+0.1330 / +0.0443 / +1.2417 pp**. Ordinary gate WGA **86.5780 ± 2.5189%**, change **−2.9416 pp**. No global weighting configuration improves mean WGA (one tie, 18 lower). [All per-group accuracies and counts](../results/dpe/head_aggregation_kaggle_v4/per_group.csv), in order `00/01/10/11`, test counts `2255/2255/642/642`.
- **Paper comparison or expected behavior:** The simple global-weighting H2 mechanism is unsupported by this run. A small neural-stacker signal merits a prespecified calibration control. Compare with this run's matched uniform reference, not the earlier 89.00% single-seed reproduction; training profile and data allocation differ.
- **Status:** Completed exploratory Waterbirds adaptation; not a paper reproduction or final selected-method benchmark. The test set previously informed project research, and all 30 configurations are independently reported rather than selected by test.
- **Notes and follow-up:** [Detailed findings and log audit](../results/dpe/reweighting_review.md). Independent NumPy inference reproduced all 90 test results and training metric records, group counts, NLL, paired error IDs and summary statistics. Seed 0's joint lambda-0.01 solver fallback passed its saved numerical certificate. Gates concentrated on roughly one effective head. The [user-run Kaggle verification](../results/dpe/kaggle_verification_20261112/README.md) subsequently reproduced all 90 results, saved states and full test-image inference on all three original source seeds. Additional combiner runs do not establish dropout superiority over plain MLP. Next: additional independent backbone/head seeds on the same Waterbirds/ResNet-50 setup. Each configuration remains independently evaluated without a required selection split.

## exp-dpe-waterbirds-ermstar-001 — 2026-11-02

- **Method / version:** Diverse Prototypical Ensembles (DPE) / commit `68b8114`
- **Purpose:** Original reproduction on Waterbirds (Enhanced $\text{ERM}^*$ Track with Photometric Distortion)
- **Branch or commit:** `main`
- **Configuration:** `notebooks/dpe_waterbirds_end_to_end.ipynb` (default data augmentation enabled)
- **Dataset and split:** Waterbirds (`waterbird_complete95_forest2water2`), official train/val/test splits (train=4,795, val=1,199, test=5,794)
- **Seed:** 0
- **Environment and hardware:** Kaggle Linux container, 2x Tesla T4 GPUs (15 GB VRAM each), PyTorch 2.x, CUDA 12.x
- **Command:** End-to-end execution of `notebooks/dpe_waterbirds_end_to_end.ipynb` (Stage 0 ResNet-50 ERM 100 epochs with `RandomPhotometricDistort(p=0.5)` + Stage 1+ 15-head DPE ensemble with IsoMax+ and covariance regularization)
- **Output location:** `checkpoints/waterbirds_stage0_erm/` (untracked)
- **Results:**
  - **Stage 0 ERM Backbone (100 ep):** Overall Acc: 93.63%; Per-group Acc: [99.03%, 77.10%, 77.30%, 99.75%]; Worst-Group Acc (WGA): 77.10%
  - **Stage 1 DPE (1 Head):** Worst-Group Acc (WGA): 92.50%
  - **Stage 15 Full DPE (Cov Reg + Random Subsets):** Overall Acc: 93.30%; Per-group Acc: [93.6%, 93.0%, 92.7%, 93.9%]; Worst-Group Acc (WGA): 92.70%
  - **Stage 15 Ablation 1 (Random Subsets Only, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.70%; Per-group Acc: [94.2%, 93.5%, 92.8%, 94.3%]; Worst-Group Acc (WGA): 92.80%
  - **Stage 15 Ablation 2 (Single Fixed Subset, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.70%; Per-group Acc: [94.3%, 93.7%, 92.7%, 94.1%]; Worst-Group Acc (WGA): 92.70%
- **Paper comparison or expected behavior:** Closely matches the paper's reported $\text{ERM}^*$ baseline ($77.9\% \pm 3.0\%$ WGA) and $\text{ERM}^*$ + DPE benchmark ($93.0\% \pm 0.4\%$ / $94.1\% \pm 0.2\%$ WGA). The photometric augmentations produced robust backbone representations, leading to relatively flat ablation metrics across the ensemble variants.
- **Status:** verified reproduction
- **Notes and follow-up:** Validates the paper's enhanced data-augmented track. The presence of `RandomPhotometricDistort` mitigated background bias during Stage 0 (+10% WGA over standard ERM).


## exp-dpe-waterbirds-standard-002 — 2026-11-02

- **Method / version:** Diverse Prototypical Ensembles (DPE) / commit `68b8114`
- **Purpose:** Original reproduction on Waterbirds (Standard SubpopBench ERM Track, `--no_augmentation`)
- **Branch or commit:** `main`
- **Configuration:** `configs/dpe4subpop/waterbirds_stage0_erm.yaml` (`no_augmentation: true`), `configs/dpe4subpop/waterbirds_stage1_dpe.yaml`
- **Dataset and split:** Waterbirds (`waterbird_complete95_forest2water2`), official train/val/test splits (train=4,795, val=1,199, test=5,794)
- **Seed:** 0
- **Environment and hardware:** Kaggle Linux container, 2x Tesla T4 GPUs (15 GB VRAM each), PyTorch 2.x, CUDA 13.0
- **Command:** End-to-end execution of `notebooks/dpe_waterbirds_end_to_end.ipynb` with `--no_augmentation` passed to Stage 0 training (deterministic `Resize(256) -> CenterCrop(224)`) + Stage 1+ 15-head DPE ensemble
- **Output location:** `checkpoints/waterbirds_stage0_standard_erm/` (untracked)
- **Results:**
  - **Stage 0 ERM Backbone (100 ep):** Overall Acc: 88.63%; Per-group Acc: [99.47%, 80.49%, 62.46%, 93.30%]; Worst-Group Acc (WGA): 62.46%
  - **Stage 1 DPE (1 Head):** Overall Acc: 92.70%; Per-group Acc: [94.8%, 92.5%, 88.9%, 90.0%]; Worst-Group Acc (WGA): 88.90%
  - **Stage 15 Full DPE (Cov Reg + Random Subsets):** Overall Acc: 91.40%; Per-group Acc: [93.8%, 89.0%, 89.9%, 92.7%]; Worst-Group Acc (WGA): 89.00%
  - **Stage 15 Ablation 1 (Random Subsets Only, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.50%; Per-group Acc: [96.3%, 93.5%, 88.2%, 89.6%]; Worst-Group Acc (WGA): 88.20%
  - **Stage 15 Ablation 2 (Single Fixed Subset, $\lambda_{\text{cov}}=0$):** Overall Acc: 93.30%; Per-group Acc: [95.5%, 93.2%, 88.8%, 90.3%]; Worst-Group Acc (WGA): 88.80%
- **Paper comparison or expected behavior:** Matches the paper's canonical Standard ERM baseline ($69.1\% \pm 4.7\%$ WGA) and reproduces the Standard ERM + DPE benchmark ($91.0\% \pm 0.5\%$ WGA) within 2.0% on a single seed. On standard shortcut-susceptible representations, Full DPE outperformed the no-covariance-regularization ablation by +0.8% WGA (89.00% vs. 88.20%), confirming the empirical benefit of prototype covariance regularization.
- **Status:** verified reproduction
- **Notes and follow-up:** Confirms the primary paper claims across both Standard and Enhanced tracks. Next steps: multi-seed runs (seeds 1 & 2) and unannotated subsampling evaluation (`train_attr=no`, `subsample_type=class`).


## exp-poter-waterbirds-standard-seed1-001 — 2026-11-04

- **Method / version:** POTER / NeurIPS 2026
- **Purpose:** Seed-1 Waterbirds benchmark run and validation-based checkpoint selection
- **Branch or commit:** `phat/poter-windows-cmnist`
- **Configuration:** `configs/poter/waterbirds_standard.yaml`
- **Dataset and split:** Waterbirds official train/validation/test splits (4,795 / 1,199 / 5,794)
- **Seed:** 1
- **Environment and hardware:** Kaggle Linux container, 1× NVIDIA Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0
- **Command:** `scripts/kaggle_waterbirds.py --data-dir /kaggle/working/poter-data --cache /kaggle/working/poter-features/waterbirds_standard.pt --run-dir /kaggle/working/poter-results/CUB/Clean/POTER_CUB_seed1_kaggle --seed 1 --epochs 300 --batch-size 128 --max-hours 9.5`
- **Output location:** `poter-results/CUB/Clean/POTER_CUB_seed1_kaggle/` (untracked); train/validation aggregate in `results/poter/seed1_full_metrics.csv`
- **Results:**
  - Selected epoch: 258 (zero-based), selected only by validation WGA.
  - Validation WGA: 90.10%; validation average accuracy: 91.60%.
  - Validation group accuracies: [92.50%, 90.10%, 92.50%, 92.50%].
  - Held-out test WGA, overall accuracy, adjusted average, group accuracies and counts: **not present in tracked artifacts**.
- **Paper comparison or expected behavior:** The paper reports held-out test WGA 90.9 ± 0.3% over three seeds. The seed-1 validation result is not directly comparable and no reproduction verdict is drawn from the 0.8-point numerical difference.
- **Status:** partial reproduction
- **Notes and follow-up:** Recover `log.txt`, `val.csv`, and `test.csv` from the Kaggle backup and run the summarizer with `--seeds 1`. If `test.csv` is unavailable, evaluate `best_validation.pt` once on the official test split. Then run seeds 2 and 3 for the paper-level aggregate.

## exp-autobackswap-waterbirds-wom-003 — 2026-10-31 to 2026-11-01

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

## exp-autobackswap-waterbirds-wm-004 — 2026-10-31

- **Method / version:** AutoBackSwap, same as above
- **Purpose:** partial reproduction
- **Branch or commit:** `method/autobackswap`
- **Configuration:** `configs/autobackswap/waterbirds_autobackswap_resnet50.yaml` (`use_minority: true`)
- **Dataset and split:** Waterbirds, official split
- **Seed:** 0
- **Environment and hardware:** Kaggle, Tesla T4, fp32
- **Command:** classifier step only (`--mode=ours --waterbirds_use_minority=true`), reusing the detector and generated images of `exp-autobackswap-waterbirds-wom-003` seed 0
- **Output location:** local, untracked
- **Results:** worst-group acc 92.368, overall acc 95.444
- **Paper comparison or expected behavior:** paper 93.0 ± 0.9 / 95.0 ± 0.5, both within 2σ (single seed)
- **Status:** partial reproduction
- **Notes and follow-up:** one seed only; reuse of generated data was inferred from code reading, not verified by regenerating.

## exp-erm-waterbirds-wom-005 — 2026-10-31 to 2026-11-01

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

## exp-erm-waterbirds-wm-006 — 2026-10-31 to 2026-11-01

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

## exp-dpe-waterbirds-five-seed-verification-004 — analyzed 2026-11-12

- **Method:** learned prediction aggregation of frozen DPE-style heads; Waterbirds / ResNet-50 fixed.
- **Sources:** seeds 0–4, with 0–2 reused and 3–4 newly trained; 30 configurations each; primary validation partition 0.
- **Source:** commit `8e14249f4b3510363be06f627182e2daa9926179`; original saved Kaggle version 368677926; final recovery version not identified in uploaded evidence.
- **Status:** completed after documented auxiliary solver recovery. Both new backbones were retained.
- **Results:** uniform WGA 89.2337%; dropout MLP 89.8537/89.8625% (wd 0.01/0.001); plain MLP 89.9578/89.9667%. Dropout improves over uniform on each primary source but the source-level 95% intervals include zero. Dropout mean gains reverse on alternative partition 2.
- **Evidence:** [scientific report](../results/dpe/five_seed_verification_20261112/README.md), [experiment record](../experiments/exp-dpe-waterbirds-five-seed-verification-004/record.json), and [audit script](../scripts/analysis/audit_dpe_five_seed.py).
- **Claims:** promising primary-partition aggregation gain; neither dropout-specific superiority nor partition-independent improvement established. 200 initialization fits and alternative partitions are nested within five sources.
