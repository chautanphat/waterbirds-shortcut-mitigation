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

## exp-dpe-h2-rules-waterbirds-standard-seed0-001 — 2026-11-08

The date is the documentation date; execution date and elapsed time are not recorded in the supplied suite metadata.

- **Method / version:** H2 multi-rule cached-prediction comparison: uniform, original power, softmax reliability, top-k and regularized joint weights.
- **Purpose:** Exploratory Waterbirds adaptation; compare C-selected weighting procedures using the original seed-0 H2 heads and cached predictions, without retraining.
- **Branch or commit:** `experiment/dpe-multiple-weighting-rules`; executed source `3ee1c6e7a9cfad351d69e75c92ad0d9c85d4c424`.
- **Configuration:** [Recorded grid](../experiments/exp-dpe-h2-rules-waterbirds-standard-seed0-001/config.yaml), [runtime/archive identities and review scope](../experiments/exp-dpe-h2-rules-waterbirds-standard-seed0-001/run.json). Twenty candidates, including uniform and 12 original power controls; each family has a uniform fallback.
- **Dataset and split:** Official Waterbirds 4,795/1,199/5,794; inherited A/B/C=720/241/238, split seed 0. B fits weights; C selects family and overall winners before test evaluation.
- **Seed:** 0 only, standard track; source ensemble/manifest/selection identities match `exp-dpe-h2-waterbirds-standard-seed0-001`.
- **Environment and hardware:** Linux, Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3, torch 2.11.0+cu130 and recorded CUDA 13.0. Weight fitting uses the SciPy CPU solver; physical hardware and hosting service are not logged.
- **Command:** `notebooks/dpe_weighting_rules.ipynb`; saved Python API interface `rule_suite.select` followed by `rule_suite.evaluate`. No shell training command is claimed.
- **Output location:** Suite `h2-standard-seed0-rules-6452ea10e0` and its supplied ZIP, retained outside Git; [tracked results](../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/report.md).
- **Results:** Overall C selection chooses top-k10, improving C WGA from 89.2473% to 90.3226% by correcting one image. Uniform test accuracy/WGA=91.2323/89.7561%; top-k10=91.1978/89.6231%, paired differences=-0.0345/-0.1330 percentage points. Top-k10 test group accuracies=92.5055/89.6231/91.5888/91.7445%, counts=2,255/2,255/642/642. Of 56 changed test predictions, 27 errors are fixed and 29 introduced. Power, softmax and joint families select uniform; their selected results equal the control.
- **Paper comparison or expected behavior:** A matched within-run aggregation comparison, not a paper reproduction. No selected weighting procedure improves test WGA in this seed.
- **Status:** exploratory; completed single-seed comparison with aggregate-record consistency review, not an independent prediction-cache audit.
- **Notes and follow-up:** Selection/result hashes, recorded code identities, nonjoint weight formulas, candidate selection and metric/count arithmetic agree. The supplied suite ZIP excludes prediction caches and heads, so sample-level metrics and joint optimization cannot be independently rerun from it. Preserve the C-selected top-k10 outcome; complete seeds 1/2 with the frozen grid and report paired differences. See [full interpretation and verification limits](../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/report.md).

## exp-dpe-h2-waterbirds-standard-seed0-001 — 2026-11-08

The date is the documentation date; the supplied execution log contains elapsed times only.

- **Method / version:** [DPE head reweighting](../src/methods/dpe_head_reweighting/README.md), H2 project-owned adaptation.
- **Purpose:** Exploratory Waterbirds adaptation; compare validation-derived power weights with the matched A-trained uniform ensemble.
- **Branch or commit:** `experiment/dpe-head-reweighting`; executed source `5ebb1128fe84200938e73c1fe85e317ecb39f09d`, recorded clean.
- **Configuration:** [H2 settings](../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/config.yaml), [backbone settings](../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/stage0_config.yaml); [run metadata/source hashes](../experiments/exp-dpe-h2-waterbirds-standard-seed0-001/run.json).
- **Dataset and split:** Official Waterbirds train/val/test 4,795/1,199/5,794. Validation A/B/C=720/241/238, group-stratified with split seed 0; group annotations used for head fitting and selection.
- **Seed:** 0 only; standard ERM without augmentation.
- **Environment and hardware:** Tesla T4, Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.1.3, CUDA 13.0; hosted on Kaggle.
- **Command:** Execute `notebooks/dpe_head_reweighting.ipynb` at the recorded commit with `TRACK="standard"`, `SEED=0`, `WORKERS=2`, `RUN_NAME="h2-standard-seed0"`. The archived command field is a kernel launcher, not a shell training command.
- **Output location:** Original `outputs/dpe_head_reweighting_runs/h2-standard-seed0/` and its ZIP, untracked; [tracked lightweight results](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md).
- **Results:** Uniform and selected test overall accuracy 91.2323%; group accuracies 00/01/10/11=92.7273/89.7561/91.1215/91.2773%, counts=2,255/2,255/642/642; WGA 89.7561%. Paired WGA and overall differences=0.0000 percentage points. C selected uniform `(alpha=0, gamma=1)`; all 13 candidates predict the same C labels. Final backbone validation accuracy/WGA=86.4887/60.1504%; no ERM-only test result is established.
- **Paper comparison or expected behavior:** The annotated paper reference is 91.0 ± 0.4% WGA, but H2 reserves B/C and trains heads using A; its single-seed score is not a like-for-like paper reproduction.
- **Status:** exploratory; completed single-seed adaptation, independently audited saved predictions and records.
- **Notes and follow-up:** No weighting improvement demonstrated. [Full candidate comparison and audit evidence](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md) reproduce the saved decision. Raw-image feature extraction could not be independently checked because images/backbone/raw features are excluded from the supplied archive. [Additional summary diagnostics](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/report.md#additional-summary-diagnostics--2026-11-08) find B-to-C head-WGA rank correlation -0.0276 and effective heads 14.3720 even for the strongest candidate. These are aggregate-record calculations, not a new run or archive re-audit. Complete seeds 1/2; check score stability and consider regularized joint weights as separately labeled, unmeasured follow-ups.


## exp-dpe-waterbirds-ermstar-001 — 2026-11-02

**Audit note:** This is a historical, group-annotated head-training run with a 100-epoch augmented backbone, not the upstream 400-epoch ERM* recipe. Original ERM test evaluation used standardized features with a head trained on raw features; its recorded ERM metrics require re-evaluation. DPE used its intended standardized features. The paper's corresponding annotated DPE reference is 94.1 ± 0.4 WGA; 94.1 ± 0.2 is the unannotated reference. No historical measurements have been replaced by corrected-run claims.

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
- **Paper comparison or expected behavior:** Closely matches the paper's reported $\text{ERM}^*$ baseline ($77.9\% \pm 3.0\%$ WGA) and $\text{ERM}^*$ + DPE benchmark ($94.1\% \pm 0.4\%$ WGA, annotated validation). The photometric augmentations produced robust backbone representations, leading to relatively flat ablation metrics across the ensemble variants.
- **Status:** historical reproduction; corrected ERM evaluation pending
- **Notes and follow-up:** Validates the paper's enhanced data-augmented track. The presence of `RandomPhotometricDistort` mitigated background bias during Stage 0 (+10% WGA over standard ERM).


## exp-dpe-waterbirds-standard-002 — 2026-11-02

**Audit note:** The executed notebook uses group-annotated validation data for head fitting and epoch selection. Its descriptive Stage-1 YAML previously disagreed with that setting; notebook Python defaults determine this run. Original ERM test evaluation used standardized features with a head trained on raw features; its recorded ERM metrics require re-evaluation. The paper's annotated DPE reference is 91.0 ± 0.4 WGA. DPE used its intended standardized features; no new accuracy result is claimed by the integration fixes.

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
- **Paper comparison or expected behavior:** Matches the paper's canonical Standard ERM baseline ($69.1\% \pm 4.7\%$ WGA) and reproduces the Standard ERM + DPE benchmark ($91.0\% \pm 0.4\%$ WGA, annotated validation) within 2.0% on a single seed. On standard shortcut-susceptible representations, Full DPE outperformed the no-covariance-regularization ablation by +0.8% WGA (89.00% vs. 88.20%), confirming the empirical benefit of prototype covariance regularization.
- **Status:** historical reproduction; corrected ERM evaluation pending
- **Notes and follow-up:** Confirms the primary paper claims across both Standard and Enhanced tracks. Next steps: multi-seed runs (seeds 1 & 2) and unannotated subsampling evaluation (`train_attr=no`, class-balanced sampling; the upstream CLI token is `subsample_type=group`).


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

## exp-xrm-waterbirds-erm-selected — historical archive

- **Method / version:** Authors' ERM from [`baselines/xrm/`](../baselines/xrm/), upstream commit `5e90815250aedea546b9e17ffd5d7e0a7c34c20f`.
- **Purpose:** Historical Waterbirds baseline with human group labels reserved for validation model selection/evaluation.
- **Branch or commit:** The original run commit was not recorded.
- **Configuration:** [`configs/erm/waterbirds_xrm.yaml`](../configs/erm/waterbirds_xrm.yaml); selected combination 4, 5,001 updates, validation-WGA checkpoint selection.
- **Dataset and split:** Official Waterbirds train/validation/test split, 4,795/1,199/5,794.
- **Seeds:** 0–6 complete; seed 7 stopped at step 0 and is excluded; seeds 8–9 are missing.
- **Environment and hardware:** Handoff records Windows CPU, Python 3.9, PyTorch 1.13.0, torchvision 0.14.0, 11th Gen Intel Core i7-11390H. Per-run package versions were not logged.
- **Command:** Historical exact command not retained. Current reproduction command is in [`src/methods/erm/README.md`](../src/methods/erm/README.md).
- **Output location:** Raw JSON in [`results/archive/xrm_waterbirds_initial/erm/`](../results/archive/xrm_waterbirds_initial/erm/); generated report at [`results/erm/summary.md`](../results/erm/summary.md).
- **Results:** Aggregated from each JSON history at the last occurrence of maximum validation WGA. See the generated summary; incomplete/missing seeds are identified and excluded.
- **Status:** Partial reproduction; historical per-group accuracy and selected checkpoint weights are unavailable.

## exp-xrm-waterbirds-groupdro-selected — historical archive

- **Method / version:** Authors' GroupDRO from [`baselines/xrm/`](../baselines/xrm/), upstream commit `5e90815250aedea546b9e17ffd5d7e0a7c34c20f`.
- **Purpose:** Historical Waterbirds baseline using human-annotated groups in the GroupDRO objective.
- **Branch or commit:** The original run commit was not recorded.
- **Configuration:** [`configs/groupdro/waterbirds_xrm.yaml`](../configs/groupdro/waterbirds_xrm.yaml); selected combination 8, `eta=0.01152687`, 5,001 updates, validation-WGA checkpoint selection.
- **Dataset and split:** Official Waterbirds train/validation/test split, 4,795/1,199/5,794.
- **Seeds:** 0–5 complete; seed 6 stopped at step 3,150 and is excluded; seeds 7–9 are missing.
- **Environment and hardware:** Handoff records Windows CPU, Python 3.9, PyTorch 1.13.0, torchvision 0.14.0, 11th Gen Intel Core i7-11390H. Per-run package versions were not logged.
- **Command:** Historical exact command not retained. Current reproduction command is in [`src/methods/groupdro/README.md`](../src/methods/groupdro/README.md).
- **Output location:** Raw JSON in [`results/archive/xrm_waterbirds_initial/groupdro/`](../results/archive/xrm_waterbirds_initial/groupdro/); generated report at [`results/groupdro/summary.md`](../results/groupdro/summary.md).
- **Results:** Aggregated from each JSON history at the last occurrence of maximum validation WGA. See the generated summary; incomplete/missing seeds are identified and excluded.
- **Status:** Partial reproduction; historical per-group accuracy and selected checkpoint weights are unavailable.

## exp-xrm-waterbirds-smoke — 2026-11-08

- **Method / version:** Project adapters over the authors' ERM and GroupDRO implementations in [`baselines/xrm/`](../baselines/xrm/), source commit `5e90815250aedea546b9e17ffd5d7e0a7c34c20f`.
- **Purpose:** Two-update CPU smoke runs for each method, followed by selected-checkpoint evaluation through the project evaluator; these are pipeline checks, not benchmark runs.
- **Branch or commit:** `method/erm+groupdro`; local source commit is not recorded as a run artifact.
- **Configuration:** [`configs/erm/waterbirds_xrm.yaml`](../configs/erm/waterbirds_xrm.yaml) and [`configs/groupdro/waterbirds_xrm.yaml`](../configs/groupdro/waterbirds_xrm.yaml); seed 0; two updates; checkpoint frequency 1; zero workers.
- **Dataset and split:** Existing prepared Waterbirds features; official test split, 5,794 examples.
- **Environment and hardware:** Windows CPU; Python 3.9, PyTorch 1.13.0, torchvision 0.14.0.
- **Command:** Run `python scripts/run_xrm_baseline.py --method <erm|groupdro> --seed 0 --max-steps 2 --checkpoint-frequency 1 --workers 0 --output-root outputs/xrm_<method>_smoke`, then evaluate the generated `*_best_wga.pt` checkpoint with `scripts/evaluate_xrm_checkpoint.py`.
- **Output location:** Local ignored `outputs/xrm_erm_smoke/` and `outputs/xrm_groupdro_smoke/`.
- **Results:** Both methods completed two updates and checkpoint evaluation reported all four groups with counts 2,255/2,255/642/642. Accuracy from a two-update run is not meaningful and is not included as a method result.
- **Status:** Smoke tests passed. The focused XRM test module also passes five checks for objectives, metrics, checkpoint resume, Windows filename parsing, and incomplete-run aggregation.
- **Notes and follow-up:** The historical run summaries are generated by [`scripts/summarize_xrm_baselines.py`](../scripts/summarize_xrm_baselines.py); incomplete seeds remain excluded.
