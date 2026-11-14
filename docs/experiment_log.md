# Experiment Log

Record every meaningful run here. This is an index, not a replacement for configurations, method READMEs, or untracked output files.

New meaningful runs follow the [experiment records contract](experiment_record_contract.md)
and [model artifacts and inference contract](model_artifact_contract.md). Link
`experiments/<experiment-id>/run.json` and its companion `model.json` when a
predictor exists. Execution status and reproduction claims are separate. Existing
entries below retain their historical structure and evidence limitations.

## Entry template

```markdown
## <experiment-id> — YYYY-MM-DD

- **Method / version:**
- **Purpose:** smoke test / original reproduction / Waterbirds adaptation / final comparison
- **Branch or commit:**
- **Configuration:** `configs/...` or `experiments/...`
- **Run record / frozen predictor:** `experiments/<experiment-id>/run.json`; `model.json` or why unavailable
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
- **Configuration:** [Tracked actual settings and provenance](../experiments/exp-dpe-waterbirds-reweighting-003/record.json). Executed implementation uses `configs/dpe_head_reweighting/aggregation_comparison.yaml` at the run commit. Backbone: ImageNet-1K V2 ResNet-50, 100 epochs, SGD LR 0.001, momentum 0.9, weight decay 0, class-balanced sampling, deterministic preprocessing. Heads: 15 × 20 epochs; combiner fits: 400 fixed steps.
- **Dataset and split:** Official Waterbirds 4,795 / 1,199 / 5,794. Validation is partitioned with fixed split seed 0 into disjoint head-training (720) and combiner-training (479) sets. Head fitting and epoch selection use only head-training; all weighting/combiner fitting uses only combiner-training. All configurations freeze before test evaluation. No separate candidate-selection split.
- **Seed:** 0, 1, 2; same validation partition and test identities across seeds.
- **Environment and hardware:** Kaggle `NvidiaTeslaT4` machine shape, CUDA for backbone/heads; CPU float64 neural combiners. Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.1.3, SciPy 1.16.3. GPU count not recorded. Papermill runtime 4 h 51 min 20 s, 2026-11-10 15:01:38–19:52:58 Asia/Saigon.
- **Command:** Run All in the [executed notebook](https://www.kaggle.com/kernels/scriptcontent/367976457/download), using `MODE="auto"`, `SEEDS=[0,1,2]`, `TRACK="standard"`, `EVALUATE_TEST=True`, `DEVICE="auto"`; `run_study` uses the recorded aggregation configuration. No model training was rerun during review.
- **Output location:** [Immutable Kaggle version](https://www.kaggle.com/code/noerula/dpe-reweighting?scriptVersionId=367976457); study `aggregation-standard-seeds012-0dcf563966c5`. Raw review evidence in `outputs/dpe-kaggle-v4/` (untracked). Historical tables and figures in [supporting-evidence archive](../results/dpe/supporting_evidence.zip) under `results/dpe/head_aggregation_kaggle_v4/`.
- **Results:** Mean ± sample SD over three seeds: uniform test WGA **89.5196 ± 0.6538%**, overall **91.2668 ± 0.3293%**. Dropout MLP (either weight decay): WGA **89.9926 ± 0.0923%**, overall **91.4049 ± 0.1726%**, paired WGA change **+0.4730 ± 0.6672 pp**. Per-seed WGA gains **+0.1330 / +0.0443 / +1.2417 pp**. Ordinary gate WGA **86.5780 ± 2.5189%**, change **−2.9416 pp**. No global weighting configuration improves mean WGA (one tie, 18 lower). All per-group accuracies and counts in the supporting archive, in order `00/01/10/11`, test counts `2255/2255/642/642`.
- **Paper comparison or expected behavior:** The simple global-weighting H2 mechanism is unsupported by this run. A small neural-stacker signal merits a prespecified calibration control. Compare with this run's matched uniform reference, not the earlier 89.00% single-seed reproduction; training profile and data allocation differ.
- **Status:** Completed exploratory Waterbirds adaptation; not a paper reproduction or final selected-method benchmark. The test set previously informed project research, and all 30 configurations are independently reported rather than selected by test.
- **Notes and follow-up:** Detailed findings and log audit in the supporting archive. Independent NumPy inference reproduced all 90 test results and training metric records, group counts, NLL, paired error IDs and summary statistics. Seed 0's joint lambda-0.01 solver fallback passed its saved numerical certificate. Gates concentrated on roughly one effective head. The user-run Kaggle verification, documented in the supporting archive, subsequently reproduced all 90 results, saved states and full test-image inference on all three original source seeds. Additional combiner runs do not establish dropout superiority over plain MLP. The subsequent five-seed verification is recorded below. Each configuration remains independently evaluated without a required selection split.

## dpe_seed0 — 2026-11-10

- **Method / version:** Original DPE (`baselines/dpe4subpop/`), 15-head IsoMax+ prototype ensemble with covariance regularization and equal probability averaging at temperature 1.0.
- **Purpose:** Original reproduction (Stage 1+ 15-head ensemble fitted on saved Stage-0 backbone features and exported to canonical v1 predictions for multi-method failure-case comparison).
- **Branch or commit:** `feat/prediction-csv-protocol`, commit `423a9d15bdbcb915f96f8caf0af627050bb9b44a`.
- **Configuration:** [`configs/dpe4subpop/waterbirds_stage1_dpe.yaml`](../configs/dpe4subpop/waterbirds_stage1_dpe.yaml), [`notebooks/dpe_waterbirds_end_to_end.ipynb`](../notebooks/dpe_waterbirds_end_to_end.ipynb).
- **Run record / frozen predictor:** [`experiments/dpe_seed0/run.json`](../experiments/dpe_seed0/run.json); [`experiments/dpe_seed0/model.json`](../experiments/dpe_seed0/model.json).
- **Dataset and split:** Official Waterbirds (`waterbird_complete95_forest2water2`), val=1,199 (group-balanced 532-sample draws per stage for fitting; full 1,199 for per-stage epoch selection), test=5,794 (`2,255 / 2,255 / 642 / 642`).
- **Seed:** 0.
- **Environment and hardware:** Local Windows 10 CPU, Python 3.11.1, NumPy 2.2.1, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu.
- **Command:** Stage 1+ `train_prototypes` loop wrapped with `capture_notebook_heads`, persisted via `save_dpe_heads` and `package_notebook_features`, and exported via `export_dpe_heads` to `experiments/dpe_seed0/predictions/test`.
- **Output location:** `experiments/dpe_seed0/predictions/test/{predictions.csv,provenance.json}` and `outputs/dpe_seed0_heads.pt` (untracked).
- **Results:** Held-out test overall accuracy: **95.7542%** (`0.957542`, 246 errors / 5,794); per-group accuracy `G1..G4`: **[97.7384%, 94.4568%, 93.9252%, 95.1713%]** (counts `2,255 / 2,255 / 642 / 642`); worst-group accuracy (WGA): **93.9252%** (`0.939252`).
- **Paper comparison or expected behavior:** Annotated DPE $\text{ERM}^*$ paper reference is `94.1 ± 0.4%` test WGA; this single-seed ensemble achieves `93.93%` test WGA.
- **Status:** completed execution; partial reproduction (single seed; Stage 1+ heads newly trained on CPU from the saved Stage-0 backbone checkpoint `checkpoints/dpe_ckpt_last.zip` because historical Kaggle sessions did not save head weights).
- **Notes and follow-up:** Verified numerical equivalence between native `IsoMaxPlusLossFirstPart` and `src.evaluation.dpe_export.predict_heads` on all 1,199 validation examples (`max_abs_diff = 5.96e-8`, `label_agreement = 1.0`).

## erm_stage0_seed0 — 2026-11-10

- **Method / version:** Shared Stage-0 ERM ResNet-50 backbone and linear classifier (`src/models/resnet.py`, raw 2048-D feature evaluation).
- **Purpose:** Corrected raw-feature Stage-0 ERM evaluation and canonical v1 test prediction export from `checkpoints/dpe_ckpt_last.zip`.
- **Branch or commit:** `feat/prediction-csv-protocol`, commit `423a9d15bdbcb915f96f8caf0af627050bb9b44a`.
- **Configuration:** [`configs/dpe4subpop/waterbirds_stage0_erm.yaml`](../configs/dpe4subpop/waterbirds_stage0_erm.yaml).
- **Run record / frozen predictor:** [`experiments/erm_stage0_seed0/run.json`](../experiments/erm_stage0_seed0/run.json); [`experiments/erm_stage0_seed0/model.json`](../experiments/erm_stage0_seed0/model.json).
- **Dataset and split:** Official Waterbirds (`waterbird_complete95_forest2water2`), val=1,199, test=5,794 (`2,255 / 2,255 / 642 / 642`).
- **Seed:** 0.
- **Environment and hardware:** Local Windows 10 CPU, Python 3.11.1, NumPy 2.2.1, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu.
- **Command:** Raw feature extraction (`Resize(256,256) -> CenterCrop(224) -> ImageNet normalize`) and linear head evaluation exported via `src.evaluation.prediction_export.write_predictions` to `experiments/erm_stage0_seed0/predictions/test`.
- **Output location:** `experiments/erm_stage0_seed0/predictions/test/{predictions.csv,provenance.json}` (untracked).
- **Results:** Held-out test overall accuracy: **92.3542%** (`0.923542`, 443 errors / 5,794); per-group accuracy `G1..G4`: **[99.7339%, 88.2483%, 76.6355%, 96.5732%]** (counts `2,255 / 2,255 / 642 / 642`); worst-group accuracy (WGA): **76.6355%** (`0.766355`). Validation overall/WGA: `93.0776%` / `78.9474%`.
- **Paper comparison or expected behavior:** Empirical accuracy (`92.35%` overall, `76.64%` WGA) aligns with the 100-epoch augmented $\text{ERM}^*$ Stage-0 track (`exp-dpe-waterbirds-ermstar-001`).
- **Status:** completed execution; partial reproduction.
- **Notes and follow-up:** `checkpoints/dpe_ckpt_last.zip` (`sha256: 84cf8067...`) is a bare state dict without embedded training logs; its raw-feature evaluation resolves the legacy notebook standardization mismatch for this saved checkpoint.

## poter_seed1 — 2026-11-10

- **Method / version:** POTER (`baselines/poter/`, NeurIPS 2026 official baseline reproduction).
- **Purpose:** Evaluate validation-selected checkpoint `best_validation.pt` (epoch 258, 0-based) on the official Waterbirds test split and export canonical v1 per-example predictions.
- **Branch or commit:** `feat/prediction-csv-protocol`, commit `423a9d15bdbcb915f96f8caf0af627050bb9b44a` (trained on `phat/poter-windows-cmnist`).
- **Configuration:** [`configs/poter/waterbirds_standard.yaml`](../configs/poter/waterbirds_standard.yaml) and `checkpoints/poter-results-backup.zip#CUB/Clean/POTER_CUB_seed1_kaggle/config.json`.
- **Run record / frozen predictor:** [`experiments/poter_seed1/run.json`](../experiments/poter_seed1/run.json); [`experiments/poter_seed1/model.json`](../experiments/poter_seed1/model.json).
- **Dataset and split:** Official Waterbirds (`waterbird_complete95_forest2water2`), test=5,794 (`2,255 / 2,255 / 642 / 642`).
- **Seed:** 1.
- **Environment and hardware:** Trained on Kaggle Linux 1× Tesla T4 (`torch 2.11.0+cu130`); exported on local Windows 10 CPU (`torch 2.5.1+cpu`).
- **Command:** `python scripts/export_predictions.py poter --checkpoint checkpoints/poter_extracted/CUB/Clean/POTER_CUB_seed1_kaggle/best_validation.pt --metadata dataset/waterbird_complete95_forest2water2/metadata.csv --data-dir dataset --run-label poter_seed1 --seed 1 --output experiments/poter_seed1/predictions/test --batch-size 128`
- **Output location:** `experiments/poter_seed1/predictions/test/{predictions.csv,provenance.json}` (untracked).
- **Results:** Held-out test overall accuracy: **91.9917%** (`0.919917`, 464 errors / 5,794); per-group accuracy `G1..G4`: **[92.9047%, 91.0421%, 92.5234%, 91.5888%]** (counts `2,255 / 2,255 / 642 / 642`); worst-group accuracy (WGA): **91.0421%** (`0.910421`).
- **Paper comparison or expected behavior:** Paper reports `90.9 ± 0.3%` test WGA across 3 seeds; seed 1 achieves `91.04%` test WGA.
- **Status:** completed execution; partial reproduction (single seed; seeds 2 and 3 pending).
- **Notes and follow-up:** Follows up [`exp-poter-waterbirds-standard-seed1-001`](#exp-poter-waterbirds-standard-seed1-001--2026-11-04) by recovering `best_validation.pt` and `test.csv` from `checkpoints/poter-results-backup.zip` and exporting full per-example test predictions.

## exp-dpe-h2-rules-waterbirds-standard-seeds012-003 — 2026-11-09

The date is the documentation date; the supplied log records elapsed times only.

- **Method / version:** H2, standard-track DPE aggregation; uniform, power, softmax, top-k and joint rules.
- **Purpose:** Exploratory three-seed comparison with matched frozen heads and a fixed 20-candidate grid.
- **Branch or commit:** `experiment/dpe-multiple-weighting-rules`, executed `dbeba0f3d1659596e605395bf5235d5d1b6c48e2`.
- **Configuration:** [Rule grid](../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/config.yaml), [study plan](../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/plan.json), and portable per-seed H2/backbone configurations linked from [run metadata](../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/run.json).
- **Dataset and split:** Official Waterbirds 4,795/1,199/5,794; A/B/C=720/241/238, fixed partition seed 0. B fits weights; C freezes choices before new test inference.
- **Seed:** 0/1/2. Seed 0 reuses the earlier H2 bundle; seeds 1/2 each train a 100-epoch backbone and 15 heads. Earlier seed-0 comparisons are not additional independent replicates.
- **Environment and hardware:** Kaggle CUDA runtime; Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0. Physical GPU model/utilization is not recorded.
- **Command:** Run All in `notebooks/dpe_weighting_rules.ipynb`; `study.run_study` delegates to `rule_suite.select`/`evaluate`.
- **Output location:** Untracked `WORK_DIR/studies/standard-seeds012-aed182e8ca.zip`; [raw result tables](../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/summary.json); [H2 hypothesis summary](../src/methods/dpe_head_reweighting/reports/h2.md).
- **Results:** Uniform test accuracy/WGA=91.2668±0.3293/89.5196±0.6538%; overall C selection=91.2553±0.3317/89.4752±0.6339%. Values are means±sample SD across training seeds. Paired WGA change=−0.0443±0.0768 pp: seed 0 selects top-10 (−0.1330 pp); seeds 1/2 select uniform (0 pp). Power, softmax and joint families select uniform in every seed. [Per-group counts and accuracies](../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/group_metrics.csv).
- **Paper comparison or expected behavior:** No selected reweighting improves test WGA. H2 changes the head-training data budget and is not a paper reproduction.
- **Status:** completed exploratory adaptation; all 60 B/C candidates refitted, selected test metrics and aggregate statistics recomputed from included prediction caches.
- **Notes and follow-up:** Joint NLL improves while WGA ties or decreases; conditional weighting and selection-stability explanations remain hypotheses. [Evidence, limitations and proposed tests](../src/methods/dpe_head_reweighting/reports/h2.md). Neural stacking/gating has not been run. Test inspection and the shared validation partition limit confirmatory interpretation.

## exp-dpe-h2-rules-waterbirds-standard-seeds012-002 — 2026-11-09

The date is the review date; the supplied corrected `download.txt` records elapsed
times, not the execution date. See the [failure audit and log identity](../src/methods/dpe_head_reweighting/AUDIT.md#multi-seed-joint-solver-failure--reviewed-2026-11-09).

- **Method / version:** H2 automated multi-seed weighting study; project-owned DPE adaptation.
- **Purpose:** Continue the fixed 20-candidate comparison across standard-track seeds 0/1/2.
- **Branch or commit:** `experiment/dpe-multiple-weighting-rules`; the exact executed commit is not printed in the log. The traceback uses the original SLSQP-only fitter.
- **Configuration:** [Study defaults](../configs/dpe_head_reweighting/study.yaml), [fixed weighting grid](../configs/dpe_head_reweighting/weighting_rules.yaml), [standard H2](../configs/dpe_head_reweighting/waterbirds_standard.yaml) and [standard backbone](../configs/erm/waterbirds_standard.yaml); no resolved configuration archive was supplied for this failed run.
- **Dataset and split:** Official Waterbirds; intended A/B/C fit/select separation from the standard H2 configuration. The epoch-100 log reports 1,199 validation examples.
- **Seed:** Planned 0/1/2; seed 0 reused an attached H2 bundle, seed 1 trained, seed 2 was not reached.
- **Environment and hardware:** Kaggle; user screenshot reports GPU T4 ×2. Python 3.13 appears in the traceback; package versions and actual multi-GPU utilization are not established by this log.
- **Command:** Run All in `notebooks/dpe_weighting_rules.ipynb`, calling `study.run_study`; no shell training command is claimed.
- **Output location:** Untracked WORK_DIR-relative `studies/standard-seeds012-865e9b8015.zip` (seed-0 partial study), `backbones/`, `features/` and `h2_sources/`; preserve the failed notebook version's saved outputs.
- **Results:** Seed 1 completed 100 backbone epochs/3,700 updates and all 15 heads. Final-epoch backbone validation overall/WGA=86.4887/62.4060%, group accuracies=99.3576/77.6824/62.4060/96.2406%, counts=467/466/133/133. These are backbone validation metrics, not DPE test results. Seed-1 final DPE test metrics are absent.
- **Paper comparison or expected behavior:** No new completed multi-seed comparison or reproduction claim.
- **Status:** partial reproduction; failed at 5,969.9 elapsed seconds during joint-weight fitting, with SLSQP `Positive directional derivative for linesearch`.
- **Notes and follow-up:** The failing lambda is not logged in this failed attempt. The completed rerun is [seeds012-003](#exp-dpe-h2-rules-waterbirds-standard-seeds012-003--2026-11-09): seed 1 trained again and all 100 logged epoch records match the earlier attempt. Its λ=0.01 line-search failure is handled by the recovery solver. The original failed-version checkpoint/cache identities remain unverified because only its log was supplied; this entry retains its failed status.

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
- **Output location:** Suite `h2-standard-seed0-rules-6452ea10e0` and its supplied ZIP, retained outside Git; [raw metrics](../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seed0-001/metrics.json).
- **Results:** Overall C selection chooses top-k10, improving C WGA from 89.2473% to 90.3226% by correcting one image. Uniform test accuracy/WGA=91.2323/89.7561%; top-k10=91.1978/89.6231%, paired differences=-0.0345/-0.1330 percentage points. Top-k10 test group accuracies=92.5055/89.6231/91.5888/91.7445%, counts=2,255/2,255/642/642. Of 56 changed test predictions, 27 errors are fixed and 29 introduced. Power, softmax and joint families select uniform; their selected results equal the control.
- **Paper comparison or expected behavior:** A matched within-run aggregation comparison, not a paper reproduction. No selected weighting procedure improves test WGA in this seed.
- **Status:** exploratory; completed single-seed comparison with aggregate-record consistency review, not an independent prediction-cache audit.
- **Notes and follow-up:** Selection/result hashes, recorded code identities, nonjoint weight formulas, candidate selection and metric/count arithmetic agree. The supplied suite ZIP excludes prediction caches and heads, so sample-level metrics and joint optimization cannot be independently rerun from it. Preserve the C-selected top-k10 outcome; complete seeds 1/2 with the frozen grid and report paired differences. See [full interpretation and verification limits](../src/methods/dpe_head_reweighting/reports/dpe_h2_rules_seed0.md).

**Follow-up:** The completed [three-seed study](#exp-dpe-h2-rules-waterbirds-standard-seeds012-003--2026-11-09) now supplies prediction bundles and solver refits. This earlier single-seed record and its original review limits are retained.

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
- **Output location:** Original `outputs/dpe_head_reweighting_runs/h2-standard-seed0/` and its ZIP, untracked; [raw metrics](../results/dpe_head_reweighting/exp-dpe-h2-waterbirds-standard-seed0-001/metrics.json).
- **Results:** Uniform and selected test overall accuracy 91.2323%; group accuracies 00/01/10/11=92.7273/89.7561/91.1215/91.2773%, counts=2,255/2,255/642/642; WGA 89.7561%. Paired WGA and overall differences=0.0000 percentage points. C selected uniform `(alpha=0, gamma=1)`; all 13 candidates predict the same C labels. Final backbone validation accuracy/WGA=86.4887/60.1504%; no ERM-only test result is established.
- **Paper comparison or expected behavior:** The annotated paper reference is 91.0 ± 0.4% WGA, but H2 reserves B/C and trains heads using A; its single-seed score is not a like-for-like paper reproduction.
- **Status:** exploratory; completed single-seed adaptation, independently audited saved predictions and records.
- **Notes and follow-up:** No weighting improvement demonstrated. [Full candidate comparison and audit evidence](../src/methods/dpe_head_reweighting/reports/dpe_h2_power_seed0.md) reproduce the saved decision. Raw-image feature extraction could not be independently checked because images/backbone/raw features are excluded from the supplied archive. [Additional summary diagnostics](../src/methods/dpe_head_reweighting/reports/dpe_h2_power_seed0.md#additional-summary-diagnostics--2026-11-08) find B-to-C head-WGA rank correlation -0.0276 and effective heads 14.3720 even for the strongest candidate. These are aggregate-record calculations, not a new run or archive re-audit. Complete seeds 1/2; check score stability and consider regularized joint weights as separately labeled, unmeasured follow-ups.

**Follow-up:** The fixed multi-rule grid, including joint weights, has subsequently been evaluated for seeds 0/1/2 in [seeds012-003](#exp-dpe-h2-rules-waterbirds-standard-seeds012-003--2026-11-09). The original power experiment remains a distinct historical record.


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
- **Paper comparison or expected behavior:** The annotated DPE reference is 94.1±0.4% test WGA. The reported 92.70% is a single historical run with a different 100-epoch backbone recipe; this is not a verified paper reproduction. The legacy ERM evaluation mismatch prevents a valid ERM-only comparison.
- **Status:** historical reproduction; corrected ERM evaluation pending
- **Notes and follow-up:** Preserve the reported DPE ablations; their 0.1-pp range is descriptive. Correct the legacy ERM evaluation before interpreting the augmentation effect or comparing it with standard ERM. This editorial clarification does not replace historical measurements.


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
- **Paper comparison or expected behavior:** The annotated DPE reference is 91.0±0.4% test WGA. The reported single-seed 89.00% is 2.0 pp lower. Full DPE exceeds the random-subset/no-covariance ablation by 0.8 pp in this run; the result does not establish a general regularization benefit. Legacy ERM metrics require corrected evaluation before a baseline comparison.
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

## exp-dpe-waterbirds-five-seed-verification-004 — analyzed 2026-11-12

- **Method:** learned prediction aggregation of frozen DPE-style heads; Waterbirds / ResNet-50 fixed.
- **Sources:** seeds 0–4, with 0–2 reused and 3–4 newly trained; 30 configurations each; primary validation partition 0.
- **Source:** commit `8e14249f4b3510363be06f627182e2daa9926179`; original saved Kaggle version 368677926; final recovery version not identified in uploaded evidence.
- **Status:** completed after documented auxiliary solver recovery. Both new backbones were retained.
- **Results:** uniform WGA 89.2337%; dropout MLP 89.8537/89.8625% (wd 0.01/0.001); plain MLP 89.9578/89.9667%. Dropout improves over uniform on each primary source but the source-level 95% intervals include zero. Dropout mean gains reverse on alternative partition 2.
- **Evidence:** [scientific report](../results/dpe/five_seed_verification_20261112/README.md), [experiment record](../experiments/exp-dpe-waterbirds-five-seed-verification-004/record.json), and [audit script](../scripts/analysis/audit_dpe_five_seed.py).
- **Claims:** promising primary-partition aggregation gain; neither dropout-specific superiority nor partition-independent improvement established. 200 initialization fits and alternative partitions are nested within five sources.

## exp-dpe-ermstar-neural — prepared, no benchmark run

400-epoch ERM* configuration from upstream Waterbirds launcher; original prototype
loss/training/subsampling definitions reused. User-approved comparison includes
full-validation annotated DPE uniform and a separate720/479 neural arm with its own
uniform baseline. Seeds0/1 run independently on two T4s; epoch/head recovery retained.
[Methods and limitations](../src/methods/dpe_ermstar/README.md),
[notebook](../notebooks/dpe_ermstar_neural_two_gpu_kaggle.ipynb),
[settings](../configs/dpe_ermstar/study.json). Synthetic logic checks only; no heavy
training or paper-reproduction/superiority claim made in this preparation.

The ERM* notebook was simplified at the researcher’s request: ordinary repository
imports, visible experiment code/configuration, stable readable directories, and
configuration/row-based recovery replace file checksums and encoded code bundles.
The shared ERM trainer now uses readable configuration and dataset-row records for recovery.

The ERM* interface now uses a conventional repository layout: a short notebook
launcher, an ordinary CLI, Python modules and a configuration file. Module strings,
notebook file-writing cells and runtime source extraction were removed. Changes are
published on `experiment/dpe-ermstar-neural`, the notebook’s default Kaggle checkout.

## DPE weighting with isolated OOF head/gate fitting — 2026-11-14

- **Execution / scope:** completed exploratory adaptation; 127 configurations
  across 41 deployable method families plus the oracle diagnostic, three-seed
  finalist confirmation and automatic winner ablation. Additional locked A18
  diagnostic uses five-fold cross-fitting. Original checkpoints preserved.
- **Source / settings:** commit `dc39c2fcafe75ddd413cd25b9f394680bb06fd1c`
  plus retained local source snapshots/hashes; settings in
  [config.yaml](../experiments/weighting/config.yaml). CPU; shared supplied
  frozen backbone. Full source/environment identities are in the output manifest.
- **Protocol:** new class-balanced heads use three-fold OOF predictions; gate
  train/select is 80/20 by class; group labels are evaluation-only. Winner D19
  locks before final test evaluation. Backbone training history is not embedded
  in the supplied checkpoint and remains an explicit provenance limitation.
- **Results:** D19 test accuracy 95.961 ± 0.422%, WGA 83.645 ± 4.083%; matched
  U0 accuracy 95.708 ± 0.117%, WGA 92.761 ± 0.552%. Supplied original group-aware
  DPE reference: accuracy 96.048%, WGA 94.393%. Three repeats share one backbone;
  SD describes refitted heads/gates, not independent backbone training. Results
  do not support replacing original DPE. Full report and uncertainty:
  [research report](../results/dpe/weighting_20261114/README.md).
- **Winner run/model records:**
  [seed 0](../experiments/exp-dpe-weighting-d19-seed0-fa5f0fb8/run.json),
  [seed 1](../experiments/exp-dpe-weighting-d19-seed1-8728316a/run.json),
  [seed 2](../experiments/exp-dpe-weighting-d19-seed2-f5b69e37/run.json).
  Each record links its model and canonical full-test prediction export.
- **Validation / recovery:** 51 new tests passed; 21 compatible historical tests
  passed, five platform-specific historical tests excluded after documented
  Windows failures. Independent audit verified 84 final predictors, nine folds,
  all 42 ranks and unchanged original checkpoint hashes. Outputs, 329 weighting
  checkpoints, failure cases and resume instructions remain in
  `outputs/weighting/`; see [workflow status](../experiments/weighting/STATUS.md).

## exp-dpe-ermstar-neural — completed two-source exploratory run

- Run: public Kaggle `noerula/dpe-exploration`, saved version 3, retrieved 2026-11-13.
- Waterbirds/ResNet-50; 400 epochs; independent seeds 0/1 on two T4 GPUs. Both completed.
- Matched uniform mean WGA 93.4161%; logistic 94.0031%; plain/dropout MLP WD .01
  93.7600/93.8265%; full-validation DPE uniform 94.0810%. Every configuration retained.
- Plain/dropout MLPs improve matched uniform on both sources; logistic wins one and
  loses one. No confirmed dropout, partition-independent, paper-reference superiority
  or H3 claim.
- [Scientific report and independent replay](../results/dpe/ermstar_neural_two_seed_20261113/README.md).

## exp-dpe-independent-frozen-20261113 — completed independent-method study

- **Purpose/status:** exploratory prototype competition, prototype importance,
  residual aggregation and cross-fitting. All 5,880 declared rows completed:
  5,222 distinct predictors plus 658 algebraic aliases; no failed configurations
  and no candidate selected for final testing.
- **Sources:** five saved normal-ERM backbones (seeds 0–4) and two saved ERM*
  backbones (0–1), kept separate. Base revision
  `22b65ec5fc8f00b33938fe2af6dc32ad2f34ecfd` plus retained local source snapshots.
  No backbone training. Lightweight missing-allocation/fold heads were fitted.
- **Execution:** local CPU workers, fixed full-validation geometry grid,
  720/479 allocation seeds 0/1/2, and five-fold crossfit. A fresh Kaggle launcher
  was prepared; no new Kaggle run was submitted. Source and portable evidence are included
  in the independent-method publication.
- **Records/artifacts:** grouped study; every source/allocation/configuration has
  its own run/model records, fitted state and canonical test CSV outside Git.
  [Configuration](../configs/dpe_independent/study.yaml),
  [report and all portable tables](../results/dpe/independent_methods_20261113/README.md),
  [read-only audit](../scripts/analysis/audit_dpe_independent.py),
  [method/launcher documentation](../src/methods/dpe_independent/README.md).
- **Results:** fixed shared-scale prototype importance at T=.05, maximum-group
  loss and λ=.01 reaches primary mean WGA 90.5518% (normal ERM) and 94.1910%
  (ERM*), but its gains reverse on another allocation. Fixed average-loss linear
  residual at λ=.1 gives normal-ERM paired mean gains +1.1141/+0.4018/+0.5567 pp
  across the three allocations; that pattern does not transfer to ERM*.
- **Verification/limits:** 219 tests passed, 9 skipped; all 5,880 canonical CSVs
  independently audited against 5,794 test identities each. Initial float32 CPU
  preflight failures were retained; float64 normalized distances meet the original
  2e-7 historical replay tolerance without changed baseline decisions. All final
  Brier fits pass the unchanged 1e-7 certificate threshold. No independent CNN
  image replay, untouched confirmation, causal shortcut-removal or novelty claim.
