# Experiment 2 — interim report: two POTER seeds + existing baselines

Updated 2026-11-18T17:21:02.515869+00:00. This replaces execution priority, not the archived scientific results in `outputs/failure_verification/`.

## Status and cost decision

**Independent-seed validation remains INCOMPLETE. Only POTER seed1 has finished.**
Local process inspection found 0 training jobs; Lab/Explorer are running and were preserved. No job was stopped, no training started, and no checkpoint was overwritten. Cloud status is **unverified**: no runtime/job ID was supplied. Local status is not evidence that a remote job is absent.

The retained seed1 checkpoint has completed epoch 299 (zero-based; 300 total); the selected checkpoint is epoch 258 by validation WGA. Its `last_training.pt` contains model, optimizer, RNG and loss state. It is a resume state for **seed1 only**, not initialization for independent seed2. Hashes/config/source fingerprints are saved in `resume_audit.json`. Older seed2/3 logs stop at epoch0 with no saved weights; their cause of stopping is unknown.

**New priority: obtain exactly one additional independently trained POTER seed (seed2), then compare seeds1/2 against existing ERM_enhanced seed0 and DPE_legacy seed0.** If a compatible in-progress seed2 is discovered, restore its full state and continue it; otherwise begin seed2 from the original ImageNet initialization, never from fine-tuned seed1. Defer the third POTER seed (seed0) and additional ERM/DPE training.

Historical 300-epoch run: mean 108.65s/epoch, peak 10.63 GiB. New expected POTER training cost **9.05 GPU-hours** on comparable T4 throughput, versus 18.11 for two additional seeds: about **50% less POTER training**. This is extrapolation from the saved run, not a new cloud benchmark or a price quote. Add dataset staging, checkpoint I/O, interruptions and 2,100-image evaluation. Baseline training cost is deferred and was not benchmarked. For an existing matching run at completed epoch e, remaining estimate is `(299-e)*108.65/3600` hours plus overhead, never reset to epoch0.

## Checkpoint search and preservation

Fresh read-only search hashed 783 candidates under the configured repository, sibling checkpoint/output roots, D:/POTER_logs, D:/DPE_Research and explicit Downloads candidates. No additional verified compatible independently trained POTER checkpoint was found. See `checkpoint_candidates.csv` for every candidate/reason. This is a bounded local search, not an authenticated search of all cloud storage. Metadata paths under `/kaggle/working` are not downloaded checkpoints.

- POTER seed1 best/last snapshots belong to one training run; duplicates do not increase n.
- The batch16/smoke seed1 checkpoints change recipe and/or repeat seed1; CMNIST checkpoints use another dataset.
- Standard-ERM seed3 and its partitioned DPE heads differ from the enhanced-ERM/full-validation legacy recipe; they remain excluded.
- Reuse current enhanced-ERM seed0 and legacy DPE seed0. Legacy DPE historical head/backbone hash binding remains absent; results retain that exploratory limitation.
- **Additional missing asset:** local OT cache `D:\POTER_features\waterbirds_standard.pt` has SHA `1ad34619e16d36c26cf2af2eb0a94c99762aa62fa1959a15e1033cb98d017f0a`, whereas seed1 records `8a7636923bffaddce16f843b38afe0a97743f19a151339ddb96e200d2c4d2258`. Byte hashes differ; tensor equivalence to the historical cache is unverified. Recover the original cloud cache before primary seed2 training. Do not silently replace it or infer equivalence from matching filenames/shapes. Local dataset metadata/splits are complete; dataset availability does not resolve this cache provenance gap.
- Resume uses the original runner's atomic epoch checkpoint, exact config/environment/source comparison and optimizer/RNG/loss restoration. Copy a durable full output archive before a cloud session ends. Never interrupt mid-epoch to save compute; incomplete-epoch work is not committed. No active job should be stopped without verifying a valid durable checkpoint first. Keep output directories separate; a full-run checkpoint is not an independent seed.

## Measured bottleneck evidence and optimization decisions

The original runner computes fixed OT weights once, then trains with weighted replacement sampling, full-backbone SGD, FP32, batch128 and workers0. Each epoch also evaluates validation **and full test** and saves checkpoints. Test results do not affect selection; only validation WGA does. Training/validation/test phase breakdown was not recorded historically, so the dominant end-to-end bottleneck is **not yet established**.

CPU-only measurements used 256 predeclared **train** images and the actual POTER augmentation; no model/backward/optimizer was executed:

| Probe | Measured time | Meaning |
|---|---:|---|
| workers0, 2 batches | 2.721s | Local input-only cost, no GPU overlap |
| workers2, 2 batches | 8.265s | Includes Windows spawn/startup; not steady-state cloud throughput |
| 64 compressed file reads | 0.0240s | Warm/local file I/O sample |
| 64 file decode+RGB | 0.0950s | JPEG decode cost |
| 64 in-memory-byte decode+RGB | 0.0902s | Exact decoded RGB equality verified |

The short workers2 probe is slower here and does not justify a cloud speedup claim. No production worker change is enabled. A prospective authorized run can use `scripts/profile_poter_training.py` to time original train/validation/test phases and host DataLoader wait without changing sample order, augmentation, optimizer or loss. It adds boundary synchronization overhead and is not a GPU-kernel profile; phase metrics remain distinct from the original epoch telemetry. It requires an explicit training confirmation. This GPU-side profiling has **not** run.

| Candidate optimization | Decision |
|---|---|
| Existing checkpoint and prediction cache | Adopt: no rerun of the 6,300 valid prediction rows. |
| Fixed OT feature cache | Preferred once recovered: reuse original SHA-bound cache; local similarly named cache does not match. No unvalidated fallback. |
| Stage unchanged compressed dataset/cache to cloud local disk | Adopt after byte-hash verification; preserve exact transforms. Staging is outside training RNG. |
| Cache decoded RGB before stochastic augmentation | Potentially method-preserving if exact pixels, ordering and RNG are verified; not enabled or speed-validated. |
| Cache augmented training tensors / freeze image features | Reject: changes stochastic augmentation or full-backbone training. |
| Extra workers/persistent workers/prefetch | Defer: same transform distribution can be retained, but RNG streams and exact resume behavior change; benchmark and prove parity before claiming recipe match. |
| Pin memory | Already enabled in original runner; not a new optimization. |
| AMP | Defer for primary seed2: autocast/gradient scaling changes numerical execution and may change optimization trajectory or validation-selected epoch. No AMP speed/memory or equivalence result measured; do not call it an FP32 replication. |
| Reduce epochs/batch, change architecture/POTER, early stop by test | Reject. Keep 300 epochs, batch128, original algorithm/selection. |
| Remove per-epoch full-test evaluation | Could reduce evaluation cost, but changes runner behavior/RNG consumption; not enabled in the exact-recipe job. |

PyTorch documents [AMP's autocast and gradient scaling](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [DataLoader workers/pinning](https://docs.pytorch.org/docs/stable/data.html), and [reproducibility/worker seeding](https://docs.pytorch.org/docs/stable/notes/randomness.html). Method-preservation judgments above are our inference from those behaviors and this runner's source; no speedup is assumed.

## Completed comparisons

All numbers below are **completed single-checkpoint comparisons**, reused from verified real predictions. They are not multi-seed means.

| Model | Seed | Sigma | AA (%) | WGA (%) | Landbird (%) | Waterbird (%) | Asymmetry (pp) |
|---|---|---|---|---|---|---|---|
| POTER | 1 | 0.0 | 93.25 | 86.00 | 88.00 | 98.50 | 0.00 |
| POTER | 1 | 0.5 | 92.00 | 83.00 | 86.00 | 98.00 | 1.50 |
| POTER | 1 | 1.0 | 88.75 | 74.00 | 78.50 | 99.00 | 10.00 |
| POTER | 1 | 2.0 | 82.50 | 58.00 | 65.00 | 100.00 | 24.50 |
| ERM_enhanced | 0 | 0.0 | 94.75 | 87.00 | 93.50 | 96.00 | 0.00 |
| ERM_enhanced | 0 | 0.5 | 93.00 | 86.00 | 93.00 | 93.00 | -2.50 |
| ERM_enhanced | 0 | 1.0 | 85.25 | 69.00 | 84.00 | 86.50 | 0.00 |
| ERM_enhanced | 0 | 2.0 | 74.00 | 39.00 | 68.00 | 80.00 | 9.50 |
| DPE_legacy | 0 | 0.0 | 96.25 | 90.00 | 93.00 | 99.50 | 0.00 |
| DPE_legacy | 0 | 0.5 | 96.25 | 91.00 | 93.00 | 99.50 | 0.00 |
| DPE_legacy | 0 | 1.0 | 90.25 | 80.00 | 81.50 | 99.00 | 11.00 |
| DPE_legacy | 0 | 2.0 | 85.00 | 64.00 | 71.00 | 99.00 | 21.50 |

POTER seed1, blurred Landbird sigma2: gray86% versus natural65%, interference **−21 pp [−30, −12]**, amplification relative to clean **−15 pp [−25, −5]**. Land→Water changes P(Waterbird) by **−8.77 pp [−10.72, −6.76]**; this does not support a simple water-background→Waterbird narrative. These CIs resample foregrounds, not training seeds.

## Pending comparisons and stopping rule

6,300 cached rows are complete (2,100 per available method). One additional POTER checkpoint requires only **2,100 new predictions**, giving **8,400** total in this cost-limited comparison. Use the same 100 foregrounds, four backgrounds, gray controls and original controls, sigmas0/.5/1/2, masks/geometry/preprocessing and pixel/tensor hashes. Exclude sigma4 from primary analysis. New inference is pending, not executed.

After seed2 reaches the fixed full 300-epoch target, select its strictly best validation-WGA checkpoint, export/hash it, then compute each seed's WGA/class accuracy/asymmetry/interference, paired foreground CIs and POTER two-seed mean±sample SD (ddof1). Report both seeds even if effects disagree. With n=2 this is **preliminary stability evidence**, not full three-seed validation or a reliable estimate of the seed distribution. Current n=1 has undefined training-seed SD; never substitute foreground CI or inference randomness.

The third POTER seed and additional baseline training stay deferred until a separate explicit decision. Opposite signs, materially different magnitudes or an inconclusive two-seed result warrant discussing more evidence, not silently starting extra jobs. Baselines remain one checkpoint each and cannot establish baseline seed variability. No test-driven hyperparameter/epoch/seed selection. The repeatedly inspected test cohort is diagnostic; future method development needs a separate development set and untouched evaluation.

## Deliverables and execution

`notebooks/experiment2_poter_cost_optimized.ipynb` is an offline-generated Kaggle notebook for **one additional POTER seed only**, with historical Waterbirds training source/config/cache hashes, environment checks, optional phase timing, resume and backup. One unrelated CelebA dataset source hash differs; the missing historical bytes and included hash are explicitly recorded, rather than fabricated. Waterbirds/CUB algorithm sources match the historical fingerprints. Default `CONFIRM_TRAINING=False` skips training; no ERM/DPE or third-seed training cell is included. Missing exact OT cache blocks primary launch; the local mismatched cache is rejected. No extra 2-epoch probe training is performed. Restore a trusted backup before rerunning; do not overwrite an existing different run. No cloud job was submitted here.

The notebook trains/exports a checkpoint. Exact counterfactual evaluation still requires importing it into this repository with the locked evaluation assets; no cloud evaluation or seed2 predictions are claimed by the notebook. `execution_plan.json` records cost priorities. `training_status.json`, `resume_audit.json`, `data_pipeline_profile.json`, checkpoint manifests and hashes record the evidence. Old results/checkpoints remain unchanged.

**Interim decision: independent-seed verification incomplete.** Finish or obtain one compatible POTER seed, then reassess; do not claim method-level reproducibility from seed1 alone.
