# POTER Execution Run Log

Mirrors `runs_log.csv`. Records every run attempt, execution parameters, resource utilization, and evaluation outcomes.

---

## 1. Setup & Environment

| Component | Setting |
| :--- | :--- |
| **Platform** | Kaggle Notebook Container |
| **GPU** | NVIDIA Tesla T4 (15.0 GiB VRAM) |
| **Environment** | PyTorch 2.11.0+cu130, CUDA 13.0 |
| **Dataset Source** | Stanford NLP Waterbirds archive (11,788 samples, SHA256 verified) |
| **Feature Extraction** | Pretrained ResNet-50 (`waterbirds_standard.pt`, 599 reference samples) |

---

## 2. Run Details

### Run 1: `probe_batch128` — Status: Success
- **Purpose:** Smoke test to verify batch size 128 fits within Tesla T4 VRAM.
- **Parameters:** `epochs=2`, `batch_size=128`, `max_hours=1`.
- **Duration:** 216.5 seconds (~3.6 minutes).
- **Peak VRAM:** 10,842 MiB (~10.59 GiB).
- **Verdict:** Success. Confirmed that batch size 128 executes safely on a 16 GB GPU.

### Run 2: `POTER_CUB_seed1_kaggle` — Status: Success
- **Purpose:** Full 300-epoch benchmark reproduction for Seed 1 on Waterbirds Clean.
- **Parameters:** `dataset=CUB`, `seed=1`, `epochs=300`, `batch_size=128`, `lr=1e-5`, `weight_decay=1.0`, `f_power=0.1`, `clip_cap=20`, `reg_scale=0.03`.
- **Duration:** 32,678.6 seconds (9.08 hours), average 108.6 seconds/epoch.
- **Peak VRAM:** 10,881 MiB (~10.63 GiB).
- **Outcomes:**
  - Completed all 300 epochs (`Epoch [0]` to `Epoch [299]`).
  - Terminal log confirmed: `Training complete.`.
  - Checkpoints committed: `best_validation.pt` (Epoch 258), `last_training.pt` (Epoch 299).
  - Validation metrics at Best Epoch (Epoch 258):
    - **Worst-Group Accuracy (WGA):** **90.10%**
    - **Average Accuracy:** **91.60%**
    - **Per-Group Accuracy:**
      - Group 0 (Landbird on Land): 92.50%
      - Group 1 (Landbird on Water, minority): 90.10%
      - Group 2 (Waterbird on Land, minority): 92.50%
      - Group 3 (Waterbird on Water): 92.50%
- **Verdict:** Success. Matches the paper's reported benchmark ($90.9 \pm 0.3\%$ WGA) within 0.8% on a single seed.

### Run 3: `summarize_all_seeds` — Status: Script Invocation Error
- **Purpose:** Automated summarization across seeds 1, 2, and 3.
- **Command:** `python scripts/summarize_waterbirds.py /kaggle/working/poter-results`
- **Output:** `summarize_waterbirds.py: error: No completed 300-epoch run for seeds [2, 3] in /kaggle/working/poter-results/CUB/Clean`
- **Analysis:** By default, the summarizer expects runs for all three seeds `[1, 2, 3]`. Because only Seed 1 had been trained, the script exited with status code 2. This error occurred post-training and post-checkpoint-backup, so no experimental artifacts were affected.
- **Remedy:** Specify `--seeds 1` when summarizing Seed 1 individually.
