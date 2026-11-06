# Reproducing POTER on Waterbirds

This guide covers running all 300 epochs for each seed on the standard Waterbirds configuration from the [official repository](https://github.com/Sung-Ho-Jo/POTER#reproducing-results). This is the canonical POTER on Waterbirds benchmark in Table 1 of the paper.

According to [paper, Appendix E](https://arxiv.org/html/2610.01028v1#A5), the authors used ImageNet-pretrained ResNet-50, SGD, learning rate `1e-5`, weight decay `1.0`, batch size `128`, 300 epochs, standard data augmentation, and three random seeds. The best model/epoch is selected strictly using Worst-Group Accuracy (WGA) on the validation set. Table 1 reports a test WGA of **$90.9 \pm 0.3\%$** on Waterbirds.

---

## Hardware Considerations

- **VRAM Requirement:** ResNet-50 with batch size 128 requires **> 10.8 GB VRAM** during the backward pass.
- **Kaggle Environment:** A single NVIDIA Tesla T4 (15–16 GB VRAM) is sufficient to run batch 128.
- **Laptop GPUs (6 GB):** Attempting batch size 128 on 6 GB VRAM will trigger a `torch.cuda.OutOfMemoryError`. If testing locally on 6 GB, reduce the batch size (e.g., to 16) and note that it is a reduced-batch exploratory run rather than a bit-for-bit reproduction of the published configuration.

---

## 1. Environment Verification

Ensure PyTorch detects a CUDA-capable GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Dataset Preparation

The official Waterbirds dataset archive (`waterbird_complete95_forest2water2.tar.gz`) from Stanford NLP contains 11,788 images:
- **Train split:** 4,795 images (Group 0: 3,498, Group 1: 184, Group 2: 56, Group 3: 1,057)
- **Validation split:** 1,199 images (Group 0: 467, Group 1: 466, Group 2: 133, Group 3: 133)
- **Test split:** 5,794 images (Group 0: 2,255, Group 1: 2,255, Group 2: 642, Group 3: 642)

Groups are defined as $\text{group} = 2 \times y + a$, where $y \in \{0, 1\}$ represents bird species and $a \in \{0, 1\}$ represents background.

Expected metadata location:
```text
data/cub/data/waterbird_complete95_forest2water2/metadata.csv
```

---

## 3. Pretrained Feature Extraction (One-time)

Run feature extraction to produce the offline cache for Optimal Transport (OT) potential computation:

```bash
python extract_features.py \
    --dataset CUB \
    --data_dir data \
    --ref_per_group 0:0,1:466,2:133,3:0 \
    --ref_seed 0 \
    --output features/waterbirds_standard.pt \
    --batch_size 32 \
    --num_workers 0 \
    --gpu 0
```

The reference set comprises 466 landbird-on-water (Group 1) and 133 waterbird-on-land (Group 2) samples from the validation split. Batch size 32 is used only for the evaluation-mode feature extraction.

---

## 4. Multi-Seed Training (300 Epochs per Seed)

Run training across seeds 1, 2, and 3:

```bash
for s in 1 2 3; do
    python -u run_expt.py \
        --dataset CUB \
        --data_dir data \
        --ot_cache features/waterbirds_standard.pt \
        --f_power 0.1 \
        --clip_cap 20 \
        --reg_scale 0.03 \
        --n_epochs 300 \
        --lr 1e-5 \
        --weight_decay 1.0 \
        --batch_size 128 \
        --augment_data \
        --seed $s \
        --gpu 0 \
        --log_every 50 \
        --log_dir results/poter/logs
done
```

On Windows CMD, replace the loop with:
```cmd
for %s in (1 2 3) do python -u run_expt.py --dataset CUB --data_dir data --ot_cache features\waterbirds_standard.pt --f_power 0.1 --clip_cap 20 --reg_scale 0.03 --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 --augment_data --seed %s --gpu 0 --log_every 50 --log_dir results\poter\logs
```

Each run requires completion through epoch 299 and printing `Training complete.`.

---

## 5. Model Selection & Metric Summarization

Model selection is strictly based on **validation Worst-Group Accuracy (WGA)**. The epoch achieving the highest validation WGA is selected (tie-breaker: earliest epoch), and test metrics are read from that exact epoch:

```bash
python scripts/summarize_waterbirds.py results/poter/logs --seeds 1 2 3
```

To summarize an individual completed seed (e.g., Seed 1):
```bash
python scripts/summarize_waterbirds.py results/poter/logs --seeds 1
```

The script reports:
- Selected epoch (zero-based)
- Validation Worst-Group Accuracy (WGA)
- Test Worst-Group Accuracy (Test WGA)
- Test Adjusted Average Accuracy: weighted average using training group proportions `[3498, 184, 56, 1057]` following the standard Waterbirds protocol.
- Mean $\pm$ sample standard deviation across evaluated seeds.


---

## 6. Required Result Artifacts

For every completed seed, retain the complete run directory containing `log.txt`, `val.csv`, `test.csv`, `config.json`, `best_validation.pt`, and `last_training.pt`. Run the single-seed summary before archiving:

```bash
python scripts/summarize_waterbirds.py results/poter/logs --seeds 1
```

Do not compare validation WGA with a published test WGA. A seed is eligible for benchmark reporting only after the validation-selected epoch has held-out test overall accuracy, per-group accuracies and counts, and test WGA. The paper-level reproduction additionally requires the three-seed test mean and sample standard deviation.
