# AutoBackSwap (preserved baseline)

Preserved copy of the official implementation of **AutoBackSwap**, plus notes on how we ran it on Waterbirds. The source files in this folder are the upstream files at one pinned commit, unchanged. Our notes and small additions are listed in "What is ours" below.

## Quick facts
| | |
|---|---|
| Date imported | 2026-11-05 |
| Paper | *Automated Background Swapping for Robustness against Spurious Backgrounds*, Roder and Schweighofer, [arXiv:2606.32018](https://arxiv.org/abs/2606.32018) |
| Source repository | https://github.com/cesro7/spur-corr, commit `b060532c2632039ffc2918828780d2b696763af6` |
| Details of the import | [`ORIGINAL_SOURCE.md`](ORIGINAL_SOURCE.md) |
| Upstream README | [`UPSTREAM_README.md`](UPSTREAM_README.md) (verbatim) |
| Reproduction status | **Partial reproduction** on Waterbirds (ResNet50). Spawrious and Spurious Vehicles were not run. |
| Results | [`results/autobackswap/summary.md`](../../results/autobackswap/summary.md) |
| Config | [`configs/autobackswap/`](../../configs/autobackswap/) |
| Experiment log | entries `exp-autobackswap-*` and `exp-erm-*` in [`docs/experiment_log.md`](../../docs/experiment_log.md) |

## What the method does
Waterbirds backgrounds (land or water) are spuriously correlated with the bird type. AutoBackSwap breaks that correlation by building new training images:

1. **Detector** (EfficientNet-B0, `--mode=ours_detector`): trained on patch labels (200 samples made from the segmentation masks) to tell which image patches are foreground.
2. **Generate** (`--mode=ours_generate`): remove the foreground, fill the hole with AOT-GAN inpainting, then paste other birds on the clean backgrounds. The new images are written to `data/waterbird_complete95_forest2water2/abs_seed<k>/`.
3. **Classifier** (`--mode=ours`): ResNet50 trained on the original plus generated images.

`--mode=erm` trains the same ResNet50 on the original images only (ERM baseline). In the paper's setting "without minority" means the training set has no bird on the opposite background; "with minority" adds those images.

Classifier settings from `methods/ours/config_wb.yaml` and `methods/erm/config_wb.yaml`: batch size 32, 50 epochs, SGD momentum 0.9, label smoothing 0.1, weighted loss, image size 224. AutoBackSwap lr 0.001 / weight decay 0.001; ERM lr 0.003 / weight decay 0.005. Model selection is early stopping on validation accuracy (`early_stopping: val_acc`). The test split is scored once, using the best-validation checkpoint.

Groups are `group = y*2 + place + 1`: 1 Landbird on Land (n=2255 in test), 2 Landbird on Water (2255), 3 Waterbird on Land (642), 4 Waterbird on Water (642).

## Setup
Tested on Kaggle (Tesla T4, 4 CPU cores) with Python 3.12, torch 2.11.0+cu126, torchvision 0.26.0+cu126.

1. **Environment**
   ```bash
   cd baselines/autobackswap
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
2. **AOT-GAN** (not copied here because it is a separate project with its own licence). Clone it at the pinned commit into `baselines/autobackswap/AOT-GAN-for-Inpainting/` and do not commit that folder:
   ```bash
   git clone https://github.com/researchmm/AOT-GAN-for-Inpainting AOT-GAN-for-Inpainting
   git -C AOT-GAN-for-Inpainting checkout 2cd1afd8fdfabb101c678f6062d14bc7d302509e
   ```
   Download the generator weights `G0000000.pt` (60,829,150 bytes) from the Google Drive folder named in `UPSTREAM_README.md` and put it in that folder.
3. **Data** (never committed). Waterbirds images and `metadata.csv`, plus the foreground segmentations used to make detector labels. See [`docs/dataset.md`](../../docs/dataset.md) and `data/README.md` in this folder. We used the WILDS copy of Waterbirds and the segmentations from data.caltech.edu record `w9d68-gec53`, symlinked as `data/waterbird_complete95_forest2water2/images` and `.../segmentations`.
4. **Memory patch for small GPUs** (optional, see below): `git apply patches/augment_waterbirds_chunk.patch`.

## Commands (what we ran, one seed `k`, "without minority")
```bash
export MPLBACKEND=Agg
FLAGS="--device=cuda:0 --dataset_type=waterbird --detector_dataset=waterbirds_from_segmentation --waterbirds_use_minority=false --random_seed=k --cuda_optimizations=false"
python run.py $FLAGS --mode=ours_detector
python run.py $FLAGS --mode=ours_generate
python run.py $FLAGS --mode=ours --run_label=ours_minority_false_seedk --results_dir=results
# ERM baseline
python run.py --device=cuda:0 --dataset_type=waterbird --waterbirds_use_minority=false --mode=erm --random_seed=k --cuda_optimizations=false --run_label=erm_minority_false_seedk --results_dir=results
```
For "with minority" use `--waterbirds_use_minority=true` and label `..._minority_true_...`. Full command lines per run are in `results/autobackswap/runs_log.csv`.

Expected output per run: `results/<run_label>_waterbird_<wom|wm>_resnet_seed<k>/` with `checkpoint.pt`, `final.pt`, `config.yaml`, `losses.txt`, `loss.png`, `test_accuracies.txt`. Checkpoints stay out of Git.

## Waterbirds notes and deviations from the upstream README
Each of these is also in `results/autobackswap/runs_log.md`.
1. **fp32** (`--cuda_optimizations=false`) instead of the default bf16 autocast + `torch.compile`; the T4 emulates bf16 and was slower with it.
2. **Chunked generation** (`patches/augment_waterbirds_chunk.patch`): upstream passes a whole class folder (about 60 images) to the separator at once and ran out of memory on a 15 GB T4. The patch passes `augment_batch_size` (16) images at a time. It is not applied to the preserved source in this folder.
3. **Data source**: the Stanford link in the upstream README was unreachable from our machines at the time, so the WILDS copy was used (its TLS certificate had expired, so it was fetched without certificate verification; no official checksum was available). `metadata.csv` is byte-identical to the one in upstream `data/`.
4. `detector_n_samples` is 200 (repo default for `waterbirds_from_segmentation`); the paper text mentions 287 hand-labelled samples.
5. "With minority" for AutoBackSwap (seed 0) reused the detector and generated images from the "without minority" run and only retrained the classifier. Basis: code reading (the flag is not read by the detector or by `augment_data.py`); not verified by regenerating.
6. Upstream `test_accuracies.txt` prints the two middle groups with swapped labels (index 1 is Landbird on Water, not "Waterbird on Land"). Our summaries use the dataset metadata names.

## Known limitations
- Partial reproduction: Waterbirds only, 3 seeds for three settings and 1 seed for AutoBackSwap with minority, against the paper's 5 seeds.
- AutoBackSwap without minority: overall accuracy is below the paper for all 3 seeds (mean 94.49 vs 95.1 ± 0.3, 2.04σ). Worst-group accuracy matches. The cause is not established.
- The classifier has no resume: a lost session loses the whole 50-epoch run.
- The detector ignores `--results_dir` and fails if its output folder already exists.

## What is ours
- `patches/augment_waterbirds_chunk.patch`: the memory patch above.
- `README.md`, `ORIGINAL_SOURCE.md`: these notes.
- Outside this folder: [`scripts/autobackswap/`](../../scripts/autobackswap/) (tools that export per-example predictions and build an error viewer; they do not modify upstream code), [`configs/autobackswap/`](../../configs/autobackswap/), [`results/autobackswap/`](../../results/autobackswap/).

## Licence and notices
The upstream repository has no licence file at the pinned commit, so no redistribution terms are stated. Before making this repository public, ask the authors or remove the copied sources and keep only the pointer in `ORIGINAL_SOURCE.md`. AOT-GAN is a separate project and was not copied.
