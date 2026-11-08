# XRM GroupDRO on Waterbirds

This project method integrates the authors' GroupDRO implementation for Waterbirds using human-annotated group labels and frozen ImageNet ResNet-50 features. The upstream implementation is shared with ERM in [`baselines/xrm/`](../../../baselines/xrm/); this separate folder contains the GroupDRO-specific adapter and reproduction instructions. The upstream source is pinned to `5e90815250aedea546b9e17ffd5d7e0a7c34c20f` from [facebookresearch/XRM](https://github.com/facebookresearch/XRM), for Pezeshki et al., *Discovering Environments with XRM* (ICML 2024). License and compatibility changes are recorded in [`ORIGINAL_SOURCE.md`](../../../baselines/xrm/ORIGINAL_SOURCE.md).

## Method and tracked settings

GroupDRO updates a distribution over the four observed target/background groups by exponentiating each group's loss, then minimizes the weighted loss. The training data therefore uses oracle group labels. The tracked selected configuration uses hyperparameter combination 8: SGD, learning rate `5.5828866e-4`, batch size 122, weight decay `4.0511940e-4`, GroupDRO step size `eta=0.01152687`, 5,001 updates, checkpoint frequency 50, and CPU execution. Model selection uses validation WGA; it never uses test metrics.

See [`configs/groupdro/waterbirds_xrm.yaml`](../../../configs/groupdro/waterbirds_xrm.yaml) for the complete config and [`xrm.py`](xrm.py) for the method adapter.

## Setup and data

Use Python 3.9 in an environment separate from the shared project environment:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r baselines/xrm/requirements.txt
python -m pip install PyYAML
python baselines/xrm/download.py --download --data_path .\data waterbirds
```

The local data directory must contain the prepared Waterbirds metadata/images and XRM's cached features under `data/waterbirds/`. Data and features are excluded from Git.

## Run and evaluate

One seed:

```powershell
python scripts/run_xrm_baseline.py --method groupdro --config configs/groupdro/waterbirds_xrm.yaml --data-path .\data --seed 0
```

The six complete archived seeds:

```powershell
python scripts/run_xrm_baseline.py --method groupdro --config configs/groupdro/waterbirds_xrm.yaml --data-path .\data --num-seeds 6
```

Two-step smoke run and four-group evaluation:

```powershell
python scripts/run_xrm_baseline.py --method groupdro --config configs/groupdro/waterbirds_xrm.yaml --data-path .\data --seed 0 --max-steps 2 --checkpoint-frequency 1 --workers 0 --output-root .\outputs\xrm_groupdro_smoke
python scripts/evaluate_xrm_checkpoint.py --method groupdro --config configs/groupdro/waterbirds_xrm.yaml --data-path .\data --checkpoint .\outputs\xrm_groupdro_smoke\GroupDRO\Waterbirds\group_labels_yes\ckpt_hpcomb_8_seed0_best_wga.pt
```

Regenerate the summary directly from archived JSON histories:

```powershell
python scripts/summarize_xrm_baselines.py --method groupdro
```

The historical archive contains six complete runs and one GroupDRO seed stopped at step 3,150 of 5,001; the aggregator marks and excludes it. Selected checkpoint handling, complete-run lists, and source details are recorded in the config, experiment record, and [`results/groupdro/summary.md`](../../../results/groupdro/summary.md).


