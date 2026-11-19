# Historical POTER seed 1 OT replay

**`VERIFIED_AND_REPLAYED`.** The supplied `waterbirds_standard.zip` is a PyTorch serialization archive. Its SHA-256 is `8a7636923bffaddce16f843b38afe0a97743f19a151339ddb96e200d2c4d2258`, exactly the cache hash recorded by the original seed-1 run. The archived Kaggle notebook version remains unknown. The validation-selected checkpoint at epoch 258 and the historical run config also matched their recorded fingerprints. See the [candidate inventory](candidate_files.csv) and [full audit](replay_audit.json).

The restricted loader checked finite float32 features for 4,795 training rows and 599 validation reference rows, with 2,048 features per row. The verified metadata, original CUB split code, and exact training label sequence establish the canonical training order. Replay used the original seed-0 reference selection with quotas `[0, 466, 133, 0]` and verified the reference label sequence and disjoint train/reference IDs. The cache stores labels but has no embedded IDs. The audit records the reference IDs in the order reconstructed from the extraction recipe.

The [weights CSV](historical_ot_weights_seed1.csv) has one positive finite weight per canonical train ID. It includes the pre-normalization dual potential, the normalized potential passed to the weight transform, the runner's float32 weight, and the sampler's normalized probability. All train IDs have `reference_membership=False` because the reference set is drawn from validation. The final weights average 1.0 and range from `3.85e-6` to `33.59`. Final mean normalization follows clipping, so the largest final weight can exceed 20.

| Training group | Replay mass | Historical draw share | Historical minus expected draws, in standard deviations |
| --- | ---: | ---: | ---: |
| G0 Landbird/Land | 22.63% | 22.67% | +1.08 |
| G1 Landbird/Water | 24.14% | 24.06% | −2.30 |
| G2 Waterbird/Land | 11.88% | 11.88% | +0.27 |
| G3 Waterbird/Water | 41.35% | 41.39% | +0.91 |

Class mass after clipping and renormalization is 46.77% Landbird and 53.23% Waterbird. The group-share check uses all 1,438,500 historical replacement draws as an aggregate plausibility check. The largest per-image probability difference between local CPU and CUDA replay was `2.12e-7`. The replay used local PyTorch 2.5.1 and CUDA 12.1, while the run config records PyTorch 2.11.0+cu130. Individual weights were not stored historically, so bitwise identity with the training-time vector is unverified.

Rerun into a fresh directory from the repository root.

```powershell
.\.venv\Scripts\python.exe scripts\replay_historical_poter_ot.py waterbirds_standard.zip --output outputs\kaggle_cache_recovery\replay_check
```

The next study remains **`BLOCKED_TRAIN_DEV_VULNERABILITY`** pending train/development masks, provenance-tracked background banks and an independent evaluation target. This replay used no test labels for selection and ran no training, embedding extraction or model inference. See [train/development readiness](../train_dev_readiness.md).
