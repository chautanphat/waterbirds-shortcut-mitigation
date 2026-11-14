# Backbone provenance resolved: ERM*, not standard ERM

On 2026-11-14, the public notebook and its saved output files were retrieved
directly from Kaggle. This resolves the earlier uncertainty about the backbone
used by the local weighting v1/v2 runs.

Source notebook: [noerula/dpe-exploration, version 3](https://www.kaggle.com/code/noerula/dpe-exploration?scriptVersionId=368838235).
The public pull API reports current version 3. Its executed launcher imports
`src.methods.dpe_ermstar.run.prepare_and_launch` and loads
`configs/dpe_ermstar/study.json`.

## Evidence chain

1. Notebook code calls the ERM* experiment launcher, using seeds 0/1.
2. The launcher loads `waterbirds_enhanced.yaml`; its backbone duration is
   overridden to 400 epochs.
3. The actual saved output `dpe_ermstar/jobs/seed0/backbone/config.yaml`
   independently records `profile: dpe_enhanced_erm`, `augmentation: true`,
   `epochs: 400`, ImageNet V2 ResNet-50, SGD .001, momentum .9, weight decay 0,
   class-balanced sampling and batch size 128.
4. Its saved summary records 14,800 updates and selects `ckpt_last.pt`.
5. The remote checkpoint was streamed to compute SHA-256, without replacing
   or modifying any local model. It matches the exact bytes of the local
   `checkpoints/kaggle_exploration/seed0/ckpt_last.pt` used in both studies.

```text
Remote and local size: 94,356,497 bytes
Remote and local SHA-256:
bce28dc72c3737ae19ffa057ba056e03d3f531180e7ad55e0b3c93bf4ea4bcaa
```

Machine-readable evidence: [identity comparison](checkpoint_identity.json),
[saved Kaggle configuration](kaggle_backbone_config.yaml),
[saved Kaggle completion summary](kaggle_backbone_summary.json).
The downloaded notebook/API evidence remains under
`outputs/backbone_provenance_audit/`. Public artifact paths were read through
the official Kaggle API; no credentials or signed download URLs are included
in this report.

## Correct interpretation of the results

The local studies reuse the verified **seed-0 ERM* backbone** and refit DPE
heads/weighting under their own no-group protocols. Three local head/gate
seeds are not three independently trained ERM* backbones. They also are not
an exact reproduction of the notebook's annotated head-training protocol.

| Local procedure on the ERM* backbone | Accuracy (%) | WGA (%) |
|---|---:|---:|
| Matched uniform, refitted heads | 95.708 ± 0.117 | 92.761 ± 0.552 |
| D19 weighting v1 | 95.961 ± 0.422 | 83.645 ± 4.083 |
| D5 reliability v2 | 96.076 ± 0.339 | 90.343 ± 1.685 |

These are exploratory measured results on a previously inspected test set.
They **must not be reported as standard ERM + DPE/D5 results**. Earlier notes
that the backbone recipe was unverified describe the previous audit status;
the exact hash comparison above supersedes that uncertainty. This does not
independently replay the original 400-epoch backbone training process.

The separate `dpe_waterbirds_end_to_end.ipynb` contains a 100-epoch
`--no_augmentation` standard-ERM recipe. Its existence does not establish
that this checkpoint came from that notebook: the matched output proves
it came from the enhanced ERM* source instead. Standard-ERM training/results
remain a separate pending task, documented in
[experiments/erm_only](../../../experiments/erm_only/README.md).
