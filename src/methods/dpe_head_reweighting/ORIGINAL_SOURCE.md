# Source and adaptation provenance

Adaptation introduced 2026-11-07 for the project-owned H2 experiment.

- DPE upstream: https://github.com/minhto2802/dpe4subpop
- Preserved upstream commit: `68b811461e238d44cced6ae6c8ce75b66cad5449`.
- Behavioral reference: `notebooks/dpe_waterbirds_end_to_end.ipynb`, cells 11,
  13 and 14 at project commit `fc6c267` (the H2 plan branch).
- `train_heads.py` adapts the notebook's group-balanced feature-only training,
  covariance/prototype penalties, entropic loss and last-on-tie epoch selection.
- `ensemble.py` follows the preserved IsoMax+ Euclidean distance implementation
  in `baselines/dpe4subpop/utils/isomaxplus.py`, originally David Macêdo (2022),
  Apache 2.0: https://github.com/dlmacedo/entropic-out-of-distribution-detection.
- DPE source copyright (c) 2025 Minh To, MIT. Dataset subsampling reference:
  Yuzhe Yang (2023), MIT, https://github.com/YyzHarry/SubpopBench.

The accompanying LICENSE preserves the baseline's MIT and third-party license
texts. Weight fitting, split manifests, experiment locks and diagnostic reporting
are project-owned H2 additions. No upstream files or reproduction notebook were
modified. See the method README for explicit behavioral differences.
