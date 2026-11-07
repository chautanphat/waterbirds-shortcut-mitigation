# ERM source and adaptation record

- DPE paper: [Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift](https://arxiv.org/abs/2505.23027), ICML 2025.
- DPE source: [minhto2802/dpe4subpop](https://github.com/minhto2802/dpe4subpop), preserved in `baselines/dpe4subpop/` from upstream commit `68b811461e238d44cced6ae6c8ce75b66cad5449`. MIT License, copyright 2025 Minh To.
- Shared recipe: this repository's `configs/dpe4subpop/waterbirds_stage0_erm.yaml` plus stage-0 behavior in `main.py`, `utils/datasets.py`, `utils/dataloaders.py`, and `utils/models.py`.
- SubpopBench source: [YyzHarry/SubpopBench](https://github.com/YyzHarry/SubpopBench), inspected at `4d3dbbe21029666ef19d040e110ec22908640c5b`. MIT License, copyright 2023 Yuzhe Yang. Waterbirds preprocessing comes from `subpopbench/dataset/datasets.py`; default comparisons refer to `hparams_registry.py`, `models/networks.py`, `learning/optimizers.py`, and `dataset/fast_dataloader.py`.

The shared implementation is project-owned and independent of either runtime. It preserves DPE's deterministic image preprocessing, class-balanced stage-0 sampling, model initialization, checkpoint-key layout, SGD settings and full-backbone training. It separates evaluation and export into project-level scripts, validates inputs/configuration, records provenance, and uses validation-only checkpoint selection. It intentionally follows the tracked DPE YAML's weight decay (`1e-4`), not the default CLI value (`0`). Raw feature caches are compatible with DPE's subsequent normalization.

The default is DPE-aligned ERM, not an exact reproduction of SubpopBench's default run, and not DPE's photometrically augmented ERM* variant. The preserved external implementations and their recorded experiments are unchanged. See [README.md](README.md) for the complete settings and deviations. License notices for the referenced/adapted code are retained in [LICENSE](LICENSE).
