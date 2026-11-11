# XRM source provenance

- **Paper:** Mohammad Pezeshki et al., *Discovering Environments with XRM*, ICML 2024, [PMLR](https://proceedings.mlr.press/v235/pezeshki24a.html).
- **Official repository:** [facebookresearch/XRM](https://github.com/facebookresearch/XRM).
- **Exact upstream commit:** `5e90815250aedea546b9e17ffd5d7e0a7c34c20f`.
- **License:** CC BY-NC, preserved as [`LICENSE`](LICENSE).
- **Upstream documentation:** [`UPSTREAM_README.md`](UPSTREAM_README.md).

`baselines/xrm/` preserves the authors' shared implementation of ERM, GroupDRO, and other algorithms. Project-owned configuration and launch/reporting code lives in the repository's separate method, config, script, experiment, and results locations.

## Local compatibility changes

The baseline source was copied from the `XRM` checkout at the exact commit above. These small local changes are documented so results remain traceable:

- `download.py`: choose CUDA when available, otherwise CPU, for feature extraction; set feature-extraction workers to zero to avoid multiprocessing issues on Windows.
- `hparams_registry.py`: default the optional `adam` argument to `False` when absent, preserving the authors' SGD recipe; expose optional step/checkpoint/worker overrides for smoke tests while keeping upstream defaults unchanged.
- `utils.py`: parse starred checkpoint filename patterns using escaped, basename-only matching so Windows path separators do not break result discovery; map checkpoints to the selected device and restore GroupDRO `q` plus Torch RNG state on resume. The DataLoader sampler position is not checkpointed, so a mid-run resume is stateful but not bitwise identical to uninterrupted training.
- `algorithms.py`: place GroupDRO group weights/losses on the model device and detach the adversarial weights from autograd; the GroupDRO objective is unchanged.
- `main.py`: add optional training-step, checkpoint-frequency, and worker overrides for smoke runs; normal runs retain the original 5,001-step Waterbirds protocol; phase 2 also preserves a separate checkpoint selected by validation WGA, retaining the last checkpoint unchanged.
- Any later changes to upstream files must be listed here and reviewed against the original implementation.

The selected Waterbirds configurations and complete-run seed sets are stored separately in `configs/erm/waterbirds_xrm.yaml` and `configs/groupdro/waterbirds_xrm.yaml`. Do not copy datasets, feature caches, checkpoints, or local output directories into this baseline folder.




