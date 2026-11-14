# Standard ERM only — verified recipe required

Current scope: standard ERM without training augmentation. ERM* is excluded.
Update: direct remote/local checkpoint hash comparison now verifies that the
previous weighting runs used ERM* (400 epochs with augmentation). See
`results/dpe/backbone_provenance_20261114/README.md`; this supersedes the earlier
unverified-recipe status described below. Standard-ERM results remain pending.
The previous 95.708% uniform / 96.076% D5 results use a supplied bare backbone
whose training recipe has not been verified. They must not be reported as
verified standard-ERM results, nor relabeled as ERM* without evidence.

`config.yaml` resolves the existing standard recipe: ImageNet V2 ResNet-50,
100 epochs, no augmentation, SGD .001/momentum .9, weight decay 0,
class-balanced sampling, seed 0, final-epoch checkpoint. No test selection.
Standard ERM still uses deterministic resize/crop/normalization preprocessing.

No verified standard checkpoint is present. The available torch build is
CPU-only; ImageNet V2 initialization is not cached (the cached ImageNet V1
weights must not silently substitute for V2). Full training is not yet run.
No standard-ERM/DPE/D5 benchmark metrics are claimed here.

Run from the repository root on an appropriate compute environment:

```powershell
.venv/Scripts/python.exe scripts/train_erm.py --config experiments/erm_only/config.yaml
```

The existing trainer saves model/optimizer/RNG state at epoch boundaries.
Resume via its Python API `run_training(config, resume=True)`; do not restart
against a nonempty output directory without resume. Use the final checkpoint
to extract new features and refit OOF DPE heads before evaluating D5. All prior
feature caches and tables belong to their old backbone and cannot substitute.
Check `state.json` for the current prepared/untrained status.
