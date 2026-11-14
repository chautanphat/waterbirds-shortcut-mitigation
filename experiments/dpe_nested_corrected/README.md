# Corrected nested DPE experiment

Status: implementation and software validation complete; the corrected 70-cell
Waterbirds experiment has **not been executed**. Original test data and backbones
were previously inspected, so a corrected rerun remains exploratory evidence.

Import [the self-contained notebook](artifacts/dpe_nested_corrected_kaggle.ipynb)
into Kaggle, enable Internet, select CPU and choose **Save & Run All**. Source is
embedded; no code dataset upload is needed. Frozen backbone features are recovered
from attached inputs or the same public outputs used by the original notebook.
Missing feature sources stop the run; backbones are not automatically retrained.

CPU is appropriate because the supplied head trainers explicitly execute on CPU.
Selecting a GPU runtime alone will not accelerate this implementation.

## Protocol and changes

- Five inner evaluation folds. All five OOF ensembles, the full-refit ensemble,
  and calibrators are rebuilt inside each inner training pool. Every inner
  evaluation row is excluded from every fitted stage used to predict it.
- Inner evaluation reproduces the actual inference graph. Foldwise calibration,
  bias after averaging, and calibration transferred to a full-refit ensemble are
  separately evaluated and independently select their settings.
- Exact empirical WGA thresholds search all attainable prediction partitions,
  move tied scores together, and apply after probability averaging. The old
  coarse-grid function remains solely for diagnostic comparison.
- Original four loss/penalty settings and fixed average/0.1 arm are retained.
  Primary selection remains balanced NLL; WGA selection remains secondary.
- Full-validation uniform, the 75-head bag, cross-fit uniform, pooled 60/40
  affine/linear controls and DFR-lite are retained. DFR fits its standardizer
  inside each CV fold and uses a fixed solver random seed.
- Predictions are checkpointed between ensembles with index manifests and
  checksums. Code, configuration, runtime and source hashes bind cache identity.
  Legacy result caches are rejected and the legacy `run` entry point is disabled.
- Partial results and task failures are explicit. Reports pair methods within
  sources and split seeds, average splits within each source, and then compare
  sources; folds are not treated as independent backbones.

The corrected run fits 41 fifteen-head ensembles per source/split, compared with
11 previously. The notebook requests all seven sources and ten split seeds and
uses an 8.5-hour soft budget checked between head fits. It may need multiple
sessions. Attach its prior output or upload `corrected_nested_resume.zip` to
resume. Download `corrected_nested_results.zip` for subsequent analysis.

## Source and validation

The `code/` tree is an intentionally frozen, independently packaged source
snapshot, including original licenses and metadata. The corrected entry point is
`code/corrected_nested.py`; the old module supplies discovery, regression checks
and fitting adapters. Neither the reference metadata nor synced project sources
were relabelled by this work.

```sh
python experiments/dpe_nested_corrected/code/corrected_selftest.py
python experiments/dpe_nested_corrected/code/corrected_selftest.py --smoke
python experiments/dpe_nested_corrected/build_notebook.py
python experiments/dpe_nested_corrected/validate_notebook.py
python experiments/dpe_nested_corrected/audit_original.py
```

Use the repository's numerical dependencies (NumPy, SciPy, scikit-learn, PyTorch,
torchvision, pandas, PyYAML and tqdm). The ten regression tests exercise narrow
threshold improvements, ties and exhaustive partitions, post-average semantics,
held-out label perturbations, training-index ancestry, corrupt/stale cache
rejection, time budgets and partial reports. Synthetic smoke tests used both real
head trainers, 41 ensembles and all 28 arms per cell, with one head epoch and
eight calibration steps. These reduced settings are confined to synthetic tests;
the experiment retains the original production settings.

Saved validation evidence is in `artifacts/validation_report.json`,
`packaged_tests.txt`, `unit_tests.txt`, `final_smoke.txt` and
`dfr_determinism.txt`. The smoke deliberately injects incompatible protocol
identities to check rejection; those expected errors in the saved log are not
Waterbirds experiment failures. A fresh validator checks the distributed
notebook, embedded ZIP, extracted source bytes, clean outputs and legacy-run guard.

`audit_original.py` reproduces the audit from the tracked original 70-cell run
and the seven public historical prediction caches in
`audit/historical_heldout_caches.zip`. No API keys or signed download URLs are
stored in this package.

See [the audit and research interpretation](../../docs/dpe_nested_audit_and_rerun.md)
and [Kaggle instructions](artifacts/RUN_INSTRUCTIONS.txt).
