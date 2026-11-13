# Execution, result counts and artifact guide

This records the completed 2026-11-13 study. The implementation and portable
evidence are published with this report; large generated artifacts remain outside
Git under the repository's artifact policy.

## What the result counts mean

A result row identifies one saved backbone source, validation allocation and
fixed prediction configuration. It is not an independently trained backbone.

| Block per backbone source | Declared rows | Aliases | Distinct predictors |
| --- | ---: | ---: | ---: |
| Full-validation geometry | 85 | 19 | 66 |
| Three split-pool allocations: 240 each | 720 | 75 | 645 |
| Five-fold cross-fitting | 35 | 0 | 35 |
| Total per source | 840 | 94 | 746 |
| Seven sources combined | 5,880 | 658 | 5,222 |

There are **seven independent backbone sources**, comprising normal-ERM seeds
0–4 and ERM* seeds 0–1. Settings, allocations, fold models and aliases share
sources and labels; they are not additional independent experimental repetitions.

An alias is an algebraically equivalent prediction construction declared before
evaluation. It references an existing prediction instead of repeating its
calculation, while retaining its own result row and canonical prediction export.
It is never identified by looking for favorable test scores.

Each geometry block has 19 aliases: 12 permuted-membership endpoint configurations
(k=1 or k=15), six local-top-15/global-pooling equivalences, and the native k=1,
T=1 original-DPE equivalence. Each split-pool block adds six H3+uniform entries
already present in its H3 geometry grid, giving 25 aliases per allocation.
Thus `(19 + 3 × 25) × 7 = 658` aliases. The exact `alias_of` identities are in
[per_seed.csv](per_seed.csv); the full declaration can be regenerated with `grid`.

## Where computation actually ran

The experiments ran in the **managed Codex cloud execution environment attached
to the chat**, using four CPU worker processes. Earlier references to running
“locally” mean that attached workspace, not the researcher's personal computer.
The initial plan proposed Kaggle for substantial computation; the completed run
used the available cloud CPUs after the saved feature artifacts were recovered.
No new Kaggle job was submitted. The provided Kaggle notebook is an alternative
launcher and was syntax-checked.

Recorded runtime: Python 3.12.14, NumPy 1.26.4, SciPy 1.12.0 and PyTorch 2.6.0+cpu
on Linux x86-64. Each worker uses one PyTorch CPU thread, with BLAS/OpenMP thread
counts constrained to one. No GPU was used for this new study. Historical source
artifacts were produced in a different GPU/software environment.

ResNet-50 image feature extraction was already completed in the saved Kaggle runs:

* [noerula/dpe-exploration](https://www.kaggle.com/code/noerula/dpe-exploration)
  supplies ERM* sources 0/1, raw validation/test features and both original arms.
* [noerula/dpe-mlp-dropout-verification](https://www.kaggle.com/code/noerula/dpe-mlp-dropout-verification)
  supplies the five normal-ERM sources, their raw features and saved head/control
  evidence. Source 0–2 and 3–4 histories remain reuse, not newly independent data.

After network access was enabled, public saved-output files were downloaded into
the execution workspace. The recovery command records hashes and public notebook
member names, without retaining expiring signed URLs in its artifact index.

Each image is represented by its saved **2,048-dimensional raw feature vector**.
The new study computed prototype distances, fitted lightweight heads/combiners,
and evaluated class predictions from these vectors. It did not retrain a CNN,
download replacement pretrained weights, or re-extract image features. New heads
were needed for missing ERM* allocation heads and the five-fold ensembles.

## Data access and checks

Fitted split-pool methods use 479 fitting examples, separate from the 720 examples
used for their heads. The three allocation seeds are 0/1/2. Full-validation heads
only receive the fixed geometry experiments. Crossfit calibrators use 1,199
out-of-fold predictions; each fold head ensemble excludes its held-out rows.
All fits in an arm freeze before that arm opens test feature arrays. Existing
historical prediction caches are also used for numerical baseline checks.

The independent audit reads every completed canonical CSV and verifies all 5,794
test image identities, labels, groups, counts and integer accuracy calculations.
It checks source-mean/SD summaries, fitted-state/head/source-snapshot hashes,
frozen-fit locks and retained NumPy replay evidence. It does **not** rerun CNN
extraction, recreate historical backbone optimization, or establish the absence
of every possible historical data leak. ERM* historical feature markers do not
record a backbone digest; selected head epochs were not exported.

The initial float32 CPU preflight and its failed acceptance checks were preserved.
The final float64 distance path passed the original 2e-7 historical probability
tolerance, with zero changed historical head/uniform decisions. Brier refinements
preserve the objective and 1e-7 optimality-gap threshold. See
[numerical_checks.json](numerical_checks.json) and the main report for details.

## Retained artifacts and availability

Git contains the readable implementation, configuration, fresh notebook, tests,
analysis scripts, this report, the figure, all 5,880 portable per-source rows,
all 1,680 recipe/allocation summaries, completion status and audit evidence.

The full generated study occupies approximately 6.4 GiB in the execution
workspace. It is not uploaded by committing this report. Raw feature arrays,
native prototype states, fitted parameters, all canonical CSVs, per-predictor
run/model records, source snapshots and detailed logs remain in that retained
study output. Its earlier failed preflight remains separately retained.
No public download location for these new large artifacts is established here;
GitHub checkout alone does not contain them. Historical input artifacts can be
recovered from the public Kaggle outputs; new artifacts can be regenerated using
the implementation. Preserve or separately archive the complete study directory
before removing its execution environment.

The generated layout uses portable roles rather than a machine-specific root:

```text
<study>/
  study.json, completion.json, workers.json
  per_configuration.csv, summary.csv, independent_audit.json
  implementation.zip, implementation.json
  jobs/<source-id>/
    implementation.zip, implementation.json, study.json
    <source-id>/<arm>/
      inputs.json, ledger.json, fit_complete.json
      cache/
      frozen/<configuration>.json
      experiments/<predictor-id>/
        run.json, model.json, metrics.json
        predictions/test/predictions.csv
        predictions/test/provenance.json
```

Fold heads and assignments also reside under each source's crossfit directory;
missing allocation heads are retained under its heads directory. The generated
root `per_configuration.csv` includes resolved artifact paths. The portable
`per_seed.csv` published here omits that machine-specific path column.

Execution preceded this publication commit. The executed source identity is the
base revision `22b65ec5fc8f00b33938fe2af6dc32ad2f34ecfd` plus the retained dirty-source
snapshot, not a retrospectively substituted publication commit.

## Reproduce or inspect

Run from a checkout with the dependencies in
[requirements.txt](../../../src/methods/dpe_independent/requirements.txt):

```bash
python -m src.methods.dpe_independent recover --output /path/to/recovered
python -m src.methods.dpe_independent inventory --roots /path/to/recovered --output /path/to/sources.json
python -m src.methods.dpe_independent grid --output /path/to/declared-grid.json
python -m src.methods.dpe_independent launch --sources /path/to/sources.json --output /path/to/study --devices cpu cpu cpu cpu
python -m src.methods.dpe_independent report --output /path/to/study
python scripts/analysis/audit_dpe_independent.py --study /path/to/study --output /path/to/study/independent_audit.json
```

Use fresh output for changed code, inputs or prediction settings. Exact matching
inputs support resumption; failed fits remain visible. A regenerated result is a
new execution and should retain its own runtime/source evidence.

The report-generation script additionally reads the retained
`numerical_recovery.json`, which links the initial preflight and its diagnostics.
Those original diagnostics are published in portable form in
`numerical_checks.json`; reproducing the final scientific grid does not require
recreating the failed preflight.

The [method README](../../../src/methods/dpe_independent/README.md) specifies model
equations, regularization, solver checks and recovery. The [results report](README.md)
explains comparisons and limits. All settings receive final scores; descriptive
grid extrema are not a selected deployment model or untouched confirmation.
