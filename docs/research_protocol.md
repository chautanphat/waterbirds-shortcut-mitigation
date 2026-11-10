# Research Protocol

## Scope

The current phase is to preserve, understand, and reproduce selected baseline methods before adapting or comparing them on Waterbirds.

For every baseline, follow: **Preserve → Understand → Reproduce → Adapt → Refactor**. Do not refactor or standardize an upstream method before its behavior is understood. An official implementation stays in `baselines/`; an adaptation or project-owned implementation belongs in `src/methods/`.

## Baseline artifact map

Use these locations for a baseline as the repository grows:

| Artifact | Location |
| --- | --- |
| Official external implementation and preserved README | `baselines/<method-name>/` |
| Project-owned, reimplemented, or integrated method | `src/methods/<method-name>/` |
| Reproducible settings | `configs/<method-name>/` |
| Run metadata and experiment record | `experiments/<experiment-id>/` and `docs/experiment_log.md` |
| Lightweight comparison tables and plots | `results/` |
| Shared method status | `docs/methods.md` |

These paths are the target convention; do not create empty directories until a baseline needs them.

## Fair comparison

- Use the same Waterbirds version, official splits, target definition, group definition, and held-out test set whenever possible.
- Preserve method-specific validation, early-stopping, and model-selection procedures when they are necessary for faithful reproduction.
- Never use the test set for training, hyperparameter selection, or checkpoint selection.
- Document every meaningful deviation from the paper or upstream implementation.

## Evaluation and reporting

Report, at minimum:

- overall test accuracy;
- accuracy for every target/background group;
- worst-group accuracy; and
- the number of examples in each evaluated group.

Each reported result must link to its method version, configuration, command, seed, environment, output location, and experiment-log entry.

For new meaningful runs, the [experiment records contract](experiment_record_contract.md)
defines `experiments/<experiment-id>/run.json`, execution/claim status, attempt
history and retained evidence. The [model artifacts and inference contract](model_artifact_contract.md)
defines the companion `model.json`, required reconstruction inputs and inference
verification. Native checkpoint formats and method-specific validation remain
unchanged; missing historical evidence must be disclosed rather than fabricated.

## Shared per-example prediction artifacts

All methods must export future per-example evaluation results using the
[predictions.csv version-1 protocol](prediction_protocol.md). Use one complete
reference split per run/seed/frozen predictor, include correct and incorrect
examples, and retain companion provenance. Validate against the reference
metadata before accepting or reporting the export. Method-specific internals,
preprocessing and validation/checkpoint-selection procedures remain as documented.

The canonical location is
`experiments/<experiment-id>/predictions/<split>/predictions.csv`, with
`provenance.json` beside it. This is a general evaluation artifact location for
all methods. Dataset images remain in the dataset directory.

The shared [export adapters](export_predictions.md) implement this boundary.
Legacy and errors-only files may remain available for historical inspection;
they are not substitutes for complete canonical exports from new runs.

## Reproduction status

- **Smoke test:** the pipeline runs on the intended data path.
- **Partial reproduction:** only part of the original experiment is reproduced.
- **Verified reproduction:** the relevant original result is compared with the paper and differences are explained.
- **Waterbirds adaptation:** the documented method has been adapted and evaluated under this protocol.

Successful execution alone is not a verified reproduction.
