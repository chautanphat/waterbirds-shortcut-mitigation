# Research Protocol

## Scope

The current phase is to preserve, understand, and reproduce selected baseline methods before adapting or comparing them on Waterbirds.

For every baseline, follow: **Preserve → Understand → Reproduce → Adapt → Refactor**. Do not refactor or standardize an upstream method before its behavior is understood.

## Baseline artifact map

Use these locations for a baseline as the repository grows:

| Artifact | Location |
| --- | --- |
| Preserved implementation and method README | `src/methods/<method-name>/` |
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

## Reproduction status

- **Smoke test:** the pipeline runs on the intended data path.
- **Partial reproduction:** only part of the original experiment is reproduced.
- **Verified reproduction:** the relevant original result is compared with the paper and differences are explained.
- **Waterbirds adaptation:** the documented method has been adapted and evaluated under this protocol.

Successful execution alone is not a verified reproduction.
