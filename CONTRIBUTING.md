# Contributing

This repository is a shared research codebase. Contributions must preserve method provenance, make experiments reproducible, and avoid introducing undocumented differences between methods.

Before contributing, read:

- [Research and Experiment Protocol](docs/research_protocol.md)
- The README inside the method you intend to modify.

## 1. Branch workflow

Do not develop directly on `main`. Create a focused branch from the latest `main`:

```text
method/<method-name>
feature/<short-description>
experiment/<short-description>
fix/<short-description>
docs/<short-description>
```

Keep each branch focused on one method, experiment, fix, or documentation task. Sync with `main` before opening a pull request where practical.

## 2. Adding a baseline or method

1. Create `src/methods/<method-name>/`.
2. Preserve the relevant upstream source files, scripts, configs, and documentation together.
3. Record the paper, official code URL, upstream version/commit, import date, and license.
4. Add a method-level README describing the mechanism, source, dependencies, original commands, and known limitations.
5. Follow the Preserve → Understand → Reproduce → Adapt → Refactor lifecycle in the research protocol.
6. Do not extract shared components until their behavior and reuse across methods have been established.

Do not commit raw datasets, checkpoints, feature caches, credentials, or large generated outputs.

## 3. Code organization

- Method-specific algorithms and training procedures belong in `src/methods/<method-name>/`.
- Shared code belongs in `src/datasets/`, `src/models/`, `src/evaluation/`, or `src/utils/` only when multiple methods genuinely use compatible behavior.
- Project-level launchers belong in `scripts/`; method-specific entry points may remain in the method directory.
- Experiment settings belong in `configs/`.
- Notebooks are for exploration and analysis; move reusable logic into tested Python modules.
- Avoid unrelated renames, formatting changes, and refactors in baseline-import pull requests.

## 4. Experiments and reproducibility

- Use configuration files for important experiment settings.
- Record the dataset and split, seed, model, optimizer, hyperparameters, method-specific settings, environment, command, checkpoint-selection procedure, and metrics.
- Report overall and group-wise results, including worst-group accuracy where applicable.
- Keep the test set held out. Follow method-specific validation procedures when required for faithful reproduction, and document them rather than silently standardizing them.
- Distinguish smoke tests, partial reproductions, and verified reproductions.
- Document deviations from the paper or upstream code.

See [docs/research_protocol.md](docs/research_protocol.md) for the full requirements.

## 5. Commits

Use concise messages that describe the change and its purpose. Examples:

```text
Import POTER upstream implementation
Document POTER Waterbirds adaptation
Add shared Waterbirds group metrics
Fix JTT stage-two sample weighting
Record GroupDRO reproduction settings
```

## 6. Pull requests

- Open a pull request into `main`; do not merge your own work without review from another teammate.
- Explain what changed and why.
- Identify the method and whether the change affects its algorithm, data pipeline, evaluation, or configuration.
- Include the exact validation commands and their outcomes.
- Include experiment details and results when relevant.
- Disclose deviations, limitations, and unverified behavior.
- Update the relevant method README, project documentation, or configuration in the same pull request.
- Keep generated artifacts and unrelated changes out of the pull request.

A pull request that changes experimental behavior must make the impact on reproducibility and comparability explicit.
