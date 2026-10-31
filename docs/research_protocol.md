# Research and Experiment Protocol

This document defines the shared rules for implementing, adapting, and comparing methods in this project. It complements the root README; it does not replace method-specific reproduction instructions.

## 1. Research principles

1. **Preserve before changing.** When importing a published method, retain its original implementation and record its provenance before refactoring or adapting it.
2. **Reproduce before refactoring.** First understand the original pipeline and verify the relevant original behavior where feasible. Then adapt it to Waterbirds and verify the adapted pipeline.
3. **Share only proven common functionality.** Do not move code into shared infrastructure merely because two files have similar names. Extract a component only when multiple methods need the same behavior and its interface and semantics are compatible.
4. **Do not silently change a paper's method.** Record adaptations, deviations, and unresolved differences from the paper and upstream implementation.
5. **Keep test data held out.** Test results must not be used to choose hyperparameters, checkpoints, thresholds, or other model decisions.

## 2. Adding or importing a method

Create a self-contained directory at `src/methods/<method-name>/`. Preserve the upstream implementation's relevant source files, scripts, documentation, and configuration files together initially. Do not copy an upstream `.git/` directory or commit datasets, checkpoints, caches, or generated outputs.

Each method directory must contain a `README.md` documenting, at minimum:

- Full method name and a short mechanism summary.
- Paper citation and paper URL.
- Official/upstream code URL and imported commit, tag, or release when available.
- Original supported datasets and intended task.
- Dependencies and environment requirements.
- Original entry points and commands.
- Waterbirds adaptation status and instructions.
- Known limitations and deviations from the paper/upstream code.

Also record provenance in an `ORIGINAL_SOURCE.md` or the method README: source URL, version/commit, import date, license, and modifications made by this project. Preserve upstream license and copyright notices.

## 3. Required lifecycle: Preserve → Understand → Reproduce → Adapt → Refactor

### Stage A — Preserve

- Keep the imported implementation self-contained under `src/methods/<method-name>/`.
- Preserve supporting files, scripts, configs, and relevant documentation, not only Python source files.
- Keep the original README as method-level documentation; do not replace the project root README with upstream documentation.
- Record provenance and license information.
- Keep original dependencies identifiable. Do not blindly merge incompatible method dependencies into a single global environment.

### Stage B — Understand

- Identify the method's entry points, training stages, losses, model components, data requirements, evaluation procedure, and output artifacts.
- Map imports and dependencies before moving or renaming files.
- Note which behavior is method-specific and which appears potentially reusable.

### Stage C — Reproduce

- Follow the original paper and upstream instructions as closely as the available environment permits.
- Record the exact command, configuration, software environment, hardware, seed, and result.
- Label outcomes accurately as a smoke test, partial reproduction, or verified reproduction. Do not call a smoke test a reproduction.
- If the original dataset or environment is unavailable, state what could not be verified.

### Stage D — Adapt to Waterbirds

- Document every change needed to support Waterbirds, including data loading, labels, group definitions, preprocessing, training, and evaluation.
- Verify that the adapted method runs end to end.
- Keep original-paper settings distinct from Waterbirds-specific settings.
- Do not assume that a method's original dataset pipeline or validation strategy transfers unchanged.

### Stage E — Refactor shared components

Refactor only after the method is understood and its Waterbirds adaptation has been tested.

- Method-specific training logic, losses, sampling/reweighting procedures, and special stages remain in `src/methods/<method-name>/`.
- Common dataset interfaces and transformations may live in `src/datasets/`.
- Common model components may live in `src/models/`.
- Common metrics and evaluation utilities may live in `src/evaluation/`.
- Genuinely generic utilities may live in `src/utils/`.
- Project-level launchers and orchestration belong in `scripts/`; method-specific entry points may remain inside the method directory.
- Configurations belong in `configs/`, organized by method or experiment.

After extraction, run the affected method's tests or smoke tests and compare behavior before and after refactoring. Do not change algorithmic behavior as an incidental consequence of cleanup.

## 4. Waterbirds comparison protocol

For cross-method comparisons, use a common, documented dataset source/version, label meaning, group definition, and held-out test set wherever the methods support the same task. Document the exact split files or split-generation procedure and the mapping from dataset metadata to labels and groups.

Some published methods prescribe different validation usage, model-selection procedures, or training stages. Do **not** silently force all methods into one validation procedure if doing so would change the method being reproduced. Instead:

- Follow and document each method's justified validation/model-selection procedure.
- Record which split is used for training, validation, model selection, and final testing.
- Keep the test set untouched until final evaluation.
- Clearly disclose protocol differences that affect comparability.
- When a common evaluation protocol is possible without changing a method, apply it consistently.

Report overall accuracy and group-wise accuracy, including worst-group accuracy, where applicable. Include per-group sample counts and metrics so aggregate numbers can be interpreted. State the exact group mapping used; do not rely on undocumented assumptions.

## 5. Experiment configuration and records

- Store reproducible settings in version-controlled configuration files under `configs/`; these are the source of truth for the run.
- Avoid burying important hyperparameters only in notebook cells or command history.
- For every meaningful run, record: method and code version, config path or config snapshot, dataset source/version and split, seed, model/backbone and pretrained weights, optimizer, learning rate, batch size, epochs/steps, method-specific settings, validation/model-selection procedure, software versions, hardware, command, output location, and status.
- Record overall and group-wise metrics, checkpoint-selection criteria, and relevant notes.
- Keep exploratory results separate from verified results. Never report an unverified run as a reproduced result.
- Store large outputs, checkpoints, raw data, and caches outside Git (for example, approved Drive storage). Commit small, useful result summaries only when appropriate.

## 6. Documentation responsibilities

- Root `README.md`: project overview, high-level structure, and getting started.
- `CONTRIBUTING.md`: branch, commit, review, and contribution workflow.
- `docs/research_protocol.md`: shared experimental and reproduction rules.
- `docs/methods.md`: concise cross-method research overview and paper references.
- `src/methods/<method-name>/README.md`: method mechanism, source, setup, commands, settings, adaptation, and limitations.
- `configs/`: actual run settings.
- Experiment logs/results: what was run and what was observed.

Update the closest relevant documentation in the same pull request as a behavior or protocol change.
