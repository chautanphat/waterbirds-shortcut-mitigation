# Background Shortcut Learning Mitigation

This project investigates methods for mitigating shortcut learning and spurious correlations in machine-learning systems.

The repository supports shared research and experimentation across multiple datasets, baselines, and mitigation methods. Method-specific implementations should remain self-contained, while this README provides the common project structure and collaboration guidelines.

## Project organization

```text
.
├── src/
│   └── methods/
│       └── <method-name>/
│           ├── README.md
│           ├── data/
│           ├── scripts/
│           └── implementation files
├── docs/
└── README.md
```

In general:

- `src/methods/<method-name>/` contains one method implementation.
- A method-level `README.md` documents its setup, commands, assumptions, and results.
- `docs/` contains project-wide notes, references, and shared documentation.
- Datasets, feature caches, logs, checkpoints, and other generated artifacts should remain outside version control unless explicitly required.

The repository currently includes a POTER-based implementation under [`src/methods/poter/`](src/methods/poter/), with detailed usage documentation in its [method README](src/methods/poter/README.md).

## Getting started

1. Clone the repository and create a local Python environment appropriate for the method you want to run.
2. Read the relevant method-level README before running experiments.
3. Prepare datasets and local output directories outside the tracked source tree.
4. Run a documented baseline before testing a new mitigation method.
5. Record the configuration and results so that another team member can reproduce the experiment.

There is no single project-wide dependency environment yet. Dependencies may differ between methods, so each method should document its own requirements clearly.

## Team workflow

### Branches

Use a separate branch for each feature, method, experiment, documentation update, or bug fix.

Recommended branch names:

```text
feature/<short-description>
method/<method-name>
experiment/<short-description>
fix/<short-description>
docs/<short-description>
```

Keep branches focused and avoid combining unrelated changes.

### Commits

Prefer small, focused commits with descriptive messages. A commit should make it clear what changed and why.

Examples:

```text
Add baseline evaluation script
Document CMNIST experiment setup
Fix validation-group accuracy calculation
Update method reproduction notes
```

### Pull requests

Before opening a pull request:

- Confirm that the relevant code or documentation works as described.
- Run the applicable tests or smoke tests.
- Include the commands used for validation.
- Summarize important configuration choices.
- Mention known limitations or unfinished work.
- Keep generated files and unrelated formatting changes out of the pull request.

For experiment changes, include the dataset, random seed, model, main hyperparameters, and evaluation metrics.

## Adding a new method

A new method should generally be added under:

```text
src/methods/<method-name>/
```

At minimum, include:

- A method-level `README.md`.
- A clear entry point for training or evaluation.
- Reproducible example commands.
- Instructions for the expected dataset layout.
- A description of required dependencies.
- Baseline and evaluation details.
- Notes about output files and generated artifacts.

Method documentation should explain how the implementation fits into the broader shortcut-learning mitigation project without assuming that readers already understand the code.

## Experiment and reproducibility guidelines

For each meaningful experiment, record:

- Dataset and dataset version or source.
- Train, validation, and test split details.
- Random seed.
- Model and pretrained checkpoint.
- Main hyperparameters.
- Hardware and software environment.
- Baseline used for comparison.
- Evaluation metrics, including group-wise metrics when relevant.
- Location or naming convention of generated outputs.

Separate exploratory results from verified results. Do not describe a smoke test as a full benchmark reproduction.

## Files that should not be committed

Do not commit the following unless the project explicitly requires them:

- Raw datasets.
- Feature caches.
- Model checkpoints.
- Training logs and generated reports.
- Local environments.
- Credentials, tokens, or private configuration.
- Temporary editor or notebook files.

The repository's `.gitignore` excludes common generated artifacts. If a new tool produces additional local files, update the ignore rules in a focused change.

## Documentation guide

- **Root README:** project overview, shared structure, and team workflow.
- **Method README:** method-specific setup, implementation details, commands, and limitations.
- **Project docs:** shared research notes, references, design decisions, and cross-method guidance.

When behavior changes, update the closest relevant documentation in the same pull request.

## References

See the documentation inside each method directory for method-specific papers, upstream repositories, and reproduction notes.
