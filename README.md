# Background Shortcut Learning Mitigation

This project explores methods for mitigating shortcut learning and spurious correlations in machine-learning systems.

The repository is intended for collaborative research across multiple datasets, baselines, and mitigation methods. The root README gives only the project-level conventions; detailed setup, implementation, and experiment guides belong in [`docs/`](docs/) and in the documentation for each method.

## Repository layout

The following is the intended repository structure. Directories are added as the project needs them; this layout is a shared convention, not a claim that every component already exists.

```text
waterbirds-shortcut-mitigation/
├── README.md                   # Short project entry point
├── requirements.txt            # Shared dependencies, when stable
├── .github/                    # Issue/PR templates and GitHub workflows
├── baselines/                  # Preserved external implementations
│   └── <method-name>/
├── configs/                    # YAML settings for datasets, baselines, and methods
├── data/                       # Download/setup instructions; no raw data in Git
├── docs/                       # Protocol, dataset, methods, and experiment documentation
├── notebooks/                  # Exploration, debugging, and visualisation
├── scripts/                    # Project-level train/evaluate/reproduce entry points
├── src/
│   ├── datasets/               # Shared data loading and transforms
│   ├── models/                 # Shared backbones and classifiers
│   ├── evaluation/             # Shared metrics and evaluation utilities
│   ├── utils/                  # Shared seeding, logging, and checkpoint utilities
│   └── methods/                # Project-owned or integrated method implementations
│       └── <method-name>/
│           └── ...             # Independent method implementations
├── tests/                      # Tests for shared code and methods
├── experiments/                # Experiment records and tracked metadata
└── results/                    # Lightweight summaries, tables, and plots
```

Use `baselines/<method-name>/` for an official external implementation that is being preserved and reproduced. Use `src/methods/<method-name>/` for a method implemented by this project, reimplemented from a paper, or integrated into the project's common framework. Store tracked experiment settings in `configs/` or `experiments/`; keep raw data, large checkpoints, caches, and local logs out of version control.

## Team workflow

For full contributor operating rules, repository conventions, and the baseline integration lifecycle, refer to [`docs/contributing.md`](docs/contributing.md).

- Create a focused branch for each feature, method, experiment, fix, or documentation change.
- Keep commits small and describe what changed.
- Open a pull request for review when the work is ready.
- Include validation steps and important configuration details in the pull request.
- Keep unrelated changes and generated files out of the pull request.

Suggested branch names:

```text
feature/<description>
method/<name>
experiment/<description>
fix/<description>
docs/<description>
```

## Adding a method

For an official external implementation, add it under `baselines/<method-name>/` and preserve its source, README, dependencies, configurations, license notices, and original execution flow. Avoid unnecessary source changes.

For a project-owned or integrated implementation, add it under `src/methods/<method-name>/` and include a method-level README with:

- summary of the method (source, mechanism, architectures, pipeline,...);
- setup and dependencies;
- dataset expectations;
- reproducible commands;
- baseline and evaluation details; and
- known limitations.

Record the dataset, split, seed, model, main hyperparameters, environment, and metrics for meaningful experiments. Store tracked configurations in `configs/` or `experiments/`, and record the exact command in the method README and experiment log. Clearly distinguish exploratory results, smoke tests, and verified benchmark results.

See [`docs/`](docs/) (including [`docs/contributing.md`](docs/contributing.md)) for detailed project guidance and the existing method directories for implementation-specific documentation.
