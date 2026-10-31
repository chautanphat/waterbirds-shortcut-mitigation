# Background Shortcut Learning Mitigation

This project explores methods for mitigating shortcut learning and spurious correlations in machine-learning systems.

The repository is intended for collaborative research across multiple datasets, baselines, and mitigation methods. The root README gives only the project-level conventions; detailed setup, implementation, and experiment guides belong in [`docs/`](docs/) and in the documentation for each method.

## Repository layout

The following is the intended repository structure. Directories are added as the project needs them; this layout is a shared convention, not a claim that every component already exists.

```text
.
├── .github/                     # Issue/PR templates and GitHub workflows
├── configs/                     # YAML configurations for datasets, baselines, and methods
├── data/                        # Dataset instructions; raw data stays outside Git
├── notebooks/                   # Exploration, debugging, and result visualization
├── src/
│   ├── datasets/                # Shared Waterbirds loaders and transforms
│   ├── models/                  # Shared backbones and classifiers
│   ├── evaluation/              # Shared accuracy and group-wise metrics
│   ├── utils/                   # Seeding, logging, checkpointing, and utilities
│   └── methods/
│       └── <method-name>/
│           └── ...              # Independent method implementations
├── scripts/                     # Training, evaluation, and reproduction entry points
├── tests/                       # Tests for datasets, metrics, and methods
├── experiments/                 # Experiment configurations and records
├── results/                     # CSV summaries, comparison tables, and plots
├── docs/                        # Protocol, dataset, methods, and experiment documentation
└── README.md
```

Keep reusable code in the shared `src/` modules. Keep method-specific code independent under `src/methods/<method-name>/`. Store tracked experiment settings in `configs/` or `experiments/`; keep raw data, large checkpoints, caches, and local logs out of version control.

## Team workflow

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

Add new methods under `src/methods/<method-name>/` and include a method-level README with:

- summary of the method (source, mechanism, architectures, pipeline,...);
- setup and dependencies;
- dataset expectations;
- reproducible commands;
- baseline and evaluation details; and
- known limitations.

Record the dataset, split, seed, model, main hyperparameters, environment, and metrics for meaningful experiments. Store tracked configurations in `configs/` or `experiments/`, and record the exact command in the method README and experiment log. Clearly distinguish exploratory results, smoke tests, and verified benchmark results.

See [`docs/`](docs/) for detailed project guidance and the existing method directories for implementation-specific documentation.
