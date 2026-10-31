# Background Shortcut Learning Mitigation

This project explores methods for mitigating shortcut learning and spurious correlations in machine-learning systems.

The repository is intended for collaborative research across multiple datasets, baselines, and mitigation methods. The root README gives only the project-level conventions; detailed setup, implementation, and experiment guides belong in [`docs/`](docs/) and in the documentation for each method.

## Repository layout

```text
.
├── src/
│   └── methods/
│       └── <method-name>/
├── docs/
└── README.md
```

- `src/methods/<method-name>/` contains a method implementation.
- `docs/` contains detailed project and experiment guides.
- Each method should document its own setup, dependencies, commands, and limitations.
- Datasets, checkpoints, logs, caches, and other generated artifacts should stay out of version control.

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

Record the dataset, split, seed, model, main hyperparameters, environment, and metrics for meaningful experiments. Clearly distinguish exploratory results, smoke tests, and verified benchmark results.

See [`docs/`](docs/) for detailed project guidance and the existing method directories for implementation-specific documentation.
