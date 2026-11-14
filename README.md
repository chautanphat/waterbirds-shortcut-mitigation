# Background Shortcut Learning Mitigation

**Kết quả DPE v2:** D5 đạt **96,076% test accuracy**, tăng **0,368 điểm %** so
với uniform cùng heads; WGA giảm **2,418 điểm %** xuống **90,343%**. D5 là
winner chọn trên validation của nghiên cứu v2. Xem
[báo cáo chi tiết D5](results/dpe/discrete_weighting_v2_20261114/README.md)
để đọc cấu hình, kết quả ba seed và sự đánh đổi về subgroup robustness.

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

## Shared ERM

Use `python scripts/train_erm.py` for the repository-wide, DPE-aligned standard ERM baseline. Its reusable dataset, backbone, evaluation, and feature-export modules live in `src/`; settings are in [`configs/erm/waterbirds_standard.yaml`](configs/erm/waterbirds_standard.yaml). See the [shared ERM guide](src/methods/erm/README.md) for setup, evaluation, and DPE handoff commands.

## Waterbirds Prediction Viewer

Open [`scripts/viewer.html`](scripts/viewer.html) directly in Chrome or Edge to inspect per-image Waterbirds predictions across **DPE**, **AutoBackSwap**, **POTER**, and **ERM** (no local server required):

1. **Predictions auto-load on open:** All 13 exported test runs under `experiments/` are pre-bundled in the HTML viewer for instant comparison across subpopulation groups (`G1–G4`), **Common Failures (All Wrong)**, **Methods Disagree**, **Compare A vs B**, and the **Bird Species Failure Ranking**.
2. **Load bird photos (optional):** Click **`2. 🖼️ Load Bird Photos (data/waterbirds/)`** in the top bar and select your local `data/waterbirds` folder to render the bird photographs on each card.

See the [Waterbirds Prediction Viewer Guide](docs/viewer.md), [Export Predictions Guide](docs/export_predictions.md), and [Shared Prediction CSV Protocol](docs/prediction_protocol.md) for details on exporting new methods (such as GroupDRO) and validating `predictions.csv` artifacts.

## Team workflow

For full contributor operating rules, repository conventions, and the baseline integration lifecycle, refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).

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

See [`docs/`](docs/) and [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed project guidance and the existing method directories for implementation-specific documentation.

## DPE experiment evidence

The [five-seed verification report](results/dpe/five_seed_verification_20261112/README.md) gives the current results, uncertainty and supported claims. The [results index](results/dpe/README.md) separates current evidence from historical checks; [notebook notes](notebooks/README.md) describe available workflows.
