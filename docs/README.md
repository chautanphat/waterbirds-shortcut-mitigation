# Project Documentation

These documents define the shared research infrastructure for reproducing and comparing Waterbirds baselines. Read them before starting a method or reporting a result.

## Desired repository structure

This is the target structure for the project. It describes where work should go as the repository grows; it does not require every directory to exist yet.

```text
waterbirds-shortcut-mitigation/
├── README.md                   # Short project entry point
├── AGENTS.md                   # Project rules for contributors and agents
├── requirements.txt            # Shared dependencies, when stable
├── .github/                    # Issue/PR templates and GitHub workflows
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
│   └── methods/                # Independent implementations of each method
│       └── <method-name>/
│           └── ...             # Independent method implementations
├── tests/                      # Tests for shared code and methods
├── experiments/                # Experiment records and tracked metadata
└── results/                    # Lightweight summaries, tables, and plots
```

### Placement rules

- Keep an imported or reproduced baseline inside `src/methods/<method-name>/` until its behavior is understood.
- Put code in shared `src/` modules only after more than one method genuinely uses it without changing behavior.
- Put reproducible settings in `configs/` and experiment metadata in `experiments/`.
- Keep raw data, checkpoints, caches, and large local outputs out of Git.
- Put project-wide rules in `docs/`; put method-specific setup and commands beside the method.

| Document | Purpose |
| --- | --- |
| [`research_protocol.md`](research_protocol.md) | Fair-comparison, validation, testing, and reporting rules. |
| [`dataset.md`](dataset.md) | Waterbirds source, groups, splits, local-data expectations, and verification. |
| [`methods.md`](methods.md) | Method inventory and the rule for documenting a baseline. |
| [`experiment_log.md`](experiment_log.md) | Traceable index of meaningful runs. |

## Baseline contribution checklist

1. Preserve the upstream implementation under `src/methods/<method-name>/`.
2. Record its source, paper, license, and original behavior in that method's README.
3. Use a tracked configuration in `configs/` or `experiments/`.
4. Run a smoke test before claiming a reproduction.
5. Add the method summary and experiment-log entry in the same pull request.

## Method-entry template

When the team starts a baseline, copy this template into [`methods.md`](methods.md):

```markdown
## <Method name>

### Overview
- **Full name:**
- **Paper:**
- **Source:**
- **Year:**
- **Implementation location:** `src/methods/<method-name>/`

### Mechanism
Briefly explain the method's central idea.

### Implementation
- Backbone:
- Loss function:
- Training procedure:
- Special requirements:

### Experimental Settings
- Dataset and split:
- Configuration:
- Batch size / learning rate / epochs / optimizer:
- Random seeds:

### Evaluation
- Overall accuracy:
- Per-group accuracy:
- Worst-group accuracy:

### Status
- Original reproduction:
- Waterbirds adaptation:

### Notes
Important deviations, limitations, and reproduction observations.
```

Keep detailed setup and commands in the relevant method README; keep this documentation focused on the shared protocol and research record.
