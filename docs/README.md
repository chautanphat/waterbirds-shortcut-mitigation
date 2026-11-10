# Project Documentation

These documents define the shared research infrastructure for reproducing and comparing Waterbirds baselines. Read them before starting a method or reporting a result.

## Desired repository structure

This is the target structure for the project. It describes where work should go as the repository grows; it does not require every directory to exist yet.

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
│   └── methods/                # Project-owned or integrated implementations
│       └── <method-name>/
│           └── ...             # Independent method implementations
├── tests/                      # Tests for shared code and methods
├── experiments/                # Experiment records and tracked metadata
└── results/                    # Lightweight summaries, tables, and plots
```

### Placement rules

- Keep an imported or reproduced official implementation inside `baselines/<method-name>/` until its behavior is understood.
- Put a project-owned, reimplemented, or integrated method inside `src/methods/<method-name>/`.
- Put code in shared `src/` modules only after more than one method genuinely uses it without changing behavior.
- Put reproducible settings in `configs/` and experiment metadata in `experiments/`.
- Keep raw data, checkpoints, caches, and large local outputs out of Git.
- Put project-wide rules in `CONTRIBUTING.md` and `docs/`; put method-specific setup and commands beside the method.

| Document | Purpose |
| --- | --- |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributor operating rules, repository structure, baseline lifecycle, and workflow conventions. |
| [`research_protocol.md`](research_protocol.md) | Fair-comparison, validation, testing, and reporting rules. |
| [`dataset.md`](dataset.md) | Waterbirds source, groups, splits, local-data expectations, and verification. |
| [`methods.md`](methods.md) | Method inventory and the rule for documenting a baseline. |
| [`experiment_log.md`](experiment_log.md) | Traceable index of meaningful runs. |
| [DPE H2 experiment report](../src/methods/dpe_head_reweighting/reports/h2.md) | Summary of head-weighting experiments, results, interpretation and follow-up hypotheses. |
| [Shared ERM](../src/methods/erm/README.md) | DPE-aligned repository-wide fine-tuning, evaluation, and reusable features. |
| [Failure case viewer](failure_case_viewer.md) | Local image inspection and comparison of saved per-example predictions across approaches. |
| [Export predictions](export_predictions.md) | Viewer CSV recovery and export for POTER, original DPE, AutoBackSwap, and ERM. |
| [Prediction CSV protocol](prediction_protocol.md) | Required version-1 per-example export contract and validator for all methods. |
| [Model artifacts and inference contract](model_artifact_contract.md) | Frozen-predictor description, native loading/preprocessing, selection and inference verification. |
| [Experiment records contract](experiment_record_contract.md) | Version-1 run evidence, execution/claim status, attempts, artifact links and acceptance review. |

## Baseline contribution checklist

1. Decide whether the implementation belongs in `baselines/` or `src/methods/`.
2. Record its source, paper, license, and intended role in a method README.
3. Use a tracked configuration in `configs/` or `experiments/`.
4. Run a smoke test before claiming a reproduction.
5. Add the method summary and experiment-log entry in the same pull request.
6. Export full per-example predictions using the [shared CSV protocol](prediction_protocol.md), retain provenance, and validate against the reference split metadata.
7. For new meaningful runs, retain `run.json` and the frozen predictor's `model.json` under `experiments/<experiment-id>/`, following the experiment and model contracts above. Record missing artifacts explicitly.

## Method-entry template

When the team starts a baseline, copy this template into [`methods.md`](methods.md):

```markdown
## <Method name>

### Overview
- **Full name:**
- **Paper:**
- **Source:**
- **Year:**
- **Implementation location:** `baselines/<method-name>/` or `src/methods/<method-name>/`

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
- Prediction export command and CSV/provenance paths:
- Canonical location: `experiments/<experiment-id>/predictions/<split>/`.
- Reference metadata and CSV validation command:

### Status
- Original reproduction:
- Waterbirds adaptation:

### Notes
Important deviations, limitations, and reproduction observations.
```

Keep detailed setup and commands in the relevant method README; keep this documentation focused on the shared protocol and research record.
