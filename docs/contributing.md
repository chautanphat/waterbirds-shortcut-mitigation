# Waterbirds Background Shortcut Learning Mitigation
## Project Workflow, Repository Structure, and Documentation Requirements

This document summarizes the project's established decisions and conventions. Use it as the source of truth when reviewing or preparing the final project documentation. Do not introduce new workflow rules or architectural decisions without clearly identifying them as proposals.

---

## 0. Contributor Operating Rules

### Scope and authority

- This guidance applies to all contributors (human team members and AI assistants) working on this repository.
- Work locally unless explicitly asked for a GitHub action.
- Do not add machine-specific paths, personal environment details, credentials, or local-only assumptions to tracked documentation.
- Treat the desired repository tree as a **target convention**, not authorization to create directories, move code, or refactor files before the relevant work begins.
- Preserve existing method implementations and documentation unless the requested task explicitly includes changing them.

### Before making changes

1. Inspect the relevant files and current repository state.
2. Check the applicable sections of this document before changing structure, documentation, methods, or experiment procedures.
3. Keep the change focused on the requested task.
4. If the intended structure, documentation content, or workflow is unclear, ask before editing. Do not fill gaps with new project conventions.

### Documentation authority

- `docs/contributing.md` defines project-wide operating rules and contribution guidelines.
- The root `README.md` is a concise entry point and overview.
- `docs/` contains detailed shared research infrastructure.
- Each method directory contains implementation-specific instructions and upstream preservation notes.
- When these documents need to change together, update only the parts required by the approved task and keep them consistent.

---

## 1. Project Overview

**Research topic:** Background Shortcut Learning Mitigation for the Waterbirds dataset.

**Main objective:** Investigate and evaluate methods for mitigating background-related shortcut learning. The project involves implementing and evaluating multiple methods, including established baselines, under a carefully documented experimental protocol.

**Project context:**
- Team-based research project.
- Beginner-friendly implementation is important.
- Google Colab is the primary compute environment.
- Python and deep learning experience may vary among team members.
- Reproducibility, fair comparison, and a defensible research contribution are priorities.

The project is not simply about collecting implementations or combining existing techniques. Each method should be understood, reproduced where feasible, and evaluated systematically.

## 2. Core Principles

All project development and documentation should follow these principles:

1. **Reproducibility first.** Preserve the original method and understand its implementation before modifying it.
2. **Separate original reproduction from adaptation.** Clearly distinguish reproducing a paper's original experiment from adapting its method to Waterbirds.
3. **Fair comparison.** Use a consistent and documented evaluation protocol wherever possible, while preserving method-specific procedures necessary for faithful reproduction.
4. **Avoid premature abstraction.** Do not force every method into the same implementation structure.
5. **Keep method-specific logic isolated.** Each method should remain understandable and independently maintainable.
6. **Share infrastructure only when appropriate.** Extract common components only when multiple methods genuinely need the same functionality.
7. **Keep documentation organized.** The root README should be concise; detailed instructions belong in `docs/`.
8. **Keep experiments traceable.** Every reported result should be linked to its configuration, command, seed, environment, and relevant implementation.

## 3. Desired Repository Structure

The following is the intended logical structure. It is a guide, not a requirement to create every directory immediately.

```text
waterbirds-shortcut-mitigation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── erm/
│   ├── groupdro/
│   └── ...
│
├── baselines/
│   ├── dpe/
│   ├── groupdro/
│   └── ...
│
├── data/
│   └── README.md
│
├── docs/
│   ├── research_protocol.md
│   ├── dataset.md
│   ├── methods.md
│   ├── experiment_log.md
│   └── ...
│
├── notebooks/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── ...
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── evaluation/
│   ├── utils/
│   │
│   └── methods/
│       ├── erm/
│       ├── groupdro/
│       ├── jtt/
│       └── ...
│
├── tests/
│
├── experiments/
│
└── results/
```

### 3.1. Root directory

The root should contain only important project-level files and directories.

Examples:
- `README.md`: short project introduction and navigation.
- `requirements.txt`: project-level dependencies, where appropriate.
- `.gitignore`: shared ignore rules.
- `baselines/`: preserved external implementations.
- `configs/`: experiment configurations.
- `docs/`: detailed documentation.
- `src/`: implementation.
- `scripts/`: project-level entry points.
- `tests/`: tests.

Do not place baseline implementation files directly in the root directory.

### 3.2. External baselines and project methods

The repository distinguishes preserved external implementations from project-owned code:

- `baselines/<method-name>/`: official implementations obtained from other researchers and used for faithful reproduction.
- `src/methods/<method-name>/`: methods implemented by this project, reimplemented from a paper, or integrated into the common framework.

Do not unnecessarily modify code under `baselines/`. Preserve its README, dependencies, configurations, license notices, and original execution flow. Put adaptations or standardized implementations in `src/methods/` instead.

### 3.3. Method-specific implementation

Each project-owned method belongs under:

```text
src/methods/<method_name>/
```

For example:

```text
src/methods/poter/
├── README.md
├── ORIGINAL_SOURCE.md
├── train.py
├── models.py
├── loss.py
└── ...
```

The exact files and subdirectories depend on the original implementation.

Keep method-specific components inside their own directory unless there is a clear reason to share them.

Do not require every method to have identical filenames or internal organization.

### 3.4. Shared infrastructure

The following directories are intended for components genuinely reused by multiple methods:

- `src/datasets/`: shared dataset loading and preprocessing.
- `src/models/`: shared model architectures or reusable model components.
- `src/evaluation/`: common evaluation utilities and metrics.
- `src/utils/`: general-purpose utilities.

A component should be moved into shared infrastructure only if:
- Multiple methods need it.
- Its behavior and assumptions are compatible across those methods.
- Extracting it does not change the original method's behavior.

Do not refactor a method simply to make its code look consistent with another method.

### 3.5. Scripts and configurations

Project-level scripts should provide convenient entry points for common operations.

For example:

```text
scripts/
├── train.py
├── evaluate.py
└── reproduce.py
```

A common launcher may select a method and configuration, then delegate to the appropriate method-specific implementation.

Method-specific entry points may remain inside the method directory when necessary.

Configuration files should be stored in `configs/`, preferably using YAML.

Keep experiment settings in configuration files rather than scattering important values throughout scripts.

### 3.6. Data and generated outputs

Do not commit large raw datasets, checkpoints, caches, or other generated artifacts to Git unless there is a specific reason.

The `data/` directory can contain instructions explaining how to obtain and organize the dataset.

Use `.gitignore` to exclude:
- Raw datasets.
- Checkpoints and model weights.
- Experiment outputs.
- Temporary files.
- Python caches.
- Environment-specific files.
- Credentials and secrets.

Team members should be able to reproduce the directory setup using documentation without needing the actual dataset committed to the repository.

---

## 4. Baseline Integration Workflow

Every new baseline should follow this lifecycle:

**Preserve → Understand → Reproduce → Adapt → Refactor**

Do not skip directly to adaptation or refactoring.

### Stage 1: Preserve the original implementation

When introducing a baseline:

1. Obtain the original implementation from its official repository or another clearly identified source.
2. Preserve all relevant source files.
3. Preserve associated scripts, documentation, configurations, dependency files, and license notices.
4. Preserve the original README or equivalent usage instructions.
5. Record the original repository URL and, where possible, the upstream commit or release.
6. Place the preserved external implementation under `baselines/<method_name>/`.

Do not:
- Dump baseline files into the repository root.
- Immediately distribute files across shared directories.
- Delete files merely because they appear unnecessary before understanding the implementation.
- Copy nested `.git/` directories.
- Commit raw datasets, checkpoints, caches, credentials, or unrelated generated files.

If the original implementation contains a dataset directory, inspect it carefully. Preserve dataset-loading code and documentation when relevant, but do not commit the actual dataset.

### Stage 2: Understand the implementation

Before running or modifying the method, identify:

- Main entry point.
- Training stages and execution flow.
- Model architecture and components.
- Loss functions and optimization procedure.
- Dataset requirements and preprocessing.
- Label and group definitions.
- Validation and model-selection procedures.
- Evaluation metrics.
- Checkpoint and output formats.
- Dependencies and environment requirements.
- Relevant imports and external utilities.

The goal is to understand what the method actually does, not merely how to execute its script.

### Stage 3: Reproduce the original method

Attempt to reproduce the original paper or official implementation as faithfully as the available environment permits.

Record:
- Original source and commit.
- Execution command.
- Configuration.
- Dataset and preprocessing.
- Random seed.
- Dependencies and software versions.
- Hardware and compute environment.
- Training behavior.
- Evaluation procedure.
- Results.

Clearly label the reproduction status:

- **Smoke test:** verifies that the code can execute.
- **Partial reproduction:** reproduces some components or experiments but not the complete reported result.
- **Verified reproduction:** sufficiently reproduces the relevant original experiment and provides supporting evidence.

Do not claim verified reproduction based only on successful execution.

### Stage 4: Adapt the method to Waterbirds

After understanding the original implementation, adapt it to the Waterbirds task. Keep the preserved source under `baselines/`; put the project adaptation under `src/methods/` when it becomes project-owned or integrated.

Potential adaptations include:
- Dataset loading.
- Label and group mapping.
- Training and evaluation data handling.
- Model architecture or pretrained weights.
- Training configuration.
- Evaluation metrics.
- Checkpoint selection.

Document every meaningful deviation from the original implementation.

Keep two sets of settings conceptually separate:

1. **Original reproduction settings:** settings used to reproduce the original paper or implementation.
2. **Waterbirds adaptation settings:** settings required to apply the method to the project's dataset and experimental protocol.

Do not silently change an original method's assumptions and then describe the result as an exact reproduction.

### Stage 5: Refactor reusable components

Only after the method works and its behavior is understood should reusable components be extracted.

For example:
- A dataset loader used by several methods may belong in `src/datasets/`.
- A genuinely shared evaluation metric may belong in `src/evaluation/`.
- A common model component may belong in `src/models/`.

After extraction:
1. Update imports and entry points.
2. Check that the method still runs.
3. Compare behavior before and after refactoring.
4. Update documentation.
5. Keep method-specific logic in the method directory.

The objective is maintainability without compromising fidelity or reproducibility.

---

## 5. Dataset and Fair Evaluation Protocol

The Waterbirds dataset is the shared benchmark for the project's comparisons.

The project should establish and document a common evaluation protocol.

### 5.1. Shared dataset protocol

Where possible, methods should use the same:
- Dataset version and source.
- Train, validation, and test split definitions.
- Label definitions.
- Group definitions and mapping.
- Test set.
- Preprocessing assumptions relevant to evaluation.

Record the dataset source, directory organization, split files, and any preprocessing.

### 5.2. Preserve method-specific validation procedures

A common evaluation protocol does not mean forcing every method to use an identical validation strategy.

Different methods may have different reasons for using validation data, including:
- Hyperparameter selection.
- Model selection.
- Early stopping.
- Group-specific procedures.
- Method-specific training stages.

Preserve these procedures when necessary to reproduce the method faithfully.

**Do not silently standardize or remove a method's validation procedure merely to make the code look consistent.**

The important distinction is:

- Validation data may be used according to a method's documented procedure.
- Test data must remain held out for final evaluation.
- The test set must not be used for training or hyperparameter/model selection.

Document differences in validation and model-selection procedures so that readers can understand the comparison.

### 5.3. Evaluation metrics

Report overall performance and group-wise performance.

At minimum, include:
- Overall accuracy.
- Accuracy for each Waterbirds group.
- Worst-group accuracy.
- Number of examples in each evaluated group.

Use the same group definitions and test set for comparisons wherever possible.

Do not report only overall accuracy, because it can conceal poor performance on minority groups.

---

## 6. Experiment Management and Reproducibility

Every meaningful experiment should be traceable.

Use configuration files as the source of truth for experiment settings.

A configuration should record relevant fields such as:

```yaml
method: groupdro

dataset:
  name: waterbirds
  split: official

model:
  architecture: resnet50
  pretrained: true

training:
  seed: 0
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam

evaluation:
  metric: worst_group_accuracy
```

This is an illustrative structure, not a mandatory configuration schema. Each method may require additional parameters.

For each experiment, record:

- Method and implementation version.
- Dataset version and split.
- Random seed.
- Model and pretrained weights.
- Optimizer and learning rate.
- Batch size.
- Epochs or training steps.
- Method-specific hyperparameters.
- Validation and model-selection procedure.
- Environment and hardware.
- Configuration file.
- Execution command.
- Output and checkpoint locations.
- Evaluation metrics.
- Deviations from the intended protocol.

### 6.1. Experiment status

Clearly distinguish:
- Exploratory experiments.
- Debugging and smoke tests.
- Partial reproductions.
- Verified reproductions.
- Final reported experiments.

Do not mix preliminary results with final results without labeling them.

### 6.2. Results and checkpoints

Keep large results and checkpoints outside Git unless explicitly needed.

Store experiment metadata and concise summaries in the repository so that results remain interpretable.

A result without its configuration, command, seed, and evaluation context should not be treated as fully reproducible.

---

## 7. Team Collaboration and Git Workflow

The project is maintained collaboratively through GitHub.

Use branches for development rather than committing directly to `main`.

Suggested workflow:

1. Create a branch for a specific task or method.
2. Make changes within the appropriate directories.
3. Run relevant tests or experiments.
4. Update documentation when behavior or workflow changes.
5. Open a pull request.
6. Review the changes and supporting evidence.
7. Merge only after the changes are accepted.

Keep pull requests focused. Avoid mixing unrelated documentation, implementation, and experiment changes.

When reviewing a baseline contribution, check:
- Whether an official implementation is preserved under `baselines/` and project-owned code is under `src/methods/` where appropriate.
- Whether its source and license are documented.
- Whether generated data or checkpoints were accidentally added.
- Whether Waterbirds adaptations are identified.
- Whether the experiment can be executed using the documented instructions.
- Whether results include sufficient reproducibility information.

---

## 8. Documentation Requirements

Documentation should be organized by purpose.

### 8.1. Root README.md

The root README must be **short and concise**.

It should provide:
- Project title.
- A brief description of the research objective.
- A concise overview of the repository.
- Basic setup or quick-start instructions, if stable.
- Links to detailed documentation.

Do not turn the root README into a comprehensive manual.

Detailed instructions should live in `docs/` or in the relevant method directory.

Avoid duplicating long explanations across the root README and other documentation.

### 8.2. `docs/research_protocol.md`

Describe the project's shared research and experimental protocol.

Include:
- Research objective and scope.
- Dataset and split conventions.
- Group definitions.
- Fair comparison principles.
- Validation and test-set rules.
- Evaluation metrics.
- Reproducibility requirements.
- Reporting conventions.

### 8.3. `docs/dataset.md`

Document:
- Dataset source.
- Download and setup instructions.
- Expected directory structure.
- Labels and group definitions.
- Split information.
- Preprocessing.
- Dataset verification.

Do not include the actual dataset.

### 8.4. `docs/methods.md`

Provide a concise overview of all methods and baselines.

Do not pre-populate method details before the team begins investigating that method. When a baseline begins, add an entry using the agreed method-entry template documented in `docs/README.md`.

For each method, summarize:
- Method name.
- Paper and source repository.
- Main mechanism.
- Intended role in the comparison.
- Implementation location.
- Reproduction status.
- Waterbirds adaptation status.
- Important deviations or limitations.

This file should help readers understand the method landscape without duplicating implementation-level details.

### 8.5. Method-specific README

Each baseline or project-method directory should have its own README explaining how to use that implementation.

Include:
- Date imported.
- Method overview.
- Original paper.
- Paper title and URL.
- Original implementation structure.
- Original source repository.
- Dependencies.
- Setup instructions.
- Execution commands.
- Configuration.
- Expected outputs.
- Waterbirds-specific notes.
- Known limitations.
- Dependencies and compatibility changes.
- Known deviations.
- License and notices.

Preserve the upstream README when it is part of the original implementation. Add project-specific documentation without misrepresenting the original instructions.


### 8.6. `docs/experiment_log.md`

Maintain a concise experiment index.

For each experiment, record:
- Experiment identifier.
- Method.
- Configuration.
- Date.
- Seed.
- Execution status.
- Results.
- Link or path to detailed outputs.

The configuration files and experiment artifacts remain the detailed source of truth.

### 8.7. Other documentation

Additional documents can be created when useful, such as:
- Pull request templates.
- Methods documenting template.
- Testing instructions.
- Troubleshooting guides.

Do not create documentation solely to make the repository look complete. Each file should have a clear purpose and an owner or maintenance plan.

---

## 9. Rules for Reviewing or Modifying the Repository

When contributing to or modifying this repository, follow these constraints:

1. Inspect the existing repository before proposing structural changes.
2. Follow the established directory responsibilities.
3. Do not modify the root README unless explicitly requested.
4. When editing the root README, keep it concise and move detailed guidance into `docs/`.
5. Do not move method-specific code into shared infrastructure prematurely.
6. Do not delete upstream implementation files before understanding their role.
7. Do not silently change experiment protocols or validation procedures.
8. Do not commit datasets, checkpoints, caches, or credentials.
9. Do not claim reproduction without evidence.
10. Update relevant documentation when implementation behavior or experimental procedures change.
11. Prefer small, focused changes that are easy to review.
12. Clearly report what was changed and what remains incomplete.
13. When instructions do not establish a clear answer, ask for clarification before creating structure, changing workflow, or adding documentation conventions.

---

## 10. Expected Outcome of the Final Documentation

The final documentation should make it possible for a new team member to understand:

1. What the research project is about.
2. How the repository is organized.
3. Where each method's implementation belongs.
4. How to add a new baseline.
5. How to reproduce an original implementation.
6. How to adapt a method to Waterbirds.
7. How to run and record experiments.
8. How methods are evaluated fairly.
9. How to contribute changes safely.
10. Where to find detailed instructions.

Keep the root README concise and use it as the entry point. Put detailed procedures in `docs/` and method-specific documentation in each method directory.

**Final guiding principle:** Preserve original methods, understand them before adapting them, evaluate them under a transparent protocol, and share code only when the shared abstraction is genuinely justified.
