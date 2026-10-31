# Research Methods

This is the project-wide inventory of methods being investigated. It explains the comparison landscape; method directories contain implementation details.

## How to add a method

Add a method only after the team agrees to investigate it. Then:

1. Preserve the upstream implementation under `src/methods/<method-name>/`.
2. Add a method-specific README with source, paper, license, setup, commands, and Waterbirds-specific changes.
3. Add one entry here using the template in [`README.md`](README.md).
4. Link a tracked configuration and experiment-log entry.

Do not claim an original reproduction when the work is only a smoke test or an adaptation.

Each completed method entry should point to:

- `src/methods/<method-name>/` for the implementation;
- `configs/<method-name>/` for reproducible settings;
- `experiments/<experiment-id>/` for run metadata, when present; and
- the relevant entry in `docs/experiment_log.md`.

## Planned methods

The initial comparison is expected to include ERM, GroupDRO, JTT, and other agreed baselines. No method entry is populated until its investigation begins.

## Method entries

Add approved methods below this heading using the shared template.
