# Waterbirds Dataset

## Shared task definition

Waterbirds is the shared benchmark for project comparisons.

- **Target:** bird category — waterbird or landbird.
- **Spurious attribute:** background — water or land.
- **Evaluation groups:** every target/background combination.

| Target | Background |
| --- | --- |
| Waterbird | Water |
| Waterbird | Land |
| Landbird | Land |
| Landbird | Water |

Overall accuracy is not sufficient: report per-group accuracy and worst-group accuracy on the held-out test split.

## Source and local setup

When a baseline is introduced, record the official dataset source, version or release, access requirements, and any supplied metadata/split files in that method README and experiment configuration.

Raw data must remain outside Git. Use the project `data/` area only for instructions and lightweight metadata that the project is allowed to track. Do not commit images, feature caches, checkpoints, credentials, or generated datasets.

## Documentation and configuration locations

- Keep project-wide data instructions in `data/README.md` when that directory is created.
- Keep the shared Waterbirds definition and verification rules in this file.
- Store dataset-specific reproducible settings in `configs/`.
- Keep raw data and generated features in local, untracked locations documented by the relevant configuration.

## Required dataset record

For each dataset preparation or variant, record:

- source and version;
- local path convention;
- official train, validation, and test split used;
- target and background/group mapping;
- preprocessing and augmentation;
- verification command or checks; and
- deviations from the official release.

## Verification before training

Before a meaningful run, confirm that the loader can find the expected files, the split counts are plausible, each evaluation group is present, and the target/background mapping matches the configuration. Record the check in the experiment log.
