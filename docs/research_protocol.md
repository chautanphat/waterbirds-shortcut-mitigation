# Research Protocol

## Current phase

Reproduce baseline methods from selected papers before developing or comparing new mitigation methods.

## Reproduction workflow

1. Record the paper, code source, dataset, and target result.
2. Create a focused GitHub branch for the reproduction work.
3. Save a tracked configuration in `configs/` or `experiments/`; document the exact command in the method README.
4. Document environment, data preparation, command, seed, and hardware.
5. Run a small smoke test before a full experiment.
6. Compare results with the paper and explain meaningful differences.
7. Open a pull request containing the code, documentation, configuration, and experiment-log entry.

## Evidence levels

- **Smoke test:** confirms that the pipeline runs.
- **Local reproduction:** completes the intended experiment with documented settings.
- **Verified reproduction:** results are compared against the paper and differences are explained.

Do not describe a smoke test as a reproduced benchmark.
