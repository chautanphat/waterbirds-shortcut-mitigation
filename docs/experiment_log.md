# Experiment Log

Record every meaningful run here. This is an index, not a replacement for configurations, method READMEs, or untracked output files.

## Entry template

```markdown
## <experiment-id> — YYYY-MM-DD

- **Method / version:**
- **Purpose:** smoke test / original reproduction / Waterbirds adaptation / final comparison
- **Branch or commit:**
- **Configuration:** `configs/...` or `experiments/...`
- **Dataset and split:**
- **Seed:**
- **Environment and hardware:**
- **Command:**
- **Output location:** local, untracked path
- **Results:** overall accuracy; per-group accuracy; worst-group accuracy
- **Paper comparison or expected behavior:**
- **Status:** exploratory / smoke test / partial reproduction / verified reproduction / final
- **Notes and follow-up:**
```

Use one entry per meaningful run or a short grouped entry for repeated seeds with the same configuration. Never present a result without its configuration, command, seed, and evaluation context.

Use repository-relative paths in every entry. Link implementation changes to `baselines/` or `src/methods/`, tracked settings to `configs/`, run metadata to `experiments/`, and lightweight summaries to `results/`; keep large outputs and checkpoints local and untracked.
