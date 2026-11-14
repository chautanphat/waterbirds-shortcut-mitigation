# Discrete reliability v2

Run from the repository root using the existing research environment:

```powershell
.venv/Scripts/python.exe -m experiments.discrete_weighting.run --config experiments/discrete_weighting/config.yaml --phase all --resume
```

`--status`, `--dry-run`, `--max-tasks 2` and `--max-minutes 10` are supported.
The immutable run ID binds source, config, development cache/head fingerprints
and all 29 predeclared Phase-1 configurations. Read
`outputs/discrete_weighting_v2/LATEST_RUN.txt` to locate the active run.

The authoritative records are `completed/*.json`. Every task publishes checked
artifacts using flushed/fsynced same-directory temporary files and atomic
replacement, followed by its commit marker. State, incremental CSVs, handoff
and next-agent instructions are regenerated from valid markers. Missing/corrupt
artifacts invalidate dependent tasks; abandoned temporary files are ignored.
Pure tables/calibrators checkpoint each fit; no neural training is performed.
An interruption can replay at most one unfinished small unit. Do not launch
multiple writers against the same run directory.

Verified acceptance checks include two separate real `--max-tasks 2` launches,
separate-process synthetic resumes, output corruption and truncated temp files.
Tests: `.venv/Scripts/python.exe -m pytest tests/methods/test_discrete_weighting.py -q --basetemp outputs/dw_tests`.

The completed run is `64b58d2c31b62c9d`. A prior attempt
`68ba1060eb0de9ec` stopped at a comparison-model lookup mismatch. All 275 valid
tasks were migrated and verified without repeating their computations. The
model lookup fix and migration evidence are retained; source/config changes
normally create a new run ID. `--resume-from` is restricted to pre-lock runs
with unchanged inputs, config and core reliability/journal implementations.

OOF/final heads are reused unchanged. Confidence tertiles, pooled/hierarchical
correctness tables, calibration and guard margins fit training data only.
Groups are used only after lock. All A0–A20 variants and their seed-specific
metrics are retained, with equivalent configurations deduplicated and mapped
in `SELECTION_LOCK.json`. No untouched confirmatory dataset is available.

Detailed reporting/audit: `python -m scripts.report_discrete_weighting_v2`.
Read `detailed_report.md`, `results/phase2_summary.csv`, `final_audit.json`,
`HANDOFF_STATUS.md` and `NEXT_AGENT_PROMPT.md` in the active output directory.
The model is a heuristic reliability weighting rule; historical backbone
training provenance is incomplete and the test has been repeatedly inspected.
