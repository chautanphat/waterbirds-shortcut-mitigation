# ACIIDS 2027 paper

`main.tex` is the manuscript draft (Springer LNCS). It inputs `sections/intro.tex`,
`sections/related.tex` and `sections/body.tex`, uses `refs.bib`, and pulls the
generated `evidence_table.tex` and `figures/allocation.tex`.
The earlier planning outline is kept in
`backup_before_outline_20261114/outline_main_20261114.tex`, next to the Week 2 copy.

## Build

```powershell
python build_evidence.py
pdflatex main; bibtex main; pdflatex main; pdflatex main
```

`build_evidence.py` uses only Python's standard library. It reads
`results/dpe/independent_methods_20261113/per_seed.csv` and `summary.csv`, checks
per-source WGA arithmetic and summary agreement, and writes `evidence_table.tex`,
`figures/allocation.tex` and `evidence_audit.json` (values and paired contrasts).
Tables 2 and 3 in `sections/body.tex` (per-backbone and per-group differences) are
typed in from the same CSV; re-check them if the CSV changes.

## Status markers

- Red `[[...]]`: waits for the nested-selection run (`REPORT.md`, `results_long.csv`).
- Blue `[TODO: ...]`: verify before submission (citations, head checkpoint rule,
  taxonomy source).

## Choosing the framing

`main.tex` defaults to the provisional framing `P`, which assumes no outcome. After the
run is in the repository, set `\framing` and `\contribresult` to one of:

| If the report shows | Use |
| --- | --- |
| Selected cross-fit affine beats full-validation DPE **and** both the offset-only control on the same fold ensemble (C2) and the transferred threshold (C3), with intervals above 0 on standard ERM | A |
| C2 or C3 recovers most of the gain (affine minus control interval includes 0) | B |
| Selected cross-fit affine minus full-validation DPE has an interval including 0 on standard ERM | C |

Decide from the table, not from which framing reads best, and delete the unused macros.
