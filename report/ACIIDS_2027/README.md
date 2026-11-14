# ACIIDS 2027 paper

`main.tex` is the manuscript (Springer LNCS): *Eighteen Labels Outweigh Aggregation:
Validation-Label Errors and Data Allocation for Prototype Ensembles on Waterbirds*. It inputs
`sections/intro.tex`, `sections/related.tex` and `sections/body.tex`, uses `refs.bib`, and
pulls `figures/allocation.tex`. The earlier planning outline is kept in
`backup_before_outline_20261114/outline_main_20261114.tex`, next to the Week 2 copy.

## Build

```powershell
python build_evidence.py   # regenerates figures/allocation.tex (and evidence_table.tex, no longer used)
pdflatex main; bibtex main; pdflatex main; pdflatex main
```

## Where each number comes from

| Paper element | Source |
| --- | --- |
| Table 1 (reproduction) | `results/dpe/independent_methods_20261113`, `results/dpe/nested_selection_20261115`, `results/dpe/upstream_reproduction_20261115` |
| Table 2 (worst-group errors) | `outputs/dpe_controls/*/cache/full` predictions vs `metadata.csv` (saved heads, split seed 0) |
| Table 3 (label study) | `results/dpe/label_correction_20261115/REPORT.md` |
| Table 4 (spending the validation set) | `results/dpe/nested_selection_20261115/results_long.csv` |
| Fig. 1 (allocation) | `results/dpe/independent_methods_20261113` via `build_evidence.py` |
| Recalibration = threshold | `results/dpe/threshold_controls_20261115` |
| Re-scored comparisons | `results/dpe/label_correction_20261115/rescored_variants.csv` |

Tables 1--4 are typed into `sections/body.tex`; re-check them if a source changes.

## Before submission

- Confirm the EasyChair deadline and whether review is double-blind. If it is, remove the
  author block, the e-mail addresses and the GitHub footnote in `sections/intro.tex`.
- The GitHub footnote assumes the repository is public at submission time.
- Add any AI-assistance statement the call for papers requires.
