# ACIIDS 2027 paper

`main.tex` is the manuscript (Springer LNCS): *Aggregation, Cross-Fitting or Clean
Labels? Where the Validation Set Pays Off for Prototype Ensembles on Waterbirds*. It inputs
`sections/intro.tex`, `sections/related.tex` and `sections/body.tex`, uses `refs.bib`, and
pulls `figures/levers.tex` (built by `build_levers.py`). The earlier planning outline is kept in
`backup_before_outline_20261114/outline_main_20261114.tex`, next to the Week 2 copy.

## Build

```powershell
python build_levers.py     # regenerates figures/levers.tex (Fig. 1)
python build_evidence.py   # figures/allocation.tex (no longer used in the paper)
pdflatex main; bibtex main; pdflatex main; pdflatex main
```

## Where each number comes from

| Paper element | Source |
| --- | --- |
| Fig. 1 (levers) | `build_levers.py` -> `figures/levers.tex`, `figures/levers_audit.json` (strict nested run, independent-methods study, label study) |
| Table 1 (reproduction) | `results/dpe/independent_methods_20261113`, `results/dpe/nested_selection_20261115`, `results/dpe/upstream_reproduction_20261115` |
| Table 2 (aggregation, three partitions) | `results/dpe/independent_methods_20261113/summary.csv`; dropout MLP and power rules: `results/dpe/five_seed_verification_20261112`; selected weights: `src/methods/dpe_head_reweighting/reports/h2.md` |
| Table 3 (validation-only selection) | `results/dpe/strict_nested_20261115/results/*/split*.json` (strictly nested rerun) |
| Table 4 (worst-group errors) | `outputs/dpe_controls/*/cache/full` predictions vs `metadata.csv` (saved heads, split seed 0) |
| Table 5 (label study) | `results/dpe/label_correction_20261115/REPORT.md` |
| Recalibration = threshold | `results/dpe/threshold_controls_20261115` |
| Re-scored comparisons | `results/dpe/label_correction_20261115/rescored_variants.csv` |
| Replication subsection (TODO) | Kaggle `noerula/dpe-corrected-label` -> `experiments/dpe_label_corrected_backbones/summarize.py` |

Tables are typed into `sections/body.tex`; re-check them if a source changes. The previous
framing is kept in `backup_before_levers_20261115/`.

## Before submission

- Confirm the EasyChair deadline and whether review is double-blind. If it is, remove the
  author block, the e-mail addresses and the GitHub footnote in `sections/intro.tex`.
- The GitHub footnote assumes the repository is public at submission time.
- Add any AI-assistance statement the call for papers requires.
