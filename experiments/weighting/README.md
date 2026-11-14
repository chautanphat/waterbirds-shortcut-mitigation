# Leak-isolated DPE weighting

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m experiments.weighting.run --config experiments/weighting/config.yaml --phase all --resume
```

Dependencies: torch, torchvision, numpy, pandas, scipy, scikit-learn, PyYAML,
joblib, matplotlib, pytest. Use a compatible torch/torchvision pair. The supplied
configuration uses CPU and the existing frozen ResNet-50 checkpoint. It never
downloads replacement pretrained weights or changes the old heads.

The old ensemble records the entire original validation pool as head-training
data. Splitting its predictions afterwards does not make them out of sample.
This separate implementation refits 15 heads in each of three class-stratified
folds. Each sample's gate features come from heads fitted on other samples.
OOF gate train/select is 80/20 by class. Group annotations never enter head or
gate fitting/selection. Selection uses accuracy, then balanced accuracy, then
NLL. Heads use class-balanced subsets and a fixed final epoch. This is an
explicit no-group adaptation, not a claim to reproduce annotated DPE unchanged.
For cheaper runs, `protocol: disjoint` uses a fixed 60/20/20 partition before
head fitting. Pick the protocol before running; do not change an active run.

All 42 handoff ranks have implementations: 41 deployable families plus the
label-aware coverage oracle, which cannot construct an inference predictor.
Small declared grids and two screening seeds precede three-seed finalist fits.
The winner and ablation configurations lock before test extraction/evaluation.
All feasible A0–A20 ablations then run automatically; N/A entries explain why.
Member truncation reuses the ordered heads and refits only weighting.
Full-probability features retain confidence/margin information even when their
explicit scalar features are removed; interpret those ablations accordingly.

Caches are sample-major `[samples, members, classes]`. Upstream `_predict`
returns `[members, samples, classes]`, despite the ambiguous dimension names in
the handoff. Cache identities bind checkpoint, features, IDs, config and source.
No fake distances or embeddings are substituted. U0 exactly averages member
probabilities at temperature 1, without the training entropic scale.

Outputs under `outputs/weighting/` include each candidate model, native winner,
fold/final heads, per-sample outputs, all tables, plots and the research report.
`phase1_ranked.csv` is the two-seed screening table; `phase1_finalists.csv`
retains the three-seed confirmation. Group metrics are attached to screening
outputs only after both locks. Test comparisons include matched U0 and the old
group-aware full-validation ensemble as distinct references. Latency excludes
CNN/heads; three seeds share one backbone. The supplied bare backbone has no
embedded training-history proof, which is disclosed rather than fabricated.

## Resume and handoff

`HANDOFF.md` and `progress.json` in the output directory update before each fit.
Features checkpoint each batch; heads checkpoint each completed member and RNG
state; neural gates checkpoint each epoch, optimizer and RNG. Every completed
candidate has an immutable model and result. Tables are rewritten atomically.
An interrupted current head is retried from its start (at most 20 epochs).
Completed fits and test evaluations are reused. Config/source changes are
rejected for an existing output directory; use a new output directory for a new
research recipe. Source snapshots and environment versions accompany the run.
Keep the complete output directory for handoff, not just the final checkpoint.

Validation:

```powershell
.venv/Scripts/python.exe -m pytest tests/methods/test_dpe_weighting.py -q
```

Synthetic tests are implementation checks, never benchmark measurements.

Completed-run findings are in `results/dpe/weighting_20261114/README.md`.
The supplementary A18 diagnostic and final post-hoc tables can be resumed with
`python -m experiments.weighting_ablation_crossfit` and regenerated with
`python -m scripts.finalize_dpe_weighting`, respectively. These never select a
new winner from test scores. Keep the original DPE checkpoint: the measured
winner did not improve its mean accuracy and incurred substantial WGA loss.
