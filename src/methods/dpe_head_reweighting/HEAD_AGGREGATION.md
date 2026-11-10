# Independent head-aggregation comparison

## Question

Which global weighting rules or neural combiners improve Waterbirds worst-group
accuracy over uniform averaging of the same 15 frozen prototype heads?
All configurations are separate approaches; evaluate and report every one.

## Data and training roles

| Data | Images | Purpose |
| --- | ---: | --- |
| Official training set | 4,795 | Train a ResNet-50 backbone. |
| Head-training split | 720 | Fit prototype heads and choose head epochs. |
| Combiner-training split | 479 | Fit weighting rules and neural combiners. |
| Official test set | 5,794 | Evaluate frozen configurations. |

Divide the 1,199 validation images directly into head-training (60%) and
combiner-training (40%) subsets within each bird/background group. Allocation
uses largest remainders, fixed split seed 0, sorted image IDs and a seeded
permutation. The two subsets are disjoint and cover validation exactly; the
same partition is used for seeds 0/1/2. There is no candidate-selection subset.
Group labels inform training objectives and reporting, not inference inputs.

## Fixed comparison

| Approach | Configurations |
| --- | ---: |
| Uniform averaging | 1 |
| Power weights: α=0.25/0.5/0.75/1, γ=1/2/4 | 12 |
| Softmax weights: β=10/30 | 2 |
| Top-k averaging: k=10/5 | 2 |
| Joint weights: λ=1/0.1/0.01 | 3 |
| Logistic, MLP, MLP dropout, gate, gate dropout; two weight decays each | 10 |

Power, softmax and top-k use each head's WGA on the combiner-training split.
Joint fitting minimizes worst-group NLL plus squared distance to uniform using
a checked convex solver. A solver failure stops the run; it is not replaced by
a better-scoring configuration. The power formula's defined uniform fallback
for degenerate scores is recorded under that rule's own ID.

The backbone uses ImageNet initialization and 100 ERM epochs; the final epoch
is retained. Heads use frozen raw backbone features, row standardization,
20 training epochs and group-balanced subsamples of 320 head-training images.
Head-epoch selection reuses only head-training examples. Settings are declared
in [head_training.yaml](../../../configs/dpe_head_reweighting/head_training.yaml).

Neural inputs are 15 waterbird probabilities transformed to `2*p-1`.
Logistic stacking uses 15→1 sigmoid; MLP stacking uses 15→8 tanh→1 sigmoid;
the gate uses 15→8 tanh→15 softmax and mixes original probabilities.
Training uses 400 full-batch CPU float64 AdamW steps, LR 0.01, weight decay
0.01/0.001 and equal weighting of the four group BCE means. Each decay is a
separate configuration. Initialization/dropout use training seed + 10000.
Zero output initialization starts gates uniform and stackers at probability 0.5.

Prediction dropout removes each whole head prediction with probability 0.25,
retains at least one head, and replaces dropped centered inputs with zero.
There is no inverted scaling. Gates exclude dropped heads from softmax and
renormalize available weights. Dropout is disabled during inference.

Freeze all configurations across every planned seed before test inference or
scoring. Report accuracy, group accuracy/counts, WGA and paired changes against
uniform for every configuration. There is no test-based hyperparameter tuning,
early stopping or winner replacement. Training metrics are not held-out results.
Mean ± sample SD across seeds describes training variation, not a confidence interval.

## Run

Open [dpe_head_aggregation.ipynb](../../../notebooks/dpe_head_aggregation.ipynb),
select a GPU runtime, enable Internet on Kaggle and Run All. The notebook clones
the published implementation, installs missing dependencies, downloads official
Waterbirds and trains missing sources. No attachments or code edits are needed.
Default settings run seeds 0/1/2 and test evaluation. `EVALUATE_TEST=False` runs
training only. Preserve WORK_DIR for epoch-boundary resume and artifact reuse.

The configuration is [aggregation_comparison.yaml](../../../configs/dpe_head_reweighting/aggregation_comparison.yaml);
[source preparation](../../../configs/dpe_head_reweighting/aggregation_study.yaml)
controls download, workers and backbone resume. `SOURCE` optionally accepts a
current-format source bundle or combined study ZIP. `MODE="cached"` requires
sources for every requested seed. Sources and input ZIPs are read only.

```python
from src.methods.dpe_head_reweighting.aggregation_study import run_study
run = run_study("outputs/aggregation_session", evaluate_test=True)
```

For an existing source, from the repository root:

```bash
python scripts/dpe_head_aggregation.py fit --source-run outputs/head_source --output-dir outputs/comparison
python scripts/dpe_head_aggregation.py evaluate --source-run outputs/head_source --output-dir outputs/comparison
```

## Artifacts and reproducibility

Each source contains a two-split `experiment/manifest.json`, frozen
`checkpoints/ensemble.pt`, a verified `experiment/heads.json`, and
`experiment/predictions/combiner_training.npz`. Evaluation adds `test.npz`.
There is no intermediate candidate-selection stage or selection record.

Each comparison writes `models.json` (exact neural states, global weights,
training histories, diagnostics, input identities and configuration),
`candidate_comparison.csv` (training metrics), and, after evaluation,
`metrics.json`, `run.json` and `report.md`. Model records are data-only JSON;
frozen head checkpoints load with PyTorch's weights-only loader.
Deterministic replay checks states and identities before reuse. Edited inputs,
models or results are rejected. Cross-version bitwise equality is not claimed.

Study outputs add `plan.json`, `progress.json`, `summary.json`, `per_seed.csv`
and a ZIP updated after every seed. `source_archives/` retains reusable source
bundles. Images, backbone/resume checkpoints and raw feature arrays remain in
WORK_DIR outside the compact download. A source missing test predictions needs
the matching verified raw-feature handoff; preserve WORK_DIR or a complete source.

## Attribution and limits

Neural combination and prediction dropout are motivated by Pineda Arango et al.,
[*Regularized Neural Ensemblers*, AutoML 2025](https://proceedings.mlr.press/v293/arango25a.html).
This is independently written project code, not a reproduction of its complete
architecture, dropout algorithm or theoretical guarantees. The binary models,
group-balanced loss and neutral masking are project choices.

The current protocol needs a real Waterbirds run. Software checks use synthetic
data and a tiny backbone substitute, and do not establish benchmark performance.
The comparison remains exploratory because the shared test set has previously
informed project research. Renaming data roles does not reset that history.

Run `python -m pytest tests/methods/dpe_head_reweighting -q` for software checks.
Historical experiments and their protocols are documented separately under
`reports/`, `archive/` and `notebooks/archive/`.
