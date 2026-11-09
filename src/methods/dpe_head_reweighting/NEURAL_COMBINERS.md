# Neural combiners over frozen DPE heads

**Status:** implemented project adaptation; synthetic software validation only.
Waterbirds experiments are pending. The existing H2 result records and global
weighting suite are unchanged. Work began on 2026-11-09.

## Question and attribution

Can a learned decision boundary or input-dependent head weights improve
Waterbirds worst-group accuracy over uniform averaging of the **same 15 heads**?
This follows the negative global-weighting results in [H2](reports/h2.md).

The main methodological reference is Pineda Arango et al.,
[*Regularized Neural Ensemblers*, AutoML 2025](https://proceedings.mlr.press/v293/arango25a.html).
It motivates neural combination of model predictions and prediction dropout.
This code is independently written and is **not an implementation or reproduction
of their complete paper**. Our binary architectures, CPU optimizer, small grid,
group-balanced loss, neutral replacement and H2 B/C selection protocol are
project choices. We do not claim the paper's diversity guarantees for them.
In particular, our tiny binary MLPs replace the paper's class-shared stacking
network and Deep Set averaging architecture; neutral masking without inverted
scaling also differs from its dropout algorithm.

## Predeclared first comparison

The configuration is [neural_combiners.yaml](../../../configs/dpe_head_reweighting/neural_combiners.yaml).
There are **11 candidates**, including uniform:

| Procedure | Architecture | Prediction dropout | Weight decay |
| --- | --- | --- | --- |
| Uniform | Mean of 15 probability vectors | None | None |
| Logistic stacking | 15 → 1, sigmoid output | 0 | 0.01, 0.001 |
| MLP stacking | 15 → 8 tanh units → 1 sigmoid | 0 | 0.01, 0.001 |
| MLP with dropout | Same MLP | 0.25 | 0.01, 0.001 |
| Softmax gate | 15 → 8 tanh units → 15 softmax weights | 0 | 0.01, 0.001 |
| Gate with dropout | Same gate | 0.25 | 0.01, 0.001 |

Inputs are each head's waterbird probability, transformed to `2*p-1`. No
backbone features, group labels or targets are inputs at inference. The gate
returns a nonnegative, sum-to-one weighted average of original probabilities.
Stackers return a new probability and can learn negative coefficients or
nonlinear interactions; they do not have convex head weights.

Every candidate trains for **400 full-batch CPU float64 AdamW steps**, LR 0.01,
with the declared weight decay. Loss is the mean BCE over each of the four B
groups, then averaged equally across groups. This is **group-balanced BCE**, not
the previous joint fitter's maximum-group NLL and not direct WGA optimization.
Probabilities are clamped to `[1e-12, 1-1e-12]` only for the training BCE.
Model initialization and prediction dropout use source training seed + 10000;
candidate hyperparameters do not trigger seed searches. Output layers initialize
to zero, so the gate starts uniform and stackers start at probability 0.5.

Prediction dropout samples one keep/drop decision per example and head, removing
the whole head prediction. At least one randomly chosen head is retained if all
would be dropped. Dropped stacker/gate inputs become neutral probability 0.5
(centered input zero), with no inverted scaling. A gate also excludes dropped
heads from its softmax and renormalizes the retained weights. At inference every
head is available and dropout is disabled. Both class probabilities come from
the same retained binary prediction, so dropout cannot break normalization.

## Fit, select, evaluate

- Keep the original backbone, heads, head order and A/B/C partition fixed.
- Fit only on B; training code accepts no C/test data. No C-based early stopping.
- Use C WGA, then C overall accuracy, to freeze one choice per family and one
  overall choice. Uniform is eligible in every family and wins a complete tie;
  otherwise declared order wins, preferring larger weight decay.
- No B+C refit. Every planned seed's choices are saved before any test cache is
  read by the study runner. Evaluate uniform and only frozen family/overall
  choices. Do not choose a family from its test result.
- Report all seeds and procedures, including uniform fallbacks and failures to
  improve. Sample SD across seeds is not a confidence interval.

The B/C diagnostic includes a label-aware head oracle: how many uniform errors
have at least one correct head? It measures possible room for convex gating;
the oracle is not a deployable predictor. B/C group NLL accompanies WGA to expose
the previous loss/accuracy trade-off. Test outputs include group accuracy and
counts, paired deltas, corrected/new error IDs and gate concentration. Effective
head counts are computed **per example**, then summarized; stackers do not receive
an invented effective-head count.

## Run the next experiment

Open [dpe_neural_combiners.ipynb](../../../notebooks/dpe_neural_combiners.ipynb)
in a compatible checkout. For a fresh hosted clone, the implementation branch
must first be published; until then set `REPO_PATH` to a checkout containing this
change. A CPU runtime suffices; there is no backbone/head retraining.

Attach the previous combined H2 study ZIP **including `source_archives/`**,
individual original H2 ZIPs, or extracted runs. A summary-only ZIP or the tracked
JSON/CSV results cannot supply the per-head predictions. `SOURCE` accepts a path
or list of paths; automatic discovery searches the notebook's input roots.
Inputs are read only. The runner extracts nested source ZIPs into separate
managed storage and checks the original B/C selection against the caches.

Default `SEEDS=[0,1,2]` requires all three sources; use `[0]` explicitly for a
pilot. `EVALUATE_TEST=False` fits/freezes and exports all C comparisons without
reading test predictions. Set it to `True` and rerun for final frozen-choice
evaluation. This project has already inspected the common test set, so these
follow-ups remain **exploratory**, including later training seeds.

The equivalent single-seed CLI, from the repository root:

```bash
python scripts/dpe_neural_combiners.py select \
  --source-run outputs/h2_sources/h2-standard-seed0 \
  --output-dir outputs/dpe_neural_runs/standard-seed0-v1
python scripts/dpe_neural_combiners.py evaluate \
  --source-run outputs/h2_sources/h2-standard-seed0 \
  --output-dir outputs/dpe_neural_runs/standard-seed0-v1
```

Use `--config` for a separately declared configuration. Dependencies are existing
Torch, NumPy, SciPy and PyYAML; no new model-training library is needed.

For a multi-seed run or prior combined ZIP:

```python
from src.methods.dpe_head_reweighting.neural_study import run_study

run = run_study("outputs/neural_session", source="outputs/previous_h2_study.zip",
                seeds=[0, 1, 2], evaluate_test=False)
# After reviewing C selections, keep the declared recipe fixed for evaluation:
run = run_study("outputs/neural_session", source="outputs/previous_h2_study.zip",
                seeds=[0, 1, 2], evaluate_test=True)
```

The study never downloads data or trains missing sources. Missing B/C/test
predictions require the original H2 bundle or the existing H2 recovery workflow.
An incomplete source with B/C can still freeze choices; missing test caches are
reported after freezing, without generating them in this runner.

## Outputs and verification

Each comparison writes `selection.json` with exact data-only model states,
training histories, hyperparameters, B/C diagnostics and input/code/software
identities; `candidate_comparison.csv` includes every C candidate. Evaluation
adds `metrics.json`, `run.json` and `report.md`. Only those selected configurations
are scored on test. Saved selections are replayed deterministically before reuse;
edited states, selections, configs, inputs and final metrics are rejected. This
guarantees consistency within the recorded CPU software, not cross-version or
cross-hardware bitwise equality.

The notebook/API adds `plan.json`, `progress.json`, per-seed comparison directories,
and a ZIP updated after every frozen/evaluated seed. Completed evaluations add
`summary.json`, `per_seed.csv` and an aggregate `report.md`. The ZIP retains the
new combiner states and reports; **retain source prediction bundles separately**.

Run the functional checks with:

```bash
python -m pytest tests/methods/dpe_head_reweighting/test_neural_combiners.py -q
```

The checks use synthetic complementary-expert tasks, reversed signals, B/C
isolation, source/checkpoint tampering, dropout normalization, deterministic
replay, multi-seed source ZIPs and notebook execution. They establish software
behavior only. Record actual Waterbirds runs in `docs/experiment_log.md` and keep
their reports beside this method or in the experiment/result directories.
