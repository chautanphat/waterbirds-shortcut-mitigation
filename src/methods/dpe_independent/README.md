# Independent DPE experiment suite

Project adaptation on frozen Waterbirds ResNet-50 features, based on repository
revision `22b65ec5fc8f00b33938fe2af6dc32ad2f34ecfd`. This is an implementation of
the agreed prototype-competition/aggregation study, not a literature reproduction.
The preserved IsoMax+/DPE implementation and attribution remain under
`src/methods/dpe_ermstar` and `baselines/dpe4subpop`. Those implementations are
unchanged. There is no backbone-training entry point in this suite.

The [completed seven-source study](../../../results/dpe/independent_methods_20261113/README.md)
retains all 5,880 declared rows, with no failures, and an independent CSV audit.
It ran in the attached managed Codex cloud workspace on CPU. The
[execution and artifact guide](../../../results/dpe/independent_methods_20261113/execution.md)
explains result counts, aliases, runtime, input recovery and artifact availability.
The fresh Kaggle notebook is also provided.

## Protocol

Every fixed configuration is evaluated independently. There is no candidate
selection, best-temperature selection, best-initialization selection, early
stopping, or test-driven exclusion. Configuration is in
`configs/dpe_independent/study.yaml`. Defaults declare:

* 85 geometry configurations per full-validation ensemble;
* 240 configurations per 720/479 allocation, at partition seeds 0, 1 and 2;
* 35 crossfit configurations per source using five stratified folds.

Across five normal-ERM and two ERM* backbones this is 5,880 rows, including explicit
algebraic aliases. Aliases share predictions but retain separate rows and exports.
Normal ERM and ERM* summaries remain separate. Partitions/folds are not additional
independent backbone sources.

H3 uses balanced cyclic subsets of original prototype pairs, including one fixed
seed-0 permutation. Native inference scores are negative normalized Euclidean
distances times each saved absolute scale. Shared scores use the median absolute
scale. Temperatures 1, .1 and .05 are fixed alternatives, not selected settings.
Local top-k selects within each class per image and pools with log-mean-exp.

Prototype weights normalize separately within each class and use mean class-wise
KL shrinkage toward uniform. Temperature and bias controls are separate fits;
their penalties are squared log-temperature displacement from 1 and squared bias.
Temperature bounds are [.01,10] and bias bounds [-5,5]. Temperature-fitting controls
start at 1; fixed-temperature bias controls cover every declared temperature.

Residual models initialize at uniform predictions. Their regularizer is
`(a-1)^2 + b^2` plus the mean squared correction parameters, where present.
New differentiable fits use 400 full-batch CPU float64 Adam steps at .01 with
explicit penalties .01/.1. Maximum-group objectives take the maximum of the four
group means. Existing neural anchors preserve AdamW, width 8, .01/.001 decays and
.25 prediction dropout. Existing maximum-group NLL weights are a baseline;
average/maximum-group Brier fits are new and require simplex feasibility plus a
supporting-hyperplane optimality gap at most 1e-7.

Each allocation freezes all fitted records before opening test feature arrays.
Full-validation heads only receive geometry methods. Crossfit fits a shared affine
calibrator on 1,199 out-of-fold probabilities, applies it separately to each fold
ensemble at test time, then averages. Prototype-specific weights are not transferred
between independently trained fold ensembles. New heads preserve the source recipe;
normal-ERM diversity decays by stage, while ERM* uses its preserved constant penalty.

## Recover, inspect, and run

Run from the repository root with these dependencies available:

```bash
python -m src.methods.dpe_independent recover --output /path/to/recovered
python -m src.methods.dpe_independent inventory --roots /path/to/recovered --output /path/to/sources.json
python -m src.methods.dpe_independent grid --output /path/to/declared-grid.json
python -m src.methods.dpe_independent launch --sources /path/to/sources.json --output /path/to/study --devices cuda:0 cuda:1
python -m src.methods.dpe_independent report --output /path/to/study
```

The public saved runs are `noerula/dpe-exploration` and
`noerula/dpe-mlp-dropout-verification`. Recovery retrieves raw features, head states,
manifests, and existing prediction/model evidence, excluding images and backbone
training checkpoints. No credentials are required for these public outputs.
Inventory verifies normal-ERM feature/head bindings and control-head identities,
and rejects ambiguous conflicting artifacts. Saved ERM* feature markers lack a
backbone digest; that historical limitation is retained in each record.

If a feature cache is unavailable but the exact backbone and images are retained,
`extract --checkpoint ... --sha256 ... --data ... --output ...` regenerates raw
features without training. Its SHA-256 check precedes model construction. Retain
the extraction identity and explicitly bind regenerated caches in the source
manifest; they are not silently substituted for historical cache bytes.

CPU execution is supported using `--devices cpu` (or `cpu cpu` for two processes).
Device/library differences from historical GPU runs are recorded. CPU scoring and
fitting use the same rules. Row z-scoring preserves float32; normalized Euclidean
distances accumulate in float64, avoiding observed CPU float32 cdist drift. Original
head probabilities are rounded to float32 to preserve the historical input format.
Original probability replay tolerance remains 2e-7 and
independent frozen-model NumPy replay tolerance 1e-10; changed baseline decisions
are always rejected. Synthetic tests are not Waterbirds results.

The eight-hour limit saves completed predictor/fold outputs. Rerun the exact command
with unchanged code, inputs and configuration to resume. A partial normal-ERM fold
head ensemble restarts deterministically; completed folds are reused. ERM* also
saves per-head RNG state. Such restarts do not count as new independent runs.
Failed fits stay visible and are not silently retried with different settings.

## Outputs and verification

Each predictor retains `run.json`, `model.json`, `metrics.json`, its fitted JSON,
and `experiments/<id>/predictions/test/{predictions.csv,provenance.json}` under its
worker directory. Distances are regenerated from raw features, never inferred from
binary probabilities. Input identities, cache hashes and frozen-fit hashes reject
changed caches or states. An implementation ZIP retains the exact executed code.

The root report contains all configuration rows, recipe/allocation/source-level
summaries, source counts, and explicit failures or blockers. No test winner is
selected. Feature-based availability does not imply image-level replay, background
causality, formal confirmation, or superiority over the paper's published results.

```bash
python -m pytest tests/methods/dpe_independent -q
```

For Kaggle, build a source ZIP with `bundle.build`, attach it as
`dpe-independent-code/dpe-independent-code.zip`, and run the fresh notebook
`notebooks/dpe_independent_methods_kaggle.ipynb`. It uses regular imports and all
default configurations. No GitHub push is needed.
