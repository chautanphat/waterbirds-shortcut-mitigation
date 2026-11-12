# ERM* + DPE with neural aggregation

Prepared Kaggle experiment, **not a new benchmark result**. Use
[dpe_ermstar_neural_two_gpu_kaggle.ipynb](../../../notebooks/dpe_ermstar_neural_two_gpu_kaggle.ipynb):
use the `experiment/dpe-ermstar-neural` branch selected by `REPO_BRANCH`, enable Internet,
select GPU T4 ×2, and Run All. Defaults are two source seeds, 0 and 1,
in independent GPU processes. Adding seed 2 queues a third source; it does not make
two GPUs cooperate on one model or change batch sizes. Results remain exploratory.

## Configuration sources and reuse

The preserved upstream launchers are the scientific configuration reference:

- [train_all.sh](../../../baselines/dpe4subpop/scripts/train_all.sh): Waterbirds,
  ResNet-50, ImageNet initialization, **400 ERM epochs** with augmentation.
- [train_all_pe.sh](../../../baselines/dpe4subpop/scripts/train_all_pe.sh) and
  [train_pe.sh](../../../baselines/dpe4subpop/scripts/train_pe.sh): 15 heads
  (`range(1,16)`), 20 epochs, BS256, LR .001, covariance coefficient 5e5,
  entropic scale 30, prototype penalty 10, group annotations and group subsampling.
  CLI covariance is constant. The demo notebook's inverse schedule is not applied.
- [main.py defaults](../../../baselines/dpe4subpop/main.py): ERM BS128,
  SGD .001/momentum .9/optimizer WD0, no scheduler, evaluation BS512.
- [upstream model code](../../../baselines/dpe4subpop/utils/models.py) and
  [dataset code](../../../baselines/dpe4subpop/utils/datasets.py): ImageNet V2,
  random resized crop, photometric distortion .5, horizontal flip .5;
  deterministic square resize 256, center crop 224 and ImageNet normalization
  for validation/test; class-balanced replacement sampling for ERM.

[study.json](../../../configs/dpe_ermstar/study.json) reuses the existing neural
configuration and head settings. The existing shared ERM config is loaded with the
launcher-required 400 epochs and evaluation BS512 explicitly applied. This is
**not** the older project 100-epoch enhanced adaptation. No old 100-epoch backbone
is automatically accepted as a completed run.

The project shared ERM trainer/model/transforms/sampling are reused for the matching
recipe with atomic optimizer/RNG checkpoint recovery. This is a maintained
implementation of the recipe, not a claim of identical upstream RNG trajectories.
The upstream CLI trainer itself is not copied or modified to add resumability.
The shared trainer records readable configuration, dataset rows and source paths
for recovery. There is no file-checksum mode in the trainer used here.

For heads, the original `train_prototypes`, IsoMax+ class/loss and dataframe
`subsample` routines are copied without changing their math into
[upstream.py](upstream.py), with attribution to their preserved source files.
Normal Python imports load them; there is no runtime source extraction or execution
of generated code. A test compares these definitions with the preserved originals.
The replacement feature-only evaluation callback uses the worker's actual device
with safe batch shapes. A gradient hook rejects nonfinite training.
The small wrapper checkpoints after each completed head and restores NumPy/Torch RNG.
Selected epochs are not returned by the upstream function: exported `selected_epochs`
uses **−1 = not exported**, never a guessed epoch. The selected prototype tensors
are exactly those returned by the upstream function.

## Two arms per backbone

| Arm | Head pool / selection pool | Aggregation fitting | Reported procedures |
| --- | --- | --- | --- |
| Paper-configuration reference | All 1199 validation examples; group-balanced subsets | None | Original uniform |
| Neural adaptation | Fixed group-stratified 720-example pool | Separate 479-example pool | Matched uniform + ten neural configurations |

The upstream launcher used here is the **annotated-validation** protocol. Do not
label this an unannotated reproduction. The two arms use the same frozen 400-epoch
backbone, but independently trained head ensembles. Comparing a neural result to
full-validation DPE changes allocation and optimization budget as well as aggregation:
full-validation balanced subsets have 532 examples and two drop-last updates/epoch;
split-head subsets have 320 and one update/epoch. Report the full reference **and**
the neural arm's matched uniform. An observed gap between arms is not a causal
estimate of the aggregation effect.

Neural code is reused unchanged from `dpe_head_reweighting/neural_combiner.py`:
400 full-batch CPU float64 AdamW steps, LR .01, tanh hidden width8, group-balanced
BCE, weight decays .01/.001, prediction dropout .25. Logistic, MLP, dropout MLP,
gate and dropout gate are reported independently, with their matching uniform.
No extra candidate-selection split and no test-selected procedure. Both arms and
all neural fits freeze before test scoring for that source. Backbone validation
metrics do not select its handoff checkpoint: the final epoch is used.

## Two-GPU execution and recovery

The parent prepares Waterbirds once and downloads ImageNet weights once, before
starting workers. Each subprocess receives one `CUDA_VISIBLE_DEVICES` value and
uses local `cuda:0`. No model, optimizer or RNG is shared across seeds; outputs and
logs are `jobs/seedN/` and `jobs/seedN.log`. CPU fitting remains independent.
The defaults use one CPU thread and one loader worker per seed to limit contention.
The paper's batch size stays128 on each GPU; no mixed precision or DDP is introduced.

400 epochs can approach Kaggle's session limit, especially with CPU/data contention.
The notebook has a 10-hour parent budget and records interruption state before
terminating worker process groups. GPU availability does not guarantee completion
in a single session. Retain the full Kaggle output. Next session set `RESUME_FROM`
to its full `jobs/` directory, not a compact results ZIP. Restore keeps existing local files, copies only missing files through temporary
paths, and checks free disk. Use a fresh jobs directory to restore a different run.

Backbone recovery uses the shared trainer's optimizer/RNG state at the last complete
epoch. Head recovery replays at most the interrupted head's 20 cheap epochs. Full
head checkpoints are reused, while deterministic small neural fits can be replayed.
Changes to the saved configuration, pool IDs, feature shape or stored raw head
features reject head-checkpoint reuse. No file checksums or generated directory IDs
are used. Feature-cache recovery checks expected files and configuration, not byte
identity; regenerating derived features is necessary if replacing a backbone at the
same path. Preserve the notebook and checkout with each run. Each finished seed publishes its own compact
ZIP with heads, fit cache/partition, neural model states, test head probabilities,
metrics and fit-complete marker. Raw features and large backbone training states
stay in the full job directory. One worker failing does not delete another seed's
successful output; read logs and `completion.json` for failure or timeout.

## Validation and limits

Run from repository root:

```bash
python scripts/run_dpe_ermstar.py --work /path/to/output --seeds 0 1
python -m unittest discover -s tests/methods/dpe_ermstar -v
```

Tests exercise preserved head training on synthetic features, exact head-boundary
resume after an injected failure, input mismatch/empty-batch rejection, full two-arm
worker fitting/evaluation with CNN training mocked, one-GPU-per-process environments,
failed-worker output retention, GPU/seed preflight, and the actual notebook code
cells in a fresh process with synthetic workers/data. The notebook uses an ordinary repository checkout and normal imports. Implementation
lives in `src/methods/dpe_ermstar/`, the experiment configuration in
`configs/dpe_ermstar/study.json`, and the command-line entry point in
`scripts/run_dpe_ermstar.py`. The notebook only checks out the repository, sets paths
and seeds, calls `prepare_and_launch`, and displays results. It contains no module
strings, file-writing cells, encoded bundles or checksums. There is no notebook
builder; the notebook itself is the maintained launcher.

These changes must be published to the selected branch before Kaggle can use them.
If the clone lacks the entry point, the notebook stops before any data download or
training and explains the missing repository change. Keep the checkout/notebook
with each run for code provenance. Kaggle supplies torch/torchvision, NumPy, SciPy, pandas, PyYAML and tqdm.

No 400-epoch models were trained in this cloud. Actual two-T4 throughput, peak GPU
memory, live dataset/weight downloads and Kaggle-session behavior remain to be
measured. Numerical paper agreement is pending; correct configuration and passing
logic checks alone do not establish a verified reproduction or neural superiority.

## File layout

```text
configs/dpe_ermstar/study.json              # Scientific settings
src/methods/dpe_ermstar/upstream.py         # Attributed original routines
src/methods/dpe_ermstar/run.py              # Experiment and GPU workers
scripts/run_dpe_ermstar.py                 # CLI entry point
notebooks/dpe_ermstar_neural_two_gpu_kaggle.ipynb  # Short Kaggle launcher
```
