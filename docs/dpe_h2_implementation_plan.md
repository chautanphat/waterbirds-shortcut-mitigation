# DPE H2 implementation plan

**Status:** Proposed experiment; implementation and experiment runs are pending.  
**Scope:** Test H2, “Reweighting the heads,” from [the research report](../report/main.tex).  
**Maintenance:** Keep this plan current with the H2 implementation; record executions in `docs/experiment_log.md`.

H2 asks whether validation-derived weights improve DPE over uniform probability averaging, and whether low-weight heads perform poorly on background-conflicting groups. Start with the standard ERM track, then evaluate the enhanced ERM* track separately. Both aggregation rules must use exactly the same frozen backbone, heads, features, and example order within each comparison.

This is a project-owned adaptation. Follow [CONTRIBUTING.md](../CONTRIBUTING.md), [the research protocol](research_protocol.md), and [the dataset rules](dataset.md). The filenames, weight family, and validation partition below are experiment-specific proposals, not new project-wide conventions.

## 1. Current implementation and prerequisites

The implementation should use the executed [end-to-end notebook](../notebooks/dpe_waterbirds_end_to_end.ipynb) as its behavioral reference:

- It trains 15 heads, with two 2048-dimensional prototypes and a learned distance scale per head. It averages each head's softmax probabilities.
- `train_ensemble()` keeps `prototype_ensemble` locally and returns aggregate metrics, not the trained heads. Recover saved heads from previous runs if available; otherwise add export to the project-owned runner and retrain Stage 1. Summary metrics cannot reconstruct head predictions.
- Stage 1 samples its training examples from the official validation split, using group annotations by default. Head epoch selection also uses validation WGA. Consequently, the full validation set is not unseen calibration data for those existing heads.
- The notebook uses 15 heads and group-annotated validation training. The Stage-1 YAML instead specifies `num_stages: 16` and `train_attr: 'no'`. Resolve these differences explicitly in a new H2 configuration rather than treating that YAML as the executed notebook configuration.
- Features are standardized per example using their feature-wise mean and standard deviation before IsoMax+ inference. IsoMax+ then L2-normalizes features and prototypes and applies the absolute learned distance scale. Preserve this sequence, default inference temperature, and class order. The training entropic scale of 30 is not an extra inference multiplier.
- The packaged `DPE` API exposes `ensemble` and per-head predictions via `predict_proba(avg=False)`, but its dataset and training interfaces differ from the notebook. Do not substitute it without checking parity.

Before coding the experiment, inventory external checkpoints and feature caches, identify the standard/enhanced track and seed for each, and verify their provenance. Stage-0 training should only be repeated when matching artifacts cannot be recovered. The repository checkout currently contains neither a local `data/` nor `checkpoints/` directory.

## 2. Repository placement

Create these files only as their implementation begins. Preserve the source under `baselines/dpe4subpop/` and the existing reproduction notebook.

| Location | Responsibility |
| --- | --- |
| `src/methods/dpe_head_reweighting/README.md` | Mechanism, setup, data roles, commands, outputs, limitations, and differences from upstream. |
| `src/methods/dpe_head_reweighting/ORIGINAL_SOURCE.md` | Upstream URL and commit, MIT notices, notebook reference, and provenance of adapted code. |
| `src/methods/dpe_head_reweighting/ensemble.py` | Load/export frozen heads and generate per-head probabilities with reference-compatible inference. |
| `src/methods/dpe_head_reweighting/train_heads.py` | Feature-only Stage-1 runner adapted from the notebook, restricted to the declared head-training subset. |
| `src/methods/dpe_head_reweighting/reweighting.py` | Fit validation scores, construct weights, and combine probabilities. |
| `src/methods/dpe_head_reweighting/experiment.py` | Partition IDs, select a configuration, lock artifacts, evaluate, and produce diagnostics. |
| `scripts/dpe_head_reweighting.py` | Thin project-level CLI delegating to the method implementation. |
| `configs/dpe_head_reweighting/waterbirds_standard.yaml` | Primary-track settings, artifact inputs, validation partition, seeds, and weight grid. |
| `configs/dpe_head_reweighting/waterbirds_enhanced.yaml` | Secondary-track settings with explicit Stage-0 augmentation provenance. |
| `tests/methods/dpe_head_reweighting/` | Inference parity, weight/metric correctness, partition isolation, and artifact round-trip tests. |
| `experiments/exp-dpe-h2-waterbirds-<track>-seed<seed>/` | Lightweight resolved configuration, split IDs, provenance, and selection record. |
| `results/dpe_head_reweighting/` | Concise metric tables, per-head diagnostics, plots, and interpretation. |

Keep binary heads under ignored `checkpoints/` and feature/prediction caches under ignored `outputs/`. Track checksums and paths in experiment metadata, not large artifacts. Keep helpers method-local until another method actually needs them. Use the baseline's documented dependencies plus a YAML reader if required; do not create a shared requirements file just for H2.

## 3. Data roles and comparison protocol

Use official Waterbirds train/validation/test splits: 4,795 / 1,199 / 5,794 examples. Derive groups from metadata, with this explicit mapping:

| Key `(y, a)` | Report label | Meaning |
| --- | --- | --- |
| `00` | G1 | Landbird on land |
| `01` | G2 | Landbird on water |
| `10` | G3 | Waterbird on land |
| `11` | G4 | Waterbird on water |

### Pilot using recovered heads

First verify uniform-aggregation parity on recovered heads, if available, and inspect per-head validation scores. Any weights estimated from validation examples already used to train or select those heads are an **exploratory reuse pilot**. Label the overlap explicitly; such a pilot cannot establish generalization to unseen calibration data. Do not select an approach using test predictions.

### Main experiment

Partition the official validation set by `(y, a)` using a fixed split seed of 0 and stable original example IDs:

| Subset | Approximate share | Permitted use |
| --- | --- | --- |
| A: head training | 60% | Balanced Stage-1 subsampling and head epoch selection, matching the notebook's reuse within this subset. |
| B: weight fitting | 20% | Estimate each frozen head's group accuracies and derive candidate weights. |
| C: aggregation selection | 20% | Choose the aggregation configuration after all heads and candidate weight vectors are frozen. |

Stratify within each group, assign integer remainders deterministically, and save actual counts and IDs. All four groups must be present in each subset. Keep the A/B/C partition identical across training seeds and across tracks. Never pass B or C to head fitting, subset resampling, epoch selection, or feature/backbone tuning.

Stage 0 continues to use the official training split. Audit its checkpoint-selection provenance; it must not have been selected using B, C, or test labels. Use the documented final epoch where applicable. Stage 1 must be retrained on A for this main experiment, even if full-validation heads were recovered. Both uniform and weighted inference then use those same A-trained heads. This changes the Stage-1 data budget relative to the reported reproduction: compare H2 against its matched uniform control, not directly against the historical 89.00% or 92.70% WGA.

The existing reuse of A for head fitting and epoch selection remains documented. B and C provide separation for the added weighting procedure. Do not refit heads on A+B+C or recompute weights on B+C after selection in this experiment.

Test data remains held out until configuration selection is complete. Remove the reference notebook's per-stage test evaluation from the project-owned runner. Make final evaluation a separate CLI action consuming a saved selection record; changing selection inputs invalidates that record.

## 4. Weighting rule and selection

For each head `h`, compute its probabilities `p_h(y | x)` with the original IsoMax+ inference. Fit its score on B:

```text
s_h = min over g in {00, 01, 10, 11} accuracy(head h, group g, subset B)
q_h = s_h ** gamma / sum_j(s_j ** gamma)
w_h = (1 - alpha) / H + alpha * q_h
p_weighted(y | x) = sum_h w_h * p_h(y | x)
```

Use accuracies as fractions. Require finite scores and probabilities, nonnegative weights, and weights summing to one. If all scores are zero, fall back to uniform weights and record the event. A missing group is an input error, not a zero-accuracy group.

Evaluate this fixed, small grid on C:

- Uniform baseline: `alpha = 0`.
- Weighted candidates: `alpha in {0.25, 0.5, 0.75, 1.0}` and `gamma in {1, 2, 4}`.

This gives 13 distinct configurations. `alpha` shrinks weights toward uniform; `gamma` controls how strongly strong heads dominate. Select maximum C WGA, then maximum C overall accuracy, then smaller `alpha`, then smaller `gamma` (use `gamma = 1` for the uniform record). Apply a fixed numerical tie tolerance of `1e-12`.

Save the selected hyperparameters, exact weight vector, head order, input hashes, configuration hash, and selection metrics before opening test labels. Average probabilities; averaging logits or voting would test a different aggregation rule. H1 prototype re-pairing, H3 joint prototype probabilities, retraining with learned weights, and temperature tuning are outside this experiment.

## 5. Configuration and artifact contracts

The new YAML is the source of truth and must be read by the CLI. Record:

- Track, source commit, method version, and official dataset/metadata checksum.
- Stage-0 checkpoint and feature-cache paths/hashes, seed, model, pretrained weights, transforms, feature normalization, and extraction ordering.
- Stage-1 head count `15`, epochs `20`, batch size `256`, SGD learning rate `0.001`, momentum `0.9`, entropic scale `30`, `cov_reg = 500000`, and prototype penalty `wd_weight = 10`.
- The notebook's covariance schedule, group-balanced validation subsampling (`train_attr = yes`), per-stage resampling, and head epoch-selection rule (including its last-epoch-on-tie behavior).
- Split seed `0`, A/B/C ratios and roles, head/backbone seeds `[0, 1, 2]`, weighting grid, selection metric, and tie rules.
- Runtime device, dependency versions, output paths, and classification tie behavior (class-index argmax, matching reference).

Use a typed validation step to reject unknown keys and incompatible settings. Check that the group-balanced A subset yields at least one training batch when `drop_last=True`; do not silently run zero optimizer steps or change batch size.

Save a versioned ensemble artifact containing `prototypes[H, 2, 2048]`, `distance_scales[H]`, inference temperatures, head IDs, selected epoch per head, class/group mapping, preprocessing identity, and source provenance. Export detached CPU tensors and verify loading reproduces probabilities. Reject unexpected shapes, class orders, nonfinite values, zero-variance feature rows, or mismatched provenance.

Prediction caches should contain `probabilities[H, N, 2]`, stable original example IDs, split role, labels/groups for permitted subsets, and ensemble/preprocessing hashes. Verify exact ID alignment with metadata rather than relying on array length. Distinguish per-row standardization from IsoMax+ L2 normalization and apply each exactly once. Process a final batch of one without squeezing away its batch dimension.

## 6. Implementation sequence and completion gates

1. **Verify artifacts and inference.** Recover matching artifacts where possible. Build the method-local ensemble adapter and reproduce the reference head logits and uniform probabilities on a fixed feature fixture, with `atol=1e-6`, `rtol=1e-5` on CPU. Resolve any mismatch before fitting weights; record backend differences for GPU checks.
2. **Make Stage 1 reproducible.** Adapt the notebook's feature-only loop into the project method with explicit A IDs, epoch selection on A, deterministic seeds, export of all 15 heads, and no test loader in training. Document every deviation. Test a short synthetic run before the full head training.
3. **Fit and select weights.** Implement B-only scoring, the 13 candidate configurations, C-only selection, uniform fallback, and the persisted selection record. Ensure the selection command runs without any test feature or label file.
4. **Run the standard seed-0 experiment.** Train/recover the valid Stage-0 artifacts, train heads on A, fit on B, select on C, and verify the complete trace. Freeze the experiment specification before final test evaluation.
5. **Complete the seed matrix.** Run standard seeds 1 and 2 and enhanced seeds 0, 1, and 2 with the same predeclared rules. Each seed should have a matching Stage-0 seed and separately trained heads; if one backbone is reused across seeds, label results as head-seed variability instead of independent full-pipeline seeds. Report standard and enhanced tracks separately.
6. **Evaluate and document.** Evaluate the matched uniform control and selected aggregation on the same held-out test predictions for each seed, then produce diagnostics and update the relevant documentation. If uniform wins selection, retain it and report that outcome.

### Proposed commands

These are interfaces to implement; they do not exist yet. Run from the repository root. Each action uses the same resolved configuration and accepts a seed override recorded in metadata.

```bash
python scripts/dpe_head_reweighting.py prepare --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
python scripts/dpe_head_reweighting.py train-heads --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
python scripts/dpe_head_reweighting.py fit-select --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
python scripts/dpe_head_reweighting.py evaluate --config configs/dpe_head_reweighting/waterbirds_standard.yaml --seed 0
```

`prepare` verifies externally supplied Stage-0 artifacts and writes the split manifest; it does not implicitly download data or launch backbone training. `evaluate` requires the matching saved selection record and writes both control and selected-method results. Add a thin Colab notebook only if needed for the run; experiment logic remains in Python modules.

## 7. Required validation

- Reference parity for individual logits, head probabilities, uniform averaging, and group metrics; include learned scales and the singleton final batch.
- Hand-calculated probability fixtures verifying the weight formula, shrinkage, deterministic selection ties, all-zero fallback, and accuracy/WGA calculations.
- Failures for missing groups, nonfinite scores/features/probabilities, invalid weights, reordered/missing/duplicate example IDs, incompatible artifacts, and an empty training loader.
- Pairwise disjoint A/B/C/test IDs and full validation partition coverage; verify head fitting and epoch selection receive only A, weight fitting only B, and selection only C. A changed test fixture must not alter selected weights.
- Save/load probability parity and protection against using a selection record with changed heads, split IDs, or configuration.
- A short CPU smoke run on synthetic grouped features, followed by the real standard seed-0 pipeline. A smoke pass is not an experimental result.

Do not rerun the historical reproduction or broaden the baseline implementation solely to accommodate H2.

## 8. Results and interpretation

For every seed and track, report uniform and selected-method overall accuracy, per-group accuracy, WGA, group counts, and paired changes in percentage points. Report the selected `alpha`, `gamma`, weight vector, and effective head count `1 / sum(w_h ** 2)`. Aggregate each track using mean and sample standard deviation over seeds, including paired WGA changes. Never pool standard and enhanced seeds into one result.

For the hypothesis's mechanism, save per-head B and C group accuracies, assigned weight, and minority-group accuracy `min(acc_G2, acc_G3)`. Plot weight against minority-group accuracy and identify low-weight heads. The relationship on B is partly built into a WGA-based weighting rule; check its persistence on C and, after selection, descriptively on test. Such an association alone does not establish that a head uses background features. Record fixed errors, new errors, and per-group changes for the weighted ensemble after final evaluation.

The primary outcome is the held-out standard-track WGA change relative to the matched uniform control; report enhanced-track results as a separate secondary outcome. A positive mean paired change with consistent seed-level improvements is evidence to investigate further, not proof from three seeds. Report regressions and selection of uniform just as clearly. Do not expand the grid or change the partition after inspecting test outcomes; a revised approach needs a separately recorded experiment.

When runs exist, add the adaptation entry to `docs/methods.md`, execution records to `docs/experiment_log.md`, and concise summaries under `results/dpe_head_reweighting/`. Record the exact command, resolved configuration, implementation commit, seeds, hardware/software, artifact hashes/locations, data counts, validation decisions, and status. Preserve the original DPE reproduction entry and its historical results.

## 9. Compute and handoff

Feature-only scoring and head reweighting can run on CPU; existing notebook estimates place Stage-1 training at roughly 1–2 minutes on a T4, which must be measured again for this runner. Missing Stage-0 artifacts dominate cost: the notebook estimates roughly 1.8–2 hours per standard backbone run. Inventory artifacts before scheduling up to six track/seed combinations; enhanced runtime depends on its recorded training recipe.

The implementation is ready for review when it reproduces uniform inference, enforces the declared data roles, exports reloadable heads and weights, passes the focused checks, and provides one complete standard seed-0 experiment record. Scientific reporting remains preliminary until the planned seed matrix is complete. This planning change does not start training or claim H2 improves accuracy.
