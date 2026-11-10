# Experiment records contract — version 1

Every new meaningful experiment MUST retain enough evidence to identify what ran, how a predictor was selected, and which claims its artifacts support. This contract formalizes existing configuration, `run.json` and experiment-log practice. It complements the [model artifact/inference](model_artifact_contract.md) and [prediction](prediction_protocol.md) contracts; it does not standardize method internals or replace native logs.

**MUST/MUST NOT** denote requirements; **SHOULD** denotes preferred behavior with documented exceptions; **MAY** denotes optional information. The experiment author maintains the record and supplies evidence; a result reviewer checks it before accepting a research claim. Contract maintainers keep cross-document requirements consistent.

## Locations and responsibilities

```text
configs/<method>/...                         # Reusable settings, where applicable
experiments/<experiment-id>/run.json           # Tracked execution/evidence record
experiments/<experiment-id>/model.json         # Tracked frozen-predictor description
experiments/<experiment-id>/predictions/<split>/
    predictions.csv                          # Generated, ignored by Git
    provenance.json                          # Generated, ignored by Git
docs/experiment_log.md                        # Concise index and interpretation
results/...                                  # Lightweight summaries, where needed
```

Existing method-specific configuration locations remain valid. `run.json` MUST reference the actual resolved configuration, including command/notebook overrides, rather than only a reusable default. Retain the exact executed configuration or an immutable source plus a complete override record; hash the retained evidence. Repository paths SHOULD be relative to the repository root; external artifact references MUST include a usable location, identity/hash and retrieval context without secrets. Retain large weights, datasets, caches and archives outside Git. Their digests and retrieval references belong in the tracked record.

An experiment ID MUST be stable and unique, and map to one method/recipe/main seed/frozen predictor. Different seeds or predictor settings use distinct IDs. A suite or grouped experiment-log entry MAY link several child IDs; it MUST NOT combine their predictions into one canonical CSV. Existing historical grouped IDs remain historical and must not be silently relabeled as v1 single-predictor records.

## Required run record interface

New records use UTF-8 JSON objects with finite numbers, these root fields and their stated types. Additional fields MAY preserve method-specific evidence. A required object may contain references to retained structured evidence rather than duplicate it. No single training-configuration schema is imposed.

| Field | Type and content |
| --- | --- |
| `record_contract_version` | Integer `1`; independent of model and prediction versions. |
| `experiment_id` | String matching its directory, model description and index mapping. |
| `method` | String identifying method/recipe and reproduction versus adaptation. |
| `seed` | Nonnegative integer matching model/prediction records. |
| `purpose` | String describing the question and intended claim: exploratory, original reproduction, Waterbirds adaptation or final comparison. |
| `execution` | Object with `status`, timestamps/timezone or explicit missing-date explanation, invocation, effective overrides, duration when available, and attempt history. |
| `source` | Object identifying repository/revision, dirty state and retained local-change evidence when dirty; include upstream revision where relevant. |
| `configuration` | Object identifying resolved settings, their location/hash and any differences from tracked defaults. Include training/model/method hyperparameters or a complete referenced configuration. |
| `dataset` | Object identifying release/source, metadata location/hash, official/reference split membership and counts, label/background/group mappings, and any subsets/partitions with identities and seeds. |
| `randomness` | Object recording main and additional stage/data-partition seeds, deterministic settings and known nondeterminism. |
| `environment` | Object recording Python, relevant library/CUDA versions, operating system, hardware/accelerator, hosting environment and precision; include dependency lock/environment evidence where available. |
| `selection` | Object recording validation/group-annotation use, candidate space, metric/direction/tie-breaking, selection outcome and evidence. Link the corresponding model description when a predictor exists. |
| `artifacts` | Object mapping model description, logs, configurations, metrics, canonical predictions/provenance, archives and source inputs to their locations and hashes where applicable. Record availability, not merely filenames. |
| `results` | Object with split-specific metric references, units, sample/group counts, aggregation/uncertainty procedure where applicable, and known verification scope. |
| `assessment` | Object with reproduction status, claim scope, limitations, deviations and missing evidence. Execution success and reproduction status MUST remain separate. |
| `relationships` | Object identifying parent/source experiments, reused components, superseded records and grouped-study membership where applicable; use an empty object when none apply. |

A planned, running, failed or incomplete historical record MAY use `null` for facts not yet known, but MUST explain each missing field in `assessment`. Empty objects or placeholder strings do not satisfy completed-run evidence requirements. A failed run need not fabricate model/prediction files: explicitly record that they were not produced. Never infer execution dates from file modification times or documentation dates without labeling that inference.

## Execution, attempts and change history

`execution.status` MUST be one of `planned`, `running`, `completed`, `failed` or `interrupted`. These describe execution only. A completed exploratory run is not automatically a verified reproduction; a final comparison is a claim scope, not an execution status.

- Before running, record the intended configuration, source, seed, data use and selection rule. Fill outcomes and hashes from actual execution afterward.
- Record notebook path/revision, relevant cells or callable entry points and effective variable overrides. A Jupyter kernel-launch command alone is not the experiment invocation.
- Record the executed commit. If local changes were used, retain the patch/source snapshot and its hash; a later clean commit must not be substituted as the executed revision. Preserve execution-time configuration hashes even when a portable configuration is later rewritten for documentation.
- Retain each failed/interrupted attempt's status, invocation, evidence and failure reason. Do not replace failure evidence with a successful rerun. An attempt history under one ID MAY describe retries/resumption of the same intended run, but it MUST unambiguously identify which attempt produced the accepted predictor and output hashes.
- Independent successful reruns or any change to seed, recipe, selected predictor or inference settings require new IDs. A true resume MUST identify its parent checkpoint, saved optimizer/scheduler/RNG state as applicable, continuation settings and limitations. A restart from scratch is a retry, not a resume.
- Records may be completed as evidence arrives; preserve corrections and reasons through tracked revisions. Actual completed prediction artifacts MUST NOT be overwritten. Corrected exports use a new experiment ID and a supersession relationship as required by the prediction contract.

Synthetic unit tests and routine implementation checks do not need experiment-log entries. A meaningful dataset experiment, including a failed experiment, does. Keep diagnostic checks separate from measured research claims.

## Data use and selection evidence

Record the actual role of each split/partition: backbone fitting, head fitting, calibration/weight fitting, model selection and final evaluation. Include group-annotation use, partition identities/counts and split seeds. Preserve each method's native validation procedure; do not silently impose another method's rule.

The candidate space, selected metric, direction, tie rule and fallback behavior MUST be traceable. Record selected epoch/step numbering and all required ensemble components. Test data MUST remain excluded from training, tuning and model selection. If test inspection motivates another method or configuration, label the subsequent work exploratory and disclose the dependency; do not describe it as a fresh untouched-test comparison.

Reused backbones, generated training images, heads or feature caches MUST link to their source experiment/artifact hashes and explain their role. Unknown provenance must remain unknown. A recovered prediction CSV MUST identify its original source commit/file/hash and recovery command, and state that inference/training was not rerun.

## Results, review and claims

Results MUST distinguish train, validation and test, and full reference sets versus subsets. Link full predictions and provenance when available. Report overall accuracy, each group accuracy, worst-group accuracy and group counts using the canonical mapping. State units explicitly: canonical prediction metrics are fractions; displayed percentages and percentage-point differences must be labeled. Missing measurements are `null`/unavailable, never zero.

For multi-seed summaries, retain each seed's record, exact seed set, aggregation formula, number of runs and uncertainty definition (for example sample standard deviation versus standard error). Disclose failures, exclusions and missing seeds; do not select seeds by their test scores. Do not compare validation scores with paper test scores as evidence of reproduction.

`assessment` MUST distinguish exploratory, smoke-test, partial-reproduction, verified-reproduction and final-reporting claims using the project's [research protocol](research_protocol.md). Record what was actually checked: aggregate arithmetic, prediction identity/metrics, model reloading, or raw-image inference. A CSV validator pass, successful training, or matching aggregate score alone does not establish verified reproduction. Describe deviations and missing artifacts without rewriting historical measurements.

Before accepting a new completed result, the author/reviewer MUST check:

1. ID, main seed and recipe agree across `run.json`, `model.json`, prediction provenance/CSV and the experiment index.
2. Executed source, resolved configuration, environment, invocation, data membership and selection evidence are retained or explicitly unavailable.
3. Required model artifacts and inference verification satisfy the model contract for any reconstruction/equivalence claim.
4. Prediction CSVs satisfy reference coverage and content validation; paths, sidecar hashes and provenance receive the separate review required by the prediction contract.
5. Reported metrics can be recomputed from the retained evidence and have correct split/units/counts; missing groups and unknowns are visible.
6. Status, deviations, failed attempts, reuse and review limitations support the stated claim. Missing evidence prevents the corresponding claim, not documentation of the run itself.

## Experiment index template

Use the [experiment log](experiment_log.md) as a concise index; keep detailed evidence in the run record. For new entries, include:

```markdown
## <experiment-id> — <execution date or explicitly labeled documentation date>

- **Method / purpose:**
- **Execution / reproduction status:**
- **Seed / source revision:**
- **Configuration / invocation:**
- **Dataset / split / selection rule:**
- **Run record:** experiments/<experiment-id>/run.json
- **Frozen predictor:** experiments/<experiment-id>/model.json, or why unavailable
- **Predictions / results:** paths, split, accuracy/WGA/group counts and units
- **Verification / limitations / follow-up:**
```

## Adoption, compatibility and enforcement

Apply v1 to the next meaningful run; do not retrofit every historical record or invent missing facts. Legacy `run.json` layouts remain valid historical evidence. If mapping one into this contract, retain the original, identify the mapping and all missing information, and only set `record_contract_version: 1` when the new record follows this interface. Model descriptions may be incomplete if truthfully labeled.

This is a documentation and review contract; there is currently no shared `run.json` or `model.json` schema validator or automatic record generator. The prediction validator checks CSV/reference content only. Contract version fields MUST NOT be interpreted as proof that evidence was reviewed. Incompatible interface changes require a new version and migration guidance; compatible clarifications must preserve existing meanings.
