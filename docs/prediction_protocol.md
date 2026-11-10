# Shared prediction CSV protocol — version 1

This is the canonical `predictions.csv` format for **all methods from now on**: POTER, original DPE, AutoBackSwap, ERM, project adaptations, and future baselines. It standardizes exported evaluation artifacts. Each method retains its own implementation, preprocessing, fitting procedure, validation use, and checkpoint-selection rule.

The shared constants and validator are in [`src/evaluation/prediction_protocol.py`](../src/evaluation/prediction_protocol.py). Export commands and method-specific artifact requirements are in [Export predictions](export_predictions.md).

The [model artifacts and inference contract](model_artifact_contract.md) defines
the frozen predictor's `model.json`; the [experiment records contract](experiment_record_contract.md)
defines the corresponding `run.json`. Both live at `experiments/<experiment-id>/`
and supply linked evidence for prediction provenance. Their contract versions are
independent of this CSV schema. CSV validation alone does not validate those records.

## Status, terminology and responsibilities

This document is the normative contract for new Waterbirds prediction artifacts. **MUST** and **MUST NOT** are acceptance requirements; **SHOULD** is the preferred behavior with any exception documented; **MAY** is optional. Examples illustrate the contract and do not establish benchmark results.

- **Experiment:** a recorded method/recipe/seed and one frozen predictor, identified by `<experiment-id>`.
- **Frozen predictor:** the checkpoint or ensemble together with its preprocessing, class mapping, inference rule, temperature/thresholds and other prediction-affecting settings.
- **Reference split:** an explicit set of image identities, ground-truth labels and backgrounds defined by retained metadata.
- **Artifact:** the pair `predictions.csv` and `provenance.json` for one experiment and reference split.
- **Consumer:** evaluation/reporting code, a comparison script, or the [Waterbirds Prediction Viewer](viewer.md).

| Party | Contract responsibility |
| --- | --- |
| Method adapter | Preserve native inference; map classes/paths/groups explicitly; align predictions to image identities; supply truthful provenance. |
| Shared writer | Serialize v1 CSV and companion provenance; check CSV content; refuse an existing destination. It cannot establish training history or independently prove complete reference coverage. |
| Contributor/report author | Verify reference coverage, artifact location, provenance, predictor identity and experiment-log links before accepting a result. |
| Consumer | Join by stable identity, preserve unknown/missing values, check compatibility before comparison, and distinguish partial/legacy artifacts from accepted full-split exports. |
| Contract maintainer | Keep constants, writer, validator, examples and compatibility documentation consistent; document changes and migrations. |

## Artifact boundary and location

One file represents **one run, one seed, one reference evaluation split, and one frozen predictor**. Export every evaluated example, including correct predictions, exactly once. Do not mix seeds, checkpoints, ensemble configurations, or train/validation/test splits in the same file.

The required location for all new method exports is:

```text
experiments/<experiment-id>/predictions/<split>/predictions.csv
experiments/<experiment-id>/predictions/<split>/provenance.json
```

`<experiment-id>` identifies one method/recipe/seed/frozen predictor. `<split>` is `train`, `val`, or `test`. Use the same experiment ID as the configuration and experiment-log record. Distinct predictor variants need distinct experiment IDs. For example:

```text
experiments/
├── erm_standard_seed0/
│   ├── config.yaml
│   └── predictions/
│       ├── val/predictions.csv
│       ├── val/provenance.json
│       ├── test/predictions.csv
│       └── test/provenance.json
├── poter_seed1/
│   └── predictions/test/{predictions.csv,provenance.json}
└── dpe_seed0/
    └── predictions/test/{predictions.csv,provenance.json}
```

Prediction directories are generated artifacts ignored by Git; experiment configurations, lightweight run records and summaries remain tracked as appropriate. Dataset images stay in the dataset location, independently of predictions. The viewer is one consumer of these general evaluation artifacts; it does not determine their storage location. Historical exports may remain in their original locations, including the previously recovered `outputs/failure_cases/` bundle; new exports use the canonical experiment location above.

An exploratory subset must identify its reference subset and exploratory status in provenance. Its metrics must not be presented as full official-split results. Official-split comparisons use the complete common metadata split.

The following storage invariants apply:

1. The directory's `<split>` MUST equal every CSV row's `split` and the provenance `split`.
2. The experiment record MUST map `<experiment-id>` to the CSV's `run_label`, seed and frozen predictor. `run_label` need not equal the directory name; its mapping MUST be unambiguous and stable.
3. Different splits under one experiment MUST describe the same frozen predictor. A different checkpoint, head set, combiner or inference configuration requires a distinct experiment ID.
4. Only one producer may write a destination at a time. A completed artifact MUST NOT be overwritten in place. A corrected/repeated export uses a new experiment ID and records what it supersedes; retain the original evidence.
5. Prediction files MUST NOT contain or embed dataset images, model weights or credentials. Images and model artifacts remain at their separately documented locations.
6. A producer running in Colab/Kaggle MAY use another absolute checkout root, but MUST preserve the `experiments/<experiment-id>/predictions/<split>/` relative structure. Machine-specific absolute paths in local provenance are references, not portable image identities.

## Exact CSV schema

Write UTF-8 CSV, with a header and these **11 columns in this order**. A UTF-8 BOM and either LF or CRLF line endings are accepted. Quote fields using a standard CSV writer. Do not add an index column, aliases, or extra columns to a version-1 canonical file. Additional information belongs in the companion provenance file.

```csv
image_path,true_label,predicted_label,correct,group,group_name,spurious_attribute,confidence,run_label,seed,split
001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg,waterbird,landbird,0,4,Waterbird on Water,water,0.93,poter_seed1_test,1,test
```

The example prediction is illustrative, not a reported model result.

| Column | Required value |
| --- | --- |
| `image_path` | Stable relative image identity: `<species>/<image>.jpg`, using forward slashes. Same identity across every method. No absolute path, drive letter, URL, `.`/`..` segment, `images/` prefix, or dataset-directory prefix. |
| `true_label` | Exactly `landbird` or `waterbird`. Ground truth from reference metadata, never inferred from the background. |
| `predicted_label` | Exactly `landbird` or `waterbird`, from the frozen predictor. |
| `correct` | Exactly `1` when the two labels agree, otherwise `0`. |
| `group` | Exactly `1`, `2`, `3`, or `4`, derived from **ground truth** and true background. |
| `group_name` | Exact name from the mapping below, including capitalization. |
| `spurious_attribute` | Exactly `land` or `water`, from reference metadata. |
| `confidence` | Predicted-class probability as a finite number in `[0,1]`; **blank if unavailable**. The column is always present. Never substitute logits, a percentage, `NaN`, an arbitrary score, or made-up confidence. |
| `run_label` | Nonempty identifier distinguishing the method, recipe/variant and frozen predictor. Identical on every row. Different predictor configurations need different labels/directories. |
| `seed` | Nonnegative decimal integer, such as `0` or `1`; identical on every row. Record additional stage/split seeds in provenance. |
| `split` | Exactly `train`, `val`, or `test`; identical on every row and consistent with metadata. |

No cell may contain leading/trailing whitespace or control characters. Every cell except an unavailable `confidence` must have a value. Paths must be unique within a file. Row order should follow the reference metadata split; validators and comparisons use image identity, not row number.

The numerical target/background mapping is:

| Ground-truth target | True background | `y` | `a` | `group` | Exact `group_name` |
| --- | --- | --- | --- | --- | --- |
| landbird | land | 0 | 0 | 1 | Landbird on Land |
| landbird | water | 0 | 1 | 2 | Landbird on Water |
| waterbird | land | 1 | 0 | 3 | Waterbird on Land |
| waterbird | water | 1 | 1 | 4 | Waterbird on Water |

Thus **`group = 1 + 2*y + a`**. Internal zero-based groups, string groups such as `01`, numeric labels, or method-specific split names must be converted at the export boundary. Preserve the method's internal representation.

### Identity and completeness invariants

For a CSV with image-identity set `P` and reference-split set `R`, acceptance requires **`P = R`**, with exactly one row for every member. Matching row counts alone is insufficient. Ground truth, background and split MUST also agree for each identity. Predictions MUST NOT be matched by a guessed row order, numeric batch index, basename alone, or truncated path.

For cross-method comparisons, consumers MUST verify the same dataset release/reference membership and label/background/split mapping. Metadata files with different byte hashes may represent the same records after a documented format conversion; that equivalence requires explicit identity/label/split verification. Matching path strings alone does not prove dataset equivalence.

Ordering SHOULD follow reference metadata. The shared probability writer accepts identified input arrays and reorders them explicitly. If no IDs are supplied, alignment is the adapter's responsibility; this mode MUST NOT be used for an array of uncertain order.

## Prediction and confidence semantics

- Use the predictor fixed by the method's documented procedure. Do not choose a checkpoint, head count, temperature, ensemble weighting, or threshold using test failures.
- For binary class probabilities, array column 0 is landbird and column 1 is waterbird. The shared writer uses argmax for the label and its probability for confidence; equal probabilities select column 0. If a native implementation uses another documented tie rule, preserve and record it.
- For original DPE, first compute each head's temperature-1 softmax, then average probabilities equally, and take argmax/max of that average. The training entropic scale of 30 is not an inference multiplier. Do not average head confidences or logits in place of the original rule.
- For a different ensemble or method, preserve its documented inference rule and record it. Never silently replace it with the DPE rule.
- If only class labels are available, leave confidence blank and describe why in provenance. Do not invent a probability. The viewer can display these predictions; positive confidence thresholds exclude rows with missing confidence.
- Save all predictions. An optional `errors.csv` is a derived subset with `correct == 0`, not a substitute for the complete canonical file. A filtered CSV downloaded from the viewer is an inspection artifact, not automatically a new canonical export.

## Provenance and reporting

Keep `provenance.json` beside every newly exported run. Include or link the existing experiment records for:

- schema version (`prediction_schema_version: 1`);
- method, implementation/source revision, run label, main seed and other relevant stage seeds;
- dataset source/version, metadata file/hash, split and whether it is official/full or exploratory/subset;
- chosen checkpoint path/hash, or all ensemble/head artifacts plus selection record and hashes;
- configuration path/hash, export command, preprocessing and inference/aggregation rule;
- checkpoint-selection procedure and confidence semantics/limitations;
- export environment/hardware, file hash and overall/per-group accuracy, worst-group accuracy and group counts;
- experiment-log entry and relevant deviations or unavailable historical provenance.

The shared writer records basic input provenance supplied by the adapter, export environment, command, schema version, CSV hash and metrics. The adapter/contributor must retain any remaining training or selection records; an export cannot recreate missing historical evidence. Absence of a group makes four-group WGA unavailable. Successful export/validation does not establish verified reproduction.

### Companion JSON interface

`provenance.json` MUST be UTF-8 JSON with an object at the root, finite numeric values, and no secrets. New artifacts MUST include the root fields below with the specified types. Additional JSON fields MAY supply method-specific evidence without redefining these fields. The shared writer emits this envelope automatically:

| Field | Type and meaning |
| --- | --- |
| `format_version` | Integer `1`; version of this JSON envelope, distinct from the prediction schema version. |
| `prediction_schema_version` | Integer `1`; the CSV contract in this document. |
| `run_label`, `seed`, `split` | String, nonnegative integer, and split string; MUST agree with the CSV. |
| `group_convention` | String describing `1 + 2*y + background`. |
| `confidence` | String documenting predicted-class probability and unavailable values. |
| `metrics` | Object containing `count`, `errors`, `accuracy`, `worst_group_accuracy`, and `per_group`. Accuracies are fractions in `[0,1]`, not percentages. |
| `metrics.per_group` | Object keyed by strings `1`–`4`; entries contain `name`, `count`, and `accuracy`. Empty-group accuracy is JSON `null`; four-group WGA is `null` if any group is absent. |
| `provenance` | Extensible method-adapter object holding the input artifact identities/hashes and linked research evidence listed above. |
| `environment` | Object recording export `python`, `numpy` and `platform`; retain method-library, accelerator and training-environment evidence in `provenance` or linked run records. |
| `command` | Array of strings representing the export invocation. A notebook caller MUST additionally record its notebook/cell invocation or equivalent execution record. |
| `exporter_sha256` | Lowercase, 64-character SHA-256 hex digest of the export implementation source used (the shared writer when using the provided adapters). |
| `csv_sha256` | Lowercase, 64-character SHA-256 hex digest of the exact CSV bytes, including header, encoding and line endings. |

The adapter object intentionally has method-specific fields: a single checkpoint and a prototype ensemble do not have identical source artifacts. It MUST identify every artifact needed to reproduce the frozen predictor, their hashes, the reference metadata/hash, and the selection/inference procedure, either directly or through resolvable retained run records. References SHOULD be repository-relative paths, artifact URIs or immutable revisions where available. Mutable labels such as “latest” are insufficient identity by themselves.

Missing historical evidence MUST be listed explicitly as unavailable, with the result's limitations. An empty adapter object or an unexplained missing checkpoint/configuration is not sufficient for reproducibility acceptance even though the low-level writer can serialize it. Recovery of an existing CSV MUST identify the original CSV/source revision and disclose that it did not run model inference; it MUST NOT invent unavailable checkpoint or training hashes.

Metrics MUST be recomputable from the CSV. Let `c_i` be the integer `correct` value: overall accuracy is `sum(c_i)/N`; each group uses its own count; WGA is the minimum of all four available group accuracies. JSON `null` is used for unavailable quantities, never zero or `NaN`. Additional method-specific metrics MAY be stored separately without altering the CSV header.

## Export and validate

Existing POTER, original DPE, AutoBackSwap and ERM adapters use [`write_predictions`](../src/evaluation/prediction_export.py). A future method can use the same writer after producing its own predictions:

```python
from src.evaluation.prediction_export import read_metadata, write_predictions

rows = read_metadata("data/waterbirds/metadata_waterbirds.csv", split="test")
# probabilities: your already-frozen method's [N, 2] array, in class order [0, 1].
# image_ids: its N corresponding image paths; the writer aligns them to metadata.
write_predictions(
    "experiments/my_method_seed0/predictions/test",
    rows,
    probabilities,
    ids=image_ids,
    run_label="my_method_seed0_test",
    seed=0,
    provenance=run_provenance,
)
```

Here `probabilities`, `image_ids`, and `run_provenance` are inputs from the method's evaluation and experiment records; this snippet does not run inference. For label-only methods, pass `probabilities=None`, supply metadata-ordered `predicted_labels` (0/1), and omit confidence or supply explicit `None` entries. The shared writer refuses existing output directories and validates its generated CSV before publishing it to the destination.

### Producer API preconditions and guarantees

For `write_predictions(output, rows, probabilities, ...)`:

- `output` MUST be the new canonical split directory. The writer is path-agnostic for reuse/testing; it does not enforce experiment-directory naming. The caller owns that storage-contract check.
- `rows` MUST be a nonempty reference split with unique canonical `image_path` and binary integer `y`/`a`, plus one `split` value. Use `read_metadata` rather than constructing these from predictions.
- Probability mode requires a finite floating-point `[N,2]` array in class order `[landbird, waterbird]`, bounded in `[0,1]`, with rows summing to one within absolute tolerance `1e-6`. IDs, when supplied, MUST cover exactly the reference identities. Predictions are determined before confidence formatting; the CSV writer formats confidence to up to ten significant decimal digits.
- Label-only mode requires exactly `N` binary predictions in reference order. `ids` is not an alignment mechanism in this mode. Confidence is omitted entirely or supplied as an aligned sequence of probabilities/explicit `None` values. Nonfinite numbers are invalid, not a substitute for `None`.
- `run_label`, seed and `provenance` MUST describe the supplied frozen predictor. The writer does not infer them from an output directory or select a model.

On success, the writer returns the provenance envelope and exposes both files together by renaming a staged directory on the same filesystem. Consumers SHOULD ignore temporary directories and SHOULD read completed artifacts only after the producer finishes. An existing output raises `FileExistsError`; invalid inputs or CSV contract failures prevent publication. Storage/process failures may leave temporary staging material; its presence does not signify a completed artifact. Do not use staging names as experiment IDs or canonical inputs.

Before accepting or reporting an export, run:

```sh
python scripts/validate_predictions.py experiments/my_method_seed0/predictions/test/predictions.csv --metadata data/waterbirds/metadata_waterbirds.csv
```

To validate every canonical CSV under an output directory:

```sh
python scripts/validate_predictions.py experiments --metadata data/waterbirds/metadata_waterbirds.csv
```

The validator requires only Python's standard library. It checks the exact header, types, label/group consistency, correctness, uniqueness, and one run/seed/split. With `--metadata`, it also verifies complete image coverage and every ground-truth target/background against that split. Without `--metadata`, it checks structure/content only and reports `metadata_coverage_verified: false`; that is insufficient evidence of full-split coverage. It does not verify checkpoint history, the truth of supplied provenance, or whether the metadata is the official release.

### Validation and acceptance boundaries

| Level | Evidence | What it establishes |
| --- | --- | --- |
| CSV conformance | Validator succeeds without metadata | File structure and internal consistency only. |
| Reference conformance | Validator succeeds with retained reference metadata | CSV conformance plus exact reference membership and labels/backgrounds/split. |
| Artifact acceptance | Contributor checks canonical location, companion JSON, identity agreement, CSV hash, recomputed metrics and linked predictor/run evidence | The pair is suitable for traceable project use, subject to explicitly recorded limitations. |
| Research claim | Experiment records meet the research protocol and support the stated reproduction/reporting status | A justified research result; it is not implied by any file validator alone. |

`validate_predictions.py` prints one JSON result per discovered CSV. Exit code `0` means every inspected file passed the requested CSV/reference checks; exit code `1` means at least one failed; argparse usage/discovery errors use exit code `2`. The fields `valid` and `metadata_coverage_verified` refer only to the inspected CSV/reference checks. They do **not** certify companion provenance or storage layout. Directory discovery recursively finds files named `predictions.csv`; an explicitly supplied filename is validated as a CSV regardless of its basename.

Before accepting a new artifact, the contributor MUST:

1. Confirm the canonical destination, experiment identity, split-directory agreement and frozen predictor.
2. Verify full reference coverage with `--metadata`, retaining the command/result in the run record.
3. Check that companion schema version, run label, seed and split agree with the CSV, and that `csv_sha256` matches the exact stored bytes.
4. Recompute/check the overall and per-group metrics/counts; distinguish missing groups and exploratory subsets.
5. Check predictor/configuration/metadata hashes and links, inference semantics, selection procedure, missing evidence and experiment-log references.
6. Make the completed pair available to consumers without rewriting historical artifacts.

Consumers MUST NOT treat an absent row in another method's export as a correct prediction. They MUST NOT combine different predictors simply because their labels or directory names look similar. Missing confidence remains unknown. Displayed percentages MUST be converted explicitly from stored accuracy fractions. The viewer's compatibility import does not perform the artifact-acceptance checklist on the user's behalf.

## Adoption and compatibility

All new methods and future exports use this protocol. Include the export command, CSV/provenance paths, reference metadata and successful validation command in method documentation and experiment records. Preserve upstream implementations and historical outputs; use an external export adapter or a new output directory for conversions.

The viewer remains tolerant of legacy aliases, numeric labels/groups, missing split fields and errors-only files for historical inspection. That import tolerance does not change the canonical export contract. Version 1 retains the existing shared exporter's column order and one-based groups, so current complete exports remain readable. Future schema changes require an explicit version and migration documentation.

Changes to header order, field meaning, label/group mapping, null representation or the unit of an existing field MUST NOT be made silently under version 1. The CSV contains no embedded version column: its interpretation is tied to this contract and the companion `prediction_schema_version`. An incompatible change requires a new documented version, compatible reader/export tooling or an explicit conversion, and preserved source artifacts. Clarifications that do not change accepted data semantics may remain in version 1. Unknown future versions MUST NOT be assumed equivalent to v1 for research acceptance.

When maintaining this contract, run `python -m unittest discover -s tests -p "test_prediction*.py" -v` and `node --test tests/test_viewer.cjs`. These checks cover the shared writer, validator, legacy AutoBackSwap adapter, documentation example and viewer download template.
