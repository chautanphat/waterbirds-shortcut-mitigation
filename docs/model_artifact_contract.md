# Model artifacts and inference contract — version 1

This contract defines what must be retained to reconstruct a **frozen predictor** and export its predictions. It applies to new POTER, original DPE, AutoBackSwap, ERM and future method artifacts. It standardizes their description and acceptance boundary; checkpoint formats, architectures and method-specific selection procedures remain native.

**MUST/MUST NOT** denote requirements; **SHOULD** denotes the preferred behavior with exceptions documented; **MAY** denotes optional information. This is a new contract for future records, not a claim that historical artifacts already conform. The method contributor maintains the record; a result reviewer checks acceptance. Changes to this contract must keep the [experiment-record](experiment_record_contract.md) and [prediction](prediction_protocol.md) contracts consistent.

## Location and identity

Retain the lightweight, tracked predictor description at:

```text
experiments/<experiment-id>/model.json
```

Link it from `run.json` in the same directory. Model weights, feature arrays and training archives remain outside Git at their existing documented locations. This contract does not require moving or converting them. A reference MUST identify a retrievable location and SHA-256 of the exact bytes; a basename or mutable “latest” link alone is insufficient. Machine-specific paths MAY be accompanied by a portable archive location and member path. Never include credentials or signed access tokens.

One experiment ID identifies one seed/recipe/frozen predictor, as in the prediction contract. A frozen predictor includes weights **and** every prediction-affecting setting: preprocessing, feature transformations, class mapping, heads, aggregation, temperature and decision rule. Different checkpoint or inference settings require a distinct experiment ID; identical weights alone do not establish predictor equivalence. Shared backbones MAY be referenced by several predictors without copying their bytes.

## Predictor description interface

`model.json` MUST be UTF-8 JSON with an object root and finite numbers. New v1 records use the fields below. Additional method-specific fields MAY extend them. References may point to retained evidence rather than duplicate long configuration files, but they MUST resolve to the actual executed settings and immutable identities.

| Field | Type and required content |
| --- | --- |
| `model_contract_version` | Integer `1`. Independent of prediction and run-record versions. |
| `experiment_id` | String matching the experiment directory and `run.json`. |
| `method` | String distinguishing original implementation, recipe and project adaptation. |
| `seed` | Nonnegative integer agreeing with the run and prediction records; additional stage seeds belong in `run.json`. |
| `availability` | `complete` or `incomplete`. Complete means all reconstruction inputs are retained and locatable, not that numerical equivalence has been verified. |
| `implementation` | Object identifying source repository/revision, architecture constructor/entry point and any local patch evidence. A branch name alone is insufficient. |
| `artifacts` | Array of artifact objects: unique `id`, `role`, `location`, lowercase 64-hex `sha256`, serialization `format`, and `loader` entry point/version. Include all required backbone, classifier, heads, learned scales, configuration and fitted preprocessing parameters. |
| `input` | Object describing dataset/reference metadata identity, input kind (`images` or `features`), input shape/dtype, and ordered preprocessing/feature transformations. |
| `inference` | Object describing native class order, canonical class mapping, evaluation mode, numerical precision, logits/probability conversion, aggregation and final decision/tie rule. |
| `selection` | Object describing the method's checkpoint/head-selection procedure, data used, selected epoch/step or head set, metric/direction/tie rule where applicable, and retained selection evidence. Explicitly describe fixed-final-step methods. |
| `verification` | Object with `status` (`not_checked`, `passed`, or `failed`), verification evidence and known tolerances/limitations. |
| `limitations` | Array of strings. Empty only when no known limitation exists; missing historical facts MUST be named explicitly. |

For an incomplete historical record, unavailable values MAY be JSON `null`, with a field-specific explanation in `limitations`. Missing weights mean `availability: "incomplete"`; do not invent a digest or silently use another checkpoint. Recovered predictions may remain useful under the prediction contract, but do not establish reconstructable inference. Unknown is distinct from a setting that is deliberately disabled.

Artifact IDs MUST let the loader distinguish each component and its dependencies. An ensemble MUST specify the complete ordered head membership and any learned weights/scales; “15 heads” without their identities is insufficient. A checkpoint containing several components MAY be listed once if the loader and component keys are explicit. A multi-file serialization MUST identify every required file, or an immutable archive plus exact required members.

## Input and preprocessing requirements

The description MUST preserve the executed pipeline, including:

- Image decoding, RGB/channel order, resize shape versus short-edge resize, interpolation/antialias behavior, crop order/size, tensor range and normalization constants.
- Architecture and feature layer, pooling/flattening, expected feature dimension, and whether features are raw, L2-normalized or standardized. State the axes, order, formula and epsilon/correction for transformations; identify fitted statistics and the data used to fit them.
- Training-time augmentation separately from evaluation preprocessing. If inference intentionally uses stochastic transforms or test-time augmentation, record its seed, repeats and aggregation.
- Dataset release, metadata path/hash and class mapping. Feature-based inference MUST identify the producing backbone/preprocessing and preserve image IDs; array length alone does not establish alignment.

Method/library defaults MAY be referenced through pinned source and configuration, but the relevant defaults must be recoverable. A generic label such as “ImageNet transforms” is insufficient. Do not change a method's input representation merely to match another method. In particular, the shared ERM classifier trained on raw embeddings must receive raw embeddings; original DPE's recorded feature standardization belongs to its own inference pipeline.

## Loading and inference requirements

1. Verify required artifact hashes and constructor/configuration compatibility before evaluating. Resolve the exact selected artifact, not a filename guessed from another run.
2. Load the complete intended state. Missing/unexpected keys, shape mismatches or unsupported formats MUST fail acceptance. Explicit method-specific key conversion is permitted when recorded and verified; silently dropping parameters is not.
3. Do not download or initialize replacement pretrained weights to compensate for missing state. If reconstruction genuinely depends on external weights, identify, retain and verify them as artifacts.
4. Use native evaluation mode and disable gradient tracking for ordinary inference. Record deliberate exceptions, including adaptation or batch-dependent inference, with the method's data-use procedure. Export MUST NOT introduce additional fitting or use test labels for selection.
5. Preserve probability semantics, class mapping, temperatures, head membership/weights and tie-breaking. Original DPE uses an equal mean of each head's temperature-1 softmax probabilities; its training entropic scale is not an inference temperature. AutoBackSwap and other methods retain their own rules.
6. Associate each prediction with its image identity. Apply the canonical class/path/group mapping only at the export boundary, then use the [prediction contract](prediction_protocol.md).

Prefer state-dictionary or other data-only loading supported by the native implementation. Legacy full-object/pickle checkpoints require an explicitly trusted source and loader opt-in; loading can execute code. Record that requirement rather than silently enabling it in an adapter.

## Selection and lifecycle

Selection MUST preserve the method's documented validation use, including any group annotations, partitions, per-head selection and early stopping. Record the selected position's indexing convention and why that artifact was selected. `last` and `best` are not interchangeable. Test performance MUST NOT select among checkpoints, temperatures, heads or ensemble variants.

Save required heads and fitted components before their training scope exits. A log of ensemble accuracy or backbone feature files cannot replace discarded prototype heads. Backbone-only checkpoints MUST NOT be described as full DPE predictors. Save resume state separately when needed: optimizer/scheduler/RNG state can support resuming, but is not normally required for frozen inference. Conversely, a resume checkpoint is usable for inference only when the documented loader reconstructs the selected predictor from it.

Publish a completed description only after artifacts are retained and hashed. Completed predictor descriptions MUST NOT silently change to point at different bytes. Corrected descriptions that only clarify evidence may retain the ID with an auditable revision; changing the actual predictor requires a new ID and a supersession link in the experiment record. Existing canonical prediction artifacts remain immutable.

## Acceptance and verification

A reviewer MUST distinguish these checks:

| Check | Required evidence |
| --- | --- |
| Reconstruction inputs complete | All required artifacts/configurations are retained, locatable and hash-matched; input/class/selection definitions are resolved. |
| Loader works | Strict reconstruction in the recorded compatible environment; no silent parameter replacement. |
| Inference equivalence | Native inference and reloaded/export-adapter inference on the same identified, nonempty fixture; preprocessing/features, labels and available probabilities compared. |
| Export conformance | Complete reference-split CSV checked using the prediction validator, plus provenance and experiment links reviewed. |

For equivalence, retain the fixture identities, source revision, command/environment, native and reloaded artifact hashes, label agreement, maximum probability difference and the **predeclared** tolerance. Use validation or a separate non-test fixture where possible. Numerical tolerance depends on precision/device and MUST be justified; do not increase it after failure without reporting the discrepancy. Any label disagreement must be reported, including ties. An image-based predictor's check SHOULD cover decoding through the final output; a head-only feature check MUST disclose that image extraction was not checked.

`verification.status: "passed"` requires retained successful equivalence evidence for the declared scope. Availability alone, a successful `load_state_dict`, a synthetic unit test or an aggregate metric match is insufficient. A failed check MUST remain visible until resolved; it does not authorize selecting another predictor using test outcomes. Record partial checks as `not_checked` for full equivalence and explain their scope.

## Adoption and enforcement

For the next meaningful run: retain native weights/components, write `model.json`, link it from `run.json`, verify reconstruction and inference, then export predictions. Historical records MAY be annotated as incomplete without retraining or rewriting their measurements.

This is a documentation and review contract. Existing export adapters perform some loading/input checks and emit prediction provenance; they do **not** currently read or validate `model.json`. The prediction CSV validator does not certify model reconstruction, hashes, selection or inference equivalence. Reviewers must check those separately. Incompatible changes to this interface require a new model-contract version and documented migration; do not reinterpret v1 records silently.
