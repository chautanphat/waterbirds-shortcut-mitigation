# Inputs required for independent CCW confirmation

B inputs are complete: nine verified240-image matching-head B caches already exported. No B/model regeneration is needed.

No verified untouched cohort is currently available. Provide `experiments/ccw_v6/fresh_inputs/CCW_FRESH_COHORT.json` describing the following assets:

- Explicit attestation of no prior researcher/model-selection exposure, source/cohort identity, annotation policy and collection provenance; a new filename or reused1199 validation images is insufficient.
- Nine fresh prediction caches identified by seed0/1/2 and actual outer-head ensemble fold0/1/2, with probabilities/distances/sample IDs/logits if available and the exact ordered15 head hashes recorded by the existing B models. Inference only, no retraining required.
- Same frozen ERM* backbone SHA256 `bce28dc72c3737ae19ffa057ba056e03d3f531180e7ad55e0b3c93bf4ea4bcaa`; inference/normalization/prototype-distance provenance, source and file hashes.
- A routing/assignment manifest fixed before labels: disjoint evaluation folds within seed, the same unique cohort across seeds, no reuse of original A/B/C, backbone-training or previously inspected test images.
- Separately sealed class/subgroup annotations keyed by unique sample_id, classes[0,1], group=2*y+background, with nonempty support for every evaluated subgroup and label-source audit. Do not place labels into scoring/selection APIs.

Continuation: preserve current B lock/source and all completed results. First validate and freeze new cohort file/ID/head hashes and input lineage before opening its annotations. Run the locked B models once, commit predictions, then evaluate the frozen fresh gate versus U0 and G1. Never use old C to rescue a NO_GO B decision. Adding a file does not automatically attest freshness or override the existing failure/lock.
