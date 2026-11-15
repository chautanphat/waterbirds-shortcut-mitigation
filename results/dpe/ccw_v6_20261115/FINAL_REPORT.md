# CCW v6 — not independently confirmed

**B decision: NO_GO. Fresh confirmation: BLOCKED. CCW production integration: NOT PERFORMED.**

The fixed primary is CCW_FULL (rho.5, beta1, gamma50, alpha.25, kappa25); all five preregistered variants were evaluated. Only matching B data fitted/selects. Existing Uniform and opt-in G1 source files/default behavior were preserved.

## B-only selection and AA floor

| Rank | Method | Mean worst-class NLL | AA eligible every seed | Seed ΔAA versus U0 (pp) |
|---|---|---:|---|---|
| 1 | G1_frozen | 0.62063432 | True | [-0.32626427406199365, -0.1615508885298933, -0.1626016260162677] |
| 2 | CCW_FULL | 0.62518272 | True | [-0.32626427406199365, -0.1615508885298933, 0.0] |
| 3 | CCW_A | 0.62520816 | True | [-0.16313213703099683, -0.1615508885298933, 0.0] |
| 4 | G2_z | 0.63253482 | True | [0.0, 0.0, -0.1626016260162677] |
| 5 | U0 | 0.63601003 | True | [0.0, 0.0, 0.0] |

Primary FULL outranks G1: False. Locked new method: None. Primary loses to G1 / violates AA floor, or a fixed control ranks best; no rescue grid.

AA degradation exactly −1.0000 pp fails. Duplicate B image IDs across outer ensembles are averaged before seed AA and bootstrap metrics; class-only worst-class NLL averages nine outer B-OOF evaluations. No group WGA contributes to ranking.

## WGA versus G1 and Uniform

**REUSED DEVELOPMENT; NOT CONFIRMATORY; NO C-BASED TUNING.**

No new CCW C replay was authorized after the B-only NO_GO decision. C_EXPLORATORY_ONLY.csv contains only the already reproduced v5 U0/G1 baselines. CCW WGA versus U0/G1 is **not measured here**, rather than fabricated or tuned on a known C result. No full-B deployment table was refitted because no novel candidate was locked.

## Why the mechanism did or did not help

B estimated-advantage within-image ranking by seed: 0.6490634094019087, 0.634764212631741, 0.6596554394115369. Full class/margin/disagreement strata and uncertainty are in CCW_MECHANISM_AUDIT.md. Class-balanced boundary weighting and the t target do not directly estimate groups or guarantee WGA. A proxy win/loss can arise from small probability shifts without top-1 gains, or from partial geometric normalization changing which expert is emphasized. The saved score-range, weighted signed-gain, per-class losses and sparse-cell supports distinguish these effects; no IPS/head-index causal claim is inferred.

## Verification, preservation and limitations

Gate0 reproduced all9 C caches and the original per-seed Uniform/G1 AA/WGA exactly. All9 matching B caches were available and verified against ABC IDs, source hashes, ordered heads and learned scales. No head/backbone retraining or replacement OOF setup was needed. B exports contain no group fields. Thirty-six inner fits are checkpointed with normalizers, bin edges, class/boundary weights, hierarchical tables, sigma_A, IDs and source/data hashes; the first two recovery fits were reused. Method descriptors/predictions and each evaluation are independently committed.

Recovery tests forcibly killed a worker mid-inner-fold after one committed inner-fold, resumed in another process and verified bitwise model/prediction equality to uninterrupted execution, unchanged completed-file timestamps/hashes and no duplicate receipts. Atomic fsync/replace uses bounded Windows/OneDrive retries and an OS single-writer lock. Read completed/ for authoritative artifacts and state.json for the DAG.

One shared augmented400-epoch ERM* backbone, reused1199 development images, and approximately240 B examples per ensemble limit power. Seed replicates share images/backbone and are not independent datasets. The OOF table is fit on approximately3/4 of B; full-B deployment support differs. Original official labels retain the documented species-label issue. Old test arrays were never opened by this study.

Fresh-cohort gate: BLOCKED — NO FRESH DATA. No old-C point estimate can override this. No FRESH_CONFIRMATION.csv was created because no fresh evaluation was performed.

## B-only audit disposition

The recorded B loss/score decomposition is independently audited in the completion section below. Keep the frozen NO-GO; require a separately preregistered B-only follow-up before changing CCW.

## Resume

`python -m experiments.ccw_v6.run --phase all --resume`

`python -m experiments.ccw_v6.run --verify`

Read HANDOFF_STATUS.md, NEXT_AGENT_PROMPT.md and INPUT_REQUEST.md. All feasible current-data work is complete; preserve completed fits and all v1-v5 artifacts.

# Independent completion audit and measured failure mechanism

**CCW v6 is NO-GO on authorized B selection. No new CCW C evaluation and no production integration were performed. Fresh confirmation is unavailable.**

## Per-class B loss decomposition

| Method | Mean worst-class NLL | Class0 NLL | Class1 NLL | Effective experts |
|---|---:|---:|---:|---:|
| U0 | 0.63601003 | 0.63327516 | 0.63267755 | 15.0000 |
| G1_frozen | 0.62063432 | 0.62063432 | 0.60649832 | 10.8298 |
| G2_z | 0.63253482 | 0.63110408 | 0.62781035 | 14.4620 |
| CCW_A | 0.62520816 | 0.62520816 | 0.61125800 | 12.6883 |
| CCW_FULL | 0.62518272 | 0.62518272 | 0.61118192 | 12.6100 |

Partial calibration alone (CCW_A) raises the B selection proxy by +0.00457384 relative to G1. Adding the advantage table recovers only 0.00002544, leaving CCW_FULL worse than G1 by +0.00454840. Both class0 and class1 mean NLL are worse than G1. This directly identifies the calibration trade-off and the insufficient compensation from the frozen advantage term; it does not establish a causal IPS/head00 explanation.

The12-cell tables have median effective image support 14.2167; 44/432 cells are empty and 190/432 have fewer than30 distinct images. kappa25 shrinkage is substantial at these supports. Positive B advantage ranking therefore does not imply a large enough score correction to recover the proxy loss.

## Uniform-wrong B images: direction remains problematic

| Seed | Method | Distinct Uniform-wrong images | Mean signed gain sum(r*t) | Mean geometry score range | Mean advantage score range |
|---|---|---:|---:|---:|
| 0 | G1_frozen | 27 | -0.01554754 | 4.59611 | 0.00000 |
| 0 | CCW_A | 27 | -0.01089306 | 3.90641 | 0.00000 |
| 0 | CCW_FULL | 27 | -0.01099986 | 3.90641 | 0.25972 |
| 1 | G1_frozen | 26 | -0.01211185 | 3.74965 | 0.00000 |
| 1 | CCW_A | 26 | -0.00808815 | 3.40946 | 0.00000 |
| 1 | CCW_FULL | 26 | -0.00801209 | 3.40946 | 0.17338 |
| 2 | G1_frozen | 24 | -0.01565983 | 4.40359 | 0.00000 |
| 2 | CCW_A | 24 | -0.01288927 | 3.92095 | 0.00000 |
| 2 | CCW_FULL | 24 | -0.01331616 | 3.92095 | 0.27296 |

The mean true signed gain remains negative on Uniform-wrong B images for every listed seed/method. Partial normalization makes it less negative than G1, but the advantage term does not consistently turn it positive. Advantage score range never dominates geometry in these strata. The primary failure is not an unreported exploding learned score. No beta/rho/alpha change was tried after these observations.

## Integrity, tests and resume

Independent audit reconstructed training-only mu/sigma/reference medians, q20, class/boundary image weights, sigma_A, parent shrinkage and every cell for all36 fits. It verified180 method validation caches, train/validation IDs, probability mixtures, score decomposition and unchanged production source hashes. Per-seed B AA is based on613/619/615 unique image IDs rather than720 independent rows.

Nine unit/regression tests passed, including the missing-fresh-data serialization test. Source revision r2 fixes only a duplicate report keyword after491 scientific tasks had committed; their inputs, model math, splits, predictions and selection were preserved. Both forced-kill recovery exercises passed. Full separate-process resume produced zero new COMMITTED experiment tasks; all493 pre-delivery task hashes verified.

The final delivery adds this independent audit, checksum manifests and a compact package, without any new fit, search, head inference or C replay. All older v1-v5 outputs/checkpoints and existing Uniform/G1 production sources remain untouched.

## Exactly one justified next step after this audit

Keep G1 and require a separately preregistered B-only follow-up before changing CCW; the current frozen CCW result remains NO-GO and must not be rescued using old C/test outcomes.
