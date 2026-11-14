# DPE v5 — prototype geometry aggregation

**Engineering promotion: PASS.** WGA is primary; every seed must retain ΔAA strictly greater than −1.00 percentage point.

## Nested verification

| Candidate | Seed | U WGA % | Candidate WGA % | ΔWGA pp | U AA % | Candidate AA % | ΔAA pp | Corrected / broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1_g50 | 0 | 90.9774 | 90.9774 | +0.0000 | 95.8299 | 95.8299 | +0.0000 | 3 / 3 |
| G1_g50 | 1 | 93.2331 | 93.9850 | +0.7519 | 95.9133 | 96.0801 | +0.1668 | 3 / 1 |
| G1_g50 | 2 | 93.2331 | 93.9850 | +0.7519 | 95.4962 | 95.6631 | +0.1668 | 3 / 1 |
| S2_t0.025 | 0 | 90.9774 | 90.9774 | +0.0000 | 95.8299 | 95.9133 | +0.0834 | 5 / 4 |
| S2_t0.025 | 1 | 93.2331 | 93.2331 | +0.0000 | 95.9133 | 95.8299 | -0.0834 | 3 / 4 |
| S2_t0.025 | 2 | 93.2331 | 93.9850 | +0.7519 | 95.4962 | 95.4128 | -0.0834 | 2 / 3 |

**G1_g50: PASS.** Mean ΔWGA +0.501253 pp; mean ΔAA +0.111204 pp. WGA positive in 2/3 seeds. Criteria: {"mean_WGA_positive": true, "positive_WGA_seeds_at_least2": true, "AA_floor_every_seed": true, "provenance": true}.
Paired mean ΔWGA 95% CI [-0.21465950283433965, 1.2531328320802024] pp; ΔAA CI [-0.055601890464277895, 0.3058103975535187] pp. Unique image IDs resampled identically across seeds; no independent-backbone or selection uncertainty included.

**S2_t0.025: FAIL.** Mean ΔWGA +0.250627 pp; mean ΔAA -0.027801 pp. WGA positive in 1/3 seeds. Criteria: {"mean_WGA_positive": true, "positive_WGA_seeds_at_least2": false, "AA_floor_every_seed": true, "provenance": true}.
Paired mean ΔWGA 95% CI [-0.7090645194287186, 1.002506265664164] pp; ΔAA CI [-0.27800945232138163, 0.22240756185710012] pp. Unique image IDs resampled identically across seeds; no independent-backbone or selection uncertainty included.

## 1. Does distance rank the right experts when uniform fails?

| Seed | −min distance rank | Confidence rank | Distance margin rank | Mixed uniform-wrong images |
|---|---:|---:|---:|---:|
| 0 | 0.3281 | 0.2836 | 0.2844 | 43 |
| 1 | 0.3119 | 0.2783 | 0.2785 | 35 |
| 2 | 0.3698 | 0.3409 | 0.3428 | 37 |

Min-distance routing diagnostic: **NO**. See phase0_geometry_audit.md for all-sample versus Uniform-wrong rankings, paired difference intervals and meta_select sensitivity. Correct-vs-wrong pairs are averaged within each image first; bootstrap units are distinct images. All-wrong images contribute no pair-ranking statistic and are reported separately with oracle headroom.

Source-level distance audit confirms logit_jk = −abs(learned_scale_j)×D_jk. Features are standardized per image then L2-normalized; prototypes are L2-normalized for distance. Learned scales and dmin mean/SD vary by actual head. A small absolute distance is closeness to some prototype, not a correctness probability. Confidence and entropy can confidently favor a wrong shortcut expert. G2 fits each actual OOF fold/head separately and does not pool slots from unrelated heads.

## 2. Does class-wise Soft-min improve WGA?

Nested S2_t0.025: mean ΔWGA +0.250627 pp; gate FAIL. This is class-specific logsumexp/prototype contribution aggregation, not a common expert-weighted average.

All S1/S2/S3 screening taus .025/.1/.5, S4 tau.1 and residual alphas .1/.25/.5 were retained including losing candidates. T_class remains1; no unplanned calibration grid was added. S2 uses actual logits; S3 removes per-head class-common distance offsets. S1 and S2 are distinct when learned scales vary.

## 3. AA constraint and 4. survival under nesting

Each gate records every seed’s exact ΔAA, its strict floor, pooled WGA sign and seed consistency. The two recipes were locked by class-only worst-class NLL and AA before subgroup WGA. No post-hoc relaxation, reranking by group or expanded sweep occurred. AA and class recall are also reported with full confusion matrices; improving majority-class accuracy cannot substitute for subgroup robustness.

135 new actual heads were trained from scratch on A using the frozen no-group adapter: class-balanced subsample, final epoch20, SGD .001/momentum.9, batch256 including final partial batch, entropic training scale30, prototype penalty10 and inverse-decayed IPS starting5e5. Backbone was frozen. Every epoch has atomic model/optimizer/RNG state; every head-final and prediction cache has hashes. B fits scalers, C only evaluates committed predictions from the identical ordered heads. All A/B/C IDs and parameter hashes are saved. Heads from old OOF/v3/v4 were not substituted.

## 5. Failure / success mechanism

G1_g50: 9 corrected and 5 broken seed/image rows; worst true group per seed [2, 2, 2]. These counts repeat images across seeds and are not independent samples. Per-group flip CSVs identify whether changes help the actual worst subgroup or shift which group is worst.
Distance/entropy weights encode geometric proximity or confidence, not observed correctness. Any reversal on Uniform-wrong images explains why overall expert ranking can look useful while routing harms the difficult subgroup. Effective expert count and wrong-expert mass quantify concentration separately from signal quality.
S2_t0.025: 10 corrected and 11 broken seed/image rows; worst true group per seed [2, 2, 2]. These counts repeat images across seeds and are not independent samples. Per-group flip CSVs identify whether changes help the actual worst subgroup or shift which group is worst.
Class-wise pooling can let different heads support different class scores. Learned-scale/common-offset effects and class-score softness differ from uniform probability averaging; class-specific effective-prototype counts and S2/S3 controls are saved. This mechanism must not be described as one reliability weight per expert.

## Scope and historical context

All three seeds share one verified augmented400-epoch ERM* backbone; these are not standard-ERM or independent-backbone results. The original1199 development images, meta_select and these deterministic nested split identities have prior researcher inspection. New head fitting and strict A/B/C isolation remove within-run leakage, but do not turn reused development into a scientifically untouched confirmation. No historical test array was opened by this study.

Historical repeatedly inspected test: matched uniform AA95.708%, WGA92.761%; D19 AA95.961%, WGA83.645%; D5 AA96.076%, WGA90.343%. LCMSW and v4 piecewise reliability failed their stronger nested gates. These are context only, with different fitted ensembles/protocols, not v5 confirmation or a pure cross-method effect.

## One next step

Evaluate the prelocked recipe/protocol on a newly reserved development cohort with evaluation-only subgroup annotations, without further tuning on these development folds or the historical test.

## Persistence / reproduce

` .venv/Scripts/python.exe -m experiments.geometry_aggregation_v5.run --phase all --resume `

Forced epoch1 process kill/restart matched uninterrupted final tensors; the recovered head was reused. Journal hashes detect missing/corrupt artifacts and source/config mismatch. All source snapshots, failed diagnostics, checkpoints, predictions, selections and decisions remain intact. Read state.json, HANDOFF_STATUS.md and NEXT_AGENT_PROMPT.md for the current committed boundary.

Production integration: COMPLETE for G1_g50 as opt-in distance mode; default remains uniform. S2 is not promoted.

## Verified opt-in use

`model.predict_proba(target="val", aggregation="distance")` enables only the frozen G1 recipe gamma50/alpha.25. Omitting aggregation preserves the prior uniform probability averaging, member probabilities and raw-logit API. `model.evaluate(target="val", aggregation="distance")` also opts in explicitly.

Versioned metadata is at `baselines/dpe4subpop/src/dpe/verified_distance_v1.json`. It states exact normalized-distance/logit representation, class order,15 members, fixed parameters and verification limits. G1 needs no trainable normalizer/calibrator. The implementation computes normalized geometric distances once, then uses learned head scale for probabilities while retaining unscaled distance for routing.

Prediction-cache CLI: `PYTHONPATH=baselines/dpe4subpop/src python -m dpe.geometry --input-cache CACHE.npz --output NEW.npz --aggregation distance`. Its default is uniform; it loads only named probabilities/distances/IDs, accepts no labels/groups, and refuses to overwrite an existing output.

Production acceptance verifies exact probabilities/weights on all nine saved C caches, the actual DPE class with real newly trained heads/features, unchanged uniform/member/raw-logit outputs, metadata roundtrip and CLI behavior. `production/integration_receipt.json` records the test log and source hashes.

Revision r2 adds bounded retries for Windows/OneDrive atomic rename locks. The local training adapter has AST-identical functions to the historical adapter; no model, grid, split or selection math changed. The incomplete head resumed from its last valid epoch, completed tasks were retained and the original sources/manifests/logs remain preserved. Revision r3 wires conditional production acceptance into the single resume command after the nested gate; it does not recompute experiments.


# Supplemental completion record

G1 passes the frozen engineering gate; S2 fails the two-positive-seeds rule. No subgroup WGA selected a recipe or chose between candidates. The first passing prelocked recipe is the only opt-in production method.

Minimum distance slightly outranks confidence on Uniform-wrong examples, but both remain below .5; therefore the audit does not establish a dependable expert router. The mild uniform residual can still make a few useful boundary corrections. This is compatible with a small engineering gain and does not imply general correctness estimation.

## G1_g50

Mean ΔWGA +0.501253 pp; ΔAA +0.111204 pp; WGA CI [-0.21465950283433965, 1.2531328320802024]. Corrected/broken by true group across seed/image rows: {"broken": {"0": 2, "1": 2, "2": 1, "3": 0}, "corrected": {"0": 1, "1": 2, "2": 3, "3": 3}}

## S2_t0.025

Mean ΔWGA +0.250627 pp; ΔAA -0.027801 pp; WGA CI [-0.7090645194287186, 1.002506265664164]. Corrected/broken by true group across seed/image rows: {"broken": {"0": 4, "1": 4, "2": 2, "3": 1}, "corrected": {"0": 1, "1": 4, "2": 3, "3": 2}}

The complete report includes the raw per-seed AA/WGA, strict gate criteria, source geometry and numerical controls. Every baseline prediction was checked against the legacy Torch uniform reduction: reductions can differ from NumPy by <=2e-7, but all nested labels are identical; original Uniform API is unchanged. This does not alter a metric tolerance or AA floor.

13 tests passed, including all nine C caches, actual DPE class inference, legacy defaults/member/raw-logit outputs, label-free metadata/CLI, forced real-head recovery, atomic retry and identical adapter mathematics. Separate full resume validated414 completed tasks and produced zero experiment commits. All135 new heads retain epoch20 optimizer/RNG state. Original v1-v4 results are preserved.

Figures under figures/ improve spacing/contrast while retaining the original committed PNGs and CSV sources. Prediction outputs, fitted G2 scalers (exploratory controls), all selected/losing candidates, ABC/head manifests and source snapshots are preserved. Readable heatmap quantile edges remain in phase0/seed*/meta_*/train_only_bin_edges.json.

Scope: one verified augmented ERM* backbone, reused1199 development images/split identities, 3 head seeds rather than independent backbones. WGA interval crosses zero; benefits depend on few subgroup examples. No previously inspected test inputs were opened. Exactly one next step remains the report’s fresh-cohort confirmation with the recipe frozen.
