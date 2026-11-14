# LCMSW v3 — source reproduction passes, nested promotion fails

**Gate A: PASS. Gate B: FAIL. Production integration: PROHIBITED and not performed.**

The newly supplied original analysis/archive were recursively found under baselines/dpe4subpop. Original cv_sensitivity.py and robust_static.py executed with only filesystem path substitutions. Results match archived CSVs and the independent calculation with 0.0 pp discrepancy. No separate original SHA manifest or epigraph_verify.py was supplied; this is explicitly documented, all archive members were hashed on receipt, the bundle export manifest verified, and the original 18 independent epigraph/direct-max checks retained.

Gate A reproduces +1.789882 pp WGA and +0.444815 pp AA, with 48 flips (32 corrected, 16 broken). These are non-nested development sensitivity results, not deployment/test improvements.

## Truly held-out head-and-weight evaluation

For each of three seeds and three outer class-stratified folds, split the outer training pool 70/30 into A=head fit and B=weight fit; C is the held-out outer fold. Train 15 heads from scratch only on A, fit frozen PCA/KMeans/minimax weights only on B, and predict C with the exact same ordered heads. Every A/B/C ID set is disjoint. Groups are used only after outer predictions are committed. No historical test arrays were read and no hyperparameters were changed.

| Seed | Uniform AA % | LCMSW AA % | ΔAA pp | Uniform WGA % | LCMSW WGA % | ΔWGA pp |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 95.8299 | 94.9124 | -0.9174 | 90.9774 | 92.9185 | +1.9410 |
| 1 | 95.9133 | 94.9124 | -1.0008 | 93.2331 | 92.9185 | -0.3146 |
| 2 | 95.4962 | 94.9124 | -0.5838 | 93.2331 | 92.4812 | -0.7519 |
| Mean | 95.7465 | 94.9124 | -0.8340 | 92.4812 | 92.7727 | +0.2915 |

Sample SD across the three head/pipeline seeds:
- uniform: accuracy 0.2207 pp, wga 1.3023 pp
- candidate: accuracy 0.0000 pp, wga 0.2524 pp

## Exact failure reasons

- Positive mean WGA: PASS (+0.2915 pp).
- WGA improvement in at least 2/3 seeds: FAIL (only seed 0 improves; seeds 1 and 2 decline).
- Mean ΔAA > −1.00 pp: PASS (−0.8340 pp).
- Every seed ΔAA > −1.00 pp: FAIL (seed 1 is −1.0008340284 pp). This narrowly crosses the frozen boundary; no rounding tolerance or criterion relaxation was introduced. The WGA consistency requirement fails independently.
- Provenance/leakage/identity audit: PASS; 135 trained heads, nine A/B/C manifests, exact paired-head hashes, capped weights and pooled metric arithmetic verified.

Paired group-stratified unique-image bootstrap 95% CI: ΔWGA [-1.9671092968472697, 1.830017640668191] pp; ΔAA [-1.3066444259104808, -0.3336113427856547] pp. The same sampled image IDs are aligned across all three seeds. WGA uncertainty crosses zero; AA indicates harm conditional on these fixed pipelines. These intervals exclude independent-backbone and method-selection uncertainty.

## Failure analysis and scope

Across the three seeds, 13 predictions were corrected and 43 were broken relative to matched uniform. Landbird-on-water accuracy falls in all three seeds. In seed 0, improving waterbirds-on-land changes which group is worst; in seed 1, landbirds-on-water become the worst group. Real subgroup WGA therefore does not track the latent worst-cluster CE objective consistently.

A contains about 559 examples and B about 240. Small class-balanced head subsets produce fewer updates per epoch than full-development training. The existing no-group head adapter retains the final partial batch, uses fixed epoch 20, SGD .001/momentum .9, entropic scale 30, prototype penalty 10 and inverse-decayed IPS starting at 5e5. This exact prior-adapter schedule was frozen and retained; no group-based early stopping or weighting occurs.

Gate B removes the OOF-slot transfer problem by fitting weights on the actual heads used on C. It also changes head/weight sample sizes relative to the exploratory sensitivity check. The loss of the earlier +1.79 pp WGA effect cannot be attributed uniquely to one of those changes. It demonstrates that the stronger verification does not sustain the claimed promotion under this protocol.

The shared backbone is the previously verified 400-epoch augmented ERM*, not standard ERM. Development data have prior researcher inspection; this is nested engineering verification, not a fresh confirmatory dataset or independent-backbone scientific claim. No production interfaces, flags or default changes were implemented.

## Crash recovery and preserved work

All 61 earlier tasks were reused. New original-source runs and comparisons were added as tasks. A real head process was killed after epoch 1 and resumed from that epoch; final tensors exactly matched uninterrupted training, and the recovered head was reused. A missing joblib serialization import stopped the first fold after head/prediction checkpoints; revision r2 records the import-only correction, preserves r1 sources/manifest and resumes at preprocessing without retraining those heads. Every head has atomic epoch-level optimizer/RNG/model state and a committed final artifact.

See state.json, HANDOFF_STATUS.md, NEXT_AGENT_PROMPT.md and completed/ markers. Original gate JSON files remain immutable historical decisions; verification/CURRENT_GATES.json points to the authoritative continuation decisions. Do not modify the frozen algorithm or integrate it to chase better metrics. A different method/protocol requires a new preregistered research task.

`python -m experiments.wga_verification.continue_run --resume`
