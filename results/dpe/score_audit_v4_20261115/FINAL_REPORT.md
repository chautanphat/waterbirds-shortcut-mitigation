# DPE v4 — score audit and piecewise reliability

**Promotion gate: FAIL.**

Production integration: NOT PERFORMED because the frozen gate failed.
WGA is primary; every seed must have ΔAA strictly greater than −1.00 pp. Selection is by predeclared class-only worst-class NLL on the exploratory meta_select, with a strict per-seed AA floor. Exactly one primary was locked before group/outer evaluation.

## Nested results

| Seed | Uniform WGA % | Lookup WGA % | ΔWGA pp | Uniform AA % | Lookup AA % | ΔAA pp |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 90.9774 | 90.2256 | -0.7519 | 95.9133 | 96.3303 | +0.4170 |
| 1 | 92.4812 | 92.4812 | +0.0000 | 95.9133 | 95.9133 | +0.0000 |
| 2 | 91.7293 | 90.9774 | -0.7519 | 95.4128 | 95.4962 | +0.0834 |

{
  "status": "FAIL",
  "criteria": {
    "mean_WGA_positive": false,
    "positive_WGA_at_least_2_seeds": false,
    "AA_floor_every_seed": true,
    "provenance": true
  },
  "mean_delta_WGA_pp": -0.5012531328320802,
  "mean_delta_AA_pp": 0.166805671392826,
  "positive_seeds": 0,
  "WGA_CI95_pp": [
    -1.2531328320802024,
    0.07153075822603863
  ],
  "AA_CI95_pp": [
    -0.027800945232137053,
    0.3892132332499262
  ],
  "qualification": "Nested fixed-head engineering verification on previously inspected development; not untouched confirmation"
}

Locked primary: {"alpha": 0.25, "beta": 2.0, "delta_AA_pp_by_seed": [0.4166666666666652, 0.4166666666666652, 0.0], "eligible": true, "map_id": "R4", "mean_NLL": 0.1445257203796775, "proxy": 0.15791577138978516}

## Interpretation and limits

All three seeds share one verified augmented 400-epoch ERM* backbone; these are not standard ERM results or independent backbone replicates. A/B/C are disjoint, bins/calibration/teacher lookup fit only B, exact same ordered actual heads predict B and C. All 135 heads were reused after per-head fit-ID and checkpoint-hash verification; original per-epoch optimizer/RNG checkpoints remain intact. Nine prediction stages recover logits from frozen heads and exactly reproduce cached raw probabilities.

The v3 nested C outcomes and development pool had prior inspection. No C outcome entered this run’s selection, but this reuse cannot establish a scientifically untouched confirmation or remove researcher-level selection bias. Bootstrap intervals condition on fixed trained pipelines and exclude backbone/training/selection uncertainty. Same-image resamples align across seeds. No historical test is used to train or select.

Raw confidence >=.75 has zero meta_train support. Confidence .75 is representation-dependent: R3 uses original logits at fixed T=.25, R4 fits shared member-NLL temperature. Coverage and sparse confident-wrong support are logged by predicted class in each fit folder. No temperature is applied twice. The lookup estimates E[signed h(confidence)|predicted class, confidence bin, LOO agreement bin] with kappa50 shrinkage; correctness sign is training-only. R6 isolates correctness without confidence magnitude.

D19 Phase 0 reports exact relative-score ranking, wrong-expert weight and concentration with calibrated/raw uniform controls; see phase0_mechanism_report.md. It separates score misalignment from softmax concentration descriptively, without causal attribution. Piecewise scoring is deliberately mild (alpha .10/.25); actual pairwise ranking, wrong-head mass, entropy, effective members, corrected/broken directions and subgroup accuracies are saved for every candidate/seed. Oracle results are labeled non-deployable and excluded from all rankings.

## Historical context (different protocols, not confirmation)

Historical test matched raw uniform: AA95.708%, WGA92.761%; continuous D19: AA95.961%, WGA83.645%; discrete D5: AA96.076%, WGA90.343%. Separate original group-aware DPE checkpoint: AA96.048%, WGA94.393%, one seed. None of these is a new v4 test result. LCMSW v3 source reproduction passed, but nested promotion failed: mean WGA+.2915pp, AA−.8340pp, only1/3 WGA-positive and seed1 AA−1.000834pp. LCMSW is not retuned here.

## Exactly one next step

Evaluate the frozen primary on a newly reserved development cohort with subgroup annotations kept evaluation-only, using the same A/B/C protocol; preserve these results as exploratory evidence and do not retune from the old test.

## Reproduce and resume

`.venv/Scripts/python.exe -m experiments.score_audit_v4.run --phase all --resume`

Read state.json, HANDOFF_STATUS.md, NEXT_AGENT_PROMPT.md and completed/ checksums. Source/input/protocol hashes are guarded. A v4 task process was forcibly killed and resumed without rewriting a completed artifact; the prior real DPE head epoch-1 kill/resume tensor-equality test is also preserved. Every evaluation, fit representation, lookup and report has an atomic commit. No completed v1/v2/v3 tasks were retrained or overwritten.

## Independent audit and mechanism findings

All 135 final heads and epoch-20 optimizer/RNG checkpoints passed hash, fitting-ID and epoch checks. Nine A/B/C manifests matched independently generated splits. Direct reconstruction reproduced mixtures, probability representation and pooled AA/WGA. No new heads were trained.

### Calibration versus weighting (all values in percent or percentage points)

| Seed | Raw U WGA | Calibrated U WGA | Lookup WGA | Calibration ΔWGA | Weighting ΔWGA | Total ΔWGA vs raw |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 90.9774 | 90.9774 | 90.2256 | +0.0000 | -0.7519 | -0.7519 |
| 1 | 93.2331 | 92.4812 | 92.4812 | -0.7519 | +0.0000 | -0.7519 |
| 2 | 93.2331 | 91.7293 | 90.9774 | -1.5038 | -0.7519 | -2.2556 |

Mean total lookup versus RAW uniform: ΔWGA -1.253133 pp; ΔAA +0.166806 pp. The promotion comparison uses matched CALIBRATED uniform; calibration itself changes WGA and must not be credited to weighting.

### Is expert reliability rankable?

| Seed | Within-image correct-over-wrong ranking | Effective experts | Mean maximum weight | Corrected | Broken |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8371 | 14.9386 | 0.07093 | 6 | 1 |
| 1 | 0.8300 | 14.9402 | 0.07078 | 1 | 1 |
| 2 | 0.8703 | 14.9486 | 0.07012 | 2 | 1 |

The inference-safe lookup ranks correct experts above wrong ones more often than chance overall, but this does not ensure subgroup improvement. Only a few boundary images change. Effective expert count remains about 14.94, so the new failure is not the extreme softmax concentration seen in D19. Seed 0 breaks one waterbird-on-land (group 2); seed 1 breaks one waterbird-on-water (group 3); seed 2 breaks one waterbird-on-land (group 2). All nine corrections occur in landbird-on-water (group 1). The corrections do not compensate the worst group.

In the exploratory meta_train R4 lookup, 5 supported cells have empirical correctness below .5 yet a positive smoothed signed score; see positive_score_low_correctness_cells.csv. These are correlated expert observations and often only a few distinct images. This documents sparse-cell/class-parent shrinkage limitations, not an inference-label leak or a justification to tune kappa after evaluation.

### D19 H1/H2/H3 conclusions

- H1 supported descriptively: all 586 historical changed predictions are Waterbird→Landbird. The class/confidence panels and exact checkpoint forward expose the asymmetry.
- H2 supported locally, not globally: overall historical image-mean pair ranking is .7609–.8257, but waterbird-on-land ranking is 0.2057, 0.3372, 0.3248. The score is capable of ranking globally while misranking the hard subgroup. Group labels are used here only after the fact.
- H3 descriptively supported: D19 effective expert count is2.50–3.07 and there are264 minority-wrong overturns across seed/image rows. The new mild lookup keeps almost15 effective experts and still reduces WGA, so concentration alone cannot explain every failure. These observations do not prove a causal decomposition.

### Confidence scale check

| Seed | Representation | Fitted/fixed T | Mean confidence | P95 | Maximum | >=.75 expert observations % | Distinct >=.75 images |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | R1 | 1.000000 | 0.5299 | 0.5601 | 0.6086 | 0.000 | 0 |
| 0 | R3 | 0.250000 | 0.6155 | 0.7242 | 0.8539 | 2.155 | 203 |
| 0 | R4 | 0.030278 | 0.9241 | 0.9997 | 1.0000 | 89.621 | 959 |
| 1 | R1 | 1.000000 | 0.5294 | 0.5590 | 0.6077 | 0.000 | 0 |
| 1 | R3 | 0.250000 | 0.6137 | 0.7207 | 0.8520 | 2.238 | 214 |
| 1 | R4 | 0.030015 | 0.9220 | 0.9996 | 1.0000 | 89.392 | 959 |
| 2 | R1 | 1.000000 | 0.5303 | 0.5610 | 0.6043 | 0.000 | 0 |
| 2 | R3 | 0.250000 | 0.6171 | 0.7273 | 0.8447 | 2.350 | 183 |
| 2 | R4 | 0.030342 | 0.9256 | 0.9997 | 1.0000 | 89.732 | 959 |

Coverage is measured on each seed’s meta_train only. Raw .75 is empty, fixed T=.25 has a sparse upper tail, and fitted T≈.03 makes .75 common. No test threshold fitting was done. Interval-specific counts/accuracy by predicted class and sparse confident-wrong cells are in every fit folder.

### Completion / recovery

7 focused tests passed. Full separate-process resume generated zero new COMMITTED events and validated every existing artifact hash/size. The earlier failed boolean diagnostic was fixed in recorded r2; r1 sources, manifest, failure log and two completed tasks were preserved. The final research/report task below adds evidence without overwriting any earlier completed artifact.

Current report: FINAL_RESEARCH_REPORT_r2.md. Earlier FINAL_REPORT.md is retained as the original final-stage checkpoint. Production integration remains prohibited; no DPE production source/default was changed.

## Report / initialization maintenance

The prior report checkpoint is preserved. A descriptive subgroup sentence was corrected from the saved flip tables: broken groups are 2,3,2 for seeds0,1,2, respectively. No measurements, selection, gate or experiment outputs changed. Source revision r3 only permits initialization when lineage history does not already exist; all completed tasks and prior source/manifest versions remain valid.
