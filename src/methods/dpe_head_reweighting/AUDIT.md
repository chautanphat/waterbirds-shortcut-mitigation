# Reweighting implementation audit

## Scope and evidence

The audit covers the experiment's complete execution path: shared Waterbirds
loading and transforms, class-balanced sampling, ResNet construction, ERM
optimization and checkpoint selection, ordered raw-feature extraction, Stage-0
handoff, A/B/C partitioning, sequential prototype training, aggregation, final
evaluation, artifact reuse and the portable notebook. Reference comparisons use
the preserved DPE implementation and historical reproduction notebook. Unrelated
preserved methods are not certified by this audit.

The supplied seed-0 execution log records source commit `5ebb112`, 100 backbone
epochs, 3,700 updates, all 15 heads, final evaluation and archive creation. Its
selected configuration is `alpha=0`, `gamma=1`, with all weights equal to `1/15`.
Uniform and selected test WGA are both 89.7561%, and overall accuracy is
91.2323%. These equal results follow from the selected configuration. The log
does not contain the complete candidate table or per-head prediction arrays.

All 100 logged backbone records were checked for consistent epoch/update counts,
split counts, group-weighted overall accuracy and minimum-group accuracy.

## Findings and corrections

1. **Saved results were not bound to the selection file.** The notebook checked
   model/input locks, but an edited selection could coexist with an old final
   report and still pass its reuse checks. Final reports now record the SHA-256
   of `selection.json`, and notebook reuse checks it. Before evaluation, the
   saved selection is also recomputed from the B/C prediction caches, whose
   IDs, labels, groups, ensemble identity and preprocessing are checked against
   the manifest. Inconsistent candidates or weights are rejected before test
   prediction. No evidence in the supplied log shows that this failure occurred.

2. **Shared metrics silently truncated fractional inputs.** Binary validation
   occurred after integer conversion. Fractional predictions, labels or
   attributes could be accepted as binary integers. Validation now precedes
   conversion. The actual dataset loader supplies integer labels and attributes,
   and evaluation uses class argmax; this does not explain the recorded equality.

3. **Candidate decisions were insufficiently visible.** Only the winner was
   printed. The notebook now displays all 13 candidates with C WGA, C overall
   accuracy, effective head count, probability differences and changed class
   predictions relative to uniform. The selection stage also writes
   `candidate_comparison.csv`. This distinguishes changed probabilities from
   changed classifications; weights can differ without changing accuracy.

No forced-uniform aggregation, reversed head axis, ignored selected weights or
unconditional copying of uniform metrics was found in the inspected path. This
is a code-review finding, not proof about unprovided runtime artifacts.

## Independent verification

The independent checker imports no production selection, combination or metric
functions. It checks the fixed grid, computes B scores and weights, explicitly
sums weighted C probabilities, applies the declared tie breakers, then computes
test group metrics, paired differences, changed error IDs and effective head
count. It also checks manifest/checkpoint/cache identities and disjoint IDs.
It supports the original results-bundle format from `5ebb112`.

Extract the notebook results archive and run from a repository checkout:

```bash
python scripts/audit_dpe_head_reweighting.py --run-dir path/to/extracted/run
```

This command reads the archived bundle without modifying it. It does not load
raw images, rerun ResNet or establish that cached image labels match the
original dataset. Those checks additionally require the retained dataset,
backbone and raw features. The supplied text log alone cannot support this
artifact-level verification of the particular seed-0 run.

Regression checks include nonuniform winners, uniform winners, changed
probabilities with unchanged class predictions, 100 randomized independent
selection comparisons, corrupt selection rejection, stale final-report
rejection and full notebook execution/reuse for both tracks. Notebook checks
use synthetic inputs and a tiny trained backbone; full ResNet GPU accuracy is
not established by these tests.

The complete repository test suite passed: 30 tests and two subtests, including
both notebook tracks, reuse checks, independent bundle audits and rejection of
stale reports. All project-owned Python modules/scripts and notebook code cells
also passed syntax parsing.

## Interpretation and compatibility

Subset C contains 26 examples in each waterbird/background group. One changed
prediction changes such a group accuracy by about 3.85 percentage points.
Consequently, selection can be coarse and ties can be common. The declared
tie breaker prefers uniform when WGA and overall accuracy tie; this is a
protocol choice, not evidence that reweighting is effective or ineffective in
general. The test set must not guide changes to the candidate grid.

The shared backbone recipe and the historical baseline do not promise identical
random streams: the latter also iterates the test loader every epoch. The two
aggregation conditions within H2 nevertheless use the same saved backbone,
features and trained heads.

Keep completed original runs intact. New implementation hashes invalidate
continuation through the normal experiment runner. Use the independent checker
to inspect old bundles; use a fresh H2 run directory for the updated runner.
The existing completed backbone can be reused. If extraction-source checks
reject an old feature bundle, export into a fresh input directory. A complete
rerun is not necessary merely to inspect the old candidate decision.

## Multi-seed joint-solver failure — reviewed 2026-11-09

The corrected supplied `download.txt` (SHA-256
`e69f336f09ec66629fe8a789edab586bc7a302583472e7a9f45f47fcd897ae6d`)
records seed 0 completing from an attached source and seed 1 completing 100
backbone epochs/3,700 updates, raw-feature export and all 15 heads. At 5,969.9
elapsed seconds, SLSQP rejects a joint fit with `Positive directional derivative
for linesearch`. This is a numerical optimizer failure, after training. The log
does not identify which lambda failed or provide the B/C arrays needed to rerun
that exact problem. No seed-1 final test result or seed-2 training is recorded.
The traceback's actual exception, rather than later nbconvert warnings, explains
the failed notebook status.

The joint fitter now retains SLSQP as primary and retries an unsuccessful or
invalid result with `trust-constr` on the identical convex objective/constraints.
The recovery solver uses analytic derivatives, checks feasibility, objective
against uniform and first-order optimality, and records both attempts. C still
selects among the same frozen candidate grid; test data remain held out.

The study discovers frozen pretest source bundles and matches their saved raw
feature handoff by hashes/provenance. After B/C selection is frozen, it performs
only the missing test inference with those same heads in a separate source copy.
This avoids retraining merely because the weighting implementation changed.
The failed version's complete saved outputs are required; a log or partial
summary ZIP alone cannot restore seed 1. See the method README for Kaggle steps.

Regression checks force the logged SLSQP failure and compare recovery with a
known complementary-head optimum and 15-head unequal-group soft-probability
problems for all three lambdas. Integration checks recover saved heads/features
with dataset preparation/training disabled, verify exact test predictions and
unchanged original inputs, and reject changed feature artifacts. These are
synthetic implementation checks, not a rerun of the unavailable seed-1 arrays
or a new benchmark measurement.
The full repository suite passed: 65 tests and two subtests, including notebook
execution/reuse, hosted bootstrap simulations and complete multi-seed smoke runs.

### Further crash-path checks — 2026-11-09

A deeper 108-case stress check forced the SLSQP failure across all three
lambdas, 15 heads, unequal B-group sizes (94/93/27/27), identical heads and
increasingly saturated probabilities. Four `trust-constr` fits reached their
iteration limit on NumPy 1.26.4/SciPy 1.12.0 despite near-optimal solutions.
Recovery now independently computes a global objective-gap certificate from a
convex combination of group tangents and the exact quadratic regularizer. Its
simplex dual lower bound, primal checks and first-order residual must pass the
declared `1e-7` tolerance. Raw solver termination flags remain recorded; a
success flag or feasibility alone cannot authorize acceptance. All 108 cases
pass, and a deliberately feasible but suboptimal fake result is rejected.
The same joint-fit checks also pass with isolated NumPy 2.1.3/SciPy 1.16.3.

Recovery copies and study/source ZIPs now publish atomically. Simulated interrupted
copies can be retried without stale partial files; interrupted ZIP exports retain
the previous complete download. Rule-grid errors and missing recovery handoffs
are checked before training a missing seed. The notebook uses an explicit
runtime compatibility marker to avoid importing an older managed checkout.
These checks supplement the previous failure audit; they do not supply the
missing real seed-1 prediction arrays or certify hosted-service uptime.
The final full suite passed 179 tests and two subtests. An additional joint-fit
subset passed 115 tests on NumPy 2.1.3/SciPy 1.16.3; the main suite used NumPy
1.26.4/SciPy 1.12.0. Full notebook checks use synthetic backbones and real
training loops, rather than a new GPU Waterbirds benchmark run.

## Completed real-data study — reviewed 2026-11-09

The subsequently supplied `standard-seeds012-aed182e8ca.zip` contains all three
original prediction bundles and frozen multi-rule decisions at `dbeba0f`.
The [H2 summary](../../../report/main.tex)
and [machine-readable checks](../../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/artifact_checks.json)
supersede the earlier absence of real seed-1/2 arrays for this completed rerun.
All 60 B/C candidates were recomputed, all nine joint fits reproduced and all
selected test metrics, group counts, changed errors, means and sample SDs matched.
Refitted weights matched exactly on NumPy 2.1.3/SciPy 1.16.3; source/code and
selection/cache identities also agreed. Seed 1's λ=0.01 SLSQP line-search failure
was recovered by `trust-constr` in 24 iterations, with objective-gap bound
5.53e-10. The real-data workflow completed all three seeds.

The 200 logged backbone epochs passed count/update/group-arithmetic checks.
Seed 1 trained again and its logged trajectory matches the earlier failed
attempt; this does not prove identity of the unavailable failed-version
checkpoints. Raw-image feature extraction and cross-hardware parity remain
outside the audit. Successful numerical recovery does not establish a
scientific benefit of weighting: no C-selected procedure improved test WGA.
