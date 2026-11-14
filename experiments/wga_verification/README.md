# Verify WGA first: frozen LCMSW

## Current continuation result

The original sources were subsequently supplied and recursively located under
`baselines/dpe4subpop`. Run the continuation rather than starting a new audit:

```powershell
.venv/Scripts/python.exe -m experiments.wga_verification.continue_run --resume
```

**Gate A now PASS; Gate B FAIL; production integration prohibited.** All 61
previous checkpoints were reused. Original source scripts match the archived
CSV and independent metrics exactly. Nested evaluation refits 135 heads over
three seeds × three outer folds, with disjoint A=head fit / B=weight fit /
C=holdout. WGA changes are +1.9410, −0.3146, −0.7519 pp; AA changes are
−0.9174, −1.000834, −0.5838 pp. Only one seed improves WGA, and seed 1 exceeds
the frozen AA-loss limit. No criteria were relaxed and no production changes
were made. See `results/dpe/wga_verification_v3_continuation_20261114/README.md`.

The original missing-source gate files remain immutable historical evidence.
`verification/CURRENT_GATES.json` in run `ce66e4448807` points to current gate
decisions. New `continuation_manifest.json` / `.r2.json` record source receipts,
nested splits/settings and the serialization-import fix; old snapshots remain.
A real head was killed after epoch 1, resumed identically to uninterrupted
training, and reused in the experiment. See the current handoff/state files.

The remaining sections describe the earlier independent-only audit.

This is an isolated research verifier, not production integration. It implements
the v3 handoff's frozen PCA(24, randomized, seed 0), global KMeans(2, n_init=6,
seed 1), static minimax mixture with simplex/cap .25 and L2 .02. It compares
epigraph SLSQP with direct-max SLSQP and never changes the recipe using metrics.

```powershell
.venv/Scripts/python.exe -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --phase all --resume
.venv/Scripts/python.exe -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --status
```

Journal tests must pass before fitting:

```powershell
.venv/Scripts/python.exe -m pytest tests/methods/test_wga_verification.py -q --basetemp outputs/wga_tests
```

Five checks passed, including a subprocess killed during an unfinished task,
resume without repeating its completed dependency, corrupt/missing output
invalidation, abandoned temporary files, solver constraints, uniform parity and
PCA/KMeans reproducibility. The real run was also stopped with `--max-tasks 2`
and resumed; only the remaining 59 tasks committed (61 total).

Each preprocessing fit, solver output, held-fold prediction and score has an
independent hash/size-verified completion marker. Artifacts are flushed/fsynced,
atomically replaced, verified and only then committed. A dependency graph
invalidates downstream tasks when a required artifact is missing or corrupt.
Source/config/development fingerprints define immutable run directories.
Status and continuation files are regenerated after every completed task.

## Actual outcome

Run `ce66e4448807`: independent numerical reproduction succeeds, with mean
delta WGA +1.789882 pp and AA +0.444815 pp, exactly 48 flips (32 corrected,
16 broken). Legacy 80/20 numbers also match the handoff tolerance. This is
only non-nested development sensitivity, not a new independent/test result.

**Full Gate A is FAIL:** original `DPE_WGA_Deep_Analysis.md` and
`DPE_WGA_Audit_Repro.zip` were not found in the repository or inspected
Downloads/Desktop locations. Therefore the original source script and its
manifest cannot be independently executed/compared. This missing requirement
is not silently waived because the numerical reproduction matches.

Gate B is NOT RUN due to Gate A failure. Production integration is prohibited.
No nested head fitting, real-head interruption test, deployment hook or default
DPE change has been performed. Only research verification code was added.
Once source evidence is supplied, complete the source comparison first; only
then implement/run nested fitting and its real-head crash acceptance test.
The downstream CLI fails closed; it must not be interpreted as an implemented
or completed Gate B/integration stage.

Read `outputs/wga_verification_v3/LATEST_RUN.txt`, then that run's
`FINAL_SUMMARY.md`, `verification/GATE_A.json`, `verification/GATE_B.json`,
`HANDOFF_STATUS.md` and `NEXT_AGENT_PROMPT.md`. Existing checkpoints, previous
reports, development and test data were not modified. Historical test arrays
were never read. The verified backbone is ERM*, not standard ERM.
