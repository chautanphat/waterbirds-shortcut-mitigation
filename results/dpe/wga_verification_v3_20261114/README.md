# LCMSW v3 — numbers reproduced; full Gate A failed on missing source evidence

> Historical decision, superseded by the [completed source comparison and nested
> verification](../wga_verification_v3_continuation_20261114/README.md): Gate A
> now PASS, Gate B FAIL, production integration prohibited. The original source
> files were supplied later; numerical reproduction below remains unchanged.

Independent implementation of the frozen PCA(24)/KMeans(2)/capped static minimax
recipe reproduces the reported development sensitivity gains:

| Seed | Uniform AA % | LCMSW AA % | Uniform WGA % | LCMSW WGA % |
|---|---:|---:|---:|---:|
| 0 | 94.7456 | 95.0792 | 91.7293 | 93.9850 |
| 1 | 94.8290 | 95.2460 | 92.2747 | 93.1330 |
| 2 | 95.1626 | 95.7465 | 92.4812 | 94.7368 |
| Mean | 94.9124 | 95.3572 | 92.1617 | 93.9516 |

**Mean delta WGA +1.789882 pp; mean delta AA +0.444815 pp.**
Maximum discrepancy from rounded per-seed references is 0.0000473 pp,
below the 0.03 pp tolerance. Exactly 48 prediction flips: 32 corrected,
16 broken. Legacy 80/20 mean metrics also match: uniform AA/WGA 95.00/90.72%,
candidate 95.97/92.76%. All 18 epigraph solutions converged, satisfy the
constraints and agree with direct-max objectives within 1e-6.

## Gate decisions

- Independent numerical reproduction: **PASS**.
- Full Gate A: **FAIL**. Required `DPE_WGA_Deep_Analysis.md` and
  `DPE_WGA_Audit_Repro.zip` were absent from the workspace and inspected
  Downloads/Desktop locations. Therefore original scripts and their manifest
  could not be executed/compared. This mandatory condition was not waived.
- Gate B: **NOT RUN, because Gate A failed**.
- Production integration: **PROHIBITED / NOT PERFORMED**.

These are non-nested sensitivity results: the five-fold procedure repartitions
existing three-fold OOF predictions, without retraining heads inside the new
outer folds. WGA is calculated after pooling the five held-out predictions
per seed, not by averaging fold minima. Heads at the same slot in different
original folds have different parameters; compatibility with final refits is
not assumed. Prior method nomination used development groups. No fresh
independent or scientific confirmation is claimed. The shared backbone is
verified ERM*, not standard ERM. Historical test arrays were never opened.

## Recovery evidence and continuation

Run `ce66e4448807` has 61 valid committed tasks. Five tests passed, including
a killed-process restart, corrupted/missing artifact invalidation, abandoned
temporary files and solver constraints. The actual run stopped at two tasks;
a separate process committed only the remaining 59, with zero duplicate IDs.
All artifact hashes and sizes were independently rechecked. The real-head
interruption acceptance test was not reached because nested fitting is gated.

The complete durable run is at `outputs/wga_verification_v3/ce66e4448807/`,
including state, handoff files, source snapshots, locked versions/config,
per-fit PCA/KMeans artifacts, weights and per-seed predictions.

```powershell
.venv/Scripts/python.exe -m experiments.wga_verification.run --config experiments/wga_verification/frozen.yaml --phase all --resume
```

Provide original source evidence before advancing. See
[reproduction record](REFERENCE_REPRODUCTION.md),
[reference comparison](reference_comparison.csv), [solver audit](solver_audit.csv),
[Gate A](GATE_A.json), [Gate B](GATE_B.json),
and [recovery acceptance](recovery_acceptance.json).
