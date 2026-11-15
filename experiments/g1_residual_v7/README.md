# G1 residual v7 — precision weighting

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

The frozen five-method matrix is U0, G1_frozen, pooled G1_R, primary G1_RH,
and margin-gated G1_RHM. Continuous six-column ridge (penalty 10), exact-head
predicted-class partial pooling (kappa 25) and nested predicted-OOF scaling
modify only the G1 expert score. Gamma=50, alpha=.25 and lambda=.5 remain fixed.
The scale off-switch at s_R<.002 is explicit. No architecture, prototype,
backbone, IPS or production-default change is made.

```powershell
.venv/Scripts/python.exe -m pytest experiments/g1_residual_v7/tests -q
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase inventory --resume
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase B_crossfit --resume
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase exploratory_c --resume
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase all --resume
.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --verify
```

Four B inner folds and three secondary OOF folds fit every feature/scale using
training-only IDs. Full-B refits follow the B direction decision. All registered
C methods are evaluated even if B fails. C cannot select settings or promote a
method. Three head seeds reuse 1199 development images and one ERM* backbone.

Artifacts live locally in this namespace; source/config/input fingerprints and
hashed semantic task IDs reject incompatible resumes. Completion receipts are
authoritative. `--max-tasks N` stops at a durable boundary. Atomic fsync/replace
with Windows retries, OS single-writer locks, corruption detection, real
secondary/C-fold kill-restart tests, and batchwise RNG persistence protect work.
Historical v1-v6 artifacts and the v6 NO_GO remain unchanged.

Read `EXPLORATORY_C_REPORT.md`, `metrics/FINAL_VERDICT.json`, `state.json`,
`HANDOFF_STATUS.md` and `NEXT_AGENT_PROMPT.md` after execution. All required
comparison/mechanism/flip tables are under `metrics/`; C labels are stored
separately from `WEIGHT_AND_SCORE_DECOMPOSITION.npz`. Fits, scale OOF predictions
and individual inference caches remain under `checkpoints/`. No independent
confirmation is implied without a genuinely untouched cohort.
