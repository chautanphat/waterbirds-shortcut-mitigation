# DPE geometry aggregation v5

Completed 2026-11-15. **G1 gamma50/alpha.25 passes engineering verification**:
mean WGA **+0.501253 pp**, AA **+0.111204 pp**; WGA improves in2/3 seeds,
every seed meets the strict AA floor. S2 tau.025 fails consistency (1/3 seeds).
WGA95% CI **[-0.214660,+1.253133] pp** crosses zero; no significant or fresh
confirmatory improvement is claimed. One shared frozen **ERM*** backbone.

```powershell
.venv/Scripts/python.exe -m experiments.geometry_aggregation_v5.run --phase all --resume
.venv/Scripts/python.exe -m experiments.geometry_aggregation_v5.run --status
.venv/Scripts/python.exe -m scripts.report_geometry_aggregation_v5
```

Aliases: `--phase audit|screen|nested` (or0/1/2); `--max-tasks N` stops at a
durable task boundary. Completed artifacts are SHA/size validated and reused.
Source/config/input changes fail explicitly. Sources/manifests r1/r2/r3 remain
preserved. Local atomic rename retries handle transient Windows/OneDrive locks;
an OS writer lock prevents concurrent runs. Every head epoch stores model,
optimizer, selected indices and RNG state. Forced epoch1 kill/restart matched
uninterrupted tensors. Full resume recomputed zero experiment tasks.

The smaller explicit grid in the repository handoff resolves the pasted grid’s
“where specified” alpha values:25 configs including U0 and diagnostic Top1,
75 seed evaluations. No WGA-driven expansion. Top group-blind recipe plus the
best eligible opposite family were locked: G1_g50 and S2_t0.025. Scalers fit
actual OOF fold/head identities on meta_train; no fictitious pooled head slots.

Nested evaluation trains135 **new** A-only heads, with B/C predicted from the
same ordered parameters. No old OOF/v3/v4 head is substituted. A/B/C splits are
disjoint, normalizers fit B only, groups evaluate committed predictions only.
The1199 development images and split identities had prior researcher inspection.

Production opt-in (default stays Uniform):

```python
probs = model.predict_proba(target="val", aggregation="distance")
metrics = model.evaluate(target="val", aggregation="distance")
```

Only the verified G1 recipe is enabled; class-wise S2 is not promoted. Versioned
metadata: `baselines/dpe4subpop/src/dpe/verified_distance_v1.json`.
The real DPE API, all9 cached C predictions, metadata/CLI and legacy defaults
passed13 tests. `netcal==1.3.5` (already a vendor requirement) was installed with
`--no-deps`; experiment NumPy/PyTorch versions were preserved.

Reports: `results/dpe/geometry_aggregation_v5_20261115/FINAL_REPORT.md`.
Checkpoints/state: `outputs/geometry_aggregation_v5/run1/`.
Package: `outputs/geometry_aggregation_v5/DPE_Geometry_v5_results.zip`.
The package contains the new heads and epoch checkpoints; workspace external
development inputs remain necessary to resume from their guarded source paths.
