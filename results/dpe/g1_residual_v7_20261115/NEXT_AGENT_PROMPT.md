Continue the frozen G1 residual v7 study, not a G1-only confirmation.
Read the v7 handoff, config.lock.json, EXPERIMENT_LOCK.json, state.json and HANDOFF_STATUS.md.
Run .venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase all --resume
Validate with --verify; honor every existing receipt; never rerun completed fits.
Exactly U0, G1_frozen, G1_R, primary G1_RH, G1_RHM. No lambda1 sensitivity or hyperparameter grid.
Four B inner folds; three secondary scale folds entirely inside B-training. C labels/groups evaluation-only.
Run all preregistered C exploratory diagnostics even if B_DIRECTION_NO_GO; never promote.
Resume all bootstrap/permutation batches with their saved RNG states. Preserve v1-v6 original artifacts.
If DONE_EXPLORATORY, read EXPLORATORY_C_REPORT.md; no fresh data means INDEPENDENT_CONFIRMATION_NOT_PERFORMED.
