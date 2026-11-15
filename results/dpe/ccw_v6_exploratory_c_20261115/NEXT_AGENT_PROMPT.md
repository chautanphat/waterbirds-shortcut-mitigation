Continue only the explicitly requested CCW exploratory C diagnostic.
Read experiments/ccw_v6/exploratory_c/state.json, PROTOCOL_DECISIONS.md and HANDOFF_STATUS.md.
Run .venv/Scripts/python.exe -m experiments.ccw_v6.run --phase exploratory_c --resume
Reuse all valid completion receipts and source/input/config fingerprints; never repeat completed fits.
Original experiments/ccw_v6/LOCKED_CANDIDATE.json must remain byte-identical NO_GO.
Fit only B; C inference accepts probabilities/distances and B metadata, never labels/groups.
Use C labels/groups only after predictions commit. Preserve the full exploratory banner.
Exactly five frozen methods; no parameter optimization or production promotion.
If complete, read EXPLORATORY_C_REPORT.md and do not rerun scientific tasks.
