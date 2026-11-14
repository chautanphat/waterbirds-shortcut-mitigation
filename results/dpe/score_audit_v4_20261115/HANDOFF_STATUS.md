# Score audit v4 checkpoint

{
  "active_phase": "complete",
  "active_task": null,
  "completed": 225,
  "error": null,
  "failures_resolved": "r1 descriptive NumPy bug fixed in r2; r3 initialization maintenance; all completed tasks retained",
  "full_resume": "PASS",
  "independent_audit": "PASS",
  "last_completed": "final:package",
  "manifest_sha256": "006ba66d8b2cd8ad1efb22b5c9e4eaf50daa847d22f9ff1a8119e48f42ee0148",
  "mean_delta_AA_pp": 0.166805671392826,
  "mean_delta_WGA_pp": -0.5012531328320802,
  "next_pending": "None under frozen v4 protocol; do not integrate or retune",
  "production_integration": "NOT_PERFORMED",
  "promotion_gate": "FAIL",
  "report": "FINAL_RESEARCH_REPORT_r2.md",
  "resume_command": ".venv/Scripts/python.exe -m experiments.score_audit_v4.run --phase all --resume",
  "status": "COMPLETE_GATE_FAILED",
  "updated_utc": "2026-11-14T18:17:38.642747+00:00",
  "package": "DPE_WGA_v4_results.zip"
}

Completion markers and checksums are authoritative. Preserve v1/v2/v3 outputs. Never tune on test/groups. Phase 2 folds were previously inspected in v3; disclose this.
