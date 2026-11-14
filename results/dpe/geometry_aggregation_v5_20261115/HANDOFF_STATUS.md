# DPE geometry v5 - complete

Authoritative report: FINAL_RESEARCH_REPORT.md (also results/dpe/geometry_aggregation_v5_20261115/FINAL_REPORT.md).
Status: COMPLETE. 424 checksum-validated tasks; 135 newly trained heads; 13 acceptance tests passed.
G1_g50 (gamma50, alpha.25): engineering PASS, mean WGA +0.501253 pp, AA +0.111204 pp; WGA positive in2/3 seeds.
S2_t0.025: FAIL (WGA positive in1/3 seeds). No S2 production promotion.
Production opt-in distance mode implemented and tested; Uniform remains default.
WGA95% CI [-0.214660,+1.253133] pp crosses zero. Reused development, one shared ERM* backbone; no fresh test claim.
All completed hashes validated. Full separate-process resume recomputed zero experiments.
Windows/OneDrive rename failures were recovered with atomic retry in source revision r2; r3 adds conditional production acceptance. Old manifests/sources/results are preserved.

Resume from repository root:
.venv/Scripts/python.exe -m experiments.geometry_aggregation_v5.run --phase all --resume
Packaging/report refresh:
.venv/Scripts/python.exe -m scripts.report_geometry_aggregation_v5

No experiment or integration work remains under this frozen protocol. Do not retune from existing C/test outcomes.
