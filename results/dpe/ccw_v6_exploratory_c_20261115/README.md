# CCW v6 archived exploratory results

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

Read [the branch research disposition](../../../DPE_V4_RESEARCH_STATUS.md) and [the full report](EXPLORATORY_C_REPORT.md). The original B NO_GO remains unchanged. Broader variant exploration is on hold; the next research direction is direct G1 failure analysis.

The NPZ contains all five methods, three seeds, 1199 aligned image IDs, probabilities, all 15 expert weights and evaluation labels/groups. Head order is head_sha256[seed, fold[seed,image], expert]. See PREDICTION_SCHEMA.json. Small fitted tables and all original completion receipts are retained; receipt artifact paths refer to the original local run, not a complete standalone resume bundle. Original images, per-fold duplicate caches, trained heads, acceptance controls and result ZIPs are excluded.

The checkpoint state and handoffs are historical export snapshots. They do not mean a clone contains the omitted local inputs. Run scripts/export_ccw_v6_branch.py only from a workspace containing the completed original study.
