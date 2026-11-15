# G1 baseline / G1 + R reference evidence

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

Read [the branch report](../../../DPE_V5_G1_BASELINE_AND_R_REFERENCE.md), [original v7 report](EXPLORATORY_C_REPORT.md) and [mechanism supplement](DELIVERY_SUPPLEMENT.md). G1 is the branch comparison baseline; pooled G1_R is the reference method. The preregistered v7 primary remains RH and its original B_DIRECTION_NO_GO / EXPLORATORY_C_WGA_EQUAL verdict is unchanged.

metrics/ contains all required result tables and the curated C decomposition NPZ with all five methods, 15 expert weights and probabilities. C_EVALUATION_LABELS.npz is a separate evaluator-only output. All 180 small ridge/head model JSONs, head-effect tables, split provenance and historical completion receipts are included. Checksums and receipts describe original artifacts, including dependencies intentionally omitted from this Git export. This is an archival export, not a complete standalone resume bundle. Inputs, prototype/backbone weights, individual inference/OOF duplicates, uncertainty batch arrays and ZIPs remain in the original workspace. Exported state/handoffs are snapshots; cloning does not restore the omitted local data.

No experiments were rerun. scripts/export_g1_residual_v7_branch.py verifies source receipts; --verify-index checks the staged Git blobs byte-for-byte against EXPORT_MANIFEST.json.
