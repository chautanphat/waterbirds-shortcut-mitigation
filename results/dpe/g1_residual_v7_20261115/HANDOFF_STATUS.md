# G1 residual v7 checkpoint

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

{
  "B_decision": {
    "AA_floor_every_seed": true,
    "C_permitted_regardless": true,
    "direction_pass": false,
    "gain_CI95": [
      -0.002486722185398428,
      -0.0012001044465132853
    ],
    "mean_gain_difference_Uwrong": -0.00186525055530677,
    "mean_worst_class_NLL": {
      "G1_R": 0.6192495454227579,
      "G1_RH": 0.6191920585331361,
      "G1_RHM": 0.6202393594825689,
      "G1_frozen": 0.6206343304304318,
      "U0": 0.6360100111456725
    },
    "per_seed_delta_AA_G1_pp": [
      0.16313213703099683,
      -0.1615508885298822,
      0.3252032520325243
    ],
    "per_seed_delta_AA_U0_pp": [
      -0.16313213703099683,
      -0.3231017770597755,
      0.1626016260162566
    ],
    "per_seed_gain_difference": [
      -0.002206993463940641,
      -0.0018597604226813864,
      -0.001528997779298282
    ],
    "positive_seeds": 0,
    "primary": "G1_RH",
    "promotion": false,
    "proxy_pass": true,
    "proxy_tolerance": 0.003,
    "role": "EXPLORATORY_ONLY \u2014 REUSED DEVELOPMENT \u2014 NOT INDEPENDENT CONFIRMATION",
    "status": "B_DIRECTION_NO_GO",
    "uncertainty": "DIRECTIONAL_IMAGE_BOOTSTRAP_INTERVAL",
    "worst_class_NLL_delta": -0.0014422718972957327
  },
  "active_phase": "complete",
  "active_task": null,
  "best_B_only_observation": {
    "AA_floor_every_seed": true,
    "C_permitted_regardless": true,
    "direction_pass": false,
    "gain_CI95": [
      -0.002486722185398428,
      -0.0012001044465132853
    ],
    "mean_gain_difference_Uwrong": -0.00186525055530677,
    "mean_worst_class_NLL": {
      "G1_R": 0.6192495454227579,
      "G1_RH": 0.6191920585331361,
      "G1_RHM": 0.6202393594825689,
      "G1_frozen": 0.6206343304304318,
      "U0": 0.6360100111456725
    },
    "per_seed_delta_AA_G1_pp": [
      0.16313213703099683,
      -0.1615508885298822,
      0.3252032520325243
    ],
    "per_seed_delta_AA_U0_pp": [
      -0.16313213703099683,
      -0.3231017770597755,
      0.1626016260162566
    ],
    "per_seed_gain_difference": [
      -0.002206993463940641,
      -0.0018597604226813864,
      -0.001528997779298282
    ],
    "positive_seeds": 0,
    "primary": "G1_RH",
    "promotion": false,
    "proxy_pass": true,
    "proxy_tolerance": 0.003,
    "role": "EXPLORATORY_ONLY \u2014 REUSED DEVELOPMENT \u2014 NOT INDEPENDENT CONFIRMATION",
    "status": "B_DIRECTION_NO_GO",
    "uncertainty": "DIRECTIONAL_IMAGE_BOOTSTRAP_INTERVAL",
    "worst_class_NLL_delta": -0.0014422718972957327
  },
  "completed": 591,
  "error": null,
  "independent_confirmation": "INDEPENDENT_CONFIRMATION_NOT_PERFORMED",
  "last_completed": "delivery_package#5429610d39bbf2e13949",
  "manifest_sha256": "b4460f6681bd3eddc29356a8ce10deb5ca0b29962d0d06aa37ce68f4f8bdc4c8",
  "next_pending": "All existing-data research and delivery complete; read report/supplement; no tuning or promotion.",
  "original_v6_NO_GO": "UNCHANGED",
  "package": "experiments\\g1_residual_v7\\DPE_G1_Residual_v7_Results.zip",
  "production_promotion": false,
  "report": "experiments\\g1_residual_v7\\EXPLORATORY_C_REPORT.md",
  "resume_command": ".venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase all --resume",
  "role": "EXPLORATORY_ONLY \u2014 REUSED DEVELOPMENT \u2014 NOT INDEPENDENT CONFIRMATION",
  "status": "DONE_EXPLORATORY",
  "supplement": "experiments\\g1_residual_v7\\DELIVERY_SUPPLEMENT.md",
  "updated_utc": "2026-11-15T04:25:08.338854+00:00"
}

Task receipts are authoritative. All feature statistics and scale CV fit only the owning B training IDs. Preserve v1-v6; never tune from C.

.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase all --resume
