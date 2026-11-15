# Explicitly authorized separate research-only diagnostic

EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION

{
  "config": {
    "schema_version": 1,
    "rho": 0.5,
    "beta": 1.0,
    "gamma": 50.0,
    "alpha": 0.25,
    "kappa": 25.0,
    "sigma_floor": 0.02,
    "std_ddof": 0,
    "z_quantiles": [
      0.3333333333333333,
      0.6666666666666666
    ],
    "boundary_quantile": 0.2
  },
  "methods": [
    "U0",
    "G1_frozen",
    "G2_z",
    "CCW_A",
    "CCW_FULL"
  ],
  "bootstrap": {
    "repeats": 2000,
    "seed": 20261115,
    "unit": "same unique image resampled across all seeds, paired, group-stratified; fixed fitted pipelines"
  },
  "production_promotion": false,
  "original_gate_modified": false
}

All B examples fit each frozen full-B model. The original recipe uses distance-z bins, predicted class and disagreement with Uniform; its B-only boundary confidence quantile weights fitting examples. No extra confidence binning or search is introduced. C labels/groups are evaluation-only. The original NO_GO remains authoritative for promotion.
