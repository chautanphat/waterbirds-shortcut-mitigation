# Phase 1 — exploratory group-blind selection

Six maps screened at alpha=.10/beta=1; two maps shortlisted by mean worst-class NLL, then six alpha/beta combinations per map. Per-seed AA decline must be strictly below 1 pp. No group WGA enters ranking.

Locked primary: {
  "map_id": "R4",
  "alpha": 0.25,
  "beta": 2.0,
  "eligible": true,
  "proxy": 0.15791577138978516,
  "mean_NLL": 0.1445257203796775,
  "delta_AA_pp_by_seed": [
    0.4166666666666652,
    0.4166666666666652,
    0.0
  ]
}

All losing candidates and matched representation-specific uniform controls appear in phase1_all.csv. Raw confidence >=.75 is empty in all training seeds; coverage tables give distinct image counts, wrong-tail counts and all transformed scales. R4 fits shared temperature on original logits once, using meta_train member NLL only. R5 quantiles and all score bins/tables fit meta_train only.

Group metrics were computed only after phase1:lock. meta_select has prior exploratory inspection and OOF heads differ across folds; pooled lookup has no expert identity. Confidence intervals resample distinct images within subgroup, conditional on fitted pipelines; no 15-fold sample-size claim. Seed splits are not pooled for fitting or significance.
