# Phase 0 — D19 mechanism audit

**HISTORICAL TEST / DIAGNOSTIC ONLY. No fitting or candidate selection uses these rows.**

Exact raw logits were recovered from the saved direct-softmax D19 checkpoints and the saved calibrated probabilities; forward weights match cached weights. Centered logits remove the common within-image offset. Raw and calibrated confidence views have separately predeclared fixed edges. All expert rows, counts, source tables, PNGs and checkpoint hashes are under phase0/seed*/.

| Seed | Image-mean pair ranking | Mean w correct | Mean w wrong | Effective experts | Corrected | Broken | Minority wrong overturn |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7609 | 0.06853 | 0.04587 | 3.07 | 121 | 132 | 128 |
| 1 | 0.8232 | 0.06961 | 0.03525 | 2.89 | 94 | 76 | 74 |
| 2 | 0.8257 | 0.07023 | 0.03142 | 2.50 | 100 | 63 | 62 |

Changed labels Waterbird→Landbird: 586/586 across seed/image rows. Seeds share images and backbone; this is not an independent sample count.

| Hypothesis | Conclusion | Evidence / limit |
|---|---|---|
| H1 predicted-class bias | Supported descriptively | Flip direction table and class/confidence score panels show asymmetric preference. This is not a causal intervention. |
| H2 preference/correctness misalignment | Assessed by ranking and reliability panels | The exact per-seed image-mean ranking, AUROC, Spearman and score-decile correctness tables quantify misalignment; relative preference is not a calibrated correctness probability. A nonmonotonic confidence curve alone is not a bug. |
| H3 softmax over-concentration | Supported as a mechanism; causality undetermined | Effective expert count, incorrect-head mass and red minority-overturn cases show where concentrated final weights reverse a correct majority. Calibration and preference are separate contributors; raw and calibrated uniform controls are both saved. |

All 8 visualization families are saved per seed: class/confidence score, correctness, weight and support; class/confidence/agreement score/correctness/support; score-decile correctness with 95% distinct-image bootstrap intervals; subgroup wrong-weight/effective-count/entropy and flip tables; maximum-weight/consensus scatter and histogram. An asterisk marks fewer than 30 distinct images. Expert AUROC/Spearman are descriptive correlated-observation summaries, with no independent-expert significance claim. ECE uses 15 confidence bins weighted by images; NLL averages one ensemble true-class log probability per image.

Phase 1 code has no import or read path into these diagnostic rows. D19 is not rescored on development because its final heads differ from fold-specific OOF heads; those would not be an honest matched weighting comparison. D5 historical aggregate remains a diagnostic control only.
