# v7 delivery: measured mechanism and remaining weakness

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

**B_DIRECTION_NO_GO — EXPLORATORY_C_WGA_EQUAL (primary RH) — INDEPENDENT_CONFIRMATION_NOT_PERFORMED**

The following is a descriptive audit of the already locked results. No candidate, fit, scale or configuration was changed.

B-OOF contains 47 distinct image IDs that are Uniform-wrong in at least one seed after per-seed pooling; per-seed counts are {0: 27, 1: 24, 2: 23}. Overlap means these counts must not be added as independent samples.

Scale matching did have influence: all 36 B-inner pooled/head scales and all nine full-B pooled/head scales passed the fixed numerical scale guard. RH mean B expert-weight total variation is about 0.02573, and residual/geometry centered RMS about 0.36694. The measured failure therefore persists despite a substantially stronger score residual than CCW v6.

B_DIRECTION_DECOMPOSITION.csv contrasts all images, Uniform-wrong and Uniform-correct by true class, with valid per-image ranking counts. It is group-free. Poor ranking/direction on Uniform-wrong images cannot be rescued by average NLL improvement on the entire set. Head/class support was not sparse under the specified <30-image diagnostic, and same-ensemble sign agreement was substantial; these facts do not establish head-aware routing succeeds on the hard subset.

On C, pooled R WGA is +0.250627 pp relative to G1, with paired 95% CI [-0.1430615164520736, 0.751879699248122] pp. The gain comes from one group-2 correction in seed 0. RH also makes that correction but breaks a group-2 image in seed 2, giving no mean WGA gain. All group-2 changes, including breakages, are listed below. All five methods retain group 2 as worst group for every seed.

| Seed | Method | Image ID | Corrected G1 | Broken G1 |
|---:|---|---|---|---|
| 0 | G1_R | 003.Sooty_Albatross/Sooty_Albatross_0034_1154.jpg | True | False |
| 0 | G1_RH | 003.Sooty_Albatross/Sooty_Albatross_0034_1154.jpg | True | False |
| 2 | G1_RH | 143.Caspian_Tern/Caspian_Tern_0006_145594.jpg | False | True |
| 0 | G1_RHM | 003.Sooty_Albatross/Sooty_Albatross_0034_1154.jpg | True | False |
| 2 | G1_RHM | 143.Caspian_Tern/Caspian_Tern_0006_145594.jpg | False | True |

RHM improves C mean AA by 0.027801 pp over RH but leaves the same mean WGA. On B-OOF it has the same classification corrections/breakages as RH while suppressing score magnitude; high-margin B images had no observed correction/breakage for either. This supports neither an AA guarantee nor a replacement of the primary.

No primary swap to pooled R follows its C result. Both pooled and head-aware B direction differences were negative on Uniform-wrong images in all seeds. The residual permutation null frequently matches/exceeds observed RH WGA. An oracle gap remains feasibility evidence only. Exact head identities, ridge fits, B target values and label-independent feature code are all retained for further review.

A later study would need a separately preregistered, B-grounded response to the signed-direction failure. The current result does not justify changing lambda, ridge, shrinkage, features or guard thresholds using these C labels. Existing G1/Uniform behavior and all v1-v6 artifacts remain unchanged.
