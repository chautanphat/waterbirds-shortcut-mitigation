# Phase 0 — geometry/reliability audit

**Exploratory development only. No historical test rows are read. Groups are post-hoc diagnostics of the four fixed scores; never a selection input.**

## Uniform-wrong images: the critical diagnostic

| Seed | Split | Score | Wrong images | Mixed-expert images | Image-mean ranking | 95% CI | Paired Δ versus confidence |
|---|---|---|---:|---:|---:|---|---:|
| 0 | meta_train | negative_min_distance | 52 | 43 | 0.3281 | [0.2678, 0.3854] | +0.0445 |
| 0 | meta_train | negative_entropy | 52 | 43 | 0.2838 | [0.2325, 0.3379] | +0.0002 |
| 0 | meta_train | confidence | 52 | 43 | 0.2836 | [0.2325, 0.3377] | +0.0000 |
| 0 | meta_train | distance_margin | 52 | 43 | 0.2844 | [0.2337, 0.3384] | +0.0009 |
| 0 | meta_select | negative_min_distance | 11 | 11 | 0.4080 | [0.2950, 0.5323] | +0.0730 |
| 0 | meta_select | negative_entropy | 11 | 11 | 0.3351 | [0.2255, 0.4457] | +0.0000 |
| 0 | meta_select | confidence | 11 | 11 | 0.3351 | [0.2255, 0.4457] | +0.0000 |
| 0 | meta_select | distance_margin | 11 | 11 | 0.3351 | [0.2255, 0.4457] | +0.0000 |
| 1 | meta_train | negative_min_distance | 47 | 35 | 0.3119 | [0.2527, 0.3698] | +0.0336 |
| 1 | meta_train | negative_entropy | 47 | 35 | 0.2785 | [0.2226, 0.3390] | +0.0002 |
| 1 | meta_train | confidence | 47 | 35 | 0.2783 | [0.2226, 0.3379] | +0.0000 |
| 1 | meta_train | distance_margin | 47 | 35 | 0.2785 | [0.2226, 0.3380] | +0.0001 |
| 1 | meta_select | negative_min_distance | 15 | 13 | 0.3441 | [0.2391, 0.4544] | +0.0327 |
| 1 | meta_select | negative_entropy | 15 | 13 | 0.3114 | [0.2062, 0.4156] | +0.0000 |
| 1 | meta_select | confidence | 15 | 13 | 0.3114 | [0.2062, 0.4156] | +0.0000 |
| 1 | meta_select | distance_margin | 15 | 13 | 0.3114 | [0.2062, 0.4156] | +0.0000 |
| 2 | meta_train | negative_min_distance | 48 | 37 | 0.3698 | [0.3030, 0.4381] | +0.0289 |
| 2 | meta_train | negative_entropy | 48 | 37 | 0.3409 | [0.2737, 0.4118] | +0.0000 |
| 2 | meta_train | confidence | 48 | 37 | 0.3409 | [0.2737, 0.4118] | +0.0000 |
| 2 | meta_train | distance_margin | 48 | 37 | 0.3428 | [0.2757, 0.4139] | +0.0018 |
| 2 | meta_select | negative_min_distance | 10 | 8 | 0.3749 | [0.2835, 0.4660] | +0.0638 |
| 2 | meta_select | negative_entropy | 10 | 8 | 0.3111 | [0.2647, 0.3632] | +0.0000 |
| 2 | meta_select | confidence | 10 | 8 | 0.3111 | [0.2663, 0.3637] | +0.0000 |
| 2 | meta_select | distance_margin | 10 | 8 | 0.3063 | [0.2601, 0.3600] | -0.0048 |

Distance-routing diagnostic: **NO** under the rule frozen above. The full table contains all-image results, paired difference intervals and subgroup rows; images without both expert outcomes have no pair-ranking estimate and are counted separately. The oracle-any-correct fraction is upper-bound headroom, not evidence of an attainable label-free router.

## Geometry and scaling

Source inspection and actual checkpoint inference verify normalized Euclidean distances after per-image embedding standardization/L2 normalization and prototype L2 normalization. The learnable positive scale differs across actual heads: logits are −abs(scale_j)×distance, not automatically −distance. Entropic scale30 belongs to training only. Per-fold head means, SDs, scale and raw prototype norms appear in head_scale_audit.csv/PNG. Normalized distances share a Euclidean unit but have different head distributions; G2 therefore fits separate location/scale for every actual fold/head using meta_train only.

The distance-margin conditional table checks -dmin ranking within train-derived margin strata on uniform-wrong images. It is descriptive and low support may preclude a strong conclusion about independent information. Binary confidence and negative entropy have theoretically identical ordering apart from finite-precision ties. Probability reconstruction is checked against original logits, not negative distances alone.

## Plots and uncertainty

Every seed/split has distance×confidence and distance×agreement correctness/distance/agreement heatmaps with distinct-image counts, class×distance source tables, correctness/predicted-class distributions and negative-distance/confidence scatter. Quantile edges are fitted on meta_train; asterisks mark <30 distinct images. Group-specific oracle headroom and minority-correct counts are in the ranking table. Bootstrap intervals resample image-level pair statistics, never individual expert rows; repeated images across seeds are not independent backbone replicates.

Continue the bounded preregistered screen regardless of this diagnostic; do not widen the grid or infer robustness from nearest-prototype intuition.
