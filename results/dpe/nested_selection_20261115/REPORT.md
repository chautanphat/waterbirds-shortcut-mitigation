# Nested selection results

Cells: 70 (source x split). Splits present: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Primary selection rule: `nll`; secondary: `wga`. Test WGA in %, differences in percentage points (pp).
Intervals are pointwise 95% t-intervals over sources of the per-source mean over split seeds; ERM* has two sources, so it is descriptive only. The test set and the seven backbones were inspected before this run.

## Mean test WGA

| arm | normal_erm (n=5) | ermstar (n=2) |
| --- | ---: | ---: |
| `fullval_uniform` | 89.88 | 93.88 |
| `bag75_uniform` | 89.94 | 94.05 |
| `fullval_thrwga_oof` | 89.88 | 93.88 |
| `fullval_threshold_oof_fixed` | 89.59 | 93.57 |
| `fullval_threshold_oof_sel_nll` | 89.31 | 93.39 |
| `fullval_affine_oof_fixed` | 89.59 | 93.57 |
| `fullval_affine_oof_sel_nll` | 89.31 | 93.30 |
| `crossfit_uniform` | 90.01 | 94.06 |
| `crossfit_thrwga` | 90.01 | 94.06 |
| `crossfit_threshold_fixed` | 90.05 | 93.99 |
| `crossfit_threshold_sel_nll` | 89.77 | 93.86 |
| `crossfit_affine_fixed` | 90.05 | 93.99 |
| `crossfit_affine_sel_nll` | 89.71 | 93.79 |
| `crossfit_affine_sel_wga` | 89.80 | 93.90 |
| `pool_uniform` | 89.51 | 93.59 |
| `pool_affine_fixed` | 90.04 | 93.78 |
| `pool_affine_sel_nll` | 89.83 | 93.64 |
| `pool_affine_sel_wga` | 89.81 | 93.68 |
| `pool_linear_fixed` | 90.19 | 93.56 |
| `pool_linear_sel_nll` | 89.02 | 93.12 |
| `pool_linear_sel_wga` | 89.79 | 93.27 |
| `dfr_lite` | 88.37 | 92.60 |
| `crossfit_threshold_sel_wga` | 89.88 | 93.92 |
| `fullval_affine_oof_sel_wga` | 89.40 | 93.44 |
| `fullval_threshold_oof_sel_wga` | 89.40 | 93.45 |
| `fullval_saved` (paper-style, seed fixed) | 89.60 | 94.08 |

## Reproduction of the paper's numbers (test WGA, %)

| recipe | sources | backbone | backbone alone (ours) | backbone alone (paper) | with DPE | with DPE (ours) | with DPE (paper) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_erm | 5 | ERM | n/a (backbone not retrained here) | 69.1 | full-validation DPE, heads retrained per split seed | 89.88 (SD over sources 0.29) | 91.0 |
| ermstar | 2 | ERM* | n/a (backbone not retrained here) | 77.9 | full-validation DPE, heads retrained per split seed | 93.88 (SD over sources 0.11) | 94.1 |

## Paired differences (pp)

### normal_erm

| comparison | mean | 95% t-interval | sources > 0 | per source |
| --- | ---: | ---: | ---: | ---: |
| `crossfit_affine_sel_nll` - `fullval_uniform` | -0.17 | [-0.33, -0.02] | 0/5 | -0.23, -0.06, -0.28, -0.28, -0.01 |
| `crossfit_affine_sel_nll` - `bag75_uniform` | -0.23 | [-0.38, -0.08] | 0/5 | -0.34, -0.09, -0.37, -0.16, -0.19 |
| `crossfit_affine_sel_wga` - `fullval_uniform` | -0.08 | [-0.17, +0.01] | 1/5 | -0.09, -0.06, -0.16, -0.14, +0.03 |
| `crossfit_affine_sel_wga` - `bag75_uniform` | -0.14 | [-0.25, -0.03] | 0/5 | -0.20, -0.09, -0.25, -0.02, -0.14 |
| `crossfit_affine_fixed` - `fullval_uniform` | +0.17 | [-0.01, +0.34] | 4/5 | +0.22, +0.29, +0.19, -0.07, +0.20 |
| `crossfit_affine_fixed` - `bag75_uniform` | +0.11 | [-0.01, +0.22] | 5/5 | +0.11, +0.26, +0.09, +0.05, +0.03 |
| `crossfit_uniform` - `fullval_uniform` | +0.13 | [-0.28, +0.53] | 4/5 | +0.08, -0.33, +0.59, +0.10, +0.20 |
| `crossfit_uniform` - `bag75_uniform` | +0.07 | [-0.32, +0.46] | 3/5 | -0.04, -0.36, +0.50, +0.22, +0.03 |
| `pool_affine_sel_nll` - `fullval_uniform` | -0.05 | [-0.17, +0.06] | 2/5 | -0.07, -0.13, +0.05, -0.15, +0.05 |
| `pool_affine_sel_nll` - `bag75_uniform` | -0.11 | [-0.20, -0.02] | 0/5 | -0.19, -0.16, -0.05, -0.03, -0.13 |
| `pool_linear_sel_nll` - `fullval_uniform` | -0.86 | [-1.32, -0.39] | 0/5 | -0.87, -1.37, -1.03, -0.63, -0.39 |
| `pool_linear_sel_nll` - `bag75_uniform` | -0.92 | [-1.38, -0.45] | 0/5 | -0.98, -1.40, -1.12, -0.51, -0.56 |
| `dfr_lite` - `fullval_uniform` | -1.51 | [-1.83, -1.20] | 0/5 | -1.92, -1.43, -1.53, -1.47, -1.23 |
| `dfr_lite` - `bag75_uniform` | -1.57 | [-1.92, -1.23] | 0/5 | -2.04, -1.46, -1.62, -1.35, -1.40 |
| `crossfit_affine_sel_nll` - `crossfit_uniform` | -0.30 | [-0.80, +0.20] | 1/5 | -0.31, +0.27, -0.87, -0.37, -0.22 |
| `crossfit_affine_sel_nll` - `crossfit_affine_fixed` | -0.34 | [-0.49, -0.18] | 0/5 | -0.45, -0.35, -0.47, -0.20, -0.22 |
| `crossfit_affine_fixed` - `crossfit_uniform` | +0.04 | [-0.44, +0.52] | 3/5 | +0.15, +0.62, -0.40, -0.17, +0.00 |
| `pool_linear_sel_nll` - `pool_affine_sel_nll` | -0.81 | [-1.24, -0.37] | 0/5 | -0.79, -1.24, -1.07, -0.49, -0.43 |
| `crossfit_affine_sel_nll` - `dfr_lite` | +1.34 | [+1.08, +1.60] | 5/5 | +1.70, +1.37, +1.25, +1.19, +1.21 |
| `crossfit_affine_sel_nll` - `fullval_threshold_oof_sel_nll` | +0.40 | [+0.17, +0.63] | 5/5 | +0.50, +0.31, +0.20, +0.33, +0.67 |
| `crossfit_affine_sel_nll` - `fullval_affine_oof_sel_nll` | +0.40 | [+0.16, +0.65] | 5/5 | +0.47, +0.33, +0.16, +0.37, +0.70 |
| `crossfit_affine_sel_nll` - `crossfit_threshold_sel_nll` | -0.06 | [-0.11, -0.02] | 0/5 | -0.08, -0.06, -0.11, -0.05, -0.02 |
| `crossfit_affine_sel_nll` - `fullval_thrwga_oof` | -0.17 | [-0.33, -0.02] | 0/5 | -0.23, -0.06, -0.28, -0.28, -0.01 |
| `fullval_threshold_oof_sel_nll` - `fullval_uniform` | -0.57 | [-0.76, -0.39] | 0/5 | -0.73, -0.37, -0.48, -0.60, -0.68 |
| `fullval_affine_oof_sel_nll` - `fullval_uniform` | -0.58 | [-0.77, -0.39] | 0/5 | -0.70, -0.39, -0.44, -0.65, -0.71 |
| `fullval_thrwga_oof` - `fullval_uniform` | +0.00 | [+0.00, +0.00] | 0/5 | +0.00, +0.00, +0.00, +0.00, +0.00 |
| `crossfit_threshold_sel_nll` - `fullval_uniform` | -0.11 | [-0.24, +0.02] | 2/5 | -0.15, +0.00, -0.17, -0.23, +0.00 |
| `crossfit_thrwga` - `fullval_uniform` | +0.13 | [-0.28, +0.53] | 4/5 | +0.08, -0.33, +0.59, +0.10, +0.20 |
| `crossfit_affine_sel_nll` - saved full-val DPE | +0.12 | [-0.30, +0.53] | 4/5 | +0.16, -0.44, +0.16, +0.47, +0.23 |
| `crossfit_affine_fixed` - saved full-val DPE | +0.45 | [+0.07, +0.84] | 4/5 | +0.61, -0.08, +0.62, +0.67, +0.45 |
| `crossfit_uniform` - saved full-val DPE | +0.41 | [-0.42, +1.25] | 4/5 | +0.46, -0.70, +1.03, +0.84, +0.45 |
| `dfr_lite` - saved full-val DPE | -1.23 | [-1.77, -0.69] | 0/5 | -1.54, -1.80, -1.09, -0.72, -0.98 |

### ermstar

| comparison | mean | 95% t-interval | sources > 0 | per source |
| --- | ---: | ---: | ---: | ---: |
| `crossfit_affine_sel_nll` - `fullval_uniform` | -0.08 | [n/a, n/a] | 1/2 | -0.34, +0.17 |
| `crossfit_affine_sel_nll` - `bag75_uniform` | -0.26 | [n/a, n/a] | 0/2 | -0.33, -0.18 |
| `crossfit_affine_sel_wga` - `fullval_uniform` | +0.03 | [n/a, n/a] | 1/2 | -0.18, +0.23 |
| `crossfit_affine_sel_wga` - `bag75_uniform` | -0.15 | [n/a, n/a] | 0/2 | -0.17, -0.12 |
| `crossfit_affine_fixed` - `fullval_uniform` | +0.11 | [n/a, n/a] | 1/2 | -0.10, +0.33 |
| `crossfit_affine_fixed` - `bag75_uniform` | -0.06 | [n/a, n/a] | 0/2 | -0.09, -0.03 |
| `crossfit_uniform` - `fullval_uniform` | +0.18 | [n/a, n/a] | 2/2 | +0.18, +0.19 |
| `crossfit_uniform` - `bag75_uniform` | +0.01 | [n/a, n/a] | 1/2 | +0.19, -0.17 |
| `pool_affine_sel_nll` - `fullval_uniform` | -0.24 | [n/a, n/a] | 0/2 | -0.38, -0.10 |
| `pool_affine_sel_nll` - `bag75_uniform` | -0.41 | [n/a, n/a] | 0/2 | -0.37, -0.45 |
| `pool_linear_sel_nll` - `fullval_uniform` | -0.76 | [n/a, n/a] | 0/2 | -0.90, -0.62 |
| `pool_linear_sel_nll` - `bag75_uniform` | -0.93 | [n/a, n/a] | 0/2 | -0.89, -0.98 |
| `dfr_lite` - `fullval_uniform` | -1.28 | [n/a, n/a] | 0/2 | -1.44, -1.11 |
| `dfr_lite` - `bag75_uniform` | -1.45 | [n/a, n/a] | 0/2 | -1.43, -1.47 |
| `crossfit_affine_sel_nll` - `crossfit_uniform` | -0.27 | [n/a, n/a] | 0/2 | -0.51, -0.02 |
| `crossfit_affine_sel_nll` - `crossfit_affine_fixed` | -0.20 | [n/a, n/a] | 0/2 | -0.23, -0.16 |
| `crossfit_affine_fixed` - `crossfit_uniform` | -0.07 | [n/a, n/a] | 1/2 | -0.28, +0.14 |
| `pool_linear_sel_nll` - `pool_affine_sel_nll` | -0.52 | [n/a, n/a] | 0/2 | -0.51, -0.53 |
| `crossfit_affine_sel_nll` - `dfr_lite` | +1.19 | [n/a, n/a] | 2/2 | +1.11, +1.28 |
| `crossfit_affine_sel_nll` - `fullval_threshold_oof_sel_nll` | +0.40 | [n/a, n/a] | 2/2 | +0.33, +0.48 |
| `crossfit_affine_sel_nll` - `fullval_affine_oof_sel_nll` | +0.49 | [n/a, n/a] | 2/2 | +0.42, +0.56 |
| `crossfit_affine_sel_nll` - `crossfit_threshold_sel_nll` | -0.07 | [n/a, n/a] | 0/2 | -0.05, -0.09 |
| `crossfit_affine_sel_nll` - `fullval_thrwga_oof` | -0.08 | [n/a, n/a] | 1/2 | -0.34, +0.17 |
| `fullval_threshold_oof_sel_nll` - `fullval_uniform` | -0.49 | [n/a, n/a] | 0/2 | -0.67, -0.31 |
| `fullval_affine_oof_sel_nll` - `fullval_uniform` | -0.57 | [n/a, n/a] | 0/2 | -0.76, -0.39 |
| `fullval_thrwga_oof` - `fullval_uniform` | +0.00 | [n/a, n/a] | 0/2 | +0.00, +0.00 |
| `crossfit_threshold_sel_nll` - `fullval_uniform` | -0.01 | [n/a, n/a] | 1/2 | -0.29, +0.26 |
| `crossfit_thrwga` - `fullval_uniform` | +0.18 | [n/a, n/a] | 2/2 | +0.18, +0.19 |
| `crossfit_affine_sel_nll` - saved full-val DPE | -0.29 | [n/a, n/a] | 1/2 | -0.78, +0.20 |
| `crossfit_affine_fixed` - saved full-val DPE | -0.09 | [n/a, n/a] | 1/2 | -0.55, +0.36 |
| `crossfit_uniform` - saved full-val DPE | -0.02 | [n/a, n/a] | 1/2 | -0.26, +0.22 |
| `dfr_lite` - saved full-val DPE | -1.48 | [n/a, n/a] | 0/2 | -1.88, -1.08 |

## Every setting, no selection (mean test WGA, %)

| family and setting | normal_erm | ermstar |
| --- | ---: | ---: |
| crossfit_affine average-l0.1 | 90.05 | 93.99 |
| crossfit_affine maximum-l0.1 | 89.43 | 93.77 |
| crossfit_affine average-l0.01 | 89.71 | 93.79 |
| crossfit_affine maximum-l0.01 | 89.52 | 93.82 |
| crossfit_threshold average-l0.1 | 90.05 | 93.99 |
| crossfit_threshold maximum-l0.1 | 89.43 | 93.79 |
| crossfit_threshold average-l0.01 | 89.70 | 93.79 |
| crossfit_threshold maximum-l0.01 | 89.43 | 93.79 |
| fullval_affine_oof average-l0.1 | 89.59 | 93.57 |
| fullval_affine_oof maximum-l0.1 | 88.97 | 93.31 |
| fullval_affine_oof average-l0.01 | 89.31 | 93.30 |
| fullval_affine_oof maximum-l0.01 | 89.10 | 93.40 |
| fullval_threshold_oof average-l0.1 | 89.59 | 93.57 |
| fullval_threshold_oof maximum-l0.1 | 89.00 | 93.30 |
| fullval_threshold_oof average-l0.01 | 89.30 | 93.30 |
| fullval_threshold_oof maximum-l0.01 | 89.00 | 93.30 |
| pool_affine average-l0.1 | 90.04 | 93.78 |
| pool_affine maximum-l0.1 | 89.31 | 93.61 |
| pool_affine average-l0.01 | 89.83 | 93.64 |
| pool_affine maximum-l0.01 | 89.34 | 93.62 |
| pool_linear average-l0.1 | 90.19 | 93.56 |
| pool_linear maximum-l0.1 | 88.99 | 93.43 |
| pool_linear average-l0.01 | 89.02 | 93.12 |
| pool_linear maximum-l0.01 | 88.69 | 93.34 |

## What validation picked (share of source x split cells)

| arm | recipe | cells | average-l0.1 | maximum-l0.1 | average-l0.01 | maximum-l0.01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crossfit_affine_sel_nll` | normal_erm | 50 | 0% | 0% | 100% | 0% |
| `crossfit_affine_sel_nll` | ermstar | 20 | 0% | 0% | 100% | 0% |
| `crossfit_affine_sel_wga` | normal_erm | 50 | 26% | 0% | 74% | 0% |
| `crossfit_affine_sel_wga` | ermstar | 20 | 30% | 0% | 50% | 20% |
| `crossfit_threshold_sel_nll` | normal_erm | 50 | 34% | 24% | 40% | 2% |
| `crossfit_threshold_sel_nll` | ermstar | 20 | 35% | 15% | 50% | 0% |
| `crossfit_threshold_sel_wga` | normal_erm | 50 | 54% | 10% | 34% | 2% |
| `crossfit_threshold_sel_wga` | ermstar | 20 | 50% | 15% | 35% | 0% |
| `pool_affine_sel_nll` | normal_erm | 50 | 0% | 0% | 100% | 0% |
| `pool_affine_sel_nll` | ermstar | 20 | 0% | 0% | 100% | 0% |
| `pool_affine_sel_wga` | normal_erm | 50 | 4% | 0% | 78% | 18% |
| `pool_affine_sel_wga` | ermstar | 20 | 10% | 0% | 80% | 10% |
| `pool_linear_sel_nll` | normal_erm | 50 | 0% | 0% | 100% | 0% |
| `pool_linear_sel_nll` | ermstar | 20 | 0% | 0% | 100% | 0% |
| `pool_linear_sel_wga` | normal_erm | 50 | 54% | 2% | 38% | 6% |
| `pool_linear_sel_wga` | ermstar | 20 | 15% | 5% | 50% | 30% |

## Split-seed noise (mean over sources of the SD of test WGA across split seeds, pp)

| recipe | arm | SD |
| --- | ---: | ---: |
| normal_erm | `fullval_uniform` | 0.40 |
| normal_erm | `crossfit_affine_sel_nll` | 0.25 |
| normal_erm | `crossfit_uniform` | 0.57 |
| ermstar | `fullval_uniform` | 0.33 |
| ermstar | `crossfit_affine_sel_nll` | 0.18 |
| ermstar | `crossfit_uniform` | 0.34 |

## Split 0 versus the earlier saved cross-fit run (fold seed 0, original head seeds)

Small differences are expected if the earlier run used a GPU; large ones mean the pipeline is not reproducing it.

| source | quantity | this run | earlier | delta |
| --- | ---: | ---: | ---: | ---: |
| ermstar-seed0 | crossfit_uniform | 94.39 | 94.39 | +0.00 |
| ermstar-seed0 | crossfit_affine/average-l0.1 | 94.24 | 94.24 | +0.00 |
| ermstar-seed0 | crossfit_affine/maximum-l0.1 | 93.77 | 93.77 | +0.00 |
| ermstar-seed0 | crossfit_affine/average-l0.01 | 93.93 | 93.93 | -0.00 |
| ermstar-seed0 | crossfit_affine/maximum-l0.01 | 93.77 | 93.77 | +0.00 |
| ermstar-seed1 | crossfit_uniform | 92.82 | 92.82 | -0.00 |
| ermstar-seed1 | crossfit_affine/average-l0.1 | 93.79 | 93.79 | +0.00 |
| ermstar-seed1 | crossfit_affine/maximum-l0.1 | 94.06 | 94.06 | -0.00 |
| ermstar-seed1 | crossfit_affine/average-l0.01 | 93.93 | 93.93 | -0.00 |
| ermstar-seed1 | crossfit_affine/maximum-l0.01 | 94.08 | 94.08 | -0.00 |
| normal-erm-seed0 | crossfit_uniform | 90.73 | 90.73 | +0.00 |
| normal-erm-seed0 | crossfit_affine/average-l0.1 | 90.34 | 90.34 | +0.00 |
| normal-erm-seed0 | crossfit_affine/maximum-l0.1 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed0 | crossfit_affine/average-l0.01 | 89.56 | 89.56 | -0.00 |
| normal-erm-seed0 | crossfit_affine/maximum-l0.01 | 89.56 | 89.56 | -0.00 |
| normal-erm-seed1 | crossfit_uniform | 90.11 | 90.11 | -0.00 |
| normal-erm-seed1 | crossfit_affine/average-l0.1 | 90.97 | 90.97 | +0.00 |
| normal-erm-seed1 | crossfit_affine/maximum-l0.1 | 89.72 | 89.72 | +0.00 |
| normal-erm-seed1 | crossfit_affine/average-l0.01 | 90.50 | 90.50 | +0.00 |
| normal-erm-seed1 | crossfit_affine/maximum-l0.01 | 90.19 | 90.19 | -0.00 |
| normal-erm-seed2 | crossfit_uniform | 90.03 | 90.03 | -0.00 |
| normal-erm-seed2 | crossfit_affine/average-l0.1 | 89.72 | 89.72 | +0.00 |
| normal-erm-seed2 | crossfit_affine/maximum-l0.1 | 89.25 | 89.25 | +0.00 |
| normal-erm-seed2 | crossfit_affine/average-l0.01 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed2 | crossfit_affine/maximum-l0.01 | 89.25 | 89.25 | +0.00 |
| normal-erm-seed3 | crossfit_uniform | 89.36 | 89.36 | +0.00 |
| normal-erm-seed3 | crossfit_affine/average-l0.1 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed3 | crossfit_affine/maximum-l0.1 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed3 | crossfit_affine/average-l0.01 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed3 | crossfit_affine/maximum-l0.01 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed4 | crossfit_uniform | 90.47 | 90.47 | -0.00 |
| normal-erm-seed4 | crossfit_affine/average-l0.1 | 89.88 | 89.88 | +0.00 |
| normal-erm-seed4 | crossfit_affine/maximum-l0.1 | 89.41 | 89.41 | -0.00 |
| normal-erm-seed4 | crossfit_affine/average-l0.01 | 89.56 | 89.56 | -0.00 |
| normal-erm-seed4 | crossfit_affine/maximum-l0.01 | 89.56 | 89.56 | -0.00 |
