# Corrected protocol diagnostics

Protocol: `strict-nested-postmean-v2`.
Completed: 70/70 planned source × split cells.
All declared cells completed.
Primary rule remains balanced NLL; WGA is secondary. Prior test inspection makes this exploratory.
Strict inner CV rebuilds all heads and calibrators inside each training fold and evaluates the complete inference graph.
`postmean` applies one scalar threshold/bias AFTER uniform probability averaging.
`bias_foldwise` applies a bias before averaging and is a distinct predictor.
The full-validation OOF calibrator transfer is evaluated separately inside inner CV.
Original pooled 60/40 affine/linear controls are retained; DFR preprocessing is fitted inside CV folds.

## Mean test WGA and overall accuracy

Each source is first averaged across its available split seeds, then sources are weighted equally. SD is across those source means.

| arm | normal ERM WGA ± source SD | normal ERM OA | ERM* WGA ± source SD | ERM* OA |
| --- | ---: | ---: | ---: | ---: |
| bag75_uniform | 89.941 ± 0.258 (n=5) | 92.022 | 94.048 ± 0.152 (n=2) | 95.897 |
| crossfit_affine_fixed | 90.048 ± 0.334 (n=5) | 91.748 | 93.989 ± 0.200 (n=2) | 96.000 |
| crossfit_affine_sel_nll | 89.710 ± 0.344 (n=5) | 92.048 | 93.793 ± 0.253 (n=2) | 96.162 |
| crossfit_affine_sel_wga | 89.801 ± 0.314 (n=5) | 92.000 | 93.793 ± 0.231 (n=2) | 96.138 |
| crossfit_bias_foldwise_fixed | 90.048 ± 0.334 (n=5) | 91.748 | 93.989 ± 0.200 (n=2) | 96.002 |
| crossfit_bias_foldwise_sel_nll | 89.709 ± 0.319 (n=5) | 92.096 | 93.793 ± 0.253 (n=2) | 96.182 |
| crossfit_bias_foldwise_sel_wga | 89.837 ± 0.287 (n=5) | 91.975 | 93.816 ± 0.220 (n=2) | 96.156 |
| crossfit_postmean_bias_fixed | 90.048 ± 0.334 (n=5) | 91.748 | 93.989 ± 0.200 (n=2) | 96.002 |
| crossfit_postmean_bias_sel_nll | 89.709 ± 0.319 (n=5) | 92.096 | 93.793 ± 0.253 (n=2) | 96.182 |
| crossfit_postmean_bias_sel_wga | 89.837 ± 0.287 (n=5) | 91.975 | 93.816 ± 0.220 (n=2) | 96.156 |
| crossfit_postmean_thrwga | 89.378 ± 0.287 (n=5) | 90.780 | 93.509 ± 0.880 (n=2) | 95.236 |
| crossfit_uniform | 90.009 ± 0.107 (n=5) | 91.236 | 94.058 ± 0.098 (n=2) | 95.742 |
| dfr_lite | 88.155 ± 0.698 (n=5) | 91.953 | 92.645 ± 0.271 (n=2) | 95.850 |
| fullval_affine_oof_fixed | 89.589 ± 0.371 (n=5) | 92.514 | 93.569 ± 0.003 (n=2) | 96.133 |
| fullval_affine_oof_sel_nll | 89.305 ± 0.362 (n=5) | 92.806 | 93.302 ± 0.154 (n=2) | 96.264 |
| fullval_affine_oof_sel_wga | 89.380 ± 0.364 (n=5) | 92.725 | 93.388 ± 0.077 (n=2) | 96.266 |
| fullval_postmean_bias_fixed | 89.586 ± 0.375 (n=5) | 92.514 | 93.569 ± 0.003 (n=2) | 96.133 |
| fullval_postmean_bias_sel_nll | 89.358 ± 0.368 (n=5) | 92.789 | 93.388 ± 0.121 (n=2) | 96.263 |
| fullval_postmean_bias_sel_wga | 89.458 ± 0.375 (n=5) | 92.680 | 93.466 ± 0.011 (n=2) | 96.249 |
| fullval_postmean_thrwga | 89.572 ± 0.240 (n=5) | 91.626 | 93.380 ± 0.419 (n=2) | 95.451 |
| fullval_uniform | 89.882 ± 0.295 (n=5) | 92.040 | 93.876 ± 0.107 (n=2) | 95.901 |
| pool_affine_fixed | 90.038 ± 0.123 (n=5) | 91.454 | 93.783 ± 0.019 (n=2) | 95.853 |
| pool_affine_sel_nll | 89.830 ± 0.213 (n=5) | 91.766 | 93.636 ± 0.098 (n=2) | 95.998 |
| pool_affine_sel_wga | 89.810 ± 0.235 (n=5) | 91.713 | 93.684 ± 0.099 (n=2) | 95.943 |
| pool_linear_fixed | 90.186 ± 0.171 (n=5) | 91.784 | 93.557 ± 0.190 (n=2) | 95.748 |
| pool_linear_sel_nll | 89.025 ± 0.339 (n=5) | 92.479 | 93.115 ± 0.088 (n=2) | 95.900 |
| pool_linear_sel_wga | 89.789 ± 0.127 (n=5) | 92.093 | 93.268 ± 0.237 (n=2) | 95.754 |
| pool_uniform | 89.508 ± 0.331 (n=5) | 90.922 | 93.591 ± 0.689 (n=2) | 95.567 |

## normal_erm: corrected paired contrasts

| contrast | mean pp | pointwise 95% source t-interval | per source |
| --- | ---: | ---: | ---: |
| crossfit_affine_sel_nll - fullval_uniform | -0.172 | [-0.328, -0.016] | -0.229, -0.060, -0.280, -0.275, -0.014 |
| crossfit_affine_sel_nll - bag75_uniform | -0.230 | [-0.382, -0.079] | -0.343, -0.092, -0.374, -0.156, -0.187 |
| crossfit_affine_sel_nll - crossfit_uniform | -0.299 | [-0.803, +0.205] | -0.306, +0.268, -0.869, -0.372, -0.216 |
| crossfit_affine_sel_nll - crossfit_affine_fixed | -0.338 | [-0.494, -0.183] | -0.452, -0.352, -0.467, -0.202, -0.218 |
| crossfit_affine_sel_wga - fullval_uniform | -0.081 | [-0.229, +0.067] | -0.104, +0.034, -0.047, -0.275, -0.014 |
| crossfit_affine_fixed - fullval_uniform | +0.167 | [-0.007, +0.340] | +0.223, +0.292, +0.187, -0.073, +0.204 |
| crossfit_affine_fixed - crossfit_uniform | +0.039 | [-0.437, +0.515] | +0.146, +0.620, -0.402, -0.170, +0.002 |
| crossfit_uniform - fullval_uniform | +0.128 | [-0.279, +0.534] | +0.077, -0.328, +0.589, +0.097, +0.202 |
| crossfit_uniform - bag75_uniform | +0.069 | [-0.324, +0.462] | -0.037, -0.360, +0.495, +0.216, +0.029 |
| pool_affine_sel_nll - fullval_uniform | -0.051 | [-0.168, +0.065] | -0.073, -0.132, +0.047, -0.146, +0.047 |
| pool_linear_sel_nll - fullval_uniform | -0.857 | [-1.322, -0.392] | -0.867, -1.368, -1.028, -0.633, -0.388 |
| dfr_lite - fullval_uniform | -1.727 | [-2.266, -1.187] | -1.662, -1.409, -2.454, -1.378, -1.730 |
| crossfit_affine_sel_nll - crossfit_postmean_thrwga | +0.332 | [-0.320, +0.984] | +0.495, +0.742, -0.565, +0.655, +0.334 |
| crossfit_affine_sel_nll - fullval_postmean_thrwga | +0.138 | [-0.138, +0.415] | +0.413, +0.329, -0.113, +0.019, +0.043 |
| crossfit_affine_sel_nll - fullval_postmean_bias_sel_nll | +0.352 | [+0.111, +0.593] | +0.467, +0.265, +0.125, +0.280, +0.623 |
| crossfit_affine_sel_nll - crossfit_postmean_bias_sel_nll | +0.001 | [-0.077, +0.079] | -0.093, +0.083, +0.000, +0.000, +0.016 |
| crossfit_postmean_thrwga - crossfit_uniform | -0.631 | [-0.984, -0.278] | -0.800, -0.475, -0.305, -1.027, -0.550 |
| fullval_postmean_thrwga - fullval_uniform | -0.310 | [-0.588, -0.032] | -0.641, -0.389, -0.168, -0.294, -0.057 |

Exact threshold differs from 0.5 in 36/50 cells.
Mean fitted-validation WGA gain over legacy coarse search: 0.633 pp. This is not a test gain.

## normal_erm: every setting (no test-based selection)

| family | setting | mean test WGA |
| --- | ---: | ---: |
| crossfit_affine | average-l0.1 | 90.048 |
| crossfit_affine | maximum-l0.1 | 89.427 |
| crossfit_affine | average-l0.01 | 89.710 |
| crossfit_affine | maximum-l0.01 | 89.523 |
| crossfit_bias_foldwise | average-l0.1 | 90.048 |
| crossfit_bias_foldwise | maximum-l0.1 | 89.430 |
| crossfit_bias_foldwise | average-l0.01 | 89.704 |
| crossfit_bias_foldwise | maximum-l0.01 | 89.430 |
| crossfit_postmean_bias | average-l0.1 | 90.048 |
| crossfit_postmean_bias | maximum-l0.1 | 89.430 |
| crossfit_postmean_bias | average-l0.01 | 89.704 |
| crossfit_postmean_bias | maximum-l0.01 | 89.430 |
| fullval_affine_oof | average-l0.1 | 89.589 |
| fullval_affine_oof | maximum-l0.1 | 88.969 |
| fullval_affine_oof | average-l0.01 | 89.305 |
| fullval_affine_oof | maximum-l0.01 | 89.100 |
| fullval_postmean_bias | average-l0.1 | 89.586 |
| fullval_postmean_bias | maximum-l0.1 | 88.997 |
| fullval_postmean_bias | average-l0.01 | 89.296 |
| fullval_postmean_bias | maximum-l0.01 | 88.997 |
| pool_affine | average-l0.1 | 90.038 |
| pool_affine | maximum-l0.1 | 89.310 |
| pool_affine | average-l0.01 | 89.830 |
| pool_affine | maximum-l0.01 | 89.337 |
| pool_linear | average-l0.1 | 90.186 |
| pool_linear | maximum-l0.1 | 88.989 |
| pool_linear | average-l0.01 | 89.025 |
| pool_linear | maximum-l0.01 | 88.691 |

Selection counts (source × split cells):

| family | rule | cells | average-l0.1 | maximum-l0.1 | average-l0.01 | maximum-l0.01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crossfit_affine | nll | 50 | 0 | 0 | 50 | 0 |
| crossfit_affine | wga | 50 | 12 | 1 | 36 | 1 |
| crossfit_bias_foldwise | nll | 50 | 8 | 9 | 31 | 2 |
| crossfit_bias_foldwise | wga | 50 | 20 | 4 | 25 | 1 |
| crossfit_postmean_bias | nll | 50 | 8 | 10 | 31 | 1 |
| crossfit_postmean_bias | wga | 50 | 20 | 5 | 25 | 0 |
| fullval_affine_oof | nll | 50 | 0 | 0 | 50 | 0 |
| fullval_affine_oof | wga | 50 | 14 | 0 | 34 | 2 |
| fullval_postmean_bias | nll | 50 | 15 | 13 | 22 | 0 |
| fullval_postmean_bias | wga | 50 | 27 | 6 | 17 | 0 |
| pool_affine | nll | 50 | 0 | 0 | 50 | 0 |
| pool_affine | wga | 50 | 2 | 0 | 39 | 9 |
| pool_linear | nll | 50 | 0 | 0 | 50 | 0 |
| pool_linear | wga | 50 | 27 | 1 | 19 | 3 |

Split-seed variability: mean within-source SD (pp). Split seeds also change head-training seeds.

| arm | mean SD |
| --- | ---: |
| fullval_uniform | 0.405 |
| bag75_uniform | 0.193 |
| crossfit_uniform | 0.568 |
| crossfit_affine_sel_nll | 0.253 |
| crossfit_postmean_thrwga | 0.788 |
| fullval_postmean_thrwga | 0.583 |

## ermstar: corrected paired contrasts

| contrast | mean pp | pointwise 95% source t-interval | per source |
| --- | ---: | ---: | ---: |
| crossfit_affine_sel_nll - fullval_uniform | -0.083 | [n/a, n/a] | -0.338, +0.171 |
| crossfit_affine_sel_nll - bag75_uniform | -0.256 | [n/a, n/a] | -0.327, -0.184 |
| crossfit_affine_sel_nll - crossfit_uniform | -0.266 | [n/a, n/a] | -0.514, -0.017 |
| crossfit_affine_sel_nll - crossfit_affine_fixed | -0.196 | [n/a, n/a] | -0.234, -0.158 |
| crossfit_affine_sel_wga - fullval_uniform | -0.083 | [n/a, n/a] | -0.322, +0.156 |
| crossfit_affine_fixed - fullval_uniform | +0.113 | [n/a, n/a] | -0.104, +0.329 |
| crossfit_affine_fixed - crossfit_uniform | -0.070 | [n/a, n/a] | -0.280, +0.141 |
| crossfit_uniform - fullval_uniform | +0.182 | [n/a, n/a] | +0.176, +0.189 |
| crossfit_uniform - bag75_uniform | +0.010 | [n/a, n/a] | +0.187, -0.167 |
| pool_affine_sel_nll - fullval_uniform | -0.240 | [n/a, n/a] | -0.385, -0.095 |
| pool_linear_sel_nll - fullval_uniform | -0.761 | [n/a, n/a] | -0.899, -0.623 |
| dfr_lite - fullval_uniform | -1.231 | [n/a, n/a] | -1.498, -0.963 |
| crossfit_affine_sel_nll - crossfit_postmean_thrwga | +0.284 | [n/a, n/a] | -0.518, +1.085 |
| crossfit_affine_sel_nll - fullval_postmean_thrwga | +0.413 | [n/a, n/a] | -0.063, +0.888 |
| crossfit_affine_sel_nll - fullval_postmean_bias_sel_nll | +0.405 | [n/a, n/a] | +0.312, +0.498 |
| crossfit_affine_sel_nll - crossfit_postmean_bias_sel_nll | +0.000 | [n/a, n/a] | +0.000, +0.000 |
| crossfit_postmean_thrwga - crossfit_uniform | -0.549 | [n/a, n/a] | +0.004, -1.102 |
| fullval_postmean_thrwga - fullval_uniform | -0.496 | [n/a, n/a] | -0.275, -0.716 |

Exact threshold differs from 0.5 in 19/20 cells.
Mean fitted-validation WGA gain over legacy coarse search: 1.362 pp. This is not a test gain.

## ermstar: every setting (no test-based selection)

| family | setting | mean test WGA |
| --- | ---: | ---: |
| crossfit_affine | average-l0.1 | 93.989 |
| crossfit_affine | maximum-l0.1 | 93.768 |
| crossfit_affine | average-l0.01 | 93.793 |
| crossfit_affine | maximum-l0.01 | 93.824 |
| crossfit_bias_foldwise | average-l0.1 | 93.989 |
| crossfit_bias_foldwise | maximum-l0.1 | 93.785 |
| crossfit_bias_foldwise | average-l0.01 | 93.785 |
| crossfit_bias_foldwise | maximum-l0.01 | 93.785 |
| crossfit_postmean_bias | average-l0.1 | 93.989 |
| crossfit_postmean_bias | maximum-l0.1 | 93.785 |
| crossfit_postmean_bias | average-l0.01 | 93.785 |
| crossfit_postmean_bias | maximum-l0.01 | 93.785 |
| fullval_affine_oof | average-l0.1 | 93.569 |
| fullval_affine_oof | maximum-l0.1 | 93.310 |
| fullval_affine_oof | average-l0.01 | 93.302 |
| fullval_affine_oof | maximum-l0.01 | 93.403 |
| fullval_postmean_bias | average-l0.1 | 93.569 |
| fullval_postmean_bias | maximum-l0.1 | 93.302 |
| fullval_postmean_bias | average-l0.01 | 93.302 |
| fullval_postmean_bias | maximum-l0.01 | 93.302 |
| pool_affine | average-l0.1 | 93.783 |
| pool_affine | maximum-l0.1 | 93.608 |
| pool_affine | average-l0.01 | 93.636 |
| pool_affine | maximum-l0.01 | 93.619 |
| pool_linear | average-l0.1 | 93.557 |
| pool_linear | maximum-l0.1 | 93.430 |
| pool_linear | average-l0.01 | 93.115 |
| pool_linear | maximum-l0.01 | 93.338 |

Selection counts (source × split cells):

| family | rule | cells | average-l0.1 | maximum-l0.1 | average-l0.01 | maximum-l0.01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crossfit_affine | nll | 20 | 0 | 0 | 20 | 0 |
| crossfit_affine | wga | 20 | 3 | 0 | 14 | 3 |
| crossfit_bias_foldwise | nll | 20 | 2 | 11 | 7 | 0 |
| crossfit_bias_foldwise | wga | 20 | 6 | 9 | 5 | 0 |
| crossfit_postmean_bias | nll | 20 | 2 | 10 | 7 | 1 |
| crossfit_postmean_bias | wga | 20 | 6 | 8 | 5 | 1 |
| fullval_affine_oof | nll | 20 | 0 | 0 | 20 | 0 |
| fullval_affine_oof | wga | 20 | 1 | 0 | 13 | 6 |
| fullval_postmean_bias | nll | 20 | 5 | 9 | 6 | 0 |
| fullval_postmean_bias | wga | 20 | 9 | 7 | 4 | 0 |
| pool_affine | nll | 20 | 0 | 0 | 20 | 0 |
| pool_affine | wga | 20 | 2 | 0 | 16 | 2 |
| pool_linear | nll | 20 | 0 | 0 | 20 | 0 |
| pool_linear | wga | 20 | 3 | 1 | 10 | 6 |

Split-seed variability: mean within-source SD (pp). Split seeds also change head-training seeds.

| arm | mean SD |
| --- | ---: |
| fullval_uniform | 0.334 |
| bag75_uniform | 0.236 |
| crossfit_uniform | 0.340 |
| crossfit_affine_sel_nll | 0.180 |
| crossfit_postmean_thrwga | 0.496 |
| fullval_postmean_thrwga | 0.467 |
