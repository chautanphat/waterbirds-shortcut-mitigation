# Threshold controls for cross-fitted DPE

Sources: ermstar-seed0, ermstar-seed1, normal-erm-seed0, normal-erm-seed1, normal-erm-seed2, normal-erm-seed3, normal-erm-seed4.
Reproduction of historical WGA: 42/42 rows exact.

## normal_erm

| Method | Mean WGA | Per source |
| --- | ---: | --- |
| full_validation | 89.60 | 89.56, 90.65, 89.10, 89.25, 89.41 |
| fold_uniform | 90.14 | 90.73, 90.11, 90.03, 89.36, 90.47 |
| fold_affine_average-l0.01 | 89.69 | 89.56, 90.50, 89.41, 89.41, 89.56 |
| fold_offset_average-l0.01 | 89.69 | 89.56, 90.50, 89.41, 89.41, 89.56 |
| full_offset_average-l0.01 | 88.97 | 88.79, 89.72, 88.79, 88.63, 88.94 |
| fold_affine_average-l0.1 | 90.06 | 90.34, 90.97, 89.72, 89.41, 89.88 |
| fold_offset_average-l0.1 | 90.06 | 90.34, 90.97, 89.72, 89.41, 89.88 |
| full_offset_average-l0.1 | 89.16 | 89.10, 90.03, 88.94, 88.79, 88.94 |
| fold_affine_maximum-l0.01 | 89.60 | 89.56, 90.19, 89.25, 89.41, 89.56 |
| fold_offset_maximum-l0.01 | 89.35 | 89.56, 89.88, 88.47, 89.25, 89.56 |
| full_offset_maximum-l0.01 | 88.72 | 88.79, 89.41, 88.16, 88.47, 88.79 |
| fold_affine_maximum-l0.1 | 89.44 | 89.41, 89.72, 89.25, 89.41, 89.41 |
| fold_offset_maximum-l0.1 | 89.56 | 89.56, 90.19, 89.25, 89.41, 89.41 |
| full_offset_maximum-l0.1 | 88.79 | 88.63, 89.56, 88.79, 88.63, 88.32 |
| fold_threshold_wga | 89.06 | 88.29, 90.07, 89.67, 88.07, 89.22 |
| full_threshold_wga | 89.34 | 88.34, 90.65, 88.65, 90.03, 89.05 |

| Contrast (pp) | Mean | 95% t interval | Per source |
| --- | ---: | --- | --- |
| fold_affine_average-l0.1 minus full_validation | +0.47 | [+0.16, +0.77] | +0.78, +0.31, +0.62, +0.16, +0.47 |
| fold_affine_average-l0.1 minus fold_offset_average-l0.1 | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00, +0.00, +0.00 |
| fold_affine_average-l0.1 minus full_offset_average-l0.1 | +0.90 | [+0.62, +1.19] | +1.25, +0.93, +0.78, +0.62, +0.93 |
| fold_affine_average-l0.1 minus fold_threshold_wga | +1.00 | [+0.07, +1.93] | +2.05, +0.90, +0.05, +1.34, +0.65 |
| fold_affine_average-l0.1 minus full_threshold_wga | +0.72 | [-0.48, +1.92] | +2.01, +0.31, +1.07, -0.62, +0.83 |
| fold_offset_average-l0.1 minus full_validation | +0.47 | [+0.16, +0.77] | +0.78, +0.31, +0.62, +0.16, +0.47 |
| full_offset_average-l0.1 minus full_validation | -0.44 | [-0.65, -0.22] | -0.47, -0.62, -0.16, -0.47, -0.47 |
| full_threshold_wga minus full_validation | -0.25 | [-1.16, +0.65] | -1.23, +0.00, -0.45, +0.78, -0.36 |
| fold_threshold_wga minus full_validation | -0.53 | [-1.47, +0.41] | -1.27, -0.59, +0.57, -1.18, -0.18 |
| fold_uniform minus full_validation | +0.54 | [-0.37, +1.46] | +1.17, -0.54, +0.93, +0.10, +1.06 |
| fold_affine_average-l0.01 minus fold_offset_average-l0.01 | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00, +0.00, +0.00 |
| fold_affine_average-l0.1 minus fold_offset_average-l0.1 | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00, +0.00, +0.00 |
| fold_affine_maximum-l0.01 minus fold_offset_maximum-l0.01 | +0.25 | [-0.15, +0.65] | +0.00, +0.31, +0.78, +0.16, +0.00 |
| fold_affine_maximum-l0.1 minus fold_offset_maximum-l0.1 | -0.12 | [-0.38, +0.13] | -0.16, -0.47, +0.00, +0.00, +0.00 |

## ermstar

| Method | Mean WGA | Per source |
| --- | ---: | --- |
| full_validation | 94.08 | 94.39, 93.77 |
| fold_uniform | 93.60 | 94.39, 92.82 |
| fold_affine_average-l0.01 | 93.93 | 93.93, 93.93 |
| fold_offset_average-l0.01 | 93.93 | 93.93, 93.93 |
| full_offset_average-l0.01 | 92.99 | 93.61, 92.37 |
| fold_affine_average-l0.1 | 94.01 | 94.24, 93.79 |
| fold_offset_average-l0.1 | 94.01 | 94.24, 93.79 |
| full_offset_average-l0.1 | 93.38 | 93.93, 92.83 |
| fold_affine_maximum-l0.01 | 93.93 | 93.77, 94.08 |
| fold_offset_maximum-l0.01 | 93.61 | 93.61, 93.61 |
| full_offset_maximum-l0.01 | 92.37 | 92.83, 91.90 |
| fold_affine_maximum-l0.1 | 93.91 | 93.77, 94.06 |
| fold_offset_maximum-l0.1 | 93.85 | 93.77, 93.93 |
| full_offset_maximum-l0.1 | 92.83 | 93.15, 92.52 |
| fold_threshold_wga | 93.66 | 94.28, 93.04 |
| full_threshold_wga | 93.75 | 93.88, 93.61 |

| Contrast (pp) | Mean | 95% t interval | Per source |
| --- | ---: | --- | --- |
| fold_affine_average-l0.1 minus full_validation | -0.07 |  | -0.16, +0.02 |
| fold_affine_average-l0.1 minus fold_offset_average-l0.1 | +0.00 |  | +0.00, +0.00 |
| fold_affine_average-l0.1 minus full_offset_average-l0.1 | +0.63 |  | +0.31, +0.96 |
| fold_affine_average-l0.1 minus fold_threshold_wga | +0.36 |  | -0.04, +0.75 |
| fold_affine_average-l0.1 minus full_threshold_wga | +0.27 |  | +0.36, +0.18 |
| fold_offset_average-l0.1 minus full_validation | -0.07 |  | -0.16, +0.02 |
| full_offset_average-l0.1 minus full_validation | -0.70 |  | -0.47, -0.93 |
| full_threshold_wga minus full_validation | -0.33 |  | -0.51, -0.16 |
| fold_threshold_wga minus full_validation | -0.42 |  | -0.11, -0.73 |
| fold_uniform minus full_validation | -0.48 |  | +0.00, -0.95 |
| fold_affine_average-l0.01 minus fold_offset_average-l0.01 | +0.00 |  | +0.00, +0.00 |
| fold_affine_average-l0.1 minus fold_offset_average-l0.1 | +0.00 |  | +0.00, +0.00 |
| fold_affine_maximum-l0.01 minus fold_offset_maximum-l0.01 | +0.31 |  | +0.16, +0.47 |
| fold_affine_maximum-l0.1 minus fold_offset_maximum-l0.1 | +0.07 |  | +0.00, +0.13 |
