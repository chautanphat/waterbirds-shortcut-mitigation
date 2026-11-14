# Label-correction study

WGA in %, averaged over head seeds within each source, then over sources. Paired differences (pp) against the official-label fit of the same method, same test labels.

## normal_erm

| Method | Fit labels | Test: official | Test: corrected | Test: 90 removed | Errors on 90 (official) |
| --- | --- | ---: | ---: | ---: | ---: |
| dpe | official | 89.89 | 90.34 | 90.42 | 78.1 |
| dpe | relabel | 89.07 | 92.81 | 92.72 | 85.7 |
| dpe | drop | 89.07 | 92.79 | 92.69 | 85.8 |
| dfr_lite | official | 88.62 | 88.97 | 89.09 | 77.0 |
| dfr_lite | relabel | 86.70 | 90.53 | 90.49 | 84.6 |
| dfr_lite | drop | 86.54 | 90.42 | 90.38 | 84.6 |

| Contrast | Test labels | Mean (pp) | 95% t interval | Per source |
| --- | --- | ---: | --- | --- |
| dpe: relabel minus official | official | -0.83 | [-1.34, -0.31] | -1.28, -0.70, -0.37, -1.24, -0.54 |
| dpe: relabel minus official | corrected | +2.48 | [+1.80, +3.15] | +2.33, +1.70, +2.98, +2.36, +3.01 |
| dpe: drop minus official | official | -0.83 | [-1.24, -0.41] | -0.87, -0.86, -0.62, -1.33, -0.45 |
| dpe: drop minus official | corrected | +2.46 | [+1.90, +3.01] | +2.95, +2.05, +2.94, +2.10, +2.25 |
| dfr_lite: relabel minus official | official | -1.92 | [-2.41, -1.43] | -2.49, -2.11, -1.87, -1.58, -1.56 |
| dfr_lite: relabel minus official | corrected | +1.56 | [+1.19, +1.94] | +1.78, +1.74, +1.04, +1.61, +1.65 |
| dfr_lite: drop minus official | official | -2.08 | [-2.83, -1.33] | -2.96, -2.42, -1.87, -1.58, -1.56 |
| dfr_lite: drop minus official | corrected | +1.45 | [+1.07, +1.83] | +1.65, +1.74, +0.96, +1.39, +1.52 |

## ermstar

| Method | Fit labels | Test: official | Test: corrected | Test: 90 removed | Errors on 90 (official) |
| --- | --- | ---: | ---: | ---: | ---: |
| dpe | official | 93.88 | 93.45 | 94.25 | 45.0 |
| dpe | relabel | 92.74 | 95.57 | 95.84 | 59.8 |
| dpe | drop | 93.16 | 95.53 | 95.87 | 58.5 |
| dfr_lite | official | 92.68 | 94.18 | 94.97 | 48.0 |
| dfr_lite | relabel | 88.94 | 94.13 | 94.15 | 81.5 |
| dfr_lite | drop | 89.25 | 94.16 | 94.17 | 80.5 |

| Contrast | Test labels | Mean (pp) | 95% t interval | Per source |
| --- | --- | ---: | --- | --- |
| dpe: relabel minus official | official | -1.14 |  | -1.43, -0.84 |
| dpe: relabel minus official | corrected | +2.12 |  | +1.91, +2.33 |
| dpe: drop minus official | official | -0.72 |  | -0.87, -0.56 |
| dpe: drop minus official | corrected | +2.08 |  | +2.03, +2.12 |
| dfr_lite: relabel minus official | official | -3.74 |  | -3.89, -3.58 |
| dfr_lite: relabel minus official | corrected | -0.04 |  | -0.48, +0.39 |
| dfr_lite: drop minus official | official | -3.43 |  | -3.74, -3.12 |
| dfr_lite: drop minus official | corrected | -0.02 |  | -0.48, +0.43 |
