# Label-corrected backbones: results

Jobs scored: 48. Missing or failed: 0.

## Test labels: corrected

WGA (%) per backbone seed, mean over head seeds (n head seeds in brackets).

| Backbone labels / val labels | seed 5 | seed 6 | seed 7 | seed 8 |
| --- | ---: | ---: | ---: | ---: |
| official / official | 90.17 (3) | 86.40 (3) | 89.18 (3) | 87.85 (3) |
| official / corrected | 91.92 (3) | 91.57 (3) | 91.28 (3) | 90.98 (3) |
| corrected / official | 89.39 (3) | 88.94 (3) | 91.51 (3) | 87.95 (3) |
| corrected / corrected | 92.03 (3) | 91.99 (3) | 92.08 (3) | 90.78 (3) |

| Contrast | Mean (pp) | 95% t interval | Backbones | Per backbone |
| --- | ---: | --- | ---: | --- |
| primary: fix validation labels | +3.04 | [+0.59, +5.48] | 4 | s5: +1.75, s6: +5.17, s7: +2.10, s8: +3.13 |
| fix training and validation labels | +3.32 | [+0.78, +5.85] | 4 | s5: +1.86, s6: +5.59, s7: +2.90, s8: +2.92 |
| fix training labels only | +1.05 | [-1.57, +3.66] | 4 | s5: -0.78, s6: +2.53, s7: +2.33, s8: +0.10 |
| add training fix to validation fix | +0.28 | [-0.40, +0.96] | 4 | s5: +0.11, s6: +0.42, s7: +0.80, s8: -0.20 |

## Test labels: official

WGA (%) per backbone seed, mean over head seeds (n head seeds in brackets).

| Backbone labels / val labels | seed 5 | seed 6 | seed 7 | seed 8 |
| --- | ---: | ---: | ---: | ---: |
| official / official | 89.32 (3) | 86.83 (3) | 88.72 (3) | 87.56 (3) |
| official / corrected | 87.23 (3) | 86.24 (3) | 86.81 (3) | 88.58 (3) |
| corrected / official | 89.09 (3) | 88.82 (3) | 90.06 (3) | 87.79 (3) |
| corrected / corrected | 86.40 (3) | 87.59 (3) | 87.12 (3) | 86.66 (3) |

| Contrast | Mean (pp) | 95% t interval | Backbones | Per backbone |
| --- | ---: | --- | ---: | --- |
| primary: fix validation labels | -0.89 | [-3.19, +1.40] | 4 | s5: -2.10, s6: -0.59, s7: -1.91, s8: +1.02 |
| fix training and validation labels | -1.17 | [-3.61, +1.28] | 4 | s5: -2.93, s6: +0.76, s7: -1.60, s8: -0.91 |
| fix training labels only | +0.83 | [-0.79, +2.45] | 4 | s5: -0.24, s6: +2.00, s7: +1.34, s8: +0.23 |
| add training fix to validation fix | -0.27 | [-2.52, +1.98] | 4 | s5: -0.83, s6: +1.35, s7: +0.31, s8: -1.92 |
