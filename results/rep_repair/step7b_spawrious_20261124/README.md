# Direction 2 · Phase C · Step 7b Spawrious (`o2o_hard`) ResNet-50 Val Control Ladder

Training split (`85%`): **21,544** images | Held-out 8-group validation (`15%`): **3,800** images | Shifted OOD test accessed: **0**.

| Arm | Params | 8-Grp WGA (%) | Minority (`beach`) (%) | Majority (`spur`) (%) | Overall Acc (%) | Minority Macro-AUC (%) |
|---|---:|---:|---:|---:|---:|---:|
| `erm_raw` | 23,516,228 | **87.72 ± 1.52** | 94.74 | 99.73 | 99.33 | **99.54** |
| `dfr_l2` | 8,196 | **93.42 ± 1.32** | 95.83 | 99.71 | 99.40 | **99.49** |
| `dar` | 274,693 | **87.72 ± 2.74** | 93.86 | 99.37 | 98.93 | **99.33** |
| `affine_dfr_bn_l3` | 28,676 | **92.11 ± 1.32** | 95.29 | 99.74 | 99.39 | **99.56** |
| `affine_dfr_bn_l4` | 30,724 | **92.98 ± 0.76** | 96.05 | 99.67 | 99.38 | **99.56** |
| `affine_dfr_bn_all` | 61,316 | **91.23 ± 4.02** | 95.18 | 99.71 | 99.35 | **99.55** |
