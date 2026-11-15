# CCW v6 mechanism audit — B crossfit only

**B decision: NO_GO. Selected method: None.**

Every fitted table, normalizer, q20 and sigma_A uses only B-inner-train. Model selection has no group input. The weighted target is p_head(true class) minus the exact same Uniform true-class probability, not correctness. Each image distributes omega across15 heads; kappa25 is effective image-weight support.

## Advantage reliability

| Seed | Image-cluster pair ranking | 95% CI | Mixed images | Descriptive AUROC(t>0) |
|---|---:|---|---:|---:|
| 0 | 0.6490634094019087 | [0.6398013451678827, 0.6572695011622418] | 613 | 0.6713346236465885 |
| 1 | 0.634764212631741 | [0.6244266866682052, 0.6437863489499193] | 619 | 0.6508944173458225 |
| 2 | 0.6596554394115369 | [0.6522110917537748, 0.6668115563298491] | 615 | 0.6567687776945218 |

Signal estimates are also stratified by predicted class, train-defined low-margin status and disagreement with Uniform. Correlation/AUROC are descriptive correlated-expert statistics; only image-cluster pair-ranking intervals are claimed. Positive correlation with relative probability is not guaranteed subgroup robustness.

## Mechanism, boundaries and support

CCW_A isolates partial distance calibration. CCW_FULL adds the smoothed signed advantage. G2_z fully standardizes distances at gamma1 and remains a control. G1 keeps raw normalized Euclidean distance at gamma50. No probability temperature or head parameter is changed.

B_gain_and_concentration.csv records true signed gain sum(r*t), low-margin/uniform-wrong strata, score dynamic ranges, effective experts, entropy, top-head/tie frequency and any advantage-range dominance. Such dominance is reported without retuning. Head00 is never penalized by index. Uniform deterministic argmax ties favor slot00; unique-top share/tie counts are reported to avoid interpreting that as learned dominance.

B_cell_tables.csv contains raw expert observations, distinct-image support and effective weighted image support for every one of12 cells. Sparse and empty cells shrink toward smoothed predicted-class parents then global mean/0; parent computation has no self-reference. Shared head slots across outer ensembles are never identified as the same expert.

The seed AA floor averages repeated B-image predictions across outer folds before evaluation. There are720 raw B rows per seed but fewer unique images; these are not720 independent examples or three independent backbone replicates. OOF lookup fits approximately180 images, whereas an allowed full-B deployment refit uses240.

No new CCW C replay occurs after a NO_GO. All failing fixed variants are retained. No optional random controls or unplanned rho/beta/alpha search was added.
