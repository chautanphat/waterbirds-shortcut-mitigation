# Stage A — completed before FDAT

**TEST ORACLE / IN-SAMPLE; no held-out or safe-bound claim.**

|Epsilon pp|Search|Threshold|Original WGA %|Mean blur WGA %|Gain pp|Fraction unconstrained gain|
|---|---|---|---|---|---|---|
|0|grid_0.001|0.51|91.2773|72.6667|+1.0000|0.0667|
|1|grid_0.001|0.52|90.3427|74.0000|+2.3333|0.1556|
|2|grid_0.001|0.559|89.0966|77.3333|+5.6667|0.3778|
|0|exact_finite_scores|0.506795763969|91.4330|72.6667|+1.0000|0.0652|
|1|exact_finite_scores|0.519467711449|90.4984|74.0000|+2.3333|0.1522|
|2|exact_finite_scores|0.558038711548|89.0966|77.3333|+5.6667|0.3696|

Objective uses only sigma0.5/1/2 with equal weight. Original full5794 WGA constrains selection. Tie: original WGA, mean blur AA, proximity to0.5, lower t. Epsilon is a deterministic inspected-test bound, not a statistical safety guarantee.

Predeclared "most" means >=80% of unconstrained global gain at epsilon<=2pp. This is a research triage definition, not a publication criterion. Stage B is still executed as requested.
