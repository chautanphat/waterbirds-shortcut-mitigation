# DPE experiment results

The [five-seed report](five_seed_verification_20261112/README.md) is the current scientific assessment. Core per-seed results, alternative-partition results, controls, combiner repetitions and uncertainty are alongside it.

The [H2 literature note](h2_aggregation_literature.md) summarizes related work without presenting new experiment results. Earlier three-seed reports, historical tables, verification logs, and archived verification notebooks are retained in [supporting_evidence.zip](supporting_evidence.zip). Paths within the ZIP reproduce their former repository locations, and `SHA256SUMS` records each file’s hash.

The supporting archive is **not** the original uploaded experiment bundle: predictions, model checkpoints and raw data remain outside Git. Retain the original verification ZIP and Kaggle output for a complete audit or recovery.

[ERM* + DPE neural aggregation: two-seed report](ermstar_neural_two_seed_20261113/README.md):
all 24 test results independently replayed. Plain/dropout MLPs improve matched
uniform on both sources; logistic has the highest neural mean but mixed paired
effects. No neural mean exceeds the full-validation DPE reference. Exploratory H2
evidence only; H1/H3 on these ERM* sources are not tested.
