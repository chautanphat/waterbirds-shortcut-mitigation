# Research handoff

User requested all feasible Phase 1 weighting experiments followed automatically
by winner ablation, with checkpointed progress and no repeated confirmations.

**Completed:** 127 Phase-1 configurations across two screening seeds, all 42
ranks including the oracle diagnostic, three-seed finalist confirmation,
72 fitted main-ablation rows plus explicit N/A entries, and the additional
three-seed A18 five-fold diagnostic. The locked winner is D19. It scores
95.961% mean test accuracy versus 95.708% matched U0 and 96.048% original DPE.
WGA falls from 92.761% matched U0 to 83.645%; original DPE WGA is 94.393%.
This does not support replacing original DPE. Preserve that checkpoint.

Independent artifact audit passed for 84 final predictors, nine OOF folds and
unchanged original checkpoint hashes. All three canonical winner exports cover
the full 5,794-example test split and passed metadata validation. New tests:
51 passed; compatible historical tests: 21 passed, five platform-specific tests
excluded after documented Windows failures.

The supplied old ensemble records all 1,199 validation examples in its
`training_pool_ids`, so it must not generate gate-training data on that pool.
The separate `experiments.weighting` package refits heads using three-fold
class-only cross-fitting and uses groups for post-hoc evaluation only.
Original checkpoints and upstream implementation are preserved.

The complete Waterbirds dataset is now present at the path in `config.yaml`.
Validation features were extracted successfully. A repository-local `.venv`
contains the research dependencies. PyTorch is CPU-only even though the machine
has an RTX 4050. The bare backbone's training history is not embedded; disclose
this provenance limitation in claims.

Run/resume:

```powershell
.venv/Scripts/python.exe -m experiments.weighting.run --config experiments/weighting/config.yaml --phase all --resume
```

Read `outputs/weighting/HANDOFF.md`, `progress.json`, `last_error.json` (if any),
and selection/ablation locks first. Never delete completed results or select
configurations using test metrics. Config/source hashes protect resume;
preserve the saved source snapshot when handing off. Test metrics must not be
invented if an external interruption prevents completion.

Final report: `outputs/weighting/REPORT.md`; tracked copy:
`results/dpe/weighting_20261114/README.md`. The full output directory contains
326 main fitted-model checkpoints, the additional A18 models, all frozen heads,
features, per-sample predictions, corrected/broken sample IDs and result tables.
Supplemental resume: `python -m experiments.weighting_ablation_crossfit`.
Post-hoc report regeneration: `python -m scripts.finalize_dpe_weighting`.

Per-image delivery bundle: `outputs/weighting/prediction_bundle.zip`.
Its `final_test.csv` contains 17,382 image/seed rows with all 15 experts' raw
and calibrated probabilities, gate weights and correctness. Three slim NPZ
caches each have `probs [5794,15,2]`, labels, IDs and weights. Original aggregate
results are copied as `final_test_metrics.csv`, avoiding ambiguity with the
per-image CSV. D19 mixes `calibrated_probs`, not raw `probs`; reconstruction
against all saved gate outputs passed at tolerance 1e-12. Pairwise error-overlap
matrices, the original DPE cache and provenance hashes are included.
Re-export: `python -m scripts.export_dpe_prediction_bundle` (no training).
