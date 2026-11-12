# Historical DPE verification notebooks

These files preserve experiment history. **Do not use them as current launch instructions.** Verification has completed; read the [five-seed report](../../../results/dpe/five_seed_verification_20261112/README.md) and use the lightweight audit in `scripts/analysis/`.

| Notebook | Historical role / limitation |
| --- | --- |
| `dpe_aggregation_verification_kaggle.ipynb` | Successful original three-source replay. |
| `dpe_aggregation_five_seed_verification_kaggle.ipynb` | Original five-source run; failed in a seed-4 auxiliary joint solver after both new backbones finished. |
| `dpe_aggregation_five_seed_recovery_kaggle.ipynb` | Recovery requiring attached full output; superseded for this run. |
| `dpe_aggregation_five_seed_auto_recovery_kaggle.ipynb` | Obsolete latest-output lookup; failed when Kaggle returned an empty latest output. |
| `dpe_aggregation_five_seed_pinned_recovery_kaggle.ipynb` | Pinned recovery manifest for the original saved version. Signed download URLs were redacted; the unmodified file is retained only in local ignored output storage. |

The final uploaded bundle records a completed recovery. Notebook presence alone is not evidence that every historical variant was run successfully. Large checkpoints and caches are deliberately outside Git. Preserved source code and self-contained notebook contents remain unchanged apart from their archive locations.
