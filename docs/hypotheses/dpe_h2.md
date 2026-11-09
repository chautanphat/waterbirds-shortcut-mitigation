# H2: Does reweighting DPE heads improve worst-group accuracy?

**Status:** not supported by the tested global weighting rules in an exploratory
three-seed study. Updated 2026-11-09.

## Hypothesis and comparison

H2 proposes that validation-derived head weights improve Waterbirds worst-group
accuracy (WGA) over uniform averaging. The comparison holds the backbone,
features, 15 DPE heads and test images fixed within each seed; only aggregation
changes. Historical standard/enhanced DPE reproductions used a different
head-training data budget and are not matched H2 controls.

Validation roles are A=720 for head fitting, B=241 for weight fitting and C=238
for selection. The 20 candidates comprise uniform, 12 power rules, softmax
β=10/30, top-k=10/5 and joint λ=1/0.1/0.01. C selects by WGA, then overall
accuracy; complete ties retain uniform. Only frozen winners are evaluated on
the official 5,794-image test set. Seeds 0/1/2 share one validation partition.

## Evidence across experiments

- **Initial power-weighting study:** all 13 candidates gave identical C labels
  in seed 0. Uniform was selected, producing no paired test improvement.
- **Expanded seed-0 study:** top-10 corrected one additional C image and was
  selected, but reduced test WGA by 0.1330 percentage points (pp). It fixed 27
  test errors and introduced 29.
- **Interrupted multi-seed attempt:** seed 1 finished training but an optimizer
  failure prevented final evaluation. This was a software failure, not evidence
  for or against H2.
- **Completed three-seed study:** seed 0 was reused and seeds 1/2 trained. Power,
  softmax and joint families selected uniform in every seed. Top-k selected
  uniform in seeds 1/2. Reused seed-0 results are not additional replicates.

| Seed | Uniform test WGA (%) | Overall C choice | Selected test WGA (%) | Paired change (pp) |
| --- | ---: | --- | ---: | ---: |
| 0 | 89.7561 | Top-10 | 89.6231 | −0.1330 |
| 1 | 90.0222 | Uniform | 90.0222 | 0.0000 |
| 2 | 88.7805 | Uniform | 88.7805 | 0.0000 |

Uniform achieved **89.5196 ± 0.6538% WGA** and **91.2668 ± 0.3293% overall
accuracy**. Overall C selection achieved **89.4752 ± 0.6339% WGA** and
**91.2553 ± 0.3317% accuracy**. The paired WGA change was **−0.0443 ± 0.0768 pp**:
zero improvements, two ties and one decrease. All ± values are sample SD across
training seeds, not confidence intervals. Full family/group results are in the
[raw result tables](../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/summary.json)
and [per-seed CSV](../../results/dpe_head_reweighting/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/per_seed.csv).

## Interpretation

**The optimizer improved its objective without improving the classification
metric.** Joint weighting minimizes worst B-group negative log likelihood
(NLL), with regularization toward uniform. Lower NLL can reward increased
confidence without increasing the number of correct predictions.

For seed 1, joint λ=0.01 reduced worst-group NLL on both B (0.6566→0.6283) and C
(0.6563→0.6418), while C WGA fell from 89.25% to 84.95%. B WGA also decreased,
88.89%→85.19%, so the trade-off already occurred on the fitting set. Improved C
NLL makes a pure B-overfitting explanation insufficient. This finite-sample
result does not establish that log loss is generally unsuitable.

**Aggressive weighting concentrated the ensemble.** Seed 1 placed approximately
82.87% on one head and 17.13% on another. Across seeds, joint λ=0.01 reduced
effective head count from 15 to 1.84/1.40/3.27 and lowered C WGA. Stronger
regularization, λ=1, changed zero C classifications in every seed, so uniform
won the ties. The heads themselves differ: mean pairwise test disagreement was
11.49/11.46/10.71%. Equal selected scores therefore do not imply duplicated
heads or inactive weighting code.

## Conclusion

The tested global weighting procedures did not improve held-out WGA over the
matched uniform ensemble. Uniform remains the reference supported by these
comparisons. The study does not demonstrate statistical superiority of uniform,
rule out other combiners, or identify a unique cause of the negative result.

## Follow-up hypotheses — untested

| Hypothesis | Motivation | Discriminating experiment |
| --- | --- | --- |
| Objective sensitivity | Lower NLL did not translate into higher WGA. | Compare predeclared loss or decision-threshold variants, fitting on B and selecting on C WGA. |
| Weight/selection instability | B contains only 27 examples per waterbird group; one C image changes group accuracy by 1.08 or 3.85 pp. | Use group-stratified B resampling and repeated A/B/C partitions, retraining heads for each new A; measure weight dispersion and selection stability. |
| Input-dependent head strengths | Distinct heads may be useful on different images even when one global weight vector gives no gain. | Compare a regularized probability/logit stacker and a small softmax gating network against uniform, with matched heads and separate B/C roles. |

Stacking and input-dependent gating are established frameworks
([Wolpert, 1992](https://doi.org/10.1016/S0893-6080(05)80023-1);
[Jacobs et al., 1991](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts-bitmap.pdf)).
Their benefit here remains a hypothesis. Neural stacking/gating has not been run;
adding a network does not itself resolve the NLL–WGA trade-off.

## Scope and supporting evidence

There are three training seeds, one shared validation partition and one shared
test set. Seed 0 had already informed follow-up design. The results and hypotheses
are exploratory; further comparisons on the inspected test set must retain that
label. Sixty candidates and selected test metrics were recomputed from supplied
prediction caches; image-to-feature inference was not independently rerun.

- [Original power-study audit](evidence/dpe_h2_power_seed0.md).
- [Expanded seed-0 audit](evidence/dpe_h2_rules_seed0.md).
- [Three-seed methods, diagnostics and provenance](evidence/dpe_h2_three_seed_audit.md).
- [Experiment index](../experiment_log.md) and [recorded configurations/run identities](../../experiments/exp-dpe-h2-rules-waterbirds-standard-seeds012-003/run.json).
