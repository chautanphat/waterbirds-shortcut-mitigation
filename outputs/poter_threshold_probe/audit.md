# Asset and leakage audit

Canonical metadata maps Landbird0/Waterbird1 and group=2*y+place, Land0/Water1. Validation1199 has groups467/466/133/133; all images/IDs/labels are checked. The requested browser-renamed predictions(5).csv was not attached; canonical audited9000-row mild-blur CSV is used. Full original test5794×3 is cached separately. Its100-image subset is not added twice.

## Selection eligibility is model specific

- POTER seed1: **BLOCKED: no independent four-group development data for threshold selection**. OT reference used all466 validation group1 and all133 group2 images; unused600 have only groups0/3. Validation also selected the best checkpoint. No threshold sweep or selected threshold is produced for POTER.
- ERM enhanced seed0: clean validation sweep allowed. Train4795 used for fitting; fixed last400 checkpoint, original validation1199 for this diagnostic. Historical monitoring and reuse for threshold selection mean these validation metrics are not unbiased validation of a selected threshold.
- DPE legacy seed0: **BLOCKED: no independent development data for threshold selection**. Actual saved head provenance lists exactly all1199 validation filenames as fitting data. Those probabilities are descriptive fitting-pool diagnostics, not held-out calibration data.
- No independent development blur bank with masks exists here; joint clean-constrained threshold not run. No untouched evaluation set was located. Previously inspected test is exploratory only.

See `asset_audit.json`, `validation_membership.csv`, `checkpoint_audit.json` and `validation_inference.json` for paths, SHA256, model/seed, group membership, fitting overlap and numeric parity. POTER seed1 and ERM/DPE seed0 are unmatched; no multi-seed claim. DPE historical head→backbone hash binding remains unverified, although explicit frozen pairing and inference parity are checked.

Official labels are retained throughout selection/evaluation. Known Waterbirds label-generation issues affect species021.Eastern_Towhee,088.Western_Meadowlark,102.Western_Wood_Pewee; the100-image counterfactual cohort had already excluded them before its predictions. Full original validation/test retain canonical labels, without post-hoc correction or selecting a favorable label policy.

## Prediction schema and reuse

`validation_predictions.csv` retains source split/model/seed/checkpoint hashes, canonical source foreground ID, species, true label/group/background, sigma/control, hard prediction, probabilities/logits, raw/model/tensor hashes and eligibility. Cached test columns retain the equivalent source fields; checkpoint hashes are attached from verified manifests during analysis. Background-only rows have no bird label and are excluded from all accuracy/error denominators.

Validation: 3597 probability rows, 1199 new POTER image forwards, 1199 cached ERM feature rows reused;1199 ERM linear predictions and1199 legacy DPE ensemble probabilities newly computed. Additional live ERM checks are disclosed in inference manifest. This is inference, not training. Shared square256/bilinear/center224/ImageNet preprocessing matches Runner. No OT recomputation, generated experimental images, new blur or augmentation changes.

Probabilities are finite/in[0,1]/sum-to-one checked, p_waterbird means class1, saved argmax matches. Recomputing >=0.5 versus saved argmax differs on 0 rows (ties/roundoff counted explicitly). See `prediction_checks.json`. Diagnostic row groups are kept separate: original1199 validation, original5794 test and100-foreground repeated-composite test. Clean natural composite sigma0 is not the original unmodified photograph.
