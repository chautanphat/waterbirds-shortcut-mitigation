# Verified opt-in distance aggregation

Uniform probability averaging remains the default. For15 binary IsoMax+ heads:

```python
probs = dpe_model.predict_proba(target="val", aggregation="distance")
metrics = dpe_model.evaluate(target="val", aggregation="distance")
```

The frozen G1 rule uses normalized Euclidean feature/prototype distances:
`r=softmax(-50*min_class_distance)`, `w=.75/15+.25*r`, `p=sum(w*member_probs)`.
Member probabilities retain the actual learned logit scale at classifier
temperature1. The training entropic scale30 is not an inference temperature.
No labels, subgroup IDs, fitted calibration or normalizer enter this mode.
Raw-logit/member-output APIs remain available under the default uniform mode.

`src/dpe/verified_distance_v1.json` versions the parameters, class order,
distance representation and verification evidence. Alternative parameters are
rejected because they did not pass this promotion gate.

For a cache with `probs`, `distances`, `sample_id` (also supports nested `C_*`):

```powershell
$env:PYTHONPATH = "baselines/dpe4subpop/src"
.venv/Scripts/python.exe -m dpe.geometry --input-cache CACHE.npz --output NEW.npz --aggregation distance
```

Omitting `--aggregation` uses uniform. This inference command does not read
labels/groups and refuses to overwrite existing output.

Nested development results: mean WGA+0.501253 pp, AA+0.111204 pp, positive WGA
in2/3 head seeds. WGA95% CI[-0.214660,+1.253133] pp crosses zero. These are
engineering checks on previously inspected development with one shared ERM*
backbone, not standard ERM, independent backbone replicates or a fresh test claim.
See `results/dpe/geometry_aggregation_v5_20261115/FINAL_REPORT.md` in the repository.
