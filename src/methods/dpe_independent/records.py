"""One predictor per experiment, canonical exports, and independently replayed metrics."""
from datetime import datetime, timezone
import platform
import sys

import numpy as np
import scipy
import torch

from src.evaluation.prediction_export import sha256, write_predictions
from src.methods.dpe_head_reweighting.reweighting import GROUPS, metrics
from .storage import atomic_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def measurement(p, y, groups, uniform=None):
    y, groups = np.asarray(y), np.asarray(groups)
    result = metrics(p, y, groups)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
    brier = np.square(p[:, 1]-y)
    result.update(nll=float(nll.mean()), group_nll={g: float(nll[groups == g].mean()) for g in GROUPS},
                  group_brier={g: float(brier[groups == g].mean()) for g in GROUPS})
    result["group_balanced_nll"] = float(np.mean(list(result["group_nll"].values())))
    result["group_balanced_brier"] = float(np.mean(list(result["group_brier"].values())))
    if uniform is not None:
        before, after = uniform.argmax(1) == y, p.argmax(1) == y
        baseline = metrics(uniform, y, groups)
        result.update(delta_matched_uniform_pp=100*(result["wga"]-baseline["wga"]),
                      group_delta_pp={g: 100*(result["groups"][g]-baseline["groups"][g]) for g in GROUPS},
                      corrected_indices=np.flatnonzero(~before & after).tolist(),
                      new_error_indices=np.flatnonzero(before & ~after).tolist(),
                      changed_predictions=int(np.sum(before != after)))
    return result


def export_experiment(path, source, spec, config, state_path, head_paths, p, replay, identity, uniform,
                      diagnostics=None):
    path.mkdir(parents=True, exist_ok=True)
    difference = float(np.max(np.abs(p-replay)))
    label_changes = int(np.sum(p.argmax(1) != replay.argmax(1)))
    if difference > 1e-10 or label_changes:
        raise ValueError(f"Independent replay failed: max delta {difference}, changed labels {label_changes}")
    rows = source.test_rows
    y = np.array([r["y"] for r in rows]); groups = np.array([f"{r['y']}{r['a']}" for r in rows])
    result = measurement(p, y, groups, uniform)
    if "corrected_indices" in result:
        result["corrected_ids"] = [rows[i]["image_path"] for i in result.pop("corrected_indices")]
        result["new_error_ids"] = [rows[i]["image_path"] for i in result.pop("new_error_indices")]
    check = dict(status="passed", scope="frozen-feature inference; independent NumPy replay",
                 maximum_probability_difference=difference, changed_labels=label_changes, tolerance=1e-10,
                 limitation="No independent CNN image extraction replay")
    artifacts = [dict(id=f"heads-{i}", role="prototype_ensemble", location=str(h), sha256=sha256(h),
                      format="torch state dictionary", loader="src.methods.dpe_head_reweighting.ensemble.load")
                 for i, h in enumerate(head_paths)]
    artifacts.append(dict(id="fitted-state", role="inference_rule_and_fitted_parameters", location=str(state_path),
                          sha256=sha256(state_path), format="json", loader="src.methods.dpe_independent.runner.predict"))
    limitations = ["Exploratory reuse of sources and repeatedly inspected test set", "No image-level replay in this study"]
    if source.spec.get("backbone_sha256") is None:
        limitations.append("Historical ERM* raw-feature producer did not retain a backbone digest in its feature marker")
    model = dict(model_contract_version=1, experiment_id=path.name, method="DPE frozen-prototype project adaptation",
        seed=source.spec["seed"], availability="complete", implementation=identity["implementation"], artifacts=artifacts,
        input=dict(kind="features", shape=[None, 2048], dtype="float32", source=source.spec,
                   transformations=["per-example z-score ddof=0", "L2-normalize features and prototypes", "normalized Euclidean distance"],
                   feature_identities=identity["features"]),
        inference=dict(classes=[0, 1], configuration=spec, final_rule="argmax; ties predict landbird",
                       precision="float32 preserved row z-score; float64 normalized distances/pooling/fitting; original head probabilities rounded to float32"),
        selection=dict(configuration="fixed; no candidate selection", fitting="final fixed step or certified convex optimum",
                       heads="preserved source selection; new folds select WGA using only head-training pool"),
        verification=check, limitations=limitations)
    atomic_json(path/"model.json", model)
    atomic_json(path/"metrics.json", dict(configuration=spec, metrics=result, replay=check, diagnostics=diagnostics or {}))
    provenance = dict(method="DPE independent methods", source=source.spec, configuration=spec,
                      model=str(path/"model.json"), run=str(path/"run.json"),
                      metadata=dict(location=source.spec["metadata"], sha256=sha256(source.spec["metadata"])),
                      artifacts=artifacts, implementation=identity["implementation"],
                      selection="Every fixed configuration evaluated; no test-based selection", limitations=limitations)
    destination = path/"predictions/test"
    if not destination.exists():
        write_predictions(destination, rows, p, run_label=path.name, seed=source.spec["seed"], provenance=provenance,
                          ids=[r["image_path"] for r in rows])
    else:
        from .storage import read_json
        existing = read_json(destination/"provenance.json")
        if existing["csv_sha256"] != sha256(destination/"predictions.csv") or existing["provenance"] != provenance:
            raise ValueError("Completed canonical export changed")
    run = dict(record_contract_version=1, experiment_id=path.name, method=model["method"], seed=source.spec["seed"],
        purpose="Independent exploratory configuration; no winner selection", source=identity["implementation"],
        execution=dict(status="completed", completed_at=timestamp(), timezone="UTC", invocation=sys.argv,
                       effective_overrides={}, attempts="retained in suite ledger.json", duration_seconds=None),
        configuration=dict(resolved=config, predictor=spec), dataset=dict(metadata=source.spec["metadata"],
            metadata_sha256=sha256(source.spec["metadata"]), split="official test", counts=result["counts"],
            data_use="head/combiner validation partitions disjoint; crossfit uses out-of-fold predictions"),
        randomness=dict(source_seed=source.spec["seed"], fit_seed=source.spec["seed"]+config["seed_offset"],
                        partition="recorded in experiment ID and suite inputs", deterministic=True),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                         device=identity["device"], precision=model["inference"]["precision"], platform=platform.platform()),
        selection=model["selection"], artifacts=dict(model="model.json", metrics="metrics.json", predictions="predictions/test/predictions.csv",
            provenance="predictions/test/provenance.json", source_inputs=source.spec, fitted_state=str(state_path)),
        results=dict(test=result, units="accuracy fractions; paired deltas percentage points", verification=check),
        assessment=dict(reproduction_status="exploratory adaptation", limitations=limitations+["Per-predictor duration not measured; suite attempts retain execution timing"]),
        relationships=dict(source=source.spec["id"]))
    atomic_json(path/"run.json", run)
    return result
