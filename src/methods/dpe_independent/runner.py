"""Reuse-only study runner. All fits freeze before their arm's test evaluation."""
from pathlib import Path
import traceback

import numpy as np
import torch
from scipy.special import softmax

from src.evaluation.prediction_export import align_ids, sha256
from src.methods.dpe_head_reweighting import ensemble, neural_combiner as neural
from src.methods.dpe_head_reweighting.weighting_rules import joint_weights
from . import fitting, geometry, replay
from .grid import candidates, crossfit_candidates, combination_inputs, ORIGINAL
from .records import export_experiment, timestamp
from .sources import Source, folds
from .storage import (Budget, atomic_json, atomic_npz, frozen_json, implementation,
                      object_hash, read_json)


def input_heads(cache, spec):
    rep = spec.get("input", ORIGINAL)
    if rep["family"] == "original":
        return cache["probabilities"].astype(np.float64)
    z = geometry.scores(cache["distances"], cache["scales"], rep["scale"])
    return geometry.representation(z, rep)


def fit_one(spec, cache, y, groups, config, seed):
    family = spec["family"]
    if family in ("uniform", "h3", "local"):
        return None
    if family == "prototype":
        z = geometry.scores(cache["distances"], cache["scales"], spec["scale"])
        return fitting.fit_adam(z, y, groups, spec, config, seed)
    p = input_heads(cache, spec)
    if family == "residual":
        return fitting.fit_adam(p[:, :, 1].T, y, groups, spec, config, seed)
    if family == "brier":
        return fitting.fit_brier(p, y, groups, spec["penalty"], spec["loss"], config["solver"])
    if family == "joint":
        w, fit = joint_weights(p, y, groups, spec["penalty"], config["solver"])
        return dict(weights=w.tolist(), fit=fit)
    if family == "anchor":
        model, fit = neural.fit(p, y, groups, config, spec["kind"].split("_")[0], spec["weight_decay"],
                                .25 if spec["kind"] == "mlp_dropout" else 0., seed)
        return dict(model=model, fit=fit)
    raise ValueError("Unknown fitting family")


def predict(spec, state, cache, config, independent=False):
    family = spec["family"]
    if spec.get("alias_of") == "uniform":
        return cache["probabilities"].astype(float).mean(0)
    if family in ("h3", "local"):
        z = geometry.scores(cache["distances"], cache["scales"], spec["scale"])
        if independent:
            return replay.geometry_prediction(z, spec)
        if family == "h3":
            return geometry.subset_heads(z, spec["k"], spec["temperature"], spec["membership"]).mean(0)
        return geometry.local_topk(z, spec["k"], spec["temperature"])
    if family == "prototype":
        z = geometry.scores(cache["distances"], cache["scales"], spec["scale"])
        return replay.adam_prediction(state, z) if independent else fitting.predict_adam(state, z, config)
    p = input_heads(cache, spec)
    if family == "uniform":
        return p.mean(0)
    if family in ("brier", "joint"):
        w = np.asarray(state["weights"])
        return np.stack([p[h]*w[h] for h in range(len(w))]).sum(0) if independent else np.einsum("h,hnc->nc", w, p)
    if family == "residual":
        return replay.adam_prediction(state, p[:, :, 1].T) if independent else fitting.predict_adam(state, p[:, :, 1].T, config)
    if family == "anchor":
        return replay.anchor_prediction(state["model"], p) if independent else neural.predict(state["model"], p)[0]
    raise ValueError("Unknown prediction family")


def distance_cache(source, heads, head_path, split, indices, output, device):
    """Bind distance caches to feature bytes, exact head state, IDs and code."""
    path = output/f"{split}-distances.npz"
    identity = dict(features_sha256=sha256(source.spec["features"][split]), heads_sha256=sha256(head_path),
                    indices=np.asarray(indices).tolist(), preprocessing=ensemble.PREPROCESSING,
                    geometry_sha256=sha256(Path(geometry.__file__)), device=device)
    frozen_json(path.with_suffix(".identity.json"), identity)
    complete = path.with_suffix(".complete.json")
    if path.exists() and complete.exists():
        if sha256(path) != read_json(complete)["sha256"]:
            raise ValueError("Distance cache bytes changed")
        with np.load(path, allow_pickle=False) as value:
            result = dict(value)
    else:
        raw = source.features(split)[indices]
        d = geometry.distances(heads, raw, device)
        scales = heads["distance_scales"].numpy()
        result = dict(distances=d, scales=scales,
                      probabilities=softmax(geometry.scores(d, scales), axis=-1).astype(np.float32))
        reconstruction = softmax(geometry.scores(result["distances"], result["scales"]), axis=-1)
        delta = float(np.max(np.abs(reconstruction-result["probabilities"])))
        if delta > 2e-7:
            raise ValueError(f"Original pair probability reconstruction failed: {delta}")
        atomic_npz(path, **result)
        atomic_json(complete, dict(sha256=sha256(path)))
    if (result["distances"].shape != (15, len(indices), 2) or not np.isfinite(result["distances"]).all()
            or result["probabilities"].shape != result["distances"].shape):
        raise ValueError("Invalid distance cache")
    return result


def arm_identity(source, config, heads, implementation_record, device):
    return dict(source=source.spec, config=config, heads=[dict(path=str(h), sha256=sha256(h)) for h in heads],
                features={k: dict(path=p, sha256=sha256(p)) for k, p in source.spec["features"].items()},
                metadata_sha256=sha256(source.spec["metadata"]),
                provenance_sha256=sha256(source.spec["provenance"]),
                implementation=implementation_record, device=device)


def progress(path, rows):
    atomic_json(path, rows)


def ledger(output, specs):
    path = output/"ledger.json"
    if path.exists():
        rows = read_json(path)
        if set(rows) != {r["id"] for r in specs}:
            raise ValueError("Declared grid changed")
    else:
        rows = {r["id"]: dict(status="planned", configuration=r, attempts=[]) for r in specs}
        progress(path, rows)
    return path, rows


def run_arm(source, arm, output, config, impl, budget, device="cpu"):
    output.mkdir(parents=True, exist_ok=True)
    specs = candidates(config, fitted=arm != "full_validation")
    ledger_path, states = ledger(output, specs)
    try:
        heads, head_path = source.heads(arm, output.parent/"heads", config, device)
        _, fit_idx = source.head_indices(arm)
        identity = arm_identity(source, config, [head_path], impl, device)
        frozen_json(output/"inputs.json", identity)
        fit_cache = None
        if len(fit_idx):
            fit_cache = distance_cache(source, heads, head_path, "val", fit_idx, output/"cache", device)
        y = np.array([source.rows[i]["y"] for i in fit_idx])
        groups = np.array([source.rows[i]["group"] for i in fit_idx])
        # This entire fitting loop precedes opening test feature arrays.
        for spec in specs:
            budget.check()
            row = states[spec["id"]]
            model_path = output/"frozen"/(spec["id"]+".json")
            if model_path.exists():
                frozen = read_json(model_path)
                if frozen["configuration"] != spec or frozen["identity"] != object_hash(identity):
                    raise ValueError("Frozen predictor record changed")
                continue
            if row["status"] == "failed":
                continue
            attempt = dict(started_at=timestamp(), phase="fit")
            row["attempts"].append(attempt)
            row["status"] = "fitting"
            progress(ledger_path, states)
            try:
                model = None if "alias_of" in spec else fit_one(spec, fit_cache, y, groups, config,
                                                              source.spec["seed"]+config["seed_offset"])
                frozen_json(model_path, dict(configuration=spec, model=model, identity=object_hash(identity)))
                attempt.update(status="completed", completed_at=timestamp())
                row["status"] = "frozen"
            except Exception as exc:
                attempt.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                row.update(status="failed", error=str(exc))
            progress(ledger_path, states)
        frozen_json(output/"fit_complete.json", {r["id"]: dict(
            status="failed" if states[r["id"]]["status"] == "failed" else "frozen",
            sha256=sha256(output/"frozen"/(r["id"]+".json")) if (output/"frozen"/(r["id"]+".json")).exists() else None) for r in specs})
        budget.check()
        test_cache = distance_cache(source, heads, head_path, "test", np.arange(5794), output/"cache", device)
        historical = source.spec.get("historical_predictions", {}).get(arm)
        if historical and Path(historical).exists():
            with np.load(historical, allow_pickle=False) as value:
                order = align_ids(value["ids"], source.test_rows)
                old = value["probabilities"][:, order]
                if (not np.array_equal(value["labels"][order], [r["y"] for r in source.test_rows]) or
                    not np.array_equal(value["groups"][order].astype(str), [f"{r['y']}{r['a']}" for r in source.test_rows])):
                    raise ValueError("Historical test cache labels/groups differ from reference")
            delta = float(np.max(np.abs(test_cache["probabilities"]-old)))
            changes = int(np.sum(test_cache["probabilities"].astype(float).mean(0).argmax(1) != old.astype(float).mean(0).argmax(1)))
            atomic_json(output/"baseline_replay.json", dict(maximum_probability_difference=delta, changed_uniform_decisions=changes, tolerance=2e-7))
            if delta > 2e-7 or changes:
                raise ValueError(f"Historical baseline replay failed: delta={delta}, decisions={changes}")
        uniform = test_cache["probabilities"].astype(float).mean(0)
        prediction_cache = {}
        by_id = {r["id"]: r for r in specs}

        def prediction(spec, independent=False):
            if "alias_of" in spec:
                return prediction(by_id[spec["alias_of"]], independent)
            key = (spec["id"], independent)
            if key not in prediction_cache:
                value = read_json(output/"frozen"/(spec["id"]+".json"))
                prediction_cache[key] = predict(spec, value["model"], test_cache, config, independent)
            return prediction_cache[key]

        for index, spec in enumerate(specs):
            budget.check()
            row = states[spec["id"]]
            if row["status"] in ("failed", "completed", "alias"):
                continue
            try:
                p, check = prediction(spec), prediction(spec, True)
                model_path = output/"frozen"/(spec["id"]+".json")
                diagnostics = {}
                if spec["family"] == "local":
                    z = geometry.scores(test_cache["distances"], test_cache["scales"], spec["scale"])
                    diagnostics["prototype_usage"] = geometry.usage(z, spec["k"], spec["temperature"],
                        np.array([f"{r['y']}{r['a']}" for r in source.test_rows]))
                state = read_json(model_path)["model"]
                if state and "theta" in state.get("state", {}):
                    alpha = softmax(np.asarray(state["state"]["theta"]), axis=0)
                    diagnostics["prototype_weights"] = alpha.tolist()
                    diagnostics["effective_prototypes_per_class"] = (1/np.square(alpha).sum(0)).tolist()
                if state and "weights" in state:
                    diagnostics["effective_heads"] = float(1/np.square(state["weights"]).sum())
                if spec.get("input") or spec["family"] == "h3":
                    hp = (geometry.subset_heads(geometry.scores(test_cache["distances"], test_cache["scales"], spec["scale"]),
                          spec["k"], spec["temperature"], spec["membership"]) if spec["family"] == "h3" else input_heads(test_cache, spec))
                    diagnostics["mean_head_probability_sd"] = float(hp[:, :, 1].std(0).mean())
                    diagnostics["mean_absolute_margin"] = float(np.abs(p[:, 1]-.5).mean())
                target = output/"experiments"/(source.spec["id"]+"-"+arm+"-"+spec["id"])
                m = export_experiment(target, source, spec, config, model_path, [head_path], p, check, identity, uniform, diagnostics)
                row.update(status="alias" if "alias_of" in spec else "completed", metrics=m,
                           experiment=str(target), completed_at=timestamp())
            except Exception as exc:
                row.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            progress(ledger_path, states)
            if index % 20 == 0:
                print(source.spec["id"], arm, index+1, "/", len(specs), flush=True)
    except TimeoutError:
        raise
    except Exception as exc:
        atomic_json(output/"arm_failure.json", dict(error=str(exc), traceback=traceback.format_exc()))
        for row in states.values():
            if row["status"] not in ("completed", "alias", "failed"):
                row.update(status="blocked", error=str(exc))
        progress(ledger_path, states)
    return states


def run_crossfit(source, output, config, impl, budget, device="cpu"):
    output.mkdir(parents=True, exist_ok=True)
    specs = crossfit_candidates(config)
    ledger_path, states = ledger(output, specs)
    assignments = folds(source.rows, config["crossfit"]["folds"], config["crossfit"]["split_seed"])
    frozen_json(output/"folds.json", dict(ids=[r["id"] for r in source.rows], folds=assignments.tolist()))
    representations = [ORIGINAL, *combination_inputs(config)]
    oof = {object_hash(rep): np.full((1199, 2), np.nan) for rep in representations}
    saved_heads = []
    for fold in range(5):
        budget.check()
        train_idx, heldout = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
        heads, path = source.train_heads(train_idx, output/"heads"/f"fold{fold}"/"ensemble.pt", config, device)
        saved_heads.append((heads, path))
        cache = distance_cache(source, heads, path, "val", heldout, output/"cache"/f"fold{fold}", device)
        for rep in representations:
            oof[object_hash(rep)][heldout] = input_heads(cache, dict(input=rep)).mean(0)
    if any(not np.isfinite(value).all() for value in oof.values()):
        raise ValueError("Incomplete out-of-fold coverage")
    identity = arm_identity(source, config, [p for _, p in saved_heads], impl, device)
    frozen_json(output/"inputs.json", identity)
    y = np.array([r["y"] for r in source.rows]); groups = np.array([r["group"] for r in source.rows])
    for spec in specs:
        budget.check()
        path = output/"frozen"/(spec["id"]+".json")
        row = states[spec["id"]]
        if path.exists():
            if read_json(path)["identity"] != object_hash(identity):
                raise ValueError("Crossfit frozen inputs changed")
            continue
        if row["status"] == "failed":
            continue
        row["attempts"].append(dict(started_at=timestamp(), phase="fit"))
        try:
            q = oof[object_hash(spec["input"])][:, 1, None]
            model = None if spec["family"] == "uniform" else fitting.fit_adam(q, y, groups, spec, config,
                                                                             source.spec["seed"]+config["seed_offset"])
            frozen_json(path, dict(configuration=spec, model=model, identity=object_hash(identity)))
            row["status"] = "frozen"
            row["attempts"][-1].update(status="completed", completed_at=timestamp())
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            row["attempts"][-1].update(status="failed", error=str(exc))
        progress(ledger_path, states)
    frozen_json(output/"fit_complete.json", {s["id"]: dict(
        status="failed" if states[s["id"]]["status"] == "failed" else "frozen",
        sha256=sha256(output/"frozen"/(s["id"]+".json")) if (output/"frozen"/(s["id"]+".json")).exists() else None) for s in specs})
    caches = []
    for fold, (heads, path) in enumerate(saved_heads):
        budget.check()
        caches.append(distance_cache(source, heads, path, "test", np.arange(5794), output/"cache"/f"fold{fold}", device))
    original_uniform = np.mean([input_heads(c, dict(input=ORIGINAL)).mean(0) for c in caches], axis=0)
    for spec in specs:
        budget.check()
        row = states[spec["id"]]
        if row["status"] in ("failed", "completed"):
            continue
        try:
            state_path = output/"frozen"/(spec["id"]+".json")
            model = read_json(state_path)["model"]
            per_fold = [input_heads(c, spec).mean(0) for c in caches]
            if model is None:
                p = np.mean(per_fold, axis=0)
            else:
                p = np.mean([fitting.predict_adam(model, v[:, 1, None], config) for v in per_fold], axis=0)
            independent_folds = []
            for c in caches:
                rep = spec["input"]
                if rep["family"] == "original":
                    value = sum(c["probabilities"].astype(float))/15
                else:
                    value = replay.geometry_prediction(geometry.scores(c["distances"], c["scales"], rep["scale"]), rep)
                independent_folds.append(value if model is None else replay.adam_prediction(model, value[:, 1, None]))
            check = np.mean(independent_folds, axis=0)
            target = output/"experiments"/(source.spec["id"]+"-"+spec["id"])
            m = export_experiment(target, source, spec, config, state_path, [path for _, path in saved_heads],
                                  p, check, identity, original_uniform,
                                  dict(representation_uncalibrated_wga=neural.metrics(np.mean(per_fold, axis=0),
                                      np.array([r["y"] for r in source.test_rows]),
                                      np.array([f"{r['y']}{r['a']}" for r in source.test_rows]))["wga"]))
            row.update(status="completed", metrics=m, experiment=str(target), completed_at=timestamp())
        except Exception as exc:
            row.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        progress(ledger_path, states)
    return states


def run_study(source_specs, output, config, device="cpu", include_crossfit=True):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    budget = Budget(config["timeout_hours"])
    impl = implementation(output)
    frozen_json(output/"study.json", dict(sources=source_specs, config=config, device=device, crossfit=include_crossfit))
    for source_spec in source_specs:
        source = Source(source_spec)
        for arm in ("full_validation", *[f"partition{p}" for p in config["partitions"]]):
            budget.check()
            print("RUN", source_spec["id"], arm, flush=True)
            run_arm(source, arm, output/source_spec["id"]/arm, config, impl, budget, device)
        if include_crossfit:
            try:
                run_crossfit(source, output/source_spec["id"]/"crossfit", config, impl, budget, device)
            except TimeoutError:
                raise
            except Exception as exc:
                path = output/source_spec["id"]/"crossfit"
                atomic_json(path/"arm_failure.json", dict(error=str(exc), traceback=traceback.format_exc()))
                lp, states = ledger(path, crossfit_candidates(config))
                for row in states.values():
                    if row["status"] not in ("completed", "failed"):
                        row.update(status="blocked", error=str(exc))
                progress(lp, states)
    from .reporting import report
    return report(output)
