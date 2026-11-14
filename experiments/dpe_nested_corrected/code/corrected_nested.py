"""Strictly nested DPE selection and explicit post-average threshold controls.

Frozen backbones; official validation labels alone fit/select all adapters.
Primary rule remains balanced NLL; WGA selection is secondary. Previously
inspected test data mean this corrected run remains exploratory evidence.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import nested_selection as old

HERE = Path(__file__).resolve().parent
PROTOCOL = "strict-nested-postmean-v2"
INNER_FOLDS = 5
CROSSFIT_FOLDS = 5
SETTINGS = old.SETTINGS
FIXED = old.FIXED
RULES = old.RULES
FAMILIES = ("crossfit_affine", "crossfit_bias_foldwise", "crossfit_postmean_bias",
            "fullval_affine_oof", "fullval_postmean_bias")
PRIMARY_RULE = "nll"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_identity():
    paths = sorted(p for p in HERE.rglob("*") if p.is_file() and
                   p.suffix in (".py", ".yaml", ".json") and "__pycache__" not in p.parts)
    return digest_json({str(p.relative_to(HERE)).replace("\\", "/"): digest_file(p) for p in paths})


def bind_manifest(path, value):
    path = Path(path)
    if path.exists():
        if old.read_json(path) != value:
            raise ValueError(f"Cache identity mismatch: {path}. Use a fresh output directory.")
    else:
        old.write_json(path, value)


def prepare_protocol(out, cfg_path=None):
    import scipy
    import torch
    import sklearn
    cfg = old.adapters()["grid"].load_config(cfg_path) if cfg_path else old.adapters()["grid"].load_config()
    identity = dict(protocol=PROTOCOL, code_sha256=code_identity(), config=cfg,
                    inner_folds=INNER_FOLDS, crossfit_folds=CROSSFIT_FOLDS,
                    settings=SETTINGS, primary_rule=PRIMARY_RULE, secondary_rule="wga",
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    torch=torch.__version__, sklearn=sklearn.__version__)
    # JSON round trip normalizes tuples before comparing with a saved manifest.
    identity = json.loads(json.dumps(identity))
    bind_manifest(Path(out) / "protocol.json", identity)
    return cfg, digest_json(identity)


class BudgetReached(Exception):
    pass


def check_budget(deadline):
    if time.time() >= deadline:
        raise BudgetReached("Time budget reached between checkpointed head fits")


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def fit_threshold_exact(x, y, g):
    """Exact empirical WGA maximizer for p > t; include all-positive/all-negative.

    Evaluate every distinct prediction partition. Ties prefer the threshold
    nearest 0.5, then the smaller threshold. Equal scores always move together.
    Thresholds are stored/applied directly in probability space to preserve ties.
    """
    p = np.asarray(x, dtype=np.float64).reshape(-1)
    y, g = np.asarray(y), np.asarray(g)
    if p.shape != y.shape or y.shape != g.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid threshold inputs")
    if (p < 0).any() or (p > 1).any() or not np.isin(y, [0, 1]).all() or set(g) != set(old.GROUPS):
        raise ValueError("Expected probabilities and all four groups")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    unique = np.unique(sorted_p)
    candidates = np.unique(np.r_[np.nextafter(unique, -np.inf), unique, .5])
    cut = np.searchsorted(sorted_p, candidates, side="right")
    group_acc = []
    for group in old.GROUPS:
        mask = g[order] == group
        # Initially all predictions are positive; moving a row to negative
        # adds one correct for y=0 and removes one correct for y=1.
        changes = np.where(mask, 1 - 2 * y[order], 0)
        prefix = np.r_[0, np.cumsum(changes)]
        correct = np.sum(mask & (y[order] == 1)) + prefix[cut]
        group_acc.append(correct / mask.sum())
    wga = np.min(group_acc, axis=0)
    best = min(range(len(candidates)), key=lambda i: (-round(float(wga[i]), 12),
                                                     abs(float(candidates[i]) - .5), float(candidates[i])))
    t = float(candidates[best])
    default = min(old.group_accuracy(p > .5, y, g).values()) * 100
    coarse = old.fit_threshold_wga(p[:, None], y, g)
    coarse_wga = min(old.group_accuracy(old.logit_clip(p) + coarse["b"] > 0, y, g).values()) * 100
    return dict(kind="exact_threshold", threshold=t, fit_wga=float(wga[best] * 100),
                default_fit_wga=default, legacy_grid_fit_wga=coarse_wga,
                fit_prediction_changes=int(np.sum((p > t) != (p > .5))))


def apply_threshold(model, p):
    return np.asarray(p) > model["threshold"]


def calibration_predictions(affine, bias, fold_scores, full_scores, cfg):
    """Exactly the same inference graph in selection and final test evaluation."""
    mean = np.mean(fold_scores, axis=0)
    apply = lambda model, p: old.predict_model(model, np.asarray(p)[:, None], cfg)
    return dict(crossfit_affine=np.mean([apply(affine, p) for p in fold_scores], axis=0),
                crossfit_bias_foldwise=np.mean([apply(bias, p) for p in fold_scores], axis=0),
                crossfit_postmean_bias=apply(bias, mean),
                fullval_affine_oof=apply(affine, full_scores),
                fullval_postmean_bias=apply(bias, full_scores))


def cached_scores(source, train_idx, val_eval_idx, test_eval, seed, cfg, directory,
                  deadline, forbidden=()):
    """Fit on explicit validation indices; cache predictions, never fitted labels.

    forbidden contains the entire current inner-evaluation fold, not only the
    rows receiving OOF scores in this individual head fit.
    """
    tr, ve = np.asarray(train_idx, dtype=int), np.asarray(val_eval_idx, dtype=int)
    forbidden = np.asarray(forbidden, dtype=int)
    if len(set(tr)) != len(tr) or np.intersect1d(tr, ve).size or np.intersect1d(tr, forbidden).size:
        raise ValueError("Head training overlaps its held-out validation rows")
    stage = Path(directory)
    stage.mkdir(parents=True, exist_ok=True)
    identity = dict(train_idx=tr.tolist(), val_eval_idx=ve.tolist(),
                    forbidden_idx=forbidden.tolist(), test_eval=bool(test_eval), seed=int(seed))
    bind_manifest(stage / "indices.json", identity)
    cache = stage / "scores.npz"
    marker = stage / "complete.json"
    if cache.exists() and marker.exists():
        if old.read_json(marker)["sha256"] != digest_file(cache):
            raise ValueError(f"Prediction checkpoint checksum mismatch: {cache}")
        with np.load(cache, allow_pickle=False) as data:
            val, test = data["val"], data["test"]
        if val.shape != (len(ve),) or test.shape != ((len(source._test),) if test_eval else (0,)):
            raise ValueError("Prediction checkpoint shape mismatch")
        return val, test
    check_budget(deadline)
    heads = old.train_ensemble(source, tr, seed, cfg, stage)
    val = old.head_probabilities(heads, source._val[ve]).mean(0)[:, 1] if len(ve) else np.empty(0)
    test = old.head_probabilities(heads, source._test).mean(0)[:, 1] if test_eval else np.empty(0)
    if not np.isfinite(val).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite head probabilities")
    atomic_npz(cache, val=val, test=test)
    old.write_json(marker, dict(sha256=digest_file(cache)))
    return val, test


def build_context(source, pool, evaluation, test_eval, fold_seed, head_seed, cfg,
                  directory, deadline, forbidden=()):
    """OOF training scores + five-model and full-refit evaluation predictions.

    During inner selection, pool excludes the entire inner evaluation fold.
    Every sub-head and the full-refit head are trained inside pool only.
    """
    pool, evaluation = np.asarray(pool, dtype=int), np.asarray(evaluation, dtype=int)
    if np.intersect1d(pool, evaluation).size or np.intersect1d(pool, forbidden).size:
        raise ValueError("Context pool overlaps an inner evaluation fold")
    rows = [source.rows[i] for i in pool]
    assignment = old.adapters()["sources"].folds(rows, CROSSFIT_FOLDS, fold_seed)
    oof = np.full(len(pool), np.nan)
    fold_eval = []
    for f in range(CROSSFIT_FOLDS):
        tr = pool[assignment != f]
        te_local = np.flatnonzero(assignment == f)
        held = pool[te_local]
        eval_idx = np.r_[held, evaluation]
        pv, pt = cached_scores(source, tr, eval_idx, test_eval, head_seed, cfg,
                               Path(directory) / f"fold{f}", deadline, forbidden)
        oof[te_local] = pv[:len(held)]
        fold_eval.append(pt if test_eval else pv[len(held):])
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete OOF scores")
    pv, pt = cached_scores(source, pool, evaluation, test_eval, head_seed, cfg,
                           Path(directory) / "full_refit", deadline, forbidden)
    return dict(oof=oof, folds=np.stack(fold_eval), full=pt if test_eval else pv)


def strict_inner_scores(source, split, cfg, base, fit_seed, directory, deadline):
    """Evaluate complete procedures on inner folds unseen by every fitted stage."""
    n = len(source.rows)
    assignment = old.adapters()["sources"].folds(source.rows, INNER_FOLDS, 10_000 + split)
    predictions = {family: {old.name(s): np.full(n, np.nan) for s in SETTINGS} for family in FAMILIES}
    threshold_predictions = {family: np.zeros(n, dtype=bool) for family in
                             ("crossfit_postmean_thrwga", "fullval_postmean_thrwga")}
    threshold_logs = []
    for f in range(INNER_FOLDS):
        tr, te = np.flatnonzero(assignment != f), np.flatnonzero(assignment == f)
        context = build_context(source, tr, te, False, split, base, cfg,
                                Path(directory) / f"inner{f}", deadline, forbidden=te)
        y, g = source.val_y[tr], source.val_g[tr]
        x = context["oof"][:, None]
        for setting in SETTINGS:
            affine = old.fit_model(x, y, g, "affine", setting, cfg, fit_seed)
            bias = old.fit_model(x, y, g, "bias", setting, cfg, fit_seed)
            for family, p in calibration_predictions(affine, bias, context["folds"], context["full"], cfg).items():
                predictions[family][old.name(setting)][te] = p
        threshold = fit_threshold_exact(x, y, g)
        threshold_logs.append(dict(inner_fold=f, model=threshold))
        threshold_predictions["crossfit_postmean_thrwga"][te] = apply_threshold(threshold, context["folds"].mean(0))
        threshold_predictions["fullval_postmean_thrwga"][te] = apply_threshold(threshold, context["full"])
    scores = {}
    for family, settings in predictions.items():
        if any(not np.isfinite(p).all() for p in settings.values()):
            raise ValueError("Incomplete inner evaluation predictions")
        scores[family] = {name: old.inner_score(p, source.val_y, source.val_g) for name, p in settings.items()}
    diagnostics = {family: old.record(p, source.val_y, source.val_g) for family, p in threshold_predictions.items()}
    return scores, dict(threshold_folds=threshold_logs, inner_threshold_performance=diagnostics)


def final_predictions(source, context, inner, cfg, fit_seed):
    y, g = source.val_y, source.val_g
    x = context["oof"][:, None]
    grid = {family: {} for family in FAMILIES}
    models = {}
    for setting in SETTINGS:
        affine = old.fit_model(x, y, g, "affine", setting, cfg, fit_seed)
        bias = old.fit_model(x, y, g, "bias", setting, cfg, fit_seed)
        models[old.name(setting)] = dict(affine=affine, bias=bias)
        for family, p in calibration_predictions(affine, bias, context["folds"], context["full"], cfg).items():
            grid[family][old.name(setting)] = p
    # Freeze every choice without consulting test labels or test metrics.
    chosen = {family: {rule: old.choose(inner[family], rule) for rule in RULES} for family in FAMILIES}
    threshold = fit_threshold_exact(x, y, g)
    return grid, chosen, threshold, models


def dfr_clean(source, split):
    """Fit preprocessing inside each CV training fold, then refit on validation."""
    X = source._val.astype(np.float64)
    y, g = source.val_y, source.val_g
    assignment = old.adapters()["sources"].folds(source.rows, 5, 20_000 + split)
    predictions = {C: np.full(len(y), np.nan) for C in old.DFR_C}
    for f in range(5):
        tr, te = assignment != f, assignment == f
        mean, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        for C in old.DFR_C:
            clf = old.fit_l1((X[tr] - mean) / sd, y[tr], old.group_weights(g[tr]), C)
            predictions[C][te] = clf.predict_proba((X[te] - mean) / sd)[:, 1]
    scores = {f"C{C:g}": old.inner_score(p, y, g) for C, p in predictions.items()}
    best = min(scores, key=lambda n: (-round(scores[n]["wga"], 6), round(scores[n]["nll"], 9), float(n[1:])))
    mean, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    clf = old.fit_l1((X - mean) / sd, y, old.group_weights(g), float(best[1:]))
    p = clf.predict_proba((source._test.astype(np.float64) - mean) / sd)[:, 1]
    return p, dict(selected=best, inner=scores, nonzero=int((clf.coef_ != 0).sum()))


def pooled_predictions(source, split, cfg, base, fit_seed, directory, deadline):
    """Retain the original valid 60/40 controls, with checkpointed head scores."""
    part = old.adapters()["sources"].partition(source.rows, split)
    lookup = {r["id"]: i for i, r in enumerate(source.rows)}
    head_idx = np.array([lookup[r["id"]] for r in part["head_training"]])
    fit_idx = np.array([lookup[r["id"]] for r in part["combiner_training"]])
    if np.intersect1d(head_idx, fit_idx).size:
        raise ValueError("Pooled head/combiner overlap")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bind_manifest(directory / "indices.json", dict(head_idx=head_idx.tolist(), fit_idx=fit_idx.tolist()))
    path, marker = directory / "head_scores.npz", directory / "complete.json"
    if path.exists() and marker.exists():
        if old.read_json(marker)["sha256"] != digest_file(path):
            raise ValueError("Pooled head-score checksum mismatch")
        with np.load(path, allow_pickle=False) as value:
            qf, qt = value["fit"], value["test"]
    else:
        check_budget(deadline)
        heads = old.train_ensemble(source, head_idx, base, cfg, directory)
        qf = old.head_probabilities(heads, source._val[fit_idx])
        qt = old.head_probabilities(heads, source._test)
        atomic_npz(path, fit=qf, test=qt)
        old.write_json(marker, dict(sha256=digest_file(path)))
    xf, xt = qf[:, :, 1].T, qt[:, :, 1].T
    y, g = source.val_y[fit_idx], source.val_g[fit_idx]
    rows = [source.rows[i] for i in fit_idx]
    inner, grid, chosen = {}, {}, {}
    for kind in ("affine", "linear"):
        family = "pool_" + kind
        inner[family] = old.inner_cv_scores(xf, y, g, rows, kind, cfg, fit_seed, split)
        grid[family] = {}
        for setting in SETTINGS:
            model = old.fit_model(xf, y, g, kind, setting, cfg, fit_seed)
            grid[family][old.name(setting)] = old.predict_model(model, xt, cfg)
        chosen[family] = {rule: old.choose(inner[family], rule) for rule in RULES}
    return qt.mean(0)[:, 1], grid, inner, chosen


def run_task(task):
    spec, split, out_dir, cfg, protocol_id, deadline = task
    started = time.time()
    out = Path(out_dir)
    cell = out / "results" / spec["id"] / f"split{split:02d}.json"
    checkpoints = out / "checkpoints" / spec["id"] / f"split{split:02d}"
    try:
        import torch
        torch.set_num_threads(1)
        source = old.get_source(spec)
        source_id = dict(id=spec["id"], recipe=spec["recipe"], seed=spec["seed"],
                         features={k: digest_file(v) for k, v in spec["features"].items()},
                         metadata=digest_file(spec["metadata"]))
        fingerprint = digest_json(dict(protocol=protocol_id, source=source_id, split=split))
        bind_manifest(checkpoints / "identity.json", dict(fingerprint=fingerprint, source=source_id))
        if cell.exists():
            if old.read_json(cell).get("fingerprint") != fingerprint:
                raise ValueError("Completed result was produced by another protocol/source")
            (out / "errors" / f"{spec['id']}-split{split:02d}.txt").unlink(missing_ok=True)
            return spec["id"], split, "cached", 0.
        base = spec["seed"] + old.SPLIT_SEED_STRIDE * split
        fit_seed = spec["seed"] + cfg["seed_offset"]
        inner_path = checkpoints / "inner_scores.json"
        if inner_path.exists():
            saved = old.read_json(inner_path)
            inner, diagnostics = saved["scores"], saved["diagnostics"]
        else:
            inner, diagnostics = strict_inner_scores(source, split, cfg, base, fit_seed,
                                                     checkpoints, deadline)
            old.write_json(inner_path, dict(scores=inner, diagnostics=diagnostics))
        context = build_context(source, np.arange(len(source.rows)), np.array([], dtype=int),
                                True, split, base, cfg, checkpoints / "final", deadline)
        grid_prob, chosen, threshold, models = final_predictions(source, context, inner, cfg, fit_seed)
        prob_arms = dict(fullval_uniform=context["full"], crossfit_uniform=context["folds"].mean(0))
        full_bag = [context["full"]]
        for k in range(1, old.BAG_ENSEMBLES):
            _, p = cached_scores(source, np.arange(len(source.rows)), np.array([], dtype=int), True,
                                  base + k, cfg, checkpoints / f"bag{k}", deadline)
            full_bag.append(p)
        prob_arms["bag75_uniform"] = np.mean(full_bag, axis=0)
        selected = {}
        for family in FAMILIES:
            prob_arms[family + "_fixed"] = grid_prob[family][old.name(FIXED)]
            selected[family + "_fixed"] = old.name(FIXED)
            for rule in RULES:
                arm = family + "_sel_" + rule
                selected[arm] = chosen[family][rule]
                prob_arms[arm] = grid_prob[family][chosen[family][rule]]
        dfr_prob, dfr_info = dfr_clean(source, split)
        prob_arms["dfr_lite"] = dfr_prob
        pool_uniform, pool_grid, pool_inner, pool_chosen = pooled_predictions(
            source, split, cfg, base, fit_seed, checkpoints / "pooled", deadline)
        prob_arms["pool_uniform"] = pool_uniform
        inner.update(pool_inner)
        chosen.update(pool_chosen)
        grid_prob.update(pool_grid)
        for family in pool_grid:
            prob_arms[family + "_fixed"] = pool_grid[family][old.name(FIXED)]
            selected[family + "_fixed"] = old.name(FIXED)
            for rule in RULES:
                arm = family + "_sel_" + rule
                selected[arm] = pool_chosen[family][rule]
                prob_arms[arm] = pool_grid[family][selected[arm]]
        # Durable fit/selection record is written before test labels are scored.
        old.write_json(checkpoints / "fit_complete.json", dict(chosen=chosen, threshold=threshold,
                       models=models, dfr=dfr_info, fingerprint=fingerprint))
        ty, tg = source.test_y, source.test_g
        arms = {arm: old.record(p > .5, ty, tg, **({"selected": selected[arm]} if arm in selected else {}))
                for arm, p in prob_arms.items()}
        arms["dfr_lite"].update(dfr_info)
        arms["crossfit_postmean_thrwga"] = old.record(apply_threshold(threshold, context["folds"].mean(0)), ty, tg, **threshold)
        arms["fullval_postmean_thrwga"] = old.record(apply_threshold(threshold, context["full"]), ty, tg, **threshold)
        grid = {family: {name: old.record(p > .5, ty, tg) for name, p in settings.items()}
                for family, settings in grid_prob.items()}
        old.write_json(cell, dict(protocol=PROTOCOL, fingerprint=fingerprint, source=spec["id"],
                       recipe=spec["recipe"], seed=spec["seed"], split=split, arms=arms, grid=grid,
                       inner=inner, diagnostics=diagnostics, seconds=time.time() - started))
        error = out / "errors" / f"{spec['id']}-split{split:02d}.txt"
        error.unlink(missing_ok=True)
        return spec["id"], split, "done", time.time() - started
    except BudgetReached:
        return spec["id"], split, "deferred", time.time() - started
    except Exception:
        error = out / "errors" / f"{spec['id']}-split{split:02d}.txt"
        error.parent.mkdir(parents=True, exist_ok=True)
        error.write_text(traceback.format_exc())
        return spec["id"], split, "ERROR", time.time() - started


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    if a.workers < 1 or not math.isfinite(a.time_limit_hours) or a.time_limit_hours <= 0 or not a.splits:
        raise ValueError("Positive workers/time budget and at least one split are required")
    cfg, protocol_id = prepare_protocol(a.out, a.config)
    specs = old.load_specs(a.out, a.only)
    if not specs:
        raise ValueError("No selected sources")
    if len(set(a.splits)) != len(a.splits) or any(s < 0 or s > 9 for s in a.splits):
        raise ValueError("Choose distinct split seeds in 0..9")
    deadline = time.time() + a.time_limit_hours * 3600
    tasks = [(s, split, str(a.out), cfg, protocol_id, deadline) for split in a.splits for s in specs]
    status_counts = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        queue, running = iter(tasks), {}
        def submit():
            while len(running) < a.workers and time.time() < deadline:
                task = next(queue, None)
                if task is None:
                    break
                running[pool.submit(run_task, task)] = task
        submit()
        while running:
            future = next(as_completed(running))
            running.pop(future)
            sid, split, status, seconds = future.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            old.log(status, sid, f"split{split:02d}", f"{seconds / 60:.1f} min")
            submit()
    complete = sum((Path(a.out) / "results" / s["id"] / f"split{split:02d}.json").exists()
                   for split in a.splits for s in specs)
    old.write_json(Path(a.out) / "completion.json", dict(protocol=PROTOCOL, requested=len(tasks),
                   completed=complete, splits=a.splits, sources=[s["id"] for s in specs], statuses=status_counts))
    old.log(f"Completed {complete}/{len(tasks)} requested cells; stage checkpoints retained.")
    return 1 if status_counts.get("ERROR") else (0 if complete == len(tasks) else 3)


def cmd_summarize(a):
    # Keep the established per-source pairing and t-interval arithmetic.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = old.load_results(out)
    if any(c.get("protocol") != PROTOCOL for c in cells):
        raise ValueError("Mixed/legacy results cannot be summarized as corrected results")
    planned_sources = [s["id"] for s in old.load_specs(out)]
    present = {(cell["source"], cell["split"]) for cell in cells}
    missing = [(s, split) for s in planned_sources for split in range(10)
               if (s, split) not in present]
    fields = ["source", "recipe", "split", "arm", "wga", "overall", "selected",
              *["g" + k for k in old.GROUPS], "protocol", "fingerprint"]
    with (out / "results_long.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for arm, value in cell["arms"].items():
                writer.writerow(dict(source=cell["source"], recipe=cell["recipe"], split=cell["split"],
                                     arm=arm, wga=value["wga"], overall=value["overall"], selected=value.get("selected", ""),
                                     **{"g" + k: value["groups"][k] for k in old.GROUPS},
                                     protocol=cell["protocol"], fingerprint=cell["fingerprint"]))
    lines = ["# Corrected protocol diagnostics", "", f"Protocol: `{PROTOCOL}`.",
             f"Completed: {len(cells)}/{10 * len(planned_sources)} planned source × split cells.",
             "PARTIAL RUN: do not treat interim means as the completed experiment." if missing else "All declared cells completed.",
             "Primary rule remains balanced NLL; WGA is secondary. Prior test inspection makes this exploratory.",
             "Strict inner CV rebuilds all heads and calibrators inside each training fold and evaluates the complete inference graph.",
             "`postmean` applies one scalar threshold/bias AFTER uniform probability averaging.",
             "`bias_foldwise` applies a bias before averaging and is a distinct predictor.",
             "The full-validation OOF calibrator transfer is evaluated separately inside inner CV.",
             "Original pooled 60/40 affine/linear controls are retained; DFR preprocessing is fitted inside CV folds.", ""]
    all_arms = sorted({arm for cell in cells for arm in cell["arms"]})
    lines += ["## Mean test WGA and overall accuracy", "",
              "Each source is first averaged across its available split seeds, then sources are weighted equally. SD is across those source means.", ""]
    table = []
    for arm in all_arms:
        row = [arm]
        for recipe in ("normal_erm", "ermstar"):
            subset = [cell for cell in cells if cell["recipe"] == recipe]
            available = sorted({cell["source"] for cell in subset})
            if not available:
                row += ["n/a", "n/a"]
                continue
            wg = [np.mean([cell["arms"][arm]["wga"] for cell in subset if cell["source"] == s]) for s in available]
            oa = [np.mean([cell["arms"][arm]["overall"] for cell in subset if cell["source"] == s]) for s in available]
            row += [f"{np.mean(wg):.3f} ± {old.fmt(np.std(wg, ddof=1) if len(wg)>1 else float('nan'), 3)} (n={len(wg)})",
                    f"{np.mean(oa):.3f}"]
        table.append(row)
    lines += [old.md_table(["arm", "normal ERM WGA ± source SD", "normal ERM OA", "ERM* WGA ± source SD", "ERM* OA"], table), ""]
    for recipe in ("normal_erm", "ermstar"):
        subset = [c for c in cells if c["recipe"] == recipe]
        if not subset:
            continue
        sources = sorted({c["source"] for c in subset})
        lines += [f"## {recipe}: corrected paired contrasts", ""]
        rows = []
        comparisons = [("crossfit_affine_sel_nll", "fullval_uniform"),
                       ("crossfit_affine_sel_nll", "bag75_uniform"),
                       ("crossfit_affine_sel_nll", "crossfit_uniform"),
                       ("crossfit_affine_sel_nll", "crossfit_affine_fixed"),
                       ("crossfit_affine_sel_wga", "fullval_uniform"),
                       ("crossfit_affine_fixed", "fullval_uniform"),
                       ("crossfit_affine_fixed", "crossfit_uniform"),
                       ("crossfit_uniform", "fullval_uniform"),
                       ("crossfit_uniform", "bag75_uniform"),
                       ("pool_affine_sel_nll", "fullval_uniform"),
                       ("pool_linear_sel_nll", "fullval_uniform"),
                       ("dfr_lite", "fullval_uniform"),
                       ("crossfit_affine_sel_nll", "crossfit_postmean_thrwga"),
                       ("crossfit_affine_sel_nll", "fullval_postmean_thrwga"),
                       ("crossfit_affine_sel_nll", "fullval_postmean_bias_sel_nll"),
                       ("crossfit_affine_sel_nll", "crossfit_postmean_bias_sel_nll"),
                       ("crossfit_postmean_thrwga", "crossfit_uniform"),
                       ("fullval_postmean_thrwga", "fullval_uniform")]
        for arm, control in comparisons:
            deltas = [float(np.mean([c["arms"][arm]["wga"] - c["arms"][control]["wga"]
                                     for c in subset if c["source"] == source])) for source in sources]
            mean, lo, hi = old.t_interval(deltas)
            rows.append([arm + " - " + control, old.fmt(mean, 3, True),
                         f"[{old.fmt(lo, 3, True)}, {old.fmt(hi, 3, True)}]",
                         ", ".join(old.fmt(d, 3, True) for d in deltas)])
        lines += [old.md_table(["contrast", "mean pp", "pointwise 95% source t-interval", "per source"], rows), ""]
        thresholds = [c["arms"]["crossfit_postmean_thrwga"] for c in subset]
        lines += [f"Exact threshold differs from 0.5 in {sum(t['threshold'] != .5 for t in thresholds)}/{len(thresholds)} cells.",
                  f"Mean fitted-validation WGA gain over legacy coarse search: {np.mean([t['fit_wga'] - t['legacy_grid_fit_wga'] for t in thresholds]):.3f} pp. This is not a test gain.", ""]
        lines += [f"## {recipe}: every setting (no test-based selection)", ""]
        grid_rows, selection_rows, noise_rows = [], [], []
        for family in (*FAMILIES, "pool_affine", "pool_linear"):
            for setting in SETTINGS:
                per_source = [np.mean([cell["grid"][family][old.name(setting)]["wga"]
                                      for cell in subset if cell["source"] == source]) for source in sources]
                grid_rows.append([family, old.name(setting), f"{np.mean(per_source):.3f}"])
            for rule in RULES:
                picks = [cell["arms"][family + "_sel_" + rule]["selected"] for cell in subset]
                selection_rows.append([family, rule, len(picks), *[picks.count(old.name(s)) for s in SETTINGS]])
        lines += [old.md_table(["family", "setting", "mean test WGA"], grid_rows), "",
                  "Selection counts (source × split cells):", "",
                  old.md_table(["family", "rule", "cells", *[old.name(s) for s in SETTINGS]], selection_rows), ""]
        for arm in ("fullval_uniform", "bag75_uniform", "crossfit_uniform", "crossfit_affine_sel_nll",
                    "crossfit_postmean_thrwga", "fullval_postmean_thrwga"):
            sds = [np.std([cell["arms"][arm]["wga"] for cell in subset if cell["source"] == source], ddof=1)
                   for source in sources if sum(cell["source"] == source for cell in subset) > 1]
            if sds:
                noise_rows.append([arm, f"{np.mean(sds):.3f}"])
        lines += ["Split-seed variability: mean within-source SD (pp). Split seeds also change head-training seeds.", "",
                  old.md_table(["arm", "mean SD"], noise_rows), ""]
    if missing:
        lines += ["Missing declared cells: " + str(missing), ""]
    errors = sorted((out / "errors").glob("*.txt")) if (out / "errors").exists() else []
    if errors:
        lines += ["Task errors: " + ", ".join(p.name for p in errors), ""]
    text = "\n".join(lines)
    (out / "CORRECTED_DIAGNOSTICS.md").write_text(text)
    (out / "REPORT.md").write_text(text)
    print(text)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, type=Path)
        if name == "run":
            s.add_argument("--only", nargs="*")
            s.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--time-limit-hours", type=float, default=8.5)
            s.add_argument("--config")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    sys.exit({"run": cmd_run, "summarize": cmd_summarize}[args.command](args))
