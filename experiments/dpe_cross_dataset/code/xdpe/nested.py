"""Strictly nested validation-only DPE post-hoc study on any frozen-feature dataset.

Port of experiments/dpe_nested_corrected/code/corrected_nested.py (protocol
strict-nested-postmean-v2) without DFR-lite (PLAN.md Sect. 4). Same arms, settings, seeds,
selection rules, cross-fit and inner folds, 60/40 pooled controls and 75-head bag.

    python -m xdpe.nested discover --roots DIR [DIR ...] --dataset CelebA --out OUT [--val-subsample 1199]
    python -m xdpe.nested run --out OUT [--splits 0 1 ...] [--workers 4] [--time-limit-hours 11] [--smoke]

`--smoke` trains 3-head ensembles for 2 epochs and never scores the test set (PLAN Sect. 8.4);
use a separate --out for it. `--no-test` runs the full recipe without scoring the test set
(timing, PLAN step 0.4); its checkpoints are reused when the same --out is later run without
the flag. Test predictions are scored only after every fit and selection of a cell is written
to fit_complete.json, as in the Waterbirds run.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
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

from . import calibrate as cal
from . import multiclass as mc
from .data import Source, folds, partition, sha256
from .datasets import DATASETS, config
from .heads import probabilities, train

HERE = Path(__file__).resolve().parent
PROTOCOL = "xdpe-strict-nested-postmean-v2"
INNER_FOLDS = 5
CROSSFIT_FOLDS = 5
BAG_ENSEMBLES = 5
SPLIT_SEED_STRIDE = 1000
FAMILIES = ("crossfit_affine", "crossfit_bias_foldwise", "crossfit_postmean_bias",
            "fullval_affine_oof", "fullval_postmean_bias")
SMOKE_HEADS = dict(heads=3, epochs=2)


# ----------------------------------------------------------------------------- io

def log(*parts):
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_identity():
    files = sorted(p for p in HERE.glob("*.py"))
    return digest_json({p.name: sha256(p) for p in files})


def bind_manifest(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Cache identity mismatch: {path}. Use a fresh output directory.")
    else:
        write_json(path, value)


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


class BudgetReached(Exception):
    pass


def check_budget(deadline):
    if time.time() >= deadline:
        raise BudgetReached("Time budget reached between checkpointed head fits")


# ----------------------------------------------------------------------------- binary / K-class switch

def binary(source):
    return source.classes == 2


def summarise_heads(source, q):
    """[H, N, K] head probabilities -> the ensemble score kept in caches."""
    mean = q.mean(0)
    return mean[:, 1] if binary(source) else mean


def decide(p):
    p = np.asarray(p)
    return p > .5 if p.ndim == 1 else p.argmax(-1)


def score(p, y, g):
    return cal.inner_score(p, y, g) if np.asarray(p).ndim == 1 else mc.inner_score(p, y, g)


def fit_map(source, x, y, g, kind, setting, cfg, seed):
    if binary(source):
        return cal.fit_model(x[:, None] if x.ndim == 1 else x, y, g, kind, setting, cfg, seed)
    return mc.fit(x, y, g, kind, setting, cfg, seed)


def apply_map(source, model, p, cfg):
    if binary(source):
        p = np.asarray(p)
        return cal.predict_model(model, p[:, None] if p.ndim == 1 else p, cfg)
    return mc.predict(model, p)


# ----------------------------------------------------------------------------- heads and caches

_SOURCES = {}


def get_source(spec):
    if spec["id"] not in _SOURCES:
        _SOURCES[spec["id"]] = Source(spec)
    return _SOURCES[spec["id"]]


def train_ensemble(source, indices, seed, cfg):
    indices = np.asarray(indices)
    return train(source._val[indices], source.val_y[indices], source.val_g[indices],
                 cfg["head_training"], seed, classes=source.classes)


def cached_scores(source, train_idx, val_eval_idx, test_eval, seed, cfg, directory, deadline, forbidden=()):
    tr, ve = np.asarray(train_idx, dtype=int), np.asarray(val_eval_idx, dtype=int)
    forbidden = np.asarray(forbidden, dtype=int)
    if len(set(tr)) != len(tr) or np.intersect1d(tr, ve).size or np.intersect1d(tr, forbidden).size:
        raise ValueError("Head training overlaps its held-out validation rows")
    stage = Path(directory)
    stage.mkdir(parents=True, exist_ok=True)
    bind_manifest(stage / "indices.json", dict(train_idx=tr.tolist(), val_eval_idx=ve.tolist(),
                                               forbidden_idx=forbidden.tolist(), test_eval=bool(test_eval), seed=int(seed)))
    cache, marker = stage / "scores.npz", stage / "complete.json"
    tail = () if binary(source) else (source.classes,)
    if cache.exists() and marker.exists():
        if read_json(marker)["sha256"] != sha256(cache):
            raise ValueError(f"Prediction checkpoint checksum mismatch: {cache}")
        with np.load(cache, allow_pickle=False) as data:
            val, test = data["val"], data["test"]
        if val.shape != (len(ve), *tail) or test.shape != (((len(source._test), *tail)) if test_eval else (0, *tail)):
            raise ValueError("Prediction checkpoint shape mismatch")
        return val, test
    check_budget(deadline)
    heads = train_ensemble(source, tr, seed, cfg)
    val = summarise_heads(source, probabilities(heads, source._val[ve])) if len(ve) else np.empty((0, *tail))
    test = summarise_heads(source, probabilities(heads, source._test)) if test_eval else np.empty((0, *tail))
    if not np.isfinite(val).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite head probabilities")
    atomic_npz(cache, val=val, test=test)
    write_json(marker, dict(sha256=sha256(cache), selected_epochs=heads["selected_epochs"]))
    return val, test


def build_context(source, pool, evaluation, test_eval, fold_seed, head_seed, cfg, directory, deadline, forbidden=()):
    pool, evaluation = np.asarray(pool, dtype=int), np.asarray(evaluation, dtype=int)
    if np.intersect1d(pool, evaluation).size or np.intersect1d(pool, forbidden).size:
        raise ValueError("Context pool overlaps an inner evaluation fold")
    rows = [source.rows[i] for i in pool]
    assignment = folds(rows, CROSSFIT_FOLDS, fold_seed)
    tail = () if binary(source) else (source.classes,)
    oof = np.full((len(pool), *tail), np.nan)
    fold_eval = []
    for f in range(CROSSFIT_FOLDS):
        tr = pool[assignment != f]
        te_local = np.flatnonzero(assignment == f)
        held = pool[te_local]
        pv, pt = cached_scores(source, tr, np.r_[held, evaluation], test_eval, head_seed, cfg,
                               Path(directory) / f"fold{f}", deadline, forbidden)
        oof[te_local] = pv[:len(held)]
        fold_eval.append(pt if test_eval else pv[len(held):])
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete OOF scores")
    pv, pt = cached_scores(source, pool, evaluation, test_eval, head_seed, cfg,
                           Path(directory) / "full_refit", deadline, forbidden)
    return dict(oof=oof, folds=np.stack(fold_eval), full=pt if test_eval else pv)


# ----------------------------------------------------------------------------- calibration arms

def calibration_predictions(source, affine, bias, fold_scores, full_scores, cfg):
    """Exactly the same inference graph in selection and final test evaluation."""
    apply = lambda model, p: apply_map(source, model, p, cfg)
    return dict(crossfit_affine=np.mean([apply(affine, p) for p in fold_scores], axis=0),
                crossfit_bias_foldwise=np.mean([apply(bias, p) for p in fold_scores], axis=0),
                crossfit_postmean_bias=apply(bias, np.mean(fold_scores, axis=0)),
                fullval_affine_oof=apply(affine, full_scores),
                fullval_postmean_bias=apply(bias, full_scores))


def strict_inner_scores(source, split, cfg, base, fit_seed, directory, deadline):
    n = len(source.rows)
    assignment = folds(source.rows, INNER_FOLDS, 10_000 + split)
    predictions = {family: {cal.name(s): None for s in cal.SETTINGS} for family in FAMILIES}
    filled = {family: {cal.name(s): np.zeros(n, dtype=bool) for s in cal.SETTINGS} for family in FAMILIES}
    thresholds = ("crossfit_postmean_thrwga", "fullval_postmean_thrwga") if binary(source) else ()
    threshold_predictions = {family: np.zeros(n, dtype=bool) for family in thresholds}
    threshold_logs = []
    for f in range(INNER_FOLDS):
        tr, te = np.flatnonzero(assignment != f), np.flatnonzero(assignment == f)
        context = build_context(source, tr, te, False, split, base, cfg, Path(directory) / f"inner{f}",
                                deadline, forbidden=te)
        y, g = source.val_y[tr], source.val_g[tr]
        x = context["oof"]
        for setting in cal.SETTINGS:
            affine = fit_map(source, x, y, g, "affine", setting, cfg, fit_seed)
            bias = fit_map(source, x, y, g, "bias", setting, cfg, fit_seed)
            for family, p in calibration_predictions(source, affine, bias, context["folds"], context["full"], cfg).items():
                key = cal.name(setting)
                if predictions[family][key] is None:
                    predictions[family][key] = np.full((n, *p.shape[1:]), np.nan)
                predictions[family][key][te] = p
                filled[family][key][te] = True
        if thresholds:
            threshold = cal.fit_threshold_exact(x[:, None], y, g)
            threshold_logs.append(dict(inner_fold=f, model=threshold))
            threshold_predictions["crossfit_postmean_thrwga"][te] = cal.apply_threshold(threshold, context["folds"].mean(0))
            threshold_predictions["fullval_postmean_thrwga"][te] = cal.apply_threshold(threshold, context["full"])
    scores = {}
    for family, settings in predictions.items():
        if any(not filled[family][k].all() or not np.isfinite(p).all() for k, p in settings.items()):
            raise ValueError("Incomplete inner evaluation predictions")
        scores[family] = {k: score(p, source.val_y, source.val_g) for k, p in settings.items()}
    diagnostics = {family: cal.record(p, source.val_y, source.val_g) for family, p in threshold_predictions.items()}
    return scores, dict(threshold_folds=threshold_logs, inner_threshold_performance=diagnostics)


def final_predictions(source, context, inner, cfg, fit_seed):
    y, g = source.val_y, source.val_g
    x = context["oof"]
    grid = {family: {} for family in FAMILIES}
    models = {}
    for setting in cal.SETTINGS:
        affine = fit_map(source, x, y, g, "affine", setting, cfg, fit_seed)
        bias = fit_map(source, x, y, g, "bias", setting, cfg, fit_seed)
        models[cal.name(setting)] = dict(affine=affine, bias=bias)
        for family, p in calibration_predictions(source, affine, bias, context["folds"], context["full"], cfg).items():
            grid[family][cal.name(setting)] = p
    chosen = {family: {rule: cal.choose(inner[family], rule) for rule in cal.RULES} for family in FAMILIES}
    threshold = cal.fit_threshold_exact(x[:, None], y, g) if binary(source) else None
    return grid, chosen, threshold, models


def pool_inputs(source, q):
    """[H, N, K] head probabilities -> combiner inputs: binary [N, H]; K-class [N, H, K]."""
    return q[:, :, 1].T if binary(source) else q.transpose(1, 0, 2)


def pool_fit(source, x, y, g, kind, setting, cfg, seed):
    if binary(source):
        return cal.fit_model(x, y, g, kind, setting, cfg, seed)
    return mc.fit(x if kind == "linear" else x.mean(1), y, g, kind, setting, cfg, seed)


def pool_apply(source, model, x, cfg):
    if binary(source):
        return cal.predict_model(model, x, cfg)
    return mc.predict(model, x if model["kind"] == "linear" else x.mean(1))


def pool_inner_cv(source, x, y, g, rows, kind, cfg, seed, split_seed):
    assign = folds(rows, INNER_FOLDS, 10_000 + split_seed)
    out = {}
    for setting in cal.SETTINGS:
        oof = None
        for f in range(INNER_FOLDS):
            tr, te = assign != f, assign == f
            model = pool_fit(source, x[tr], y[tr], g[tr], kind, setting, cfg, seed)
            p = pool_apply(source, model, x[te], cfg)
            if oof is None:
                oof = np.full((len(y), *p.shape[1:]), np.nan)
            oof[te] = p
        out[cal.name(setting)] = score(oof, y, g)
    return out


def pooled_predictions(source, split, cfg, base, fit_seed, directory, deadline):
    part = partition(source.rows, split)
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
        if read_json(marker)["sha256"] != sha256(path):
            raise ValueError("Pooled head-score checksum mismatch")
        with np.load(path, allow_pickle=False) as value:
            qf, qt = value["fit"], value["test"]
    else:
        check_budget(deadline)
        heads = train_ensemble(source, head_idx, base, cfg)
        qf = probabilities(heads, source._val[fit_idx])
        qt = probabilities(heads, source._test)
        atomic_npz(path, fit=qf, test=qt)
        write_json(marker, dict(sha256=sha256(path)))
    xf, xt = pool_inputs(source, qf), pool_inputs(source, qt)
    y, g = source.val_y[fit_idx], source.val_g[fit_idx]
    rows = [source.rows[i] for i in fit_idx]
    inner, grid, chosen = {}, {}, {}
    for kind in ("affine", "linear"):
        family = "pool_" + kind
        inner[family] = pool_inner_cv(source, xf, y, g, rows, kind, cfg, fit_seed, split)
        grid[family] = {}
        for setting in cal.SETTINGS:
            model = pool_fit(source, xf, y, g, kind, setting, cfg, fit_seed)
            grid[family][cal.name(setting)] = pool_apply(source, model, xt, cfg)
        chosen[family] = {rule: cal.choose(inner[family], rule) for rule in cal.RULES}
    return summarise_heads(source, qt), grid, inner, chosen


# ----------------------------------------------------------------------------- one cell

def run_task(task):
    spec, split, out_dir, cfg, protocol_id, deadline, score_test = task
    started = time.time()
    out = Path(out_dir)
    cell = out / "results" / spec["id"] / f"split{split:02d}.json"
    checkpoints = out / "checkpoints" / spec["id"] / f"split{split:02d}"
    error = out / "errors" / f"{spec['id']}-split{split:02d}.txt"
    try:
        import torch
        torch.set_num_threads(1)
        source = get_source(spec)
        source_id = dict(id=spec["id"], dataset=spec["dataset"], seed=spec["seed"],
                         val_subsample=spec.get("val_subsample"),
                         features={k: sha256(v) for k, v in spec["features"].items()})
        fingerprint = digest_json(dict(protocol=protocol_id, source=source_id, split=split))
        bind_manifest(checkpoints / "identity.json", dict(fingerprint=fingerprint, source=source_id))
        if cell.exists():
            if read_json(cell).get("fingerprint") != fingerprint:
                raise ValueError("Completed result was produced by another protocol/source")
            error.unlink(missing_ok=True)
            return spec["id"], split, "cached", 0.
        base = spec["seed"] + SPLIT_SEED_STRIDE * split
        fit_seed = spec["seed"] + cfg["seed_offset"]
        inner_path = checkpoints / "inner_scores.json"
        if inner_path.exists():
            saved = read_json(inner_path)
            inner, diagnostics = saved["scores"], saved["diagnostics"]
        else:
            inner, diagnostics = strict_inner_scores(source, split, cfg, base, fit_seed, checkpoints, deadline)
            write_json(inner_path, dict(scores=inner, diagnostics=diagnostics))
        context = build_context(source, np.arange(len(source.rows)), np.array([], dtype=int),
                                True, split, base, cfg, checkpoints / "final", deadline)
        grid_prob, chosen, threshold, models = final_predictions(source, context, inner, cfg, fit_seed)
        prob_arms = dict(fullval_uniform=context["full"], crossfit_uniform=context["folds"].mean(0))
        full_bag = [context["full"]]
        for k in range(1, BAG_ENSEMBLES):
            _, p = cached_scores(source, np.arange(len(source.rows)), np.array([], dtype=int), True,
                                 base + k, cfg, checkpoints / f"bag{k}", deadline)
            full_bag.append(p)
        prob_arms["bag75_uniform"] = np.mean(full_bag, axis=0)
        selected = {}
        for family in FAMILIES:
            prob_arms[family + "_fixed"] = grid_prob[family][cal.name(cal.FIXED)]
            selected[family + "_fixed"] = cal.name(cal.FIXED)
            for rule in cal.RULES:
                arm = family + "_sel_" + rule
                selected[arm] = chosen[family][rule]
                prob_arms[arm] = grid_prob[family][chosen[family][rule]]
        pool_uniform, pool_grid, pool_inner, pool_chosen = pooled_predictions(
            source, split, cfg, base, fit_seed, checkpoints / "pooled", deadline)
        prob_arms["pool_uniform"] = pool_uniform
        inner.update(pool_inner)
        chosen.update(pool_chosen)
        grid_prob.update(pool_grid)
        for family in pool_grid:
            prob_arms[family + "_fixed"] = pool_grid[family][cal.name(cal.FIXED)]
            selected[family + "_fixed"] = cal.name(cal.FIXED)
            for rule in cal.RULES:
                arm = family + "_sel_" + rule
                selected[arm] = pool_chosen[family][rule]
                prob_arms[arm] = pool_grid[family][selected[arm]]
        # Durable fit/selection record, written before any test label is read.
        write_json(checkpoints / "fit_complete.json", dict(chosen=chosen, threshold=threshold,
                   models=models, fingerprint=fingerprint))
        if not score_test:
            write_json(out / "smoke" / spec["id"] / f"split{split:02d}.json",
                       dict(protocol=PROTOCOL, fingerprint=fingerprint, chosen=chosen, inner=inner,
                            arms=sorted(prob_arms) + (["crossfit_postmean_thrwga", "fullval_postmean_thrwga"] if threshold else []),
                            seconds=time.time() - started))
            error.unlink(missing_ok=True)
            return spec["id"], split, "smoke-done", time.time() - started
        ty, tg = source.test_y, source.test_g
        np.savez_compressed(checkpoints / "test_decisions.npz",
                            **{arm: decide(p) for arm, p in prob_arms.items()})
        arms = {arm: cal.record(decide(p), ty, tg, **({"selected": selected[arm]} if arm in selected else {}))
                for arm, p in prob_arms.items()}
        if threshold is not None:
            arms["crossfit_postmean_thrwga"] = cal.record(cal.apply_threshold(threshold, context["folds"].mean(0)), ty, tg, **threshold)
            arms["fullval_postmean_thrwga"] = cal.record(cal.apply_threshold(threshold, context["full"]), ty, tg, **threshold)
        grid = {family: {k: cal.record(decide(p), ty, tg) for k, p in settings.items()}
                for family, settings in grid_prob.items()}
        write_json(cell, dict(protocol=PROTOCOL, fingerprint=fingerprint, source=spec["id"], dataset=spec["dataset"],
                              seed=spec["seed"], split=split, val_subsample=spec.get("val_subsample"),
                              arms=arms, grid=grid, inner=inner, diagnostics=diagnostics,
                              seconds=time.time() - started))
        error.unlink(missing_ok=True)
        return spec["id"], split, "done", time.time() - started
    except BudgetReached:
        return spec["id"], split, "deferred", time.time() - started
    except Exception:
        error.parent.mkdir(parents=True, exist_ok=True)
        error.write_text(traceback.format_exc())
        return spec["id"], split, "ERROR", time.time() - started


# ----------------------------------------------------------------------------- commands

def protocol_config(dataset, smoke):
    cfg = config(dataset)
    if smoke:
        cfg["head_training"].update(SMOKE_HEADS)
    return cfg


def prepare_protocol(out, dataset, smoke):
    import scipy
    import torch
    cfg = protocol_config(dataset, smoke)
    identity = dict(protocol=PROTOCOL, dataset=dataset, smoke=bool(smoke), code_sha256=code_identity(), config=cfg,
                    inner_folds=INNER_FOLDS, crossfit_folds=CROSSFIT_FOLDS, bag_ensembles=BAG_ENSEMBLES,
                    settings=cal.SETTINGS, fixed=cal.FIXED, primary_rule="nll", secondary_rule="wga",
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    torch=torch.__version__)
    identity = json.loads(json.dumps(identity))
    bind_manifest(Path(out) / "protocol.json", identity)
    return cfg, digest_json(identity)


def cmd_discover(a):
    """Find complete backbones written by train_backbone.py (backbone.json + features/*.npz)."""
    found = {}
    for root in a.roots:
        for manifest in sorted(Path(root).rglob("backbone.json")):
            m = read_json(manifest)
            if m.get("dataset") != a.dataset or m.get("status") != "complete":
                continue
            base = manifest.parent
            features = dict(val=str((base / "features/val.npz").resolve()), test=str((base / "features/test.npz").resolve()))
            if not all(Path(p).is_file() for p in features.values()):
                log("skip (missing features):", base)
                continue
            hashes = {k: sha256(v) for k, v in features.items()}
            for k, v in hashes.items():
                if m["features_sha256"][k] != v:
                    raise ValueError(f"{base}: feature cache differs from its manifest")
            sid = f"{a.dataset}-seed{m['seed']}" + (f"-val{a.val_subsample}" if a.val_subsample else "")
            spec = dict(id=sid, dataset=a.dataset, seed=int(m["seed"]), features=features,
                        feature_dim=DATASETS[a.dataset]["feature_dim"], classes=DATASETS[a.dataset]["classes"],
                        checkpoint_sha256=m["checkpoint_sha256"], features_sha256=hashes, manifest=str(manifest.resolve()))
            if a.val_subsample:
                spec.update(val_subsample=a.val_subsample, val_subsample_seed=0)
            if sid in found and found[sid]["features_sha256"] != hashes:
                raise ValueError(f"Two different backbones claim {sid}")
            found.setdefault(sid, spec)
    specs = sorted(found.values(), key=lambda s: s["seed"])
    write_json(Path(a.out) / "sources.json", dict(dataset=a.dataset, sources=specs))
    log(f"found {len(specs)} {a.dataset} backbones:", [s["id"] for s in specs])
    return 0 if specs else 1


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    if a.workers < 1 or not math.isfinite(a.time_limit_hours) or a.time_limit_hours <= 0 or not a.splits:
        raise ValueError("Positive workers/time budget and at least one split are required")
    sources = read_json(Path(a.out) / "sources.json")
    specs = [s for s in sources["sources"] if not a.only or s["id"] in a.only]
    if not specs:
        raise ValueError("No selected sources")
    if len(set(a.splits)) != len(a.splits) or any(s < 0 or s > 9 for s in a.splits):
        raise ValueError("Choose distinct split seeds in 0..9")
    cfg, protocol_id = prepare_protocol(a.out, sources["dataset"], a.smoke)
    deadline = time.time() + a.time_limit_hours * 3600
    score_test = not (a.smoke or a.no_test)
    tasks = [(s, split, str(a.out), cfg, protocol_id, deadline, score_test) for split in a.splits for s in specs]
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
            log(status, sid, f"split{split:02d}", f"{seconds / 60:.1f} min")
            submit()
    folder = "results" if score_test else "smoke"
    complete = sum((Path(a.out) / folder / s["id"] / f"split{split:02d}.json").exists() for split in a.splits for s in specs)
    write_json(Path(a.out) / "completion.json", dict(protocol=PROTOCOL, requested=len(tasks), completed=complete,
               splits=a.splits, sources=[s["id"] for s in specs], statuses=status_counts, smoke=bool(a.smoke),
               test_scored=score_test))
    log(f"Completed {complete}/{len(tasks)} requested cells; stage checkpoints retained.")
    return 1 if status_counts.get("ERROR") else (0 if complete == len(tasks) else 3)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--roots", nargs="+", required=True)
    d.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--val-subsample", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--only", nargs="*")
    r.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--time-limit-hours", type=float, default=11.0)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--no-test", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    sys.exit({"discover": cmd_discover, "run": cmd_run}[args.command](args))
