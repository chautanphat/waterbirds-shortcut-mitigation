"""Nested validation-only selection for cross-fitted DPE recalibration (CPU, saved features only).

Question. With the same 1,199 validation labels that full-validation DPE uses, does
cross-fitting the 15-head ensemble and recalibrating the averaged score with a two-parameter
group-balanced affine map beat full-validation DPE, when the regularisation setting is chosen
by a validation-only rule instead of being highlighted after seeing test results?

Declared before running (do not edit after the first result is read)
  Settings (loss, penalty) : average/0.1, maximum/0.1, average/0.01, maximum/0.01
  Fixed setting            : average/0.1 (reported as its own arm, not selected)
  Selection (primary)      : inner 5-fold CV on the out-of-fold scores, lowest group-balanced
                             NLL; ties -> higher inner WGA -> earlier setting in the list above.
  Selection (secondary)    : highest inner-CV WGA, ties -> lower NLL -> list order.
  Split seeds              : 0..9 (fold split for cross-fit, 60/40 split for the pooled arms).
  Primary comparison       : selected cross-fit affine minus full-validation DPE retrained with the
                             same split seed, paired per source; normal ERM and ERM* reported
                             separately. Interval = t-interval over sources of the per-source mean
                             over split seeds (n = 5 for normal ERM; ERM* has n = 2: descriptive).
  Threshold controls       : full-validation DPE + threshold-only / affine calibrator fitted on the cross-fit OUT-OF-FOLD scores
                             (arms fullval_threshold_oof_*, fullval_affine_oof_*; selected by the same nested rule), the direct
                             WGA-maximising threshold (fullval_thrwga_oof, crossfit_thrwga), and cross-fit + threshold-only
                             (crossfit_threshold_*). They test whether the affine gain is only a decision-threshold shift.
  Controls                 : cross-fit uniform; 75-head bag (5 full-validation ensembles);
                             pooled 720/479 uniform, affine and linear residual (selected and fixed);
                             DFR-style group-balanced L1 logistic regression on the same features.

Commands (run from the folder that holds `src/`, `configs/` and this file)
  python nested_selection.py discover --roots DIR [DIR ...] --out OUT
  python nested_selection.py check    --out OUT            # saved-ensemble regression check
  python nested_selection.py run      --out OUT --splits 0 1 ... [--workers 4] [--time-limit-hours H]
  python nested_selection.py summarize --out OUT
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import contextlib
import csv
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

GROUPS = ("00", "01", "10", "11")
SETTINGS = [("average", 0.1), ("maximum", 0.1), ("average", 0.01), ("maximum", 0.01)]
FIXED = ("average", 0.1)
DFR_C = [0.01, 0.03, 0.1, 0.3, 1.0]
INNER_FOLDS = 5
PRIMARY_RULE = "nll"
RULES = ("nll", "wga")
BAG_ENSEMBLES = 5
SPLIT_SEED_STRIDE = 1000
REFERENCE = HERE / "expected_reference.json"
SHIPPED_METADATA = HERE / "data" / "metadata.csv"


def name(setting):
    return f"{setting[0]}-l{setting[1]:g}"


def now():
    return time.strftime("%H:%M:%S")


def log(*parts):
    print(f"[{now()}]", *parts, flush=True)


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


# ----------------------------------------------------------------------------- metrics

def group_accuracy(pred, y, groups):
    correct = np.asarray(pred) == np.asarray(y)
    return {g: float(correct[groups == g].mean()) for g in GROUPS}


def record(pred, y, groups, **extra):
    acc = group_accuracy(pred, y, groups)
    out = dict(wga=min(acc.values()) * 100, overall=float((np.asarray(pred) == y).mean()) * 100,
               groups={g: v * 100 for g, v in acc.items()})
    out.update(extra)
    return out


def balanced_nll(prob1, y, groups):
    p = np.clip(np.asarray(prob1, dtype=np.float64), 1e-12, 1 - 1e-12)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return float(np.mean([loss[groups == g].mean() for g in GROUPS]))


def inner_score(prob1, y, groups):
    acc = group_accuracy(prob1 > .5, y, groups)
    return dict(wga=min(acc.values()) * 100, nll=balanced_nll(prob1, y, groups))


def choose(scores, rule):
    """scores: {setting_name: {wga, nll}} in SETTINGS order. Deterministic, validation-only."""
    order = [name(s) for s in SETTINGS]
    if rule == "nll":
        key = lambda n: (round(scores[n]["nll"], 9), -round(scores[n]["wga"], 6), order.index(n))
    elif rule == "wga":
        key = lambda n: (-round(scores[n]["wga"], 6), round(scores[n]["nll"], 9), order.index(n))
    else:
        raise ValueError(rule)
    return min((n for n in order if n in scores), key=key)


# ----------------------------------------------------------------------------- repository adapters

def adapters():
    from src.methods.dpe_independent import fitting, geometry, grid, sources
    from src.methods.dpe_head_reweighting import ensemble
    from scipy.special import softmax
    return dict(fitting=fitting, geometry=geometry, grid=grid, sources=sources, ensemble=ensemble, softmax=softmax)


_SOURCE_CACHE = {}


def get_source(spec):
    key = spec["id"]
    if key not in _SOURCE_CACHE:
        A = adapters()
        s = A["sources"].Source(spec)
        s._val = s.features("val")
        s._test = s.features("test")
        s.val_y = np.array([r["y"] for r in s.rows])
        s.val_g = np.array([r["group"] for r in s.rows])
        s.test_y = np.array([r["y"] for r in s.test_rows])
        s.test_g = np.array([f"{r['y']}{r['a']}" for r in s.test_rows])
        _SOURCE_CACHE[key] = s
    return _SOURCE_CACHE[key]


def train_ensemble(source, indices, seed, cfg, scratch):
    """Same trainers as the independent study, with an explicit head-training seed."""
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        return _train_ensemble(source, indices, seed, cfg, scratch)


def _train_ensemble(source, indices, seed, cfg, scratch):
    A = adapters()
    indices = np.asarray(indices)
    rows = [source.rows[i] for i in indices]
    raw = source._val[indices]
    settings = cfg["head_training"]
    if source.spec["recipe"] == "ermstar":
        from src.methods.dpe_ermstar.run import train_heads
        progress = Path(tempfile.mkdtemp(dir=scratch)) / "head_progress.pt"
        try:
            heads = train_heads(raw, rows, progress, seed, settings, "cpu")
        finally:
            shutil.rmtree(progress.parent, ignore_errors=True)
    else:
        from src.methods.dpe_head_reweighting.train_heads import train
        identity = dict(source=source.spec["id"], seed=seed, n=len(rows), settings=settings)
        heads = train(raw, source.val_y[indices], source.val_g[indices], settings, seed, identity, "cpu", progress=False)
    A["ensemble"].validate(heads, count=15)
    return heads


def head_probabilities(heads, raw):
    """[15, N, 2] float64 probabilities, rounded through float32 exactly as the saved caches are."""
    A = adapters()
    d = A["geometry"].distances(heads, raw)
    z = A["geometry"].scores(d, heads["distance_scales"].numpy())
    return A["softmax"](z, axis=-1).astype(np.float32).astype(np.float64)


# ----------------------------------------------------------------------------- calibrators

def logit_clip(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def fit_bias(x, y, g, setting):
    """Threshold-only recalibration: sigmoid(logit(p) + b), b from the group-balanced loss (no scale)."""
    from scipy.optimize import minimize_scalar
    z, yy = logit_clip(x[:, 0]), np.asarray(y, dtype=np.float64)
    loss_name, penalty = setting

    def objective(b):
        margin = (2 * yy - 1) * (z + b)
        per = np.logaddexp(0.0, -margin)
        groups = [per[g == k].mean() for k in GROUPS]
        return (np.mean(groups) if loss_name == "average" else np.max(groups)) + penalty * b * b

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded", options=dict(xatol=1e-6))
    return dict(kind="bias", b=float(result.x))


def fit_threshold_wga(x, y, g):
    """Direct validation threshold: the logit offset that maximises worst-group accuracy (ties: smallest |b|)."""
    z, yy = logit_clip(x[:, 0]), np.asarray(y)
    best = None
    for b in np.arange(-4.0, 4.0001, 0.05):
        pred = (z + b) > 0
        wga = min((pred[g == k] == yy[g == k]).mean() for k in GROUPS)
        key = (-round(wga, 9), abs(b))
        if best is None or key < best[0]:
            best = (key, float(b))
    return dict(kind="bias", b=best[1])


def fit_model(x, y, g, kind, setting, cfg, seed):
    if kind == "bias":
        return fit_bias(x, y, g, setting)
    A = adapters()
    spec = dict(family="residual", kind=kind, loss=setting[0], penalty=setting[1], input=A["grid"].ORIGINAL)
    return A["fitting"].fit_adam(x, y, g, spec, cfg, seed)


def predict_model(model, x, cfg):
    if model.get("kind") == "bias":
        return 1.0 / (1.0 + np.exp(-(logit_clip(x[:, 0]) + model["b"])))
    return adapters()["fitting"].predict_adam(model, x, cfg)[:, 1]


def inner_cv_scores(x, y, g, rows, kind, cfg, seed, split_seed):
    """Out-of-fold probabilities for all four settings using a stratified inner 5-fold split."""
    assign = adapters()["sources"].folds(rows, INNER_FOLDS, 10_000 + split_seed)
    out = {}
    for setting in SETTINGS:
        oof = np.full(len(y), np.nan)
        for f in range(INNER_FOLDS):
            tr, te = assign != f, assign == f
            model = fit_model(x[tr], y[tr], g[tr], kind, setting, cfg, seed)
            oof[te] = predict_model(model, x[te], cfg)
        out[name(setting)] = inner_score(oof, y, g)
    return out


# ----------------------------------------------------------------------------- DFR-lite

_L1_KWARGS = []


def l1_kwargs():
    """Pick the scikit-learn spelling of 'L1 penalty' that really gives sparse coefficients (verified on a probe)."""
    if _L1_KWARGS:
        return _L1_KWARGS[0]
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 100))
    y = (X[:, 0] + X[:, 1] + rng.normal(size=400) > 0).astype(int)
    for kwargs in (dict(penalty="l1", solver="liblinear"), dict(l1_ratio=1.0, solver="liblinear")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coef = LogisticRegression(C=0.05, max_iter=500, **kwargs).fit(X, y).coef_
            if 0 < (coef != 0).sum() < 30:
                _L1_KWARGS.append(kwargs)
                return kwargs
        except (TypeError, ValueError):
            continue
    raise RuntimeError("No scikit-learn spelling of an L1 logistic regression gave sparse coefficients")


def fit_l1(X, y, weights, C):
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, max_iter=2000, tol=1e-3, **l1_kwargs()).fit(X, y, sample_weight=weights)


def group_weights(g):
    w = np.zeros(len(g))
    for group in GROUPS:
        m = g == group
        w[m] = 1.0 / m.sum()
    return w * len(g) / w.sum()


def dfr_lite(source, split_seed):
    """Group-balanced L1 logistic regression on standardized raw features; C chosen by CV on validation only."""
    A = adapters()
    X = source._val.astype(np.float64)
    mean, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mean) / sd
    Xt = (source._test.astype(np.float64) - mean) / sd
    y, g = source.val_y, source.val_g
    assign = A["sources"].folds(source.rows, 5, 20_000 + split_seed)
    scores = {}
    for C in DFR_C:
        oof = np.full(len(y), np.nan)
        for f in range(5):
            tr, te = assign != f, assign == f
            clf = fit_l1(X[tr], y[tr], group_weights(g[tr]), C)
            oof[te] = clf.predict_proba(X[te])[:, 1]
        scores[f"C{C:g}"] = inner_score(oof, y, g)
    best = min(scores, key=lambda n: (-round(scores[n]["wga"], 6), round(scores[n]["nll"], 9), float(n[1:])))
    clf = fit_l1(X, y, group_weights(g), float(best[1:]))
    pred = clf.predict_proba(Xt)[:, 1] > .5
    return record(pred, source.test_y, source.test_g, selected=best, nonzero=int((clf.coef_ != 0).sum()), inner=scores)


# ----------------------------------------------------------------------------- one (source, split) task

def run_task(args):
    spec, split, out_dir, cfg_path = args
    t0 = time.time()
    path = Path(out_dir) / "results" / spec["id"] / f"split{split:02d}.json"
    if path.exists():
        return spec["id"], split, "cached", 0.0
    try:
        import torch
        torch.set_num_threads(1)
        A = adapters()
        cfg = A["grid"].load_config(cfg_path) if cfg_path else A["grid"].load_config()
        source = get_source(spec)
        scratch = Path(out_dir) / "scratch" / f"{spec['id']}-{split}"
        scratch.mkdir(parents=True, exist_ok=True)
        y, g = source.val_y, source.val_g
        ty, tg = source.test_y, source.test_g
        base = spec["seed"] + SPLIT_SEED_STRIDE * split
        fit_seed = spec["seed"] + cfg["seed_offset"]
        arms, grid_table, inner = {}, {}, {}

        # ---- controls: full-validation DPE retrained with this split's seeds, and a 75-head bag
        full_mean = []
        for k in range(BAG_ENSEMBLES):
            heads = train_ensemble(source, np.arange(1199), base + k, cfg, scratch)
            p = head_probabilities(heads, source._test).mean(0)
            full_mean.append(p)
            if k == 0:
                arms["fullval_uniform"] = record(p.argmax(-1), ty, tg)
        arms["bag75_uniform"] = record(np.mean(full_mean, axis=0).argmax(-1), ty, tg)

        # ---- cross-fit + affine
        assign = A["sources"].folds(source.rows, 5, split)
        oof = np.full(1199, np.nan)
        fold_test = []
        for f in range(5):
            tr, te = np.flatnonzero(assign != f), np.flatnonzero(assign == f)
            heads = train_ensemble(source, tr, base, cfg, scratch)  # same seed for every fold, as in the earlier cross-fit run
            oof[te] = head_probabilities(heads, source._val[te]).mean(0)[:, 1]
            fold_test.append(head_probabilities(heads, source._test).mean(0))
        arms["crossfit_uniform"] = record(np.mean(fold_test, axis=0).argmax(-1), ty, tg)
        x = oof[:, None]
        # Out-of-fold scores: each comes from an ensemble that never saw that image. The calibrators fitted on them are applied
        # (a) to the five fold ensembles (cross-fit) and (b) to the single full-validation ensemble (no in-sample fitting).
        fullval_p1 = full_mean[0][:, 1]
        for kind, cname, fname in (("affine", "crossfit_affine", "fullval_affine_oof"),
                                   ("bias", "crossfit_threshold", "fullval_threshold_oof")):
            inner[cname] = inner_cv_scores(x, y, g, source.rows, kind, cfg, fit_seed, split)
            grid_table[cname], grid_table[fname] = {}, {}
            for setting in SETTINGS:
                model = fit_model(x, y, g, kind, setting, cfg, fit_seed)
                prob = np.mean([predict_model(model, p[:, 1, None], cfg) for p in fold_test], axis=0)
                grid_table[cname][name(setting)] = record(prob > .5, ty, tg)
                grid_table[fname][name(setting)] = record(predict_model(model, fullval_p1[:, None], cfg) > .5, ty, tg)
            for family in (cname, fname):
                arms[f"{family}_fixed"] = dict(grid_table[family][name(FIXED)], selected=name(FIXED))
                for rule in RULES:
                    pick = choose(inner[cname], rule)
                    arms[f"{family}_sel_{rule}"] = dict(grid_table[family][pick], selected=pick)
        model = fit_threshold_wga(x, y, g)
        arms["crossfit_thrwga"] = record(np.mean([predict_model(model, p[:, 1, None], cfg) for p in fold_test], axis=0) > .5, ty, tg, b=model["b"])
        arms["fullval_thrwga_oof"] = record(predict_model(model, fullval_p1[:, None], cfg) > .5, ty, tg, b=model["b"])

        # ---- pooled 60/40: heads on 720, combiner on 479
        part = A["sources"].partition(source.rows, split)
        index = {r["id"]: i for i, r in enumerate(source.rows)}
        head_idx = np.array([index[r["id"]] for r in part["head_training"]])
        fit_idx = np.array([index[r["id"]] for r in part["combiner_training"]])
        heads = train_ensemble(source, head_idx, base, cfg, scratch)
        qf = head_probabilities(heads, source._val[fit_idx])
        qt = head_probabilities(heads, source._test)
        arms["pool_uniform"] = record(qt.mean(0).argmax(-1), ty, tg)
        xf, xt = qf[:, :, 1].T, qt[:, :, 1].T
        yf, gf = y[fit_idx], g[fit_idx]
        rows_f = [source.rows[i] for i in fit_idx]
        for kind in ("affine", "linear"):
            key = f"pool_{kind}"
            inner[key] = inner_cv_scores(xf, yf, gf, rows_f, kind, cfg, fit_seed, split)
            grid_table[key] = {}
            for setting in SETTINGS:
                model = fit_model(xf, yf, gf, kind, setting, cfg, fit_seed)
                grid_table[key][name(setting)] = record(predict_model(model, xt, cfg) > .5, ty, tg)
            arms[f"{key}_fixed"] = dict(grid_table[key][name(FIXED)], selected=name(FIXED))
            for rule in RULES:
                pick = choose(inner[key], rule)
                arms[f"{key}_sel_{rule}"] = dict(grid_table[key][pick], selected=pick)

        # ---- DFR-style last-layer retraining on the same labels
        arms["dfr_lite"] = dfr_lite(source, split)
        shutil.rmtree(scratch, ignore_errors=True)
        write_json(path, dict(source=spec["id"], recipe=spec["recipe"], seed=spec["seed"], split=split,
                              arms=arms, grid=grid_table, inner=inner, seconds=time.time() - t0))
        return spec["id"], split, "done", time.time() - t0
    except Exception:
        err = Path(out_dir) / "errors" / f"{spec['id']}-split{split:02d}.txt"
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return spec["id"], split, "ERROR", time.time() - t0


# ----------------------------------------------------------------------------- commands

def cmd_discover(a):
    A = adapters()
    S = A["sources"]
    roots = [Path(r) for r in a.roots]
    if SHIPPED_METADATA.is_file():
        roots.append(SHIPPED_METADATA.parent)
        original = S.one

        def one(paths, label):  # prefer the shipped, hash-checked metadata when several candidates exist
            paths = list(paths)
            if label == "metadata":
                for p in paths:
                    if Path(p).resolve() == SHIPPED_METADATA.resolve():
                        return str(Path(p).resolve())
            return original(paths, label)
        S.one = one
    specs, blockers = S.discover(roots)
    out = Path(a.out)
    write_json(out / "sources.json", dict(sources=specs, blockers=blockers))
    log(f"found {len(specs)} sources:", [s["id"] for s in specs])
    for b in blockers:
        log("BLOCKER", b)
    return 0 if specs else 1


def load_specs(out, only=None):
    specs = read_json(Path(out) / "sources.json")["sources"]
    if only:
        specs = [s for s in specs if s["id"] in only]
    return specs


def cmd_check(a):
    """Saved full-validation and partition-0 ensembles must reproduce the saved reference numbers."""
    A = adapters()
    cfg = A["grid"].load_config()
    ref = read_json(REFERENCE)
    rows, ok = [], True
    for spec in load_specs(a.out, a.only):
        if spec.get("fresh"):
            log(f"  {spec['id']}: freshly trained backbone, no saved reference to compare with")
            continue
        source = get_source(spec)
        ty, tg = source.test_y, source.test_g
        fit_seed = spec["seed"] + cfg["seed_offset"]
        entry = dict(source=spec["id"])
        if "full_validation" in spec["ensembles"]:
            heads, _ = source.heads("full_validation", Path(a.out) / "scratch", cfg)
            entry["fullval_saved"] = record(head_probabilities(heads, source._test).mean(0).argmax(-1), ty, tg)["wga"]
        heads, _ = source.heads("partition0", Path(a.out) / "scratch", cfg)
        head_idx, fit_idx = source.head_indices("partition0")
        qf, qt = head_probabilities(heads, source._val[fit_idx]), head_probabilities(heads, source._test)
        entry["pool_uniform_p0"] = record(qt.mean(0).argmax(-1), ty, tg)["wga"]
        xf, xt = qf[:, :, 1].T, qt[:, :, 1].T
        for kind in ("affine", "linear"):
            model = fit_model(xf, source.val_y[fit_idx], source.val_g[fit_idx], kind, FIXED, cfg, fit_seed)
            entry[f"pool_{kind}_p0|{name(FIXED)}"] = record(predict_model(model, xt, cfg) > .5, ty, tg)["wga"]
        for key, value in entry.items():
            if key == "source" or key not in ref or spec["id"] not in ref[key]:
                continue
            delta = value - ref[key][spec["id"]]
            flag = abs(delta) > 0.05
            ok &= not flag
            rows.append((spec["id"], key, value, ref[key][spec["id"]], delta, "MISMATCH" if flag else "ok"))
    log("saved-ensemble regression check (test WGA, %)")
    for r in rows:
        log("  {:<18} {:<26} got {:7.3f}  expected {:7.3f}  delta {:+.3f}  {}".format(*r))
    write_json(Path(a.out) / "check.json", dict(rows=rows, ok=ok))
    log("CHECK", "PASSED" if ok else "FAILED (see MISMATCH rows)")
    return 0 if ok else 2


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    specs = load_specs(a.out, a.only)
    tasks = [(s, r, str(a.out), a.config) for r in a.splits for s in specs]
    pending = [t for t in tasks if not (Path(a.out) / "results" / t[0]["id"] / f"split{t[1]:02d}.json").exists()]
    log(f"{len(tasks)} tasks, {len(tasks) - len(pending)} already done, {len(pending)} to run on {a.workers} workers")
    start, done = time.time(), 0
    deadline = start + a.time_limit_hours * 3600
    if not pending:
        return 0
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures, queue = {}, list(pending)

        def submit():
            while queue and len(futures) < a.workers and time.time() < deadline:
                t = queue.pop(0)
                futures[pool.submit(run_task, t)] = t
        submit()
        while futures:
            for fut in as_completed(list(futures)):
                t = futures.pop(fut)
                sid, split, status, secs = fut.result()
                done += 1
                eta = (time.time() - start) / done * (len(pending) - done) / 60
                log(f"{status:6} {sid} split{split:02d} {secs / 60:5.1f} min   ({done}/{len(pending)}, eta {eta:.0f} min)")
                submit()
                break
    left = len(queue)
    if left:
        log(f"time limit reached; {left} tasks not started. Re-run the same command to resume.")
    errors = list((Path(a.out) / "errors").glob("*.txt")) if (Path(a.out) / "errors").exists() else []
    if errors:
        log(f"{len(errors)} task errors; first: {errors[0]}")
    return 0


# ----------------------------------------------------------------------------- summary

def t_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 3:
        return mean, float("nan"), float("nan")
    from scipy import stats
    half = stats.t.ppf(.975, n - 1) * values.std(ddof=1) / math.sqrt(n)
    return mean, mean - half, mean + half


def fmt(v, digits=2, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_results(out):
    cells = []
    for p in sorted((Path(out) / "results").glob("*/split*.json")):
        cells.append(read_json(p))
    return cells


def cmd_summarize(a):
    out = Path(a.out)
    cells = load_results(out)
    if not cells:
        log("no results yet")
        return 1
    ref = read_json(REFERENCE) if REFERENCE.is_file() else {}
    arms_all = sorted({k for c in cells for k in c["arms"]})
    long_rows = []
    for c in cells:
        for arm, v in c["arms"].items():
            long_rows.append(dict(source=c["source"], recipe=c["recipe"], split=c["split"], arm=arm, wga=v["wga"],
                                  overall=v["overall"], selected=v.get("selected", ""),
                                  **{f"g{k}": v["groups"][k] for k in GROUPS}))
    with (out / "results_long.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)

    sources = sorted({c["source"] for c in cells})
    recipe_of = {c["source"]: c["recipe"] for c in cells}
    by = {}
    for c in cells:
        for arm, v in c["arms"].items():
            by.setdefault((c["source"], arm), {})[c["split"]] = v["wga"]

    def per_source_mean(arm, subset):
        return {s: float(np.mean(list(by[(s, arm)].values()))) for s in subset if (s, arm) in by}

    def paired(arm, comparator, subset):
        diffs = {}
        for s in subset:
            if (s, arm) in by and (s, comparator) in by:
                common = sorted(set(by[(s, arm)]) & set(by[(s, comparator)]))
                if common:
                    diffs[s] = float(np.mean([by[(s, arm)][r] - by[(s, comparator)][r] for r in common]))
        return diffs

    def vs_saved(arm, subset):
        saved = ref.get("fullval_saved", {})
        return {s: m - saved[s] for s, m in per_source_mean(arm, subset).items() if s in saved}

    lines = ["# Nested selection results", "",
             f"Cells: {len(cells)} (source x split). Splits present: {sorted({c['split'] for c in cells})}. "
             f"Primary selection rule: `{PRIMARY_RULE}`; secondary: `wga`. Test WGA in %, differences in percentage points (pp).",
             "Intervals are pointwise 95% t-intervals over sources of the per-source mean over split seeds; "
             "ERM* has two sources, so it is descriptive only. The test set and the seven backbones were inspected before this run.", ""]
    groups = {"normal_erm": [s for s in sources if recipe_of[s] == "normal_erm"],
              "ermstar": [s for s in sources if recipe_of[s] == "ermstar"]}

    # --- absolute WGA
    header = ["arm"] + [f"{r} (n={len(sub)})" for r, sub in groups.items() if sub]
    rows = []
    order = ["fullval_uniform", "bag75_uniform", "fullval_thrwga_oof", "fullval_threshold_oof_fixed", "fullval_threshold_oof_sel_nll",
             "fullval_affine_oof_fixed", "fullval_affine_oof_sel_nll", "crossfit_uniform", "crossfit_thrwga", "crossfit_threshold_fixed",
             "crossfit_threshold_sel_nll", "crossfit_affine_fixed", "crossfit_affine_sel_nll", "crossfit_affine_sel_wga", "pool_uniform", "pool_affine_fixed", "pool_affine_sel_nll", "pool_affine_sel_wga",
             "pool_linear_fixed", "pool_linear_sel_nll", "pool_linear_sel_wga", "dfr_lite"]
    for arm in [x for x in order if x in arms_all] + [x for x in arms_all if x not in order]:
        row = [f"`{arm}`"]
        for r, sub in groups.items():
            if sub:
                m = per_source_mean(arm, sub)
                row.append(fmt(np.mean(list(m.values()))) if m else "n/a")
        rows.append(row)
    saved = ref.get("fullval_saved", {})
    if saved:
        row = ["`fullval_saved` (paper-style, seed fixed)"]
        for r, sub in groups.items():
            if sub:
                row.append(fmt(np.mean([saved[s] for s in sub if s in saved])))
        rows.append(row)
    lines += ["## Mean test WGA", "", md_table(header, rows), ""]

    # --- reproduction of the paper's numbers
    paper = {"normal_erm": ("ERM", 69.1, "ERM + DPE", 91.0), "ermstar": ("ERM*", 77.9, "ERM* + DPE", 94.1)}
    try:
        specs_all = read_json(out / "sources.json")["sources"]
    except Exception:
        specs_all = []
    erm_only = {}
    for sp in specs_all:
        if isinstance(sp.get("erm_test"), dict):
            erm_only.setdefault(sp["recipe"], {})[sp["id"]] = sp["erm_test"]["wga"]
    rows = []
    for r, sub in groups.items():
        if not sub:
            continue
        m = list(per_source_mean("fullval_uniform", sub).values())
        e = [v for k, v in erm_only.get(r, {}).items() if k in sub]
        rows.append([r, len(sub), paper[r][0], fmt(np.mean(e), 1) if e else "n/a (backbone not retrained here)", f"{paper[r][1]:.1f}",
                     "full-validation DPE, heads retrained per split seed", f"{np.mean(m):.2f} (SD over sources {np.std(m, ddof=1) if len(m) > 1 else float('nan'):.2f})",
                     f"{paper[r][3]:.1f}"])
    lines += ["## Reproduction of the paper's numbers (test WGA, %)", "",
              md_table(["recipe", "sources", "backbone", "backbone alone (ours)", "backbone alone (paper)", "with DPE", "with DPE (ours)", "with DPE (paper)"], rows), ""]

    # --- paired differences
    lines += ["## Paired differences (pp)", ""]
    comparisons = [(arm, comp) for arm in ("crossfit_affine_sel_nll", "crossfit_affine_sel_wga", "crossfit_affine_fixed",
                                           "crossfit_uniform", "pool_affine_sel_nll", "pool_linear_sel_nll", "dfr_lite")
                   for comp in ("fullval_uniform", "bag75_uniform")]
    comparisons += [("crossfit_affine_sel_nll", "crossfit_uniform"), ("crossfit_affine_sel_nll", "crossfit_affine_fixed"),
                    ("crossfit_affine_fixed", "crossfit_uniform"), ("pool_linear_sel_nll", "pool_affine_sel_nll"),
                    ("crossfit_affine_sel_nll", "dfr_lite"),
                    ("crossfit_affine_sel_nll", "fullval_threshold_oof_sel_nll"), ("crossfit_affine_sel_nll", "fullval_affine_oof_sel_nll"),
                    ("crossfit_affine_sel_nll", "crossfit_threshold_sel_nll"), ("crossfit_affine_sel_nll", "fullval_thrwga_oof"),
                    ("fullval_threshold_oof_sel_nll", "fullval_uniform"), ("fullval_affine_oof_sel_nll", "fullval_uniform"),
                    ("fullval_thrwga_oof", "fullval_uniform"), ("crossfit_threshold_sel_nll", "fullval_uniform"),
                    ("crossfit_thrwga", "fullval_uniform")]
    for r, sub in groups.items():
        if not sub:
            continue
        rows = []
        for arm, comp in comparisons:
            d = paired(arm, comp, sub)
            if not d:
                continue
            mean, lo, hi = t_interval(list(d.values()))
            wins = sum(v > 0 for v in d.values())
            rows.append([f"`{arm}` - `{comp}`", fmt(mean, 2, True), f"[{fmt(lo, 2, True)}, {fmt(hi, 2, True)}]",
                         f"{wins}/{len(d)}", ", ".join(fmt(v, 2, True) for v in d.values())])
        for arm in ("crossfit_affine_sel_nll", "crossfit_affine_fixed", "crossfit_uniform", "dfr_lite"):
            d = vs_saved(arm, sub)
            if d:
                mean, lo, hi = t_interval(list(d.values()))
                rows.append([f"`{arm}` - saved full-val DPE", fmt(mean, 2, True), f"[{fmt(lo, 2, True)}, {fmt(hi, 2, True)}]",
                             f"{sum(v > 0 for v in d.values())}/{len(d)}", ", ".join(fmt(v, 2, True) for v in d.values())])
        lines += [f"### {r}", "", md_table(["comparison", "mean", "95% t-interval", "sources > 0", "per source"], rows), ""]

    # --- all four settings, no selection
    lines += ["## Every setting, no selection (mean test WGA, %)", ""]
    rows = []
    for fam in ("crossfit_affine", "crossfit_threshold", "fullval_affine_oof", "fullval_threshold_oof", "pool_affine", "pool_linear"):
        for setting in SETTINGS:
            row = [f"{fam} {name(setting)}"]
            for r, sub in groups.items():
                vals = [c["grid"][fam][name(setting)]["wga"] for c in cells if c["source"] in sub and fam in c["grid"]]
                row.append(fmt(np.mean(vals)) if vals else "n/a")
            rows.append(row)
    lines += [md_table(["family and setting"] + [r for r, sub in groups.items() if sub], rows), ""]

    # --- selection log
    lines += ["## What validation picked (share of source x split cells)", ""]
    rows = []
    for fam in ("crossfit_affine", "crossfit_threshold", "pool_affine", "pool_linear"):
        for rule in RULES:
            arm = f"{fam}_sel_{rule}"
            for r, sub in groups.items():
                picks = [c["arms"][arm]["selected"] for c in cells if c["source"] in sub and arm in c["arms"]]
                if picks:
                    share = {name(s): picks.count(name(s)) / len(picks) for s in SETTINGS}
                    rows.append([f"`{arm}`", r, len(picks)] + [f"{100 * share[name(s)]:.0f}%" for s in SETTINGS])
    lines += [md_table(["arm", "recipe", "cells"] + [name(s) for s in SETTINGS], rows), ""]

    # --- split noise
    rows = []
    for r, sub in groups.items():
        for arm in ("fullval_uniform", "crossfit_affine_sel_nll", "crossfit_uniform"):
            sds = [np.std(list(by[(s, arm)].values()), ddof=1) for s in sub if (s, arm) in by and len(by[(s, arm)]) > 1]
            if sds:
                rows.append([r, f"`{arm}`", fmt(np.mean(sds))])
    if rows:
        lines += ["## Split-seed noise (mean over sources of the SD of test WGA across split seeds, pp)", "",
                  md_table(["recipe", "arm", "SD"], rows), ""]

    # --- regression to the saved cross-fit reference (split 0 only)
    rows = []
    for c in cells:
        if c["split"] != 0:
            continue
        for key, arm in (("crossfit_uniform", "crossfit_uniform"),):
            if c["source"] in ref.get(key, {}):
                rows.append([c["source"], key.replace("|", "/"), fmt(c["arms"][arm]["wga"]), fmt(ref[key][c["source"]]),
                             fmt(c["arms"][arm]["wga"] - ref[key][c["source"]], 2, True)])
        for setting in SETTINGS:
            key = f"crossfit_affine|{name(setting)}"
            if c["source"] in ref.get(key, {}):
                v = c["grid"]["crossfit_affine"][name(setting)]["wga"]
                rows.append([c["source"], key.replace("|", "/"), fmt(v), fmt(ref[key][c["source"]]), fmt(v - ref[key][c["source"]], 2, True)])
    if rows:
        lines += ["## Split 0 versus the earlier saved cross-fit run (fold seed 0, original head seeds)", "",
                  "Small differences are expected if the earlier run used a GPU; large ones mean the pipeline is not reproducing it.", "",
                  md_table(["source", "quantity", "this run", "earlier", "delta"], rows), ""]

    missing = [(s, r) for r in sorted({c["split"] for c in cells}) for s in sources
               if not (out / "results" / s / f"split{r:02d}.json").exists()]
    if missing:
        lines += [f"Missing cells: {missing}", ""]
    errors = sorted((out / "errors").glob("*.txt")) if (out / "errors").exists() else []
    if errors:
        lines += ["Task errors: " + ", ".join(e.name for e in errors), ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--roots", nargs="+", required=True)
    d.add_argument("--out", required=True)
    for cname in ("check", "run", "summarize"):
        s = sub.add_parser(cname)
        s.add_argument("--out", required=True)
        s.add_argument("--only", nargs="*", help="restrict to these source ids")
        if cname == "run":
            s.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
            s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)))
            s.add_argument("--time-limit-hours", type=float, default=10.0)
            s.add_argument("--config", default=None)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    sys.exit({"discover": cmd_discover, "check": cmd_check, "run": cmd_run, "summarize": cmd_summarize}[args.command](args))
