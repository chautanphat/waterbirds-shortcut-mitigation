"""Combined-levers run: strictly nested DPE post-hoc procedures fitted on corrected validation labels.

Wraps the frozen pipeline in ../dpe_nested_corrected/code (protocol strict-nested-postmean-v2)
without editing it. The only change is which validation labels the source exposes:

  official  unchanged (reproduces the strict nested run)
  relabel   the 18 validation images of species 021/088/102 relabelled landbird (group follows)
  drop      those 18 images removed (1,181 left)

Every arm is scored on official test labels, corrected test labels and the clean subset
(the 90 affected test images removed). Per-arm test probabilities are saved so the run can
be rescored without refitting. See PLAN.md for the declared contrasts.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "dpe_nested_corrected" / "code"
sys.path.insert(0, str(CODE))

import corrected_nested as cn  # noqa: E402
import nested_selection as old  # noqa: E402

SPECIES = ("021.Eastern_Towhee", "088.Western_Meadowlark", "102.Western_Wood_Pewee")
MODES = ("official", "relabel", "drop")
PROTOCOL = "combined-levers-v1"


def affected(ids):
    return np.array([str(i).split("/")[0] in SPECIES for i in ids])


def apply_mode(s, mode):
    """Change only the validation labels/rows a loaded source exposes."""
    if mode == "official":
        return s
    mask = affected([r["id"] for r in s.rows])
    if mask.sum() != 18:
        raise ValueError(f"Expected 18 affected validation images, found {mask.sum()}")
    if mode == "relabel":
        rows = []
        for r, m in zip(s.rows, mask):
            r = dict(r)
            if m:
                r["y"], r["group"] = 0, "0" + r["group"][1]
            rows.append(r)
        s.rows = rows
    elif mode == "drop":
        keep = np.flatnonzero(~mask)
        s.rows = [dict(s.rows[i]) for i in keep]
        s._val = s._val[keep]
    else:
        raise ValueError(mode)
    s.val_y = np.array([r["y"] for r in s.rows])
    s.val_g = np.array([r["group"] for r in s.rows])
    return s


_original_get_source = old.get_source
_MODE = {"value": "official"}


def get_source(spec):
    key = (spec["id"], _MODE["value"])
    cache = old._SOURCE_CACHE
    if key not in cache:
        base = _original_get_source(spec)          # cached under spec["id"], official labels
        clone = type(base).__new__(type(base))
        clone.__dict__.update(base.__dict__)
        clone.rows = [dict(r) for r in base.rows]
        cache[key] = apply_mode(clone, _MODE["value"])
    return cache[key]


old.get_source = get_source


def test_views(source):
    y, g = source.test_y, source.test_g
    mask = affected([r["image_path"] for r in source.test_rows])
    if mask.sum() != 90:
        raise ValueError("Expected 90 affected test images")
    yc, gc = y.copy(), g.astype(str).copy()
    yc[mask] = 0
    gc[mask] = np.array(["0" + x[1] for x in gc[mask]])
    return dict(official=(None, y, g), corrected=(None, yc, gc), removed=(~mask, y, g))


def score_views(pred, views, **extra):
    out = {}
    for view, (keep, y, g) in views.items():
        p = pred if keep is None else pred[keep]
        out[view] = old.record(p, y if keep is None else y[keep], g if keep is None else g[keep], **extra)
    return out


def run_task(task):
    """Same steps as corrected_nested.run_task; adds label mode, three test views and saved probabilities."""
    spec, split, out_dir, cfg, protocol_id, deadline, mode = task
    _MODE["value"] = mode
    started = time.time()
    out = Path(out_dir)
    cell = out / "results" / mode / spec["id"] / f"split{split:02d}.json"
    checkpoints = out / "checkpoints" / mode / spec["id"] / f"split{split:02d}"
    try:
        import torch
        torch.set_num_threads(1)
        source = old.get_source(spec)
        source_id = dict(id=spec["id"], recipe=spec["recipe"], seed=spec["seed"], val_labels=mode,
                         n_val=len(source.rows),
                         features={k: cn.digest_file(v) for k, v in spec["features"].items()},
                         metadata=cn.digest_file(spec["metadata"]))
        fingerprint = cn.digest_json(dict(protocol=protocol_id, wrapper=PROTOCOL, source=source_id, split=split))
        cn.bind_manifest(checkpoints / "identity.json", dict(fingerprint=fingerprint, source=source_id))
        if cell.exists():
            if old.read_json(cell).get("fingerprint") != fingerprint:
                raise ValueError("Completed result was produced by another protocol/source")
            return spec["id"], split, mode, "cached", 0.
        base = spec["seed"] + old.SPLIT_SEED_STRIDE * split
        fit_seed = spec["seed"] + cfg["seed_offset"]
        inner_path = checkpoints / "inner_scores.json"
        if inner_path.exists():
            saved = old.read_json(inner_path)
            inner, diagnostics = saved["scores"], saved["diagnostics"]
        else:
            inner, diagnostics = cn.strict_inner_scores(source, split, cfg, base, fit_seed, checkpoints, deadline)
            old.write_json(inner_path, dict(scores=inner, diagnostics=diagnostics))
        context = cn.build_context(source, np.arange(len(source.rows)), np.array([], dtype=int),
                                   True, split, base, cfg, checkpoints / "final", deadline)
        grid_prob, chosen, threshold, models = cn.final_predictions(source, context, inner, cfg, fit_seed)
        prob_arms = dict(fullval_uniform=context["full"], crossfit_uniform=context["folds"].mean(0))
        full_bag = [context["full"]]
        for k in range(1, old.BAG_ENSEMBLES):
            _, p = cn.cached_scores(source, np.arange(len(source.rows)), np.array([], dtype=int), True,
                                    base + k, cfg, checkpoints / f"bag{k}", deadline)
            full_bag.append(p)
        prob_arms["bag75_uniform"] = np.mean(full_bag, axis=0)
        selected = {}
        for family in cn.FAMILIES:
            prob_arms[family + "_fixed"] = grid_prob[family][old.name(cn.FIXED)]
            selected[family + "_fixed"] = old.name(cn.FIXED)
            for rule in cn.RULES:
                arm = family + "_sel_" + rule
                selected[arm] = chosen[family][rule]
                prob_arms[arm] = grid_prob[family][chosen[family][rule]]
        dfr_prob, dfr_info = cn.dfr_clean(source, split)
        prob_arms["dfr_lite"] = dfr_prob
        pool_uniform, pool_grid, pool_inner, pool_chosen = cn.pooled_predictions(
            source, split, cfg, base, fit_seed, checkpoints / "pooled", deadline)
        prob_arms["pool_uniform"] = pool_uniform
        inner.update(pool_inner)
        chosen.update(pool_chosen)
        for family in pool_grid:
            prob_arms[family + "_fixed"] = pool_grid[family][old.name(cn.FIXED)]
            selected[family + "_fixed"] = old.name(cn.FIXED)
            for rule in cn.RULES:
                arm = family + "_sel_" + rule
                selected[arm] = pool_chosen[family][rule]
                prob_arms[arm] = pool_grid[family][selected[arm]]
        # Fit/selection record and test probabilities are written before any test label is used.
        old.write_json(checkpoints / "fit_complete.json", dict(chosen=chosen, threshold=threshold,
                       models=models, dfr=dfr_info, fingerprint=fingerprint))
        preds = {arm: p > .5 for arm, p in prob_arms.items()}
        preds["crossfit_postmean_thrwga"] = cn.apply_threshold(threshold, context["folds"].mean(0))
        preds["fullval_postmean_thrwga"] = cn.apply_threshold(threshold, context["full"])
        cn.atomic_npz(checkpoints / "test_probabilities.npz",
                      **{k: np.asarray(v, dtype=np.float32) for k, v in prob_arms.items()},
                      threshold=np.array(threshold["threshold"]))
        views = test_views(source)
        arms = {arm: score_views(p, views, **({"selected": selected[arm]} if arm in selected else {}))
                for arm, p in preds.items()}
        old.write_json(cell, dict(protocol=cn.PROTOCOL, wrapper=PROTOCOL, val_labels=mode, n_val=len(source.rows),
                       fingerprint=fingerprint, source=spec["id"], recipe=spec["recipe"], seed=spec["seed"],
                       split=split, arms=arms, inner=inner, threshold=threshold, dfr=dfr_info,
                       seconds=time.time() - started))
        return spec["id"], split, mode, "done", time.time() - started
    except cn.BudgetReached:
        return spec["id"], split, mode, "deferred", time.time() - started
    except Exception:
        error = out / "errors" / f"{mode}-{spec['id']}-split{split:02d}.txt"
        error.parent.mkdir(parents=True, exist_ok=True)
        error.write_text(traceback.format_exc())
        return spec["id"], split, mode, "ERROR", time.time() - started


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    cfg, protocol_id = cn.prepare_protocol(a.out, None)
    specs = old.load_specs(a.sources_dir, a.only)
    if not specs:
        raise ValueError("No selected sources")
    deadline = time.time() + a.time_limit_hours * 3600
    # Order: mode first, then split, so an early stop leaves complete splits for every source.
    tasks = [(s, split, str(a.out), cfg, protocol_id, deadline, mode)
             for mode in a.modes for split in a.splits for s in specs]
    counts = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for f in as_completed(futures):
            sid, split, mode, status, seconds = f.result()
            counts[status] = counts.get(status, 0) + 1
            print(time.strftime("%H:%M:%S"), status, mode, sid, f"split{split:02d}", f"{seconds/60:.1f} min", flush=True)
    old.write_json(Path(a.out) / f"completion-{'-'.join(a.modes)}.json",
                   dict(requested=len(tasks), statuses=counts, splits=a.splits, modes=a.modes,
                        sources=[s["id"] for s in specs]))
    return 1 if counts.get("ERROR") else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sources-dir", required=True, type=Path, help="directory holding sources.json")
    p.add_argument("--only", nargs="*")
    p.add_argument("--modes", nargs="+", default=["relabel"], choices=MODES)
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--time-limit-hours", type=float, default=2.0)
    return p


if __name__ == "__main__":
    sys.exit(cmd_run(parser().parse_args()))
