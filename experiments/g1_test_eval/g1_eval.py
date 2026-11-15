"""G1 geometry weighting on full-validation DPE, scored on the Waterbirds test set.

Declared before any test result of this run is read (do not edit the protocol afterwards):

* Heads: full-validation DPE retrained with exactly the seeds of the strictly nested run
  (`results/dpe/strict_nested_20261115`): head seed = source seed + 1000 * split, split 0..9,
  the same frozen trainers and config. Reproduction check: the uniform arm must reproduce the
  recorded `fullval_uniform` official-label test WGA of every cell (|delta| <= 1e-9).
* Arms, all on the same 15 heads per cell, no fitted parameters and no retuning:
    uniform        mean of the 15 head probabilities (DPE);
    g1             w_j = (1 - a)/15 + a * softmax_j(-g * min_k d_jk), g = 50, a = 0.25 (locked G1);
    g1_a0.1, g1_a0.5   sensitivity only (reported, never selected);
    perm           G1 weights with the routing scores randomly permuted across heads per image
                   (5 fixed permutations, mean reported): tests whether geometry matters;
    conf           same mixture with routing softmax_j(g * max_c q_jc): confidence instead of distance.
  d_jk is the Euclidean distance between the per-image standardised, L2-normalised feature and the
  L2-normalised prototype k of head j (unscaled; identical to the teammate's verified G1 recipe).
* Views: official test labels (primary for this study); corrected labels and the clean subset
  (90 images of the three "tern"-substring species removed) are logged as secondary columns.
* Primary contrast: g1 - uniform, official labels, standard ERM, paired within cell, averaged over
  ten splits per backbone, 95% t interval over the five backbones.
  PASS if mean >= +0.30 pp, interval lower bound > 0, >= 4/5 backbones positive, AND
  g1 - perm has interval lower bound > 0. ERM* (two backbones) is descriptive.
* Context only: the strictly nested run's validation-selected offset on full-validation DPE
  (`fullval_postmean_bias_sel_nll`) uses the same heads when reproduction is exact, so
  g1 - offset is reported as a paired comparison.

Run inside the extracted code bundle of experiments/dpe_nested_corrected (frozen trainers).
"""
import argparse
import json
import os
import statistics as st
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nested_selection as old  # noqa: E402

PROTOCOL = 'g1-fullval-test-v1'
GAMMA, ALPHA = 50.0, 0.25
SENSITIVITY = (0.1, 0.5)
PERMUTATIONS = 5
SPECIES = ('021.Eastern_Towhee', '088.Western_Meadowlark', '102.Western_Wood_Pewee')
T95 = {1: 12.706204736, 3: 3.182446305, 4: 2.776445105}
PASS = dict(min_mean_pp=0.30, min_positive=4)


def softmax_rows(score):
    """score [15, N] -> softmax over heads (axis 0)."""
    s = score - score.max(0, keepdims=True)
    e = np.exp(s)
    return e / e.sum(0, keepdims=True)


def mix(q, route, alpha):
    """q [15,N,2] head probabilities, route [15,N] routing distribution -> [N,2]."""
    w = (1 - alpha) / q.shape[0] + alpha * route
    return (q * w[..., None]).sum(0)


def g1_route(dist):
    return softmax_rows(-GAMMA * dist.min(-1))


def arms(q, dist, seed):
    route = g1_route(dist)
    out = dict(uniform=q.mean(0), g1=mix(q, route, ALPHA))
    for a in SENSITIVITY:
        out[f'g1_a{a:g}'] = mix(q, route, a)
    rng = np.random.RandomState(seed)
    perms = []
    for _ in range(PERMUTATIONS):
        order = np.argsort(rng.rand(*route.shape), axis=0)  # independent permutation of heads per image
        perms.append(mix(q, np.take_along_axis(route, order, axis=0), ALPHA))
    out['perm'] = perms
    out['conf'] = mix(q, softmax_rows(GAMMA * q.max(-1)), ALPHA)
    return out


def score(pred, y, g):
    acc = {k: float((pred[g == k] == y[g == k]).mean()) * 100 for k in old.GROUPS if (g == k).any()}
    return dict(wga=min(acc.values()), overall=float((pred == y).mean()) * 100, groups=acc)


def views(source):
    y, g = np.asarray(source.test_y), np.asarray(source.test_g)
    rows = getattr(source, 'test_rows', None)
    out = dict(official=(np.ones(len(y), bool), y, g))
    if rows is not None:
        bad = np.array([str(r['image_path']).startswith(SPECIES) for r in rows])
        if bad.sum() != 90:
            raise ValueError(f'Expected 90 affected test images, found {bad.sum()}')
        a = np.array([int(r['a']) for r in rows])
        yc = y.copy()
        yc[bad] = 0
        out['corrected'] = (np.ones(len(y), bool), yc, np.array([f'{i}{j}' for i, j in zip(yc, a)]))
        out['clean'] = (~bad, y, g)
    return out


def run_cell(task):
    spec, split, out_dir, cfg, expected = task
    out = Path(out_dir)
    path = out / 'results' / spec['id'] / f'split{split:02d}.json'
    if path.exists():
        return spec['id'], split, 'cached', 0.0
    started = time.time()
    try:
        import torch
        torch.set_num_threads(1)
        source = old.get_source(spec)
        seed = spec['seed'] + old.SPLIT_SEED_STRIDE * split
        with tempfile.TemporaryDirectory() as scratch:
            heads = old.train_ensemble(source, np.arange(len(source.rows)), seed, cfg, scratch)
        geometry = old.adapters()['geometry']
        dist = geometry.distances(heads, source._test)            # [15, N, 2], unscaled normalised distances
        q = old.head_probabilities(heads, source._test)          # [15, N, 2], learned-scale probabilities
        cache = out / 'checkpoints' / spec['id'] / f'split{split:02d}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, q1=q[:, :, 1].astype(np.float32), dmin=dist.min(-1).astype(np.float32),
                            argmin=dist.argmin(-1).astype(np.int8))
        probs = arms(q, dist, seed)
        record = dict(protocol=PROTOCOL, source=spec['id'], recipe=spec['recipe'], split=split, seed=seed, arms={})
        for vname, (keep, y, g) in views(source).items():
            for arm, p in probs.items():
                if arm == 'perm':
                    runs = [score(x[keep].argmax(1), y[keep], g[keep]) for x in p]
                    r = dict(wga=float(np.mean([x['wga'] for x in runs])),
                             overall=float(np.mean([x['overall'] for x in runs])),
                             groups={k: float(np.mean([x['groups'][k] for x in runs])) for k in runs[0]['groups']})
                else:
                    r = score(p[keep].argmax(1), y[keep], g[keep])
                    r['changed_vs_uniform'] = int((p[keep].argmax(1) != probs['uniform'][keep].argmax(1)).sum())
                record['arms'].setdefault(arm, {})[vname] = r
        route = g1_route(dist)
        record['diagnostics'] = dict(
            mean_max_route_weight=float(route.max(0).mean()),
            mean_effective_heads=float((1 / (route ** 2).sum(0)).mean()),
            mean_tv_from_uniform=float(0.5 * np.abs(((1 - ALPHA) / 15 + ALPHA * route) - 1 / 15).sum(0).mean()))
        exp = (expected or {}).get(spec['id'], {}).get(str(split))
        if exp is not None:
            delta = record['arms']['uniform']['official']['wga'] - exp['fullval_uniform']['official']
            record['reproduction'] = dict(expected_fullval_uniform=exp['fullval_uniform']['official'],
                                          delta=delta, exact=abs(delta) <= 1e-9)
        record['seconds'] = time.time() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        old.write_json(path, record)
        return spec['id'], split, 'done', record['seconds']
    except Exception:
        err = out / 'errors' / f"{spec['id']}-split{split:02d}.txt"
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return spec['id'], split, 'ERROR', time.time() - started


def ci(d):
    m = st.mean(d)
    if len(d) < 3:
        return m, None, None
    h = T95[len(d) - 1] * st.stdev(d) / len(d) ** .5
    return m, m - h, m + h


def summarize(out, expected):
    out = Path(out)
    cells = [json.loads(p.read_text()) for p in sorted((out / 'results').glob('*/split*.json'))]
    lines = ['# G1 on full-validation DPE: test-set results', '',
             f'Protocol `{PROTOCOL}`. Cells: {len(cells)}. Primary: g1 - uniform, official labels, standard ERM.', '']
    repro = [c['reproduction']['exact'] for c in cells if 'reproduction' in c]
    lines += [f'Reproduction of recorded full-validation DPE (uniform arm): {sum(repro)}/{len(repro)} cells exact.', '']
    if repro and not all(repro):
        bad = [(c['source'], c['split'], c['reproduction']['delta']) for c in cells if not c['reproduction']['exact']]
        lines += ['**Reproduction mismatches** (heads differ from the strict run; offset comparison invalid for these):',
                  *[f'- {s} split {k}: {d:+.4f} pp' for s, k, d in bad[:20]], '']
    summary = {}
    for recipe in ('normal_erm', 'ermstar'):
        sub = [c for c in cells if c['recipe'] == recipe]
        if not sub:
            continue
        sources = sorted({c['source'] for c in sub})
        lines += [f'## {recipe} ({len(sources)} backbones, {len(sub)} cells)', '',
                  '| Contrast | View | Mean (pp) | 95% interval | Backbones > 0 | Per backbone |', '| --- | --- | ---: | --- | ---: | --- |']
        contrasts = [('g1', 'uniform'), ('g1', 'perm'), ('g1', 'conf'), ('perm', 'uniform'), ('conf', 'uniform'),
                     ('g1_a0.1', 'uniform'), ('g1_a0.5', 'uniform')]
        for a, b in contrasts:
            for view in ('official', 'clean', 'corrected'):
                d = []
                for s in sources:
                    vals = [c['arms'][a][view]['wga'] - c['arms'][b][view]['wga'] for c in sub
                            if c['source'] == s and view in c['arms'][a]]
                    if vals:
                        d.append(st.mean(vals))
                if not d:
                    continue
                m, lo, hi = ci(d)
                summary.setdefault(recipe, {})[f'{a}-{b}|{view}'] = dict(mean=m, lo=lo, hi=hi, per_source=d)
                interval = '' if lo is None else f'[{lo:+.2f}, {hi:+.2f}]'
                lines.append(f'| {a} - {b} | {view} | {m:+.2f} | {interval} | {sum(x > 0 for x in d)}/{len(d)} | '
                             + ', '.join(f'{x:+.2f}' for x in d) + ' |')
        if expected:
            for view in ('official', 'clean', 'corrected'):
                d = []
                for s in sources:
                    vals = [c['arms']['g1'][view]['wga'] - expected[s][str(c['split'])]['fullval_offset_nll'][view]
                            for c in sub if c['source'] == s and c.get('reproduction', {}).get('exact')
                            and view in c['arms']['g1'] and view in expected[s][str(c['split'])]['fullval_offset_nll']]
                    if vals:
                        d.append(st.mean(vals))
                if d:
                    m, lo, hi = ci(d)
                    interval = '' if lo is None else f'[{lo:+.2f}, {hi:+.2f}]'
                    lines.append(f'| g1 - validation-selected offset (strict run) | {view} | {m:+.2f} | {interval} | '
                                 f'{sum(x > 0 for x in d)}/{len(d)} | ' + ', '.join(f'{x:+.2f}' for x in d) + ' |')
        lines.append('')
        lines += ['Mean test WGA (%) and per-group accuracy, official labels:', '',
                  '| Arm | WGA | Overall | ' + ' | '.join(old.GROUPS) + ' | changed vs uniform |',
                  '| --- | ---: | ---: |' + ' ---: |' * len(old.GROUPS) + ' ---: |']
        for arm in ('uniform', 'g1', 'g1_a0.1', 'g1_a0.5', 'perm', 'conf'):
            src_mean = lambda f: st.mean(st.mean(f(c) for c in sub if c['source'] == s) for s in sources)
            r = lambda c: c['arms'][arm]['official']
            groups = [src_mean(lambda c, k=k: r(c)['groups'][k]) for k in old.GROUPS]
            changed = src_mean(lambda c: r(c).get('changed_vs_uniform', 0))
            lines.append(f"| {arm} | {src_mean(lambda c: r(c)['wga']):.2f} | {src_mean(lambda c: r(c)['overall']):.2f} | "
                         + ' | '.join(f'{x:.2f}' for x in groups) + f' | {changed:.1f} |')
        diag = {k: st.mean(c['diagnostics'][k] for c in sub) for k in sub[0]['diagnostics']}
        lines += ['', 'Routing diagnostics (mean): ' + ', '.join(f'{k}={v:.4f}' for k, v in diag.items()), '']
    std = summary.get('normal_erm', {})
    p, c = std.get('g1-uniform|official'), std.get('g1-perm|official')
    verdict = 'INCOMPLETE'
    if p and c and len(p['per_source']) == 5:
        ok = (p['mean'] >= PASS['min_mean_pp'] and p['lo'] > 0 and sum(x > 0 for x in p['per_source']) >= PASS['min_positive']
              and c['lo'] > 0)
        verdict = 'PASS' if ok else 'FAIL'
    lines += [f'## Declared verdict: **{verdict}**', '',
              f"Rule: g1 - uniform (official, standard ERM) mean >= +{PASS['min_mean_pp']:.2f} pp, interval > 0, "
              f">= {PASS['min_positive']}/5 backbones positive, and g1 - perm interval > 0.", '',
              'The test set and these backbones were inspected in earlier work; this is exploratory evidence, not '
              'an untouched confirmation.']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL, verdict=verdict, contrasts=summary), indent=1))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--sources', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--splits', type=int, nargs='+', default=list(range(10)))
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--expected', default=str(HERE / 'g1_expected.json'))
    s = sub.add_parser('summarize')
    s.add_argument('--out', required=True)
    s.add_argument('--expected', default=str(HERE / 'g1_expected.json'))
    a = ap.parse_args()
    expected = json.loads(Path(a.expected).read_text()) if Path(a.expected).exists() else None
    if a.cmd == 'run':
        cfg = old.adapters()['grid'].load_config()
        if expected and expected.get('_config') and json.loads(json.dumps(cfg)) != expected['_config']:
            raise SystemExit('Head-training config differs from the strict run; refusing to run.')
        specs = json.loads(Path(a.sources).read_text())['sources']
        tasks = [(spec, k, a.out, cfg, expected) for spec in specs for k in a.splits]
        tasks.sort(key=lambda t: t[0]['recipe'] != 'ermstar')  # slower ERM* cells first
        failed = 0
        with ProcessPoolExecutor(max(1, min(a.workers, os.cpu_count() or 1))) as pool:
            for fut in as_completed([pool.submit(run_cell, t) for t in tasks]):
                sid, k, status, secs = fut.result()
                failed += status == 'ERROR'
                print(f'[{time.strftime("%H:%M:%S")}] {status} {sid} split{k:02d} {secs:.0f}s', flush=True)
        print(summarize(a.out, expected))
        sys.exit(1 if failed else 0)
    print(summarize(a.out, expected))


if __name__ == '__main__':
    main()
