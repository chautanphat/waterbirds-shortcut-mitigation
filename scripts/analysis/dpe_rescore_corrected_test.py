"""Re-score every arm of the strictly nested DPE run on corrected test labels.

Rebuilds each arm's test predictions from the run's own checkpoints (cross-fit fold, full-refit,
bag and 720/479 head scores; fitted calibration maps, chosen settings and thresholds written to
fit_complete.json before test scoring), using the frozen code in experiments/dpe_nested_corrected.
Nothing is refitted except the 720/479 combiners, whose fits are deterministic given the saved
head scores and seed. Every rebuilt arm must reproduce the run's official-label record exactly
before it is re-scored. DFR-lite needs raw features and is skipped.

Corrected test labels: the 90 images of 021 Eastern Towhee, 088 Western Meadowlark and
102 Western Wood-Pewee become landbird (y=0); their group follows. Validation labels and all
fits are unchanged (official), so this isolates the test-label effect on the comparison.

Usage: python scripts/analysis/dpe_rescore_corrected_test.py <unzipped corrected_nested_resume> <out_dir>
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'experiments/dpe_nested_corrected/code'
sys.path.insert(0, str(CODE))
import corrected_nested as cn  # noqa: E402
import nested_selection as old  # noqa: E402

SPECIES = ('021.Eastern_Towhee', '088.Western_Meadowlark', '102.Western_Wood_Pewee')
STRICT = ROOT / 'results/dpe/strict_nested_20261115'
T95 = {1: 12.706204736, 4: 2.776445105}


def load(path, *keys):
    with np.load(path, allow_pickle=False) as z:
        return [z[k] for k in keys]


def wga(pred, y, g):
    return min(float((pred[g == k] == y[g == k]).mean()) * 100 for k in old.GROUPS)


def find_checkpoints(root):
    hits = sorted(Path(root).rglob('fit_complete.json'))
    if not hits:
        raise SystemExit(f'No fit_complete.json under {root}')
    return {(p.parent.parent.name, p.parent.name): p.parent for p in hits}


def rebuild(ck, spec, split, source, cfg):
    fc = old.read_json(ck / 'fit_complete.json')
    folds = np.stack([load(ck / 'final' / f'fold{f}' / 'scores.npz', 'test')[0] for f in range(cn.CROSSFIT_FOLDS)])
    full = load(ck / 'final' / 'full_refit' / 'scores.npz', 'test')[0]
    bag = [full] + [load(ck / f'bag{k}' / 'scores.npz', 'test')[0] for k in range(1, old.BAG_ENSEMBLES)]
    arms = dict(fullval_uniform=full, crossfit_uniform=folds.mean(0), bag75_uniform=np.mean(bag, axis=0))
    grid = {fam: {} for fam in cn.FAMILIES}
    for setting in cn.SETTINGS:
        m = fc['models'][old.name(setting)]
        for fam, p in cn.calibration_predictions(m['affine'], m['bias'], folds, full, cfg).items():
            grid[fam][old.name(setting)] = p
    for fam in cn.FAMILIES:
        arms[fam + '_fixed'] = grid[fam][old.name(cn.FIXED)]
        for rule in cn.RULES:
            arms[f'{fam}_sel_{rule}'] = grid[fam][fc['chosen'][fam][rule]]
    # 720/479 combiners: deterministic refit from the saved head scores.
    part = old.adapters()['sources'].partition(source.rows, split)
    lookup = {r['id']: i for i, r in enumerate(source.rows)}
    fit_idx = np.array([lookup[r['id']] for r in part['combiner_training']])
    qf, qt = load(ck / 'pooled' / 'head_scores.npz', 'fit', 'test')
    xf, xt = qf[:, :, 1].T, qt[:, :, 1].T
    y, g = source.val_y[fit_idx], source.val_g[fit_idx]
    fit_seed = spec['seed'] + cfg['seed_offset']
    arms['pool_uniform'] = qt.mean(0)[:, 1]
    for kind in ('affine', 'linear'):
        fam = 'pool_' + kind
        need = {old.name(cn.FIXED)} | {fc['chosen'][fam][r] for r in cn.RULES}
        fitted = {}
        for setting in cn.SETTINGS:
            if old.name(setting) in need:
                fitted[old.name(setting)] = old.predict_model(old.fit_model(xf, y, g, kind, setting, cfg, fit_seed), xt, cfg)
        arms[fam + '_fixed'] = fitted[old.name(cn.FIXED)]
        for rule in cn.RULES:
            arms[f'{fam}_sel_{rule}'] = fitted[fc['chosen'][fam][rule]]
    preds = {a: p > .5 for a, p in arms.items()}
    preds['crossfit_postmean_thrwga'] = cn.apply_threshold(fc['threshold'], folds.mean(0))
    preds['fullval_postmean_thrwga'] = cn.apply_threshold(fc['threshold'], full)
    return preds


def main(resume_root, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = old.read_json(STRICT / 'protocol.json')['config']
    specs = {s['id']: s for s in old.read_json(STRICT / 'sources.json')['sources']}
    checkpoints = find_checkpoints(resume_root)
    metadata = str(CODE / 'data' / 'metadata.csv')
    source = old.adapters()['sources'].Source(dict(metadata=metadata))
    source.val_y = np.array([r['y'] for r in source.rows])
    source.val_g = np.array([r['group'] for r in source.rows])
    test_y = np.array([r['y'] for r in source.test_rows])
    test_a = np.array([r['a'] for r in source.test_rows])
    affected = np.array([str(r['image_path']).startswith(SPECIES) for r in source.test_rows])
    assert affected.sum() == 90, affected.sum()
    y_cor = test_y.copy()
    y_cor[affected] = 0
    g_off = np.array([f'{a}{b}' for a, b in zip(test_y, test_a)])
    g_cor = np.array([f'{a}{b}' for a, b in zip(y_cor, test_a)])
    rows, mismatches = [], []
    for (sid, split_name), ck in sorted(checkpoints.items()):
        split = int(split_name.removeprefix('split'))
        cell = old.read_json(STRICT / 'results' / sid / f'{split_name}.json')
        preds = rebuild(ck, specs[sid], split, source, cfg)
        for arm, p in preds.items():
            off = wga(p, test_y, g_off)
            if abs(off - cell['arms'][arm]['wga']) > 1e-9:
                mismatches.append((sid, split, arm, off, cell['arms'][arm]['wga']))
            keep = ~affected
            rows.append(dict(source=sid, recipe=specs[sid]['recipe'], split=split, arm=arm,
                             wga_official=off, wga_corrected=wga(p, y_cor, g_cor),
                             wga_removed=wga(p[keep], test_y[keep], g_off[keep]),
                             affected_pred_landbird=int((~p[affected]).sum())))
        print(f'{sid} {split_name}: {len(preds)} arms', flush=True)
    with (out / 'per_arm.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = dict(cells=len(checkpoints), arms_per_cell=len(rows) // max(1, len(checkpoints)),
                  official_mismatches=len(mismatches), examples=mismatches[:10], contrasts={})
    for recipe in ('normal_erm', 'ermstar'):
        srcs = sorted({r['source'] for r in rows if r['recipe'] == recipe})
        for arm in sorted({r['arm'] for r in rows}):
            for lab in ('wga_official', 'wga_corrected', 'wga_removed', 'affected_pred_landbird'):
                d = []
                for s in srcs:
                    by = {r['split']: r for r in rows if r['source'] == s}
                    a = {r['split']: r[lab] for r in rows if r['source'] == s and r['arm'] == arm}
                    b = {r['split']: r[lab] for r in rows if r['source'] == s and r['arm'] == 'fullval_uniform'}
                    d.append(st.mean(a[k] - b[k] for k in sorted(a)))
                m = st.mean(d)
                h = T95[len(d) - 1] * st.stdev(d) / len(d) ** .5 if len(d) in (2, 5) else None
                report['contrasts'].setdefault(recipe, {}).setdefault(arm, {})[lab] = dict(
                    mean=m, lo=None if h is None or len(d) == 2 else m - h, hi=None if h is None or len(d) == 2 else m + h,
                    positive=sum(x > 0 for x in d), per_source=d)
    (out / 'summary.json').write_text(json.dumps(report, indent=1))
    print(f"official mismatches: {len(mismatches)}")
    for arm, v in sorted(report['contrasts']['normal_erm'].items()):
        o, c, r, n = v['wga_official'], v['wga_corrected'], v['wga_removed'], v['affected_pred_landbird']
        print(f"{arm:34s} official {o['mean']:+.2f} | corrected {c['mean']:+.2f} [{c['lo'] or 0:+.2f},{c['hi'] or 0:+.2f}] pos={c['positive']}"
              f" | 90 removed {r['mean']:+.2f} [{r['lo'] or 0:+.2f},{r['hi'] or 0:+.2f}] pos={r['positive']} | landbird calls on 90 {n['mean']:+.1f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
