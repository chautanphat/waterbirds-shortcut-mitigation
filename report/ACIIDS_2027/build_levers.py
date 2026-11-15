"""Build figures/levers.tex: effect of each lever against full-validation DPE (standard ERM).

Reads tracked records only; fits and selects nothing.
  * results/dpe/strict_nested_20261115/results/<source>/splitNN.json  (strictly nested rerun)
  * results/dpe/independent_methods_20261113/{summary,per_seed}.csv   (840 fixed predictors)
  * results/dpe/label_correction_20261115/per_run.csv                 (label study)
Differences are paired per source (mean over split seeds or head seeds first), then a
pointwise 95% t interval over the five standard-ERM sources.
"""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'figures' / 'levers.tex'
SOURCES = [f'normal-erm-seed{i}' for i in range(5)]
T95_4 = 2.776445105
HEAD_SEED_SD = 0.405  # mean within-source SD of full-validation DPE over ten split seeds (strict run)


def ci(values):
    m = st.mean(values)
    h = T95_4 * st.stdev(values) / len(values) ** .5
    return m, m - h, m + h


def nested(arm, ref='fullval_uniform'):
    d = []
    for s in SOURCES:
        diffs = []
        for f in sorted((ROOT / 'results/dpe/strict_nested_20261115/results' / s).glob('split*.json')):
            a = json.loads(f.read_text())['arms']
            diffs.append(a[arm]['wga'] - a[ref]['wga'])
        assert len(diffs) == 10, (s, arm)
        d.append(st.mean(diffs))
    return ci(d)


def oracle():
    """Best of the 840 fixed standard-ERM predictors, picked on test (an optimistic bound)."""
    base = ROOT / 'results/dpe/independent_methods_20261113'
    summ = [r for r in csv.DictReader((base / 'summary.csv').open(encoding='utf-8-sig')) if r['recipe'] == 'normal_erm']
    best = max(summ, key=lambda r: float(r['mean_wga']))
    rows = list(csv.DictReader((base / 'per_seed.csv').open(encoding='utf-8-sig')))

    def per(arm, cfg):
        v = {r['seed']: 100 * float(r['wga']) for r in rows
             if (r['recipe'], r['arm'], r['configuration']) == ('normal_erm', arm, cfg)}
        assert len(v) == 5
        return v
    b, f = per(best['arm'], best['configuration']), per('full_validation', 'uniform')
    return ci([b[k] - f[k] for k in sorted(f)]), len(summ), best


def labels(method):
    rows = [r for r in csv.DictReader((ROOT / 'results/dpe/label_correction_20261115/per_run.csv').open())
            if r['recipe'] == 'normal_erm' and r['method'] == method and r['test_labels'] == 'corrected']
    d = []
    for s in SOURCES:
        m = {c: st.mean(float(r['wga']) for r in rows if r['source'] == s and r['condition'] == c)
             for c in ('official', 'relabel')}
        d.append(m['relabel'] - m['official'])
    return ci(d)


(orc, n_pred, best) = oracle()
ROWS = [  # (label, (mean, lo, hi), style)
    ('Aggregation: 720 heads + 479 for the combiner', None, None),
    ('affine vs.\\ matched 720-image ensemble', nested('pool_affine_sel_nll', 'pool_uniform'), 'trap'),
    ('affine combiner', nested('pool_affine_sel_nll'), 'main'),
    ('linear residual on head scores', nested('pool_linear_sel_nll'), 'main'),
    (f'best of {n_pred} predictors, chosen on test', orc, 'oracle'),
    ('Cross-fitting: all 1,199 images', None, None),
    ('five-fold uniform', nested('crossfit_uniform'), 'main'),
    ('75-head bag (no held-out scores)', nested('bag75_uniform'), 'main'),
    ('five-fold + affine recalibration', nested('crossfit_affine_sel_nll'), 'main'),
    ('five-fold + exact WGA threshold', nested('crossfit_postmean_thrwga'), 'main'),
    ('Clean labels: 18 validation images', None, None),
    ('DPE refit (corrected test labels)', labels('dpe'), 'label'),
    ('DFR-lite refit (corrected test labels)', labels('dfr_lite'), 'label'),
]

XMIN, XMAX, SCALE = -1.5, 3.5, 0.9  # pp range and cm per pp
x = lambda v: (v - XMIN) * SCALE
lines = [r'\begin{tikzpicture}[font=\scriptsize,y=0.42cm]']
n = len(ROWS)
y0 = 0.6
lines.append(rf'\fill[gray!15] ({x(-HEAD_SEED_SD):.3f},{y0:.2f}) rectangle ({x(HEAD_SEED_SD):.3f},{-n + 0.4:.2f});')
lines.append(rf'\draw[gray] ({x(0):.3f},{y0:.2f}) -- ({x(0):.3f},{-n + 0.4:.2f});')
for t in range(int(XMIN), int(XMAX) + 1):
    lines.append(rf'\draw ({x(t):.3f},{-n + 0.4:.2f}) -- ++(0,-0.15) node[below]{{${t:+d}$}};'.replace('{$+0$}', '{$0$}'))
lines.append(rf'\draw ({x(XMIN):.3f},{-n + 0.4:.2f}) -- ({x(XMAX):.3f},{-n + 0.4:.2f});')
lines.append(rf'\node[below] at ({x((XMIN + XMAX) / 2):.3f},{-n - 0.5:.2f}) {{$\Delta$ worst-group accuracy vs.\ full-validation DPE (pp)}};')
STY = {'main': 'fill=black', 'trap': 'fill=white', 'oracle': 'fill=white', 'label': 'fill=orange!85!black'}
audit = []
for i, (name, val, sty) in enumerate(ROWS):
    yy = -i
    if val is None:
        lines.append(rf'\node[anchor=east,font=\scriptsize\bfseries] at (-0.15,{yy}) {{{name}}};')
        continue
    m, lo, hi = val
    lines.append(rf'\node[anchor=east] at (-0.15,{yy}) {{{name}}};')
    lines.append(rf'\draw[thick] ({x(lo):.3f},{yy}) -- ({x(hi):.3f},{yy});')
    shape = 'diamond' if sty == 'oracle' else 'circle'
    lines.append(rf'\node[draw,{shape},inner sep=1.4pt,{STY[sty]}] at ({x(m):.3f},{yy}) {{}};')
    lines.append(rf'\node[anchor=west] at ({x(XMAX) + 0.05:.3f},{yy}) {{${m:+.2f}$}};')
    audit.append(dict(row=name, mean=round(m, 4), lo=round(lo, 4), hi=round(hi, 4)))
lines.append(r'\end{tikzpicture}')
OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT.parent / 'levers_audit.json').write_text(json.dumps(dict(
    head_seed_sd=HEAD_SEED_SD, oracle_config=dict(arm=best['arm'], configuration=best['configuration']),
    rows=audit), indent=1), encoding='utf-8')
for a in audit:
    print(f"{a['row']:48s} {a['mean']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}]")
