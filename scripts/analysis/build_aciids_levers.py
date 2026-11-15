"""Build report/ACIIDS_2027/figures/levers.tex: every lever against full-validation DPE, scored on
official test labels and on the 5,704 test images left after removing the 90 mislabelled ones (two panels, same rows). Standard ERM only.

Inputs (tracked records; nothing is fitted or selected here):
  results/dpe/corrected_test_rescore_20261115/per_arm.csv     strict nested run, every arm on both label sets
  results/dpe/independent_methods_20261113/{summary,per_seed}.csv   840 fixed predictors (official labels only)
  results/dpe/label_correction_20261115/per_run.csv          DPE refit on corrected validation labels
  results/dpe/label_corrected_backbones_20261115/results/*/  fresh-backbone replication (authors' code)
Differences are paired per backbone (mean over seeds first); pointwise 95% t interval over backbones.
"""
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report/ACIIDS_2027/figures/levers.tex'
AUDIT = ROOT / 'results/dpe/corrected_test_rescore_20261115/levers_audit.json'
T95 = {3: 3.182446305, 4: 2.776445105}
SPECIES = ('021.Eastern_Towhee', '088.Western_Meadowlark', '102.Western_Wood_Pewee')
HEAD_SEED_SD = 0.405
STD = [f'normal-erm-seed{i}' for i in range(5)]


def ci(d):
    m = st.mean(d)
    h = T95[len(d) - 1] * st.stdev(d) / len(d) ** .5
    return m, m - h, m + h


RESCORE = list(csv.DictReader((ROOT / 'results/dpe/corrected_test_rescore_20261115/per_arm.csv').open()))


def nested(arm, label):
    d = []
    for s in STD:
        a = {r['split']: float(r[label]) for r in RESCORE if r['source'] == s and r['arm'] == arm}
        b = {r['split']: float(r[label]) for r in RESCORE if r['source'] == s and r['arm'] == 'fullval_uniform'}
        assert len(a) == len(b) == 10, (s, arm)
        d.append(st.mean(a[k] - b[k] for k in a))
    return ci(d)


def oracle():
    base = ROOT / 'results/dpe/independent_methods_20261113'
    summ = [r for r in csv.DictReader((base / 'summary.csv').open(encoding='utf-8-sig')) if r['recipe'] == 'normal_erm']
    best = max(summ, key=lambda r: float(r['mean_wga']))
    rows = list(csv.DictReader((base / 'per_seed.csv').open(encoding='utf-8-sig')))
    per = lambda arm, cfg: {r['seed']: 100 * float(r['wga']) for r in rows
                            if (r['recipe'], r['arm'], r['configuration']) == ('normal_erm', arm, cfg)}
    b, f = per(best['arm'], best['configuration']), per('full_validation', 'uniform')
    return ci([b[k] - f[k] for k in sorted(f)]), len(summ)


def label_study(test_labels):
    rows = [r for r in csv.DictReader((ROOT / 'results/dpe/label_correction_20261115/per_run.csv').open())
            if r['recipe'] == 'normal_erm' and r['method'] == 'dpe' and r['test_labels'] == test_labels]
    d = []
    for s in STD:
        m = {c: st.mean(float(r['wga']) for r in rows if r['source'] == s and r['condition'] == c)
             for c in ('official', 'relabel')}
        d.append(m['relabel'] - m['official'])
    return ci(d)


def fresh(test_labels):
    meta = {r['img_filename']: (int(r['y']), int(r['place'])) for r in csv.DictReader(
        (ROOT / 'experiments/dpe_nested_corrected/code/data/metadata.csv').open()) if r['split'] == '2'}
    cells = {}
    for job in sorted((ROOT / 'results/dpe/label_corrected_backbones_20261115/results').glob('*/job.json')):
        j = json.loads(job.read_text())
        if j['backbone_labels'] != 'official':
            continue
        with np.load(job.parent / 'test_predictions.npz', allow_pickle=False) as z:
            files = ['/'.join(f.split('/')[-2:]) for f in z['filenames']]
            pred = z['probabilities'].argmax(1)
        y = np.array([meta[f][0] for f in files])
        a = np.array([meta[f][1] for f in files])
        bad = np.array([f.startswith(SPECIES) for f in files])
        if test_labels == 'corrected':
            y[bad] = 0
        if test_labels == 'removed':
            pred, y, a = pred[~bad], y[~bad], a[~bad]
        w = min(100 * (pred[(y == i) & (a == k)] == i).mean() for i in (0, 1) for k in (0, 1))
        cells.setdefault((j['backbone_seed'], j['val_labels']), []).append(w)
    d = [st.mean(cells[(s, 'corrected')]) - st.mean(cells[(s, 'official')]) for s in (5, 6, 7, 8)]
    assert all(len(cells[(s, v)]) == 3 for s in (5, 6, 7, 8) for v in ('official', 'corrected'))
    return ci(d)


ROWS = [
    ('Combiners: heads on 720, combiner on 479', None, None, None),
    ('uniform on 720 images', nested('pool_uniform', 'wga_official'), nested('pool_uniform', 'wga_removed'), 'main'),
    ('affine combiner, selected', nested('pool_affine_sel_nll', 'wga_official'), nested('pool_affine_sel_nll', 'wga_removed'), 'main'),
    ('linear residual, selected', nested('pool_linear_sel_nll', 'wga_official'), nested('pool_linear_sel_nll', 'wga_removed'), 'main'),
    ('linear residual, fixed setting', nested('pool_linear_fixed', 'wga_official'), nested('pool_linear_fixed', 'wga_removed'), 'fixed'),
    ('Cross-fitting: all 1,199 images', None, None, None),
    ('five-fold uniform', nested('crossfit_uniform', 'wga_official'), nested('crossfit_uniform', 'wga_removed'), 'main'),
    ('75-head bag', nested('bag75_uniform', 'wga_official'), nested('bag75_uniform', 'wga_removed'), 'main'),
    ('five-fold + recalibration, selected', nested('crossfit_affine_sel_nll', 'wga_official'), nested('crossfit_affine_sel_nll', 'wga_removed'), 'main'),
    ('full DPE + out-of-fold map, selected', nested('fullval_affine_oof_sel_nll', 'wga_official'), nested('fullval_affine_oof_sel_nll', 'wga_removed'), 'main'),
    ('five-fold + exact WGA threshold', nested('crossfit_postmean_thrwga', 'wga_official'), nested('crossfit_postmean_thrwga', 'wga_removed'), 'main'),
    ('Corrected validation labels', None, None, None),
    ('DPE refit, saved backbones', label_study('official'), label_study('removed'), 'label'),
    ("DPE refit, fresh backbones ($n{=}4$)", fresh('official'), fresh('removed'), 'label'),
]

XMIN, XMAX, SCALE, GAP = -2.0, 4.0, 0.5, 0.5
W = (XMAX - XMIN) * SCALE
n = len(ROWS)
bottom = -n + 0.4
STY = {'main': 'circle,fill=black', 'fixed': 'circle,fill=white',
       'label': 'circle,fill=orange!85!black'}
L = [r'\begin{tikzpicture}[font=\scriptsize,y=0.40cm]']
audit = []
for p, title in enumerate(('Official labels', 'Clean subset')):
    x0 = p * (W + GAP)
    x = lambda v, x0=x0: x0 + (v - XMIN) * SCALE
    L.append(rf'\fill[gray!15] ({x(-HEAD_SEED_SD):.3f},0.6) rectangle ({x(HEAD_SEED_SD):.3f},{bottom:.2f});')
    L.append(rf'\draw[gray] ({x(0):.3f},0.6) -- ({x(0):.3f},{bottom:.2f});')
    L.append(rf'\draw ({x(XMIN):.3f},{bottom:.2f}) -- ({x(XMAX):.3f},{bottom:.2f});')
    for t in range(int(XMIN), int(XMAX) + 1, 1):
        lab = '0' if t == 0 else f'{t:+d}'
        L.append(rf'\draw ({x(t):.3f},{bottom:.2f}) -- ++(0,-0.15) node[below]{{${lab}$}};')
    L.append(rf'\node[font=\scriptsize\bfseries] at ({x((XMIN + XMAX) / 2):.3f},1.2) {{{title}}};')
    for i, (name, off, cor, sty) in enumerate(ROWS):
        val = off if p == 0 else cor
        if sty is None:
            continue
        if val is None:
            L.append(rf'\node[gray] at ({x(1):.3f},{-i}) {{n/a}};')
            continue
        m, lo, hi = val
        L.append(rf'\draw[thick] ({x(max(lo, XMIN)):.3f},{-i}) -- ({x(min(hi, XMAX)):.3f},{-i});')
        L.append(rf'\node[draw,{STY[sty]},inner sep=1.3pt] at ({x(m):.3f},{-i}) {{}};')
        audit.append(dict(row=name, panel=title, mean=round(m, 4), lo=round(lo, 4), hi=round(hi, 4)))
L.append(rf'\node[below] at ({(2 * W + GAP) / 2:.3f},{bottom - 0.9:.2f}) {{$\Delta$ worst-group accuracy vs.\ full-validation DPE (pp)}};')
for i, (name, *_rest) in enumerate(ROWS):
    font = r',font=\scriptsize\bfseries' if _rest[2] is None else ''
    L.append(rf'\node[anchor=east{font}] at (-0.1,{-i}) {{{name}}};')
L.append(r'\end{tikzpicture}')
OUT.write_text('\n'.join(L) + '\n', encoding='utf-8')
AUDIT.write_text(json.dumps(dict(head_seed_sd=HEAD_SEED_SD, rows=audit), indent=1), encoding='utf-8')
for a in audit:
    print(f"{a['panel'][:9]:9s} {a['row']:42s} {a['mean']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}]")
