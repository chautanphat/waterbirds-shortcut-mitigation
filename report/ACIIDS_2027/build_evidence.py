"""Regenerate outline tables from tracked records; does not fit or select models."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
ROOT = parser.parse_args().repo.resolve()
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'results/dpe/independent_methods_20261113'
rows = list(csv.DictReader((DATA / 'per_seed.csv').open(newline='', encoding='utf-8-sig')))
summaries = list(csv.DictReader((DATA / 'summary.csv').open(newline='', encoding='utf-8-sig')))
index = {(r['recipe'], r['arm'], r['configuration']): r for r in summaries}

def collect(recipe, arm, config):
    selected = sorted([r for r in rows if (r['recipe'], r['arm'], r['configuration']) ==
                       (recipe, arm, config)], key=lambda r: int(r['seed']))
    assert len(selected) == (5 if recipe == 'normal_erm' else 2)
    assert all(r['status'] == 'completed' for r in selected)
    for r in selected:
        assert abs(float(r['wga']) - min(float(r[f'group_{g}']) for g in ('00','01','10','11'))) < 1e-12
    values = [100 * float(r['wga']) for r in selected]
    summary = index[(recipe, arm, config)]
    assert abs(st.mean(values) / 100 - float(summary['mean_wga'])) < 1e-12
    assert abs(st.stdev(values) / 100 - float(summary['source_sd_wga'])) < 1e-12
    return selected, values

methods = [
    ('Full-validation uniform', 'full_validation', 'uniform'),
    ('Fixed-split uniform (P0)', 'partition0', 'uniform'),
    ('Fixed-split affine (P0)', 'partition0', 'residual-affine-average-l0.1'),
    ('Fixed-split linear residual (P0)', 'partition0', 'residual-linear-average-l0.1'),
    ('Five-fold uniform', 'crossfit', 'crossfit-original-uniform'),
    ('Five-fold affine, average, .01', 'crossfit', 'crossfit-original-affine-average-l0.01'),
    ('Five-fold affine, average, .1', 'crossfit', 'crossfit-original-affine-average-l0.1'),
    ('Five-fold affine, maximum, .01', 'crossfit', 'crossfit-original-affine-maximum-l0.01'),
    ('Five-fold affine, maximum, .1', 'crossfit', 'crossfit-original-affine-maximum-l0.1'),
]
# Resolve exact arm IDs against the tracked table rather than assuming spelling.
arms = sorted({r['arm'] for r in rows})
assert 'partition0' in arms, arms
lines, audit = [], []
for name, arm, config in methods:
    formatted = []
    for recipe in ('normal_erm', 'ermstar'):
        selected, values = collect(recipe, arm, config)
        overall = st.mean(100 * float(r['overall']) for r in selected)
        formatted.extend([f'{st.mean(values):.2f} $\\pm$ {st.stdev(values):.2f}', f'{overall:.2f}'])
        audit.append(dict(method=name, recipe=recipe, config=config, arm=arm,
                          n=len(values), mean_wga=st.mean(values), sd_wga=st.stdev(values),
                          mean_overall=overall))
    lines.append(name + ' & ' + ' & '.join(formatted) + r'\\')
table_header = r'''\begin{tabularx}{\linewidth}{@{}Xrrrr@{}}
\toprule
& \multicolumn{2}{c}{Standard ERM ($n=5$)} & \multicolumn{2}{c}{ERM$^*$ ($n=2$)}\\
Procedure & WGA & OA & WGA & OA\\\midrule
'''
(OUT / 'evidence_table.tex').write_text(table_header+'\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabularx}'+'\n', encoding='utf-8')

contrasts = []
for recipe in ('normal_erm', 'ermstar'):
    _, ref = collect(recipe, 'full_validation', 'uniform')
    _, fold = collect(recipe, 'crossfit', 'crossfit-original-uniform')
    _, cal = collect(recipe, 'crossfit', 'crossfit-original-affine-average-l0.1')
    for name, a, b in [('fold_uniform_minus_full', fold, ref), ('fold_affine_minus_fold_uniform', cal, fold)]:
        delta = [x-y for x,y in zip(a,b)]
        result = dict(recipe=recipe, contrast=name, per_source_delta_pp=delta, mean_delta_pp=st.mean(delta))
        if len(delta) == 5:
            half = 2.7764451051977987 * st.stdev(delta) / math.sqrt(5)
            result['paired_t95_pp'] = [st.mean(delta)-half, st.mean(delta)+half]
        contrasts.append(result)

plot = [r'\begin{tikzpicture}[x=1.6cm,y=0.75cm,font=\scriptsize]',
        r'\draw[->] (-0.12,0) -- (2.3,0) node[right]{allocation};',
        r'\draw[->] (0,-0.2) -- (0,3.7) node[above]{WGA (\%)};']
for y in (89,90,91,92):
    plot.append(f'\\draw[gray!20] (-0.04,{y-88.5}) -- (2.1,{y-88.5});')
    plot.append(f'\\node[left] at (-0.05,{y-88.5}) {{{y}}};')
for x in range(3):
    plot.append(f'\\node[below] at ({x},0) {{P{x}}};')
for color, config, label in [('black','uniform','Uniform'),('blue!70!black','residual-affine-average-l0.1','Affine'),
                             ('orange!85!black','residual-linear-average-l0.1','Linear residual')]:
    means = [st.mean(collect('normal_erm',f'partition{x}',config)[1]) for x in range(3)]
    coords = ' '.join(f'({x},{y-88.5:.5f})' for x,y in enumerate(means))
    plot.append(f'\\draw[thick,{color}] plot[mark=*] coordinates {{{coords}}};')
    plot.append(f'\\node[right,{color}] at (2.1,{means[-1]-88.5:.5f}) {{{label}}};')
plot.append(r'\draw[dashed,gray] (0,1.095) -- (2.05,1.095);')
plot.append(r'\node[gray,anchor=west] at (0.05,0.75) {Full-validation uniform: 89.595\%};')
plot.append(r'\end{tikzpicture}')
(OUT / 'figures/allocation.tex').write_text('\n'.join(plot), encoding='utf-8')
(OUT / 'evidence_audit.json').write_text(json.dumps(dict(source=str(DATA), table=audit, contrasts=contrasts), indent=2), encoding='utf-8')
print(json.dumps(dict(rows_checked=len(audit), contrasts=contrasts), indent=2))
