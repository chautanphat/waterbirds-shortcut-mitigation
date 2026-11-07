"""Independently audit an extracted notebook results bundle (including old runs).

Only stdlib/NumPy are used. No experiment selection, combination or metric
functions are imported. Inputs are read without modifying the recorded run.
This checks saved predictions, not backbone extraction or image labels.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(probabilities, labels, groups):
    correct = probabilities.argmax(axis=1) == labels
    counts = {g: int(np.count_nonzero(groups == g)) for g in ('00', '01', '10', '11')}
    require(all(counts.values()), 'Missing evaluation group')
    scores = {g: float(np.count_nonzero(correct & (groups == g)) / counts[g]) for g in counts}
    return dict(overall=float(correct.sum() / len(labels)), wga=min(scores.values()),
                groups=scores, counts=counts)


def equal_metrics(actual, expected):
    require(actual['counts'] == expected['counts'], 'Metric group counts differ')
    for key in ('overall', 'wga'):
        require(abs(actual[key] - expected[key]) < 1e-12, f'Metric {key} differs')
    for g in expected['groups']:
        require(abs(actual['groups'][g] - expected['groups'][g]) < 1e-12, 'Group accuracy differs')


def audit(run_dir):
    root = Path(run_dir)
    load = lambda p: json.loads(p.read_text())
    selection_path = root / 'experiment/selection.json'
    selection = load(selection_path)
    manifest = load(root / 'experiment/manifest.json')
    result = load(root / 'results/metrics.json')
    require(selection['lock'] == result['lock'], 'Selection/result locks differ')
    require(selection['lock']['manifest'] == sha(root / 'experiment/manifest.json'), 'Manifest hash differs')
    require(selection['lock']['ensemble'] == sha(root / 'checkpoints/ensemble.pt'), 'Ensemble hash differs')
    if 'selection_hash' in result:
        require(result['selection_hash'] == sha(selection_path), 'Selection changed after evaluation')
    data = {}
    for role in ('B', 'C', 'test'):
        with np.load(root / f'experiment/predictions/{role}.npz', allow_pickle=False) as f:
            p, y, g, ids = f['probabilities'].astype(np.float64), f['labels'], f['groups'], f['ids']
            require(str(f['role']) == role, 'Prediction cache role differs')
            require(str(f['ensemble_hash']) == selection['lock']['ensemble'], 'Cached ensemble hash differs')
        require(p.shape == (len(selection['head_ids']), len(y), 2), 'Cached probability shape differs')
        require(g.shape == y.shape == ids.shape and len(set(ids)) == len(ids), 'Cached rows misaligned')
        require(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(2), 1, atol=1e-6),
                'Invalid probabilities')
        require(np.isin(y, [0, 1]).all() and all(str(label) == group[0] for label, group in zip(y, g)),
                'Cached labels/groups disagree')
        if role != 'test':
            rows = manifest['parts'][role]
            require(ids.tolist() == [r['id'] for r in rows] and y.tolist() == [r['y'] for r in rows]
                    and g.tolist() == [r['group'] for r in rows], 'Cache differs from validation manifest')
        data[role] = p, y, g, ids
    id_sets = [{r['id'] for r in manifest['parts'][role]} for role in 'ABC']
    require(sum(map(len, id_sets)) == len(set.union(*id_sets)), 'Validation subsets overlap')
    require(set.union(*id_sets) == {r['id'] for r in manifest['validation']}, 'Validation partition incomplete')
    require(not set.union(*id_sets).intersection(data['test'][3]), 'Validation/test IDs overlap')
    pb, yb, gb, _ = data['B']
    pc, yc, gc, _ = data['C']
    scores = np.array([measure(head, yb, gb)['wga'] for head in pb])
    for head, expected in zip(pb, selection['per_head_B']):
        equal_metrics(measure(head, yb, gb), expected)
    for head, expected in zip(pc, selection['per_head_C']):
        equal_metrics(measure(head, yc, gc), expected)
    grid = [(0, 1)] + [(a, g) for a in (.25, .5, .75, 1) for g in (1, 2, 4)]
    require(len(selection['candidates']) == len(grid), 'Candidate grid differs')
    uniform_c = pc.mean(axis=0)
    comparison, best, selected_weights = [], None, None
    for (alpha, gamma), saved in zip(grid, selection['candidates']):
        require((saved['alpha'], saved['gamma']) == (alpha, gamma), 'Candidate grid/order differs')
        powers = scores ** gamma
        q = powers / powers.sum() if powers.sum() else np.full(len(scores), 1 / len(scores))
        w = (1-alpha) / len(scores) + alpha * q
        require(np.allclose(saved['weights'], w, rtol=0, atol=1e-12), 'Candidate weights differ')
        require(saved['fallback'] == bool(not powers.sum()), 'Fallback flag differs')
        # Explicit summation independently checks the production einsum axes.
        combined = sum(w[h] * pc[h] for h in range(len(w)))
        m = measure(combined, yc, gc)
        equal_metrics(saved['metrics'], m)
        row = dict(alpha=alpha, gamma=gamma, c_wga=m['wga'], c_overall=m['overall'],
                   changed_c_predictions=int(np.count_nonzero(combined.argmax(1) != uniform_c.argmax(1))),
                   max_probability_change=float(np.max(np.abs(combined-uniform_c))))
        comparison.append(row)
        if best is None or m['wga'] > best['c_wga'] + 1e-12 or (
                abs(m['wga']-best['c_wga']) <= 1e-12 and m['overall'] > best['c_overall'] + 1e-12):
            best, selected_weights = row, w
    saved = selection['selected']
    require((saved['alpha'], saved['gamma']) == (best['alpha'], best['gamma']), 'Wrong candidate selected')
    require(np.allclose(saved['weights'], selected_weights, rtol=0, atol=1e-12), 'Selected weights differ')
    require((result['alpha'], result['gamma']) == (best['alpha'], best['gamma']) and
            np.allclose(result['weights'], selected_weights, rtol=0, atol=1e-12), 'Test result weights differ')
    pt, yt, gt, ids = data['test']
    uniform = pt.mean(axis=0)
    weighted = sum(selected_weights[h] * pt[h] for h in range(len(pt)))
    u, v = measure(uniform, yt, gt), measure(weighted, yt, gt)
    equal_metrics(result['uniform'], u)
    equal_metrics(result['selected'], v)
    for key in ('overall', 'wga'):
        require(abs(result['delta_pp'][key] - 100*(v[key]-u[key])) < 1e-10, 'Paired metric difference differs')
    uc, vc = uniform.argmax(1) == yt, weighted.argmax(1) == yt
    require(result['fixed_ids'] == ids[~uc & vc].tolist(), 'Fixed example IDs differ')
    require(result['new_error_ids'] == ids[uc & ~vc].tolist(), 'New error IDs differ')
    require(abs(result['effective_heads']-1/np.square(selected_weights).sum()) < 1e-10,
            'Effective head count differs')
    return dict(status='passed', scope='saved predictions and records; not raw-image re-inference',
                selection_hash_recorded='selection_hash' in result,
                b_scores=scores.tolist(), selected=best, candidates=comparison,
                uniform=u, selected_test=v,
                test_prediction_changes=int(np.count_nonzero(uniform.argmax(1) != weighted.argmax(1))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Extracted notebook run directory')
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2))
