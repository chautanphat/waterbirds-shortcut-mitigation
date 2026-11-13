"""Replay saved neural aggregation with NumPy; no training or checkpoint loading.

Usage: python scripts/analysis/audit_dpe_ermstar_neural.py --artifacts DIR --output DIR
DIR must contain seed0-results.zip and seed1-results.zip from the Kaggle run.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

GROUPS = ('00', '01', '10', '11')


def prediction(candidate, heads):
    if candidate['family'] == 'uniform':
        return heads.astype(float).mean(axis=0)
    record = candidate['model']
    state = {k: np.asarray(v, dtype=float) for k, v in record['state'].items()}
    x = 2 * heads[:, :, 1].T.astype(float) - 1
    if record['kind'] == 'logistic':
        z = x @ state['network.weight'].T + state['network.bias']
    else:
        hidden = np.tanh(x @ state['network.0.weight'].T + state['network.0.bias'])
        z = hidden @ state['network.2.weight'].T + state['network.2.bias']
    if record['kind'] == 'gate':
        weights = np.exp(z - z.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        q = (weights * heads[:, :, 1].T).sum(axis=1)
    else:
        q = 1 / (1 + np.exp(-z[:, 0]))
    return np.stack([1 - q, q], axis=1)


def scores(p, y, g):
    assert p.shape == (len(y), 2) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(axis=1), 1)
    assert set(g) == set(GROUPS) and np.array_equal(y.astype(str), np.array([v[0] for v in g]))
    correct = p.argmax(axis=1) == y
    group = {v: float(correct[g == v].mean()) for v in GROUPS}
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    brier = (p[:, 1] - y) ** 2
    return dict(wga=min(group.values()), overall=float(correct.mean()), groups=group,
                counts={v: int(sum(g == v)) for v in GROUPS}, nll=float(nll.mean()),
                group_balanced_nll=float(np.mean([nll[g == v].mean() for v in GROUPS])),
                group_balanced_brier=float(np.mean([brier[g == v].mean() for v in GROUPS])))


def check_metrics(actual, saved):
    for k in ('wga', 'overall'):
        assert abs(actual[k] - saved[k]) < 1e-12, (k, actual, saved)
    assert actual['groups'] == saved['groups'] and actual['counts'] == saved['counts']


def audit(artifacts, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, observations, identity, parts_before = [], [], None, None
    for seed in (0, 1):
        with zipfile.ZipFile(artifacts / f'seed{seed}-results.zip') as z:
            def read(name):
                return json.loads(z.read(f'seed{seed}/' + name))
            def cache(name):
                with np.load(io.BytesIO(z.read(f'seed{seed}/' + name)), allow_pickle=False) as f:
                    return dict(f)
            result, plan = read('results.json'), read('plan.json')
            assert result['seed'] == seed and plan['backbone']['training']['epochs'] == 400
            assert len(result['results']) == 12
            parts = read('neural_split/partition.json')
            if parts_before is not None:
                assert parts == parts_before
            parts_before = parts
            head_ids = {r['id'] for r in parts['head_training']}
            fit_ids = {r['id'] for r in parts['combiner_training']}
            assert len(head_ids) == 720 and len(fit_ids) == 479 and not head_ids & fit_ids
            train = cache('neural_split/combiner_training.npz')
            assert train['ids'].tolist() == [r['id'] for r in parts['combiner_training']]
            assert train['labels'].tolist() == [r['y'] for r in parts['combiner_training']]
            assert train['groups'].tolist() == [r['group'] for r in parts['combiner_training']]
            models = read('neural_split/models.json')['candidates']
            for candidate in models:
                m = scores(prediction(candidate, train['probabilities']), train['labels'], train['groups'])
                check_metrics(m, candidate['training']['metrics'])
            for arm in ('paper_reference', 'neural_split'):
                data = cache(arm + '/test_heads.npz')
                p, y, g, ids = (data[k] for k in ('probabilities', 'labels', 'groups', 'ids'))
                assert p.shape == (15, 5794, 2) and len(set(ids)) == 5794
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(-1), 1)
                assert not (head_ids | fit_ids) & set(ids)
                this_identity = (ids.tolist(), y.tolist(), g.tolist())
                if identity is not None:
                    assert identity == this_identity
                identity = this_identity
                baseline = prediction(dict(family='uniform'), p)
                base_correct = baseline.argmax(-1) == y
                base_wga = scores(baseline, y, g)['wga']
                candidates = [dict(id='uniform', family='uniform')] if arm == 'paper_reference' else models
                for candidate in candidates:
                    q = prediction(candidate, p)
                    m = scores(q, y, g)
                    stored = result['results'][arm + '/' + candidate['id']]
                    check_metrics(m, stored['metrics'])
                    delta = 100 * (m['wga'] - base_wga)
                    assert abs(delta - stored['delta_matched_uniform_pp']) < 1e-12
                    correct = q.argmax(-1) == y
                    rows.append(dict(seed=seed, arm=arm, candidate=candidate['id'],
                                     wga_percent=100*m['wga'], delta_matched_uniform_pp=delta,
                                     overall_percent=100*m['overall'], worst_groups=','.join(v for v in GROUPS if m['groups'][v] == m['wga']),
                                     **{f'accuracy_{v}_percent':100*m['groups'][v] for v in GROUPS},
                                     nll=m['nll'], group_balanced_nll=m['group_balanced_nll'],
                                     group_balanced_brier=m['group_balanced_brier'],
                                     changed_decisions=int(sum(q.argmax(-1) != baseline.argmax(-1))),
                                     fixed_errors=int(sum(~base_correct & correct)), new_errors=int(sum(base_correct & ~correct))))
            observations.append(dict(seed=seed, head_pool=len(head_ids), combiner_pool=len(fit_ids),
                                     config=plan['config'], checked_test_procedures=12))
    with (output/'per_seed.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    summary=[]
    for arm, candidate in dict.fromkeys((r['arm'], r['candidate']) for r in rows):
        selected=[r for r in rows if (r['arm'], r['candidate']) == (arm, candidate)]
        summary.append(dict(arm=arm,candidate=candidate,
                            **{f'seed{r["seed"]}_wga_percent':r['wga_percent'] for r in selected},
                            mean_wga_percent=float(np.mean([r['wga_percent'] for r in selected])),
                            mean_delta_matched_uniform_pp=float(np.mean([r['delta_matched_uniform_pp'] for r in selected])),
                            wins=sum(r['delta_matched_uniform_pp']>0 for r in selected),
                            losses=sum(r['delta_matched_uniform_pp']<0 for r in selected),
                            mean_group_balanced_nll=float(np.mean([r['group_balanced_nll'] for r in selected]))))
    with (output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator="\n");writer.writeheader();writer.writerows(summary)
    (output/'audit.json').write_text(json.dumps(dict(status='passed',seeds=observations,
        checked_test_procedures=len(rows),checked_training_procedures=22,
        groups=dict(zip(GROUPS,[2255,2255,642,642])),
        checks=['NumPy forward replay of fitted models', 'all saved WGA/overall/group metrics and deltas',
                'identical test IDs/labels/groups across seeds and arms', 'disjoint head/combiner/test IDs',
                'saved combiner training metrics reproduced', 'same primary partition across seeds'],
        limits=['No raw-image inference or independent backbone/head training',
                'No independent neural refit', 'Historical fit-before-test ordering not proved by replay']),indent=2)+'\n')
    for row in summary:
        print(row)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.artifacts,args.output)
