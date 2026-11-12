"""Explicit prepare / train / select / held-out evaluation lifecycle."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml
from . import ensemble
from .reweighting import GROUPS, combine, fit_select, metrics
from .train_heads import train

TRAIN_SCHEMA = dict(heads=int, epochs=int, batch_size=int, lr=float, momentum=float,
                    entropic_scale=float, cov_reg=float, wd_weight=float)
SCHEMA = dict(version=int, track=str, seed=int, device=str, dim=int, metadata=str,
              val_features=str, test_features=str, backbone=str, provenance=str,
              experiment_dir=str, ensemble_path=str, results_dir=str, training=dict)
PROVENANCE = dict(track=str, seed=int, checkpoint_sha256=str, val_features_sha256=str,
                  model=str, pretrained_weights=str, transforms=str, extraction_order=str,
                  feature_format=str, checkpoint_selection=str, source_commit=str)


def check_keys(value, schema):
    if not isinstance(value, dict) or set(value) != set(schema):
        raise ValueError(f'Expected exactly these fields: {sorted(schema)}')
    for key, kind in schema.items():
        v = value[key]
        if kind is float:
            valid = type(v) in (float, int) and np.isfinite(v)
        else:
            valid = type(v) is kind
        if not valid:
            raise ValueError(f'Invalid type for {key}')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def config(path, seed=None):
    c = yaml.safe_load(Path(path).read_text())
    check_keys(c, SCHEMA)
    check_keys(c['training'], TRAIN_SCHEMA)
    if seed is not None:
        c['seed'] = seed
    if c['version'] != 1 or c['track'] not in ('standard', 'enhanced') or c['seed'] not in (0, 1, 2) or c['dim'] != 2048:
        raise ValueError('Unsupported experiment contract')
    if c['training'] != dict(heads=15, epochs=20, batch_size=256, lr=.001, momentum=.9,
                             entropic_scale=30., cov_reg=500000., wd_weight=10.):
        raise ValueError('Main experiment requires the predeclared Stage-1 settings')
    for k, v in c.items():
        if isinstance(v, str):
            c[k] = v.format(seed=c['seed'], track=c['track'])
    return c


def metadata(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    result = []
    for r in rows:
        y, a, split = int(r['y']), int(r.get('a', r.get('place', '-1'))), int(r['split'])
        if y not in (0, 1) or a not in (0, 1) or split not in (0, 1, 2):
            raise ValueError('Invalid Waterbirds metadata')
        filename = r.get('filename', r.get('img_filename'))
        if not filename:
            raise ValueError('Metadata requires filename or img_filename')
        result.append(dict(id=filename, y=y, group=f'{y}{a}', split=split))
    ids = [r['id'] for r in result]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError('Duplicate/empty metadata IDs')
    return result


def partition(rows):
    """Largest-remainder 60/20/20 stratification; independent of training seed."""
    rng, parts = np.random.RandomState(0), {k: [] for k in 'ABC'}
    for g in GROUPS:
        members = sorted([r for r in rows if r['group'] == g], key=lambda r: r['id'])
        order = rng.permutation(len(members))
        quotas = np.array([.6, .2, .2]) * len(members)
        counts = np.floor(quotas).astype(int)
        for i in np.argsort(-(quotas - counts), kind='stable')[:len(members) - counts.sum()]:
            counts[i] += 1
        if (counts == 0).any():
            raise ValueError('Each validation subset must contain all groups')
        start = 0
        for k, count in zip('ABC', counts):
            parts[k].extend(members[i] for i in order[start:start + count])
            start += count
    # Preserve official metadata order within each subset.
    for k in parts:
        wanted = {r['id'] for r in parts[k]}
        parts[k] = [r for r in rows if r['id'] in wanted]
    return parts


def feature_file(path, rows, dim):
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != {'ids', 'features'}:
            raise ValueError('Feature NPZ must contain only ids and raw features')
        ids, x = f['ids'].astype(str), f['features']
    expected = [r['id'] for r in rows]
    if ids.ndim != 1 or ids.tolist() != expected or len(set(ids)) != len(ids):
        raise ValueError('Feature IDs must exactly match metadata order (no duplicates/reordering)')
    if x.shape != (len(rows), dim):
        raise ValueError('Feature shape mismatch')
    ensemble.standardize(x)  # Validate before any optimizer runs.
    return x


def runtime():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src/methods/dpe_head_reweighting',
                                        'scripts/dpe_head_reweighting.py'], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, diff = 'unknown', ''
    # Include uncommitted and untracked implementation in provenance.
    files = sorted(Path('src/methods/dpe_head_reweighting').glob('*.py')) + [Path('scripts/dpe_head_reweighting.py')]
    return dict(commit=commit, diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                implementation={str(p): digest(p) for p in files}, python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                cuda=torch.version.cuda)


def prepare(c):
    out = Path(c['experiment_dir'])
    if (out / 'manifest.json').exists():
        raise ValueError('Manifest already exists; use a fresh experiment directory')
    rows = metadata(c['metadata'])
    if [sum(r['split'] == i for r in rows) for i in range(3)] != [4795, 1199, 5794]:
        raise ValueError('Expected official Waterbirds split counts 4795/1199/5794')
    val = [r for r in rows if r['split'] == 1]
    feature_file(c['val_features'], val, c['dim'])
    p = read_json(c['provenance'])
    check_keys(p, PROVENANCE)
    if p['track'] != c['track'] or p['seed'] != c['seed'] or p['checkpoint_sha256'] != digest(c['backbone']) or p['val_features_sha256'] != digest(c['val_features']):
        raise ValueError('Stage-0 provenance mismatch')
    if p['feature_format'] != 'raw' or p['checkpoint_selection'] != 'final_epoch_no_validation_selection' or p['model'] != 'resnet50' or p['extraction_order'] != 'metadata_split_order' or any(not v for v in p.values() if isinstance(v, str)):
        raise ValueError('Audit Stage-0 selection, feature format and extraction provenance')
    m = dict(version=1, config_hash=object_hash(c), metadata_hash=digest(c['metadata']),
             val_features_hash=digest(c['val_features']), backbone_hash=digest(c['backbone']),
             provenance=p, validation=val, parts=partition(val), runtime=runtime(),
             split_seed=0, ratios=[.6, .2, .2])
    write_json(out / 'resolved_config.json', c)
    write_json(out / 'manifest.json', m)
    return dict(status='prepared', counts={k: len(v) for k, v in m['parts'].items()})


def preflight(c, require_test=True):
    """Check readiness before a long run; no models are trained or scored."""
    required = ['metadata', 'backbone', 'val_features', 'provenance']
    if require_test:
        required.append('test_features')
    missing = [f'{key}: {c[key]}' for key in required if not Path(c[key]).is_file()]
    if missing:
        raise ValueError('Missing required input files:\n' + '\n'.join(missing) +
                         '\nSupply matching Stage-0 artifacts before running this experiment.')
    try:
        torch.empty(1, device=c['device'])
    except (RuntimeError, ValueError, AssertionError) as error:
        raise ValueError(f'Configured device {c["device"]!r} is unavailable; select a supported device.') from error
    for directory in (Path(c['experiment_dir']), Path(c['ensemble_path']).parent, Path(c['results_dir'])):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
    rows = metadata(c['metadata'])
    if [sum(r['split'] == i for r in rows) for i in range(3)] != [4795, 1199, 5794]:
        raise ValueError('Expected official Waterbirds split counts 4795/1199/5794')
    val = [r for r in rows if r['split'] == 1]
    feature_file(c['val_features'], val, c['dim'])
    parts = partition(val)
    counts = [sum(r['group'] == g for r in parts['A']) for g in GROUPS]
    if min(counts) * 4 < c['training']['batch_size']:
        raise ValueError('Balanced A produces zero optimizer batches with drop_last=True')
    p = read_json(c['provenance'])
    check_keys(p, PROVENANCE)
    if (p['track'] != c['track'] or p['seed'] != c['seed'] or
        p['checkpoint_sha256'] != digest(c['backbone']) or
        p['val_features_sha256'] != digest(c['val_features'])):
        raise ValueError('Stage-0 provenance mismatch')
    if (p['model'] != 'resnet50' or p['feature_format'] != 'raw' or
        p['checkpoint_selection'] != 'final_epoch_no_validation_selection' or
        p['extraction_order'] != 'metadata_split_order' or
        any(not v.strip() for v in p.values() if isinstance(v, str))):
        raise ValueError('Audit Stage-0 selection, feature format and extraction provenance')
    if require_test:
        # Validate IDs/features only; test labels never enter training/selection.
        if {r['group'] for r in rows if r['split'] == 2} != set(GROUPS):
            raise ValueError('All four groups are required in the test metadata')
        test_ids = [dict(id=r['id']) for r in rows if r['split'] == 2]
        feature_file(c['test_features'], test_ids, c['dim'])
    return dict(status='ready', device=c['device'], test_features_checked=require_test,
                counts={k: len(v) for k, v in parts.items()})


def context(c):
    m = read_json(Path(c['experiment_dir']) / 'manifest.json')
    if m['config_hash'] != object_hash(c) or m['val_features_hash'] != digest(c['val_features']) or m['backbone_hash'] != digest(c['backbone']) or m['provenance'] != read_json(c['provenance']):
        raise ValueError('Prepared inputs changed')
    if m['runtime']['implementation'] != runtime()['implementation']:
        raise ValueError('Implementation changed since prepare; use a fresh experiment')
    if m['parts'] != partition(m['validation']):
        raise ValueError('Invalid split manifest')
    x = feature_file(c['val_features'], m['validation'], c['dim'])
    return m, x


def subset(m, x, role):
    lookup = {r['id']: i for i, r in enumerate(m['validation'])}
    rows = m['parts'][role]
    return x[[lookup[r['id']] for r in rows]], np.array([r['y'] for r in rows]), np.array([r['group'] for r in rows])


def lock(c):
    return dict(config=object_hash(c), manifest=digest(Path(c['experiment_dir']) / 'manifest.json'),
                ensemble=digest(c['ensemble_path']), val_features=digest(c['val_features']),
                implementation=runtime()['implementation'])


def cache(c, role, p, rows, a):
    path = Path(c['experiment_dir']) / 'predictions' / f'{role}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, probabilities=p, ids=np.array([r['id'] for r in rows]),
                                labels=np.array([r['y'] for r in rows]), groups=np.array([r['group'] for r in rows]),
                                role=role, ensemble_hash=digest(c['ensemble_path']), preprocessing=a['preprocessing'])
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def train_action(c):
    m, x = context(c)
    if Path(c['ensemble_path']).exists():
        raise ValueError('Ensemble exists; use a fresh experiment directory/path')
    xa, ya, ga = subset(m, x, 'A')
    a = train(xa, ya, ga, c['training'], c['seed'], dict(manifest=digest(Path(c['experiment_dir']) / 'manifest.json')), c['device'], progress=True)
    ensemble.save(c['ensemble_path'], a)
    loaded = ensemble.load(c['ensemble_path'], c['dim'], c['training']['heads'])
    np.testing.assert_allclose(ensemble.predict(a, xa[:3]), ensemble.predict(loaded, xa[:3]), atol=1e-6, rtol=1e-5)
    return dict(status='heads_saved', selected_epochs=a['selected_epochs'])


def get_ensemble(c):
    a = ensemble.load(c['ensemble_path'], c['dim'], c['training']['heads'])
    if a['provenance'] != dict(manifest=digest(Path(c['experiment_dir']) / 'manifest.json')):
        raise ValueError('Ensemble belongs to a different training partition')
    return a


def select_action(c):
    m, x = context(c)
    record = Path(c['experiment_dir']) / 'selection.json'
    if record.exists():
        raise ValueError('Selection already frozen')
    a = get_ensemble(c)
    data = []
    for role in ('B', 'C'):
        raw, y, g = subset(m, x, role)
        p = ensemble.predict(a, raw, device=c['device'])
        cache(c, role, p, m['parts'][role], a)
        data.extend([p, y, g])
    result = fit_select(*data)
    result.update(lock=lock(c), head_ids=a['head_ids'], runtime=runtime())
    write_json(record, result)
    report = record.parent / 'candidate_comparison.csv'
    fields = ['alpha', 'gamma', 'c_wga', 'c_overall', 'c_prediction_changes',
              'c_probability_max_abs_delta', 'effective_heads']
    with report.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result['candidates']:
            writer.writerow(dict(alpha=row['alpha'], gamma=row['gamma'],
                                 c_wga=row['metrics']['wga'], c_overall=row['metrics']['overall'],
                                 **{k: row[k] for k in fields[4:]}))
    return result['selected']


def diagnostics(c, result, p_test, y, g, *, plots=True):
    w = result['selected']['weights']
    rows = []
    for role, scores in zip(('B', 'C', 'test'),
                                (result['per_head_B'], result['per_head_C'], [metrics(p, y, g) for p in p_test])):
        minority = [min(m['groups']['01'], m['groups']['10']) for m in scores]
        for i, m in enumerate(scores):
            rows.append(dict(head=result['head_ids'][i], role=role, weight=w[i], minority_accuracy=minority[i], **{f'acc_{k}': v for k, v in m['groups'].items()}))
    out = Path(c['results_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_head.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
        for ax, role in zip(axes, ('B','C','test')):
            ax.scatter(w, [r['minority_accuracy'] for r in rows if r['role'] == role])
            ax.set(xlabel='Head weight', title=role)
        axes[0].set_ylabel('Minority-group accuracy')
        fig.tight_layout()
        fig.savefig(out / 'head_diagnostics.png', dpi=160)
        plt.close(fig)


def validated_selection(c):
    """Recompute the frozen decision from B/C caches before accepting it."""
    record = read_json(Path(c['experiment_dir']) / 'selection.json')
    if record['lock'] != lock(c):
        raise ValueError('Frozen selection inputs changed')
    manifest = read_json(Path(c['experiment_dir']) / 'manifest.json')
    data = []
    for role in ('B', 'C'):
        rows = manifest['parts'][role]
        with np.load(Path(c['experiment_dir']) / 'predictions' / f'{role}.npz', allow_pickle=False) as f:
            if (str(f['role']) != role or str(f['ensemble_hash']) != record['lock']['ensemble'] or
                    str(f['preprocessing']) != ensemble.PREPROCESSING or
                    f['ids'].tolist() != [r['id'] for r in rows] or
                    f['labels'].tolist() != [r['y'] for r in rows] or
                    f['groups'].tolist() != [r['group'] for r in rows]):
                raise ValueError('Selection prediction cache changed')
            data.extend([f['probabilities'], f['labels'], f['groups']])
    expected = fit_select(*data)
    if any(object_hash(record[k]) != object_hash(expected[k]) for k in expected):
        raise ValueError('Saved selection disagrees with B/C predictions')
    return record


def evaluate_action(c, *, plots=True):
    record = validated_selection(c)
    out = Path(c['results_dir']) / 'metrics.json'
    if out.exists():
        raise ValueError('Test result already exists; preserve the original evaluation')
    m, _ = context(c)
    # Only this action opens test labels/features, after checking the selection lock.
    if digest(c['metadata']) != m['metadata_hash']:
        raise ValueError('Dataset metadata changed')
    rows = [r for r in metadata(c['metadata']) if r['split'] == 2]
    a = get_ensemble(c)
    if record['head_ids'] != a['head_ids'] or record['selected'] not in record['candidates']:
        raise ValueError('Inconsistent selection record or head order')
    p = ensemble.predict(a, feature_file(c['test_features'], rows, c['dim']), device=c['device'])
    cache(c, 'test', p, rows, a)
    y, g = np.array([r['y'] for r in rows]), np.array([r['group'] for r in rows])
    w = record['selected']['weights']
    uniform, weighted = combine(p, np.ones(len(p)) / len(p)), combine(p, w)
    u, v = metrics(uniform, y, g), metrics(weighted, y, g)
    uc, vc = uniform.argmax(-1) == y, weighted.argmax(-1) == y
    result = dict(track=c['track'], seed=c['seed'], uniform=u, selected=v,
                  selection_hash=digest(Path(c['experiment_dir']) / 'selection.json'),
                  delta_pp={k: 100 * (v[k] - u[k]) for k in ('overall', 'wga')},
                  group_delta_pp={k: 100 * (v['groups'][k] - u['groups'][k]) for k in GROUPS},
                  fixed_ids=[r['id'] for r, b in zip(rows, ~uc & vc) if b],
                  new_error_ids=[r['id'] for r, b in zip(rows, uc & ~vc) if b],
                  effective_heads=float(1 / np.square(w).sum()), alpha=record['selected']['alpha'],
                  gamma=record['selected']['gamma'], weights=w, lock=record['lock'],
                  test_features_hash=digest(c['test_features']), metadata_hash=m['metadata_hash'], runtime=runtime())
    diagnostics(c, record, p, y, g, plots=plots)
    write_json(out, result)
    return result


def summarize(config_path):
    records = []
    for seed in (0, 1, 2):
        c = config(config_path, seed)
        r = read_json(Path(c['results_dir']) / 'metrics.json')
        if r['track'] != c['track'] or r['seed'] != seed or r['lock'] != lock(c):
            raise ValueError('Stale or mismatched result in seed matrix')
        records.append(r)
    if len({r['metadata_hash'] for r in records}) != 1:
        raise ValueError('Seeds used different dataset metadata')
    summary = dict(track=records[0]['track'], seeds=[0, 1, 2], n=3, status='preliminary', metrics={})
    for name in ('overall', 'wga'):
        for method in ('uniform', 'selected', 'delta_pp'):
            values = [r[method][name] for r in records]
            summary['metrics'][f'{method}_{name}'] = dict(mean=float(np.mean(values)),
                                                         sample_std=float(np.std(values, ddof=1)), values=values)
    # Respect configured report locations, including notebook/exported layouts.
    path = Path(config(config_path, 0)['results_dir']).parent / 'summary.json'
    write_json(path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'prepare', 'train-heads', 'fit-select', 'evaluate', 'summarize'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    if args.action == 'summarize':
        print(json.dumps(summarize(args.config), indent=2))
        return
    c = config(args.config, args.seed)
    actions = {'check': preflight, 'prepare': prepare, 'train-heads': train_action, 'fit-select': select_action, 'evaluate': evaluate_action}
    print(json.dumps(actions[args.action](c), indent=2))
