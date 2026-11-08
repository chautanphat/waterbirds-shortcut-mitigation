"""Read-only reuse of an H2 bundle for a separately recorded weighting suite."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy
import yaml

from .experiment import digest, object_hash, read_json, runtime, write_json
from .ensemble import PREPROCESSING
from .reweighting import GROUPS, combine, fit_select, metrics, probabilities
from .weighting_rules import fit_rules, validate_config


def notebook_code_hash(path):
    """Ignore execution counts, outputs and metadata changed by Jupyter autosave."""
    notebook = read_json(path)
    code = [''.join(cell['source']).replace('\r\n', '\n') for cell in notebook['cells']
            if cell['cell_type'] == 'code']
    return object_hash(code)


def load_cache(source, role, manifest, head_ids, ensemble_hash):
    """No model loading; cache identity and row alignment are checked explicitly."""
    with np.load(source / 'experiment/predictions' / f'{role}.npz', allow_pickle=False) as f:
        p = probabilities(f['probabilities'])
        y, g, ids = f['labels'], f['groups'].astype(str), f['ids'].astype(str)
        if (str(f['role']) != role or str(f['ensemble_hash']) != ensemble_hash or
                str(f['preprocessing']) != PREPROCESSING):
            raise ValueError('Cache role, ensemble or preprocessing identity differs')
    if (p.shape != (len(head_ids), len(y), 2) or y.shape != g.shape or y.shape != ids.shape or
            len(set(ids)) != len(ids) or any(not x for x in ids)):
        raise ValueError('Misaligned cache or duplicate/empty IDs')
    for head in p:
        metrics(head, y, g)
    if role in ('B', 'C'):
        rows = manifest['parts'][role]
        if (ids.tolist() != [r['id'] for r in rows] or y.tolist() != [r['y'] for r in rows] or
                g.tolist() != [r['group'] for r in rows]):
            raise ValueError('Cache differs from validation manifest')
    else:
        validation_ids = {r['id'] for r in manifest['validation']}
        if validation_ids.intersection(ids):
            raise ValueError('Test overlaps validation')
    return p, y, g, ids


def validation_context(source, config_path):
    source = Path(source).resolve()
    required = ['experiment/manifest.json', 'experiment/selection.json',
                'checkpoints/ensemble.pt', 'experiment/predictions/B.npz',
                'experiment/predictions/C.npz']
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError('Incomplete H2 source bundle. Set source to the extracted run directory '
                         'containing experiment/ and checkpoints/. Missing: ' + ', '.join(missing))
    c = yaml.safe_load(Path(config_path).read_text())
    manifest = read_json(source / 'experiment/manifest.json')
    # Retain the old selection as provenance, not as a source of new weights.
    original = read_json(source / 'experiment/selection.json')
    head_ids = original['head_ids']
    if len(head_ids) != 15 or len(set(head_ids)) != len(head_ids):
        raise ValueError('The declared suite requires 15 distinct head IDs')
    validate_config(c, len(head_ids))
    model_hash = digest(source / 'checkpoints/ensemble.pt')
    if (original['lock']['ensemble'] != model_hash or
            original['lock']['manifest'] != digest(source / 'experiment/manifest.json')):
        raise ValueError('Original ensemble/manifest identities differ')
    parts = manifest['parts']
    sets = [{r['id'] for r in parts[role]} for role in 'ABC']
    validation = [r['id'] for r in manifest['validation']]
    if (sum(map(len, sets)) != sum(len(parts[role]) for role in 'ABC') or
            sum(map(len, sets)) != len(set.union(*sets)) or
            set.union(*sets) != set(validation) or len(validation) != len(set(validation))):
        raise ValueError('Validation partition contains duplicates, overlap or missing IDs')
    data = {role: load_cache(source, role, manifest, head_ids, model_hash) for role in ('B', 'C')}
    # Old archives did not hash prediction caches. Check their original recorded
    # B/C scores and decision before using those caches as inputs to new rules.
    expected = fit_select(*data['B'][:3], *data['C'][:3])
    if len(original['candidates']) != len(expected['candidates']):
        raise ValueError('Original candidate grid differs')
    for saved, computed in zip(original['candidates'], expected['candidates']):
        if ((saved['alpha'], saved['gamma'], saved['fallback']) !=
                (computed['alpha'], computed['gamma'], computed['fallback']) or
                not np.allclose(saved['weights'], computed['weights'], rtol=0, atol=1e-12) or
                not same_metrics(saved['metrics'], computed['metrics'])):
            raise ValueError('Original selection disagrees with B/C cache')
    if (original['selected']['alpha'], original['selected']['gamma']) != (
            expected['selected']['alpha'], expected['selected']['gamma']):
        raise ValueError('Original selected configuration disagrees with B/C cache')
    for role in ('B', 'C'):
        saved, computed = original[f'per_head_{role}'], expected[f'per_head_{role}']
        if len(saved) != len(computed) or not all(same_metrics(a, b) for a, b in zip(saved, computed)):
            raise ValueError('Original per-head scores disagree with cache')
    files = ['experiment/manifest.json', 'experiment/selection.json', 'checkpoints/ensemble.pt',
             'experiment/predictions/B.npz', 'experiment/predictions/C.npz']
    identity = dict(inputs={name: digest(source / name) for name in files},
                    config=object_hash(c), implementation=runtime()['implementation'],
                    scipy=scipy.__version__, numpy=np.__version__)
    identity['implementation']['scripts/dpe_weighting_rules.py'] = digest('scripts/dpe_weighting_rules.py')
    identity['implementation']['notebooks/dpe_weighting_rules.ipynb'] = notebook_code_hash(
        'notebooks/dpe_weighting_rules.ipynb')
    return source, c, manifest, head_ids, data, identity


def same_metrics(a, b):
    return (a['counts'] == b['counts'] and
            all(abs(a[k] - b[k]) < 1e-12 for k in ('overall', 'wga')) and
            all(abs(a['groups'][g] - b['groups'][g]) < 1e-12 for g in GROUPS))


def output_path(source, output):
    output = Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Use a separate output directory outside the original run')
    return output


def compute_selection(source, config_path):
    source, c, manifest, head_ids, data, identity = validation_context(source, config_path)
    b, bc, bg, _ = data['B']
    pc, yc, gc, _ = data['C']
    selection = fit_rules(b, bc, bg, pc, yc, gc, c)
    selection.update(version=1, config=c, lock=identity, head_ids=head_ids,
                     source_track=manifest['provenance']['track'],
                     source_seed=manifest['provenance']['seed'])
    return selection


def write_comparison(output, selection):
    fields = ['id', 'family', 'parameters', 'selected_for_family', 'selected_overall',
              'c_wga', 'c_overall', 'effective_heads', 'c_prediction_changes',
              'c_probability_max_abs_delta']
    with (output / 'candidate_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in selection['candidates']:
            writer.writerow(dict(id=r['id'], family=r['family'], parameters=json.dumps(r['parameters']),
                                 selected_for_family=r['id'] in selection['per_family_selected'].values(),
                                 selected_overall=r['id'] == selection['selected'],
                                 c_wga=r['metrics']['wga'], c_overall=r['metrics']['overall'],
                                 **{k: r[k] for k in fields[7:]}))


def select(source, config_path, output):
    """Freeze every family winner before opening any test predictions."""
    source = Path(source).resolve()
    output = output_path(source, output)
    record = output / 'selection.json'
    if record.exists():
        result = validated_selection(source, config_path, output)
        # Recover a derived CSV if execution stopped after the atomic JSON write.
        write_comparison(output, result)
        return result
    if (output / 'metrics.json').exists():
        raise ValueError('Existing test result has no frozen selection; preserve it')
    result = compute_selection(source, config_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(record, result)
    write_comparison(output, result)
    return result


def validated_selection(source, config_path, output):
    output = output_path(Path(source).resolve(), output)
    result = read_json(output / 'selection.json')
    expected = compute_selection(source, config_path)
    if object_hash(result) != object_hash(expected):
        raise ValueError('Frozen suite selection or inputs changed; use a fresh output directory')
    return result


def evaluate(source, config_path, output):
    selection = validated_selection(source, config_path, output)
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = read_json(source / 'experiment/manifest.json')
    p, y, g, ids = load_cache(source, 'test', manifest, selection['head_ids'],
                              selection['lock']['inputs']['checkpoints/ensemble.pt'])
    candidates = {r['id']: r for r in selection['candidates']}
    uniform = combine(p, candidates['uniform']['weights'])
    u = metrics(uniform, y, g)
    original_metrics = source / 'results/metrics.json'
    if original_metrics.exists():
        old = read_json(original_metrics)
        if (old['lock']['ensemble'] != selection['lock']['inputs']['checkpoints/ensemble.pt'] or
                old['lock']['manifest'] != selection['lock']['inputs']['experiment/manifest.json'] or
                not same_metrics(old['uniform'], u)):
            raise ValueError('Original uniform test metrics disagree with test cache')
    # Only C-selected configurations are scored: no test-driven candidate sweep.
    selected = dict(uniform='uniform', **selection['per_family_selected'],
                    overall_selected=selection['selected'])
    results = {}
    for family, name in selected.items():
        candidate = candidates[name]
        combined = combine(p, candidate['weights'])
        m = metrics(combined, y, g)
        uc, wc = uniform.argmax(1) == y, combined.argmax(1) == y
        results[family] = dict(candidate=name, parameters=candidate['parameters'], weights=candidate['weights'],
                               metrics=m, effective_heads=candidate['effective_heads'],
                               delta_pp={k: 100 * (m[k] - u[k]) for k in ('wga', 'overall')},
                               group_delta_pp={k: 100 * (m['groups'][k] - u['groups'][k]) for k in GROUPS},
                               changed_predictions=int(np.sum(uc != wc)),
                               fixed_ids=ids[~uc & wc].tolist(), new_error_ids=ids[uc & ~wc].tolist())
    result = dict(version=1, status='exploratory; cached-prediction comparison',
                  source_track=selection['source_track'], source_seed=selection['source_seed'],
                  selection_hash=digest(output / 'selection.json'), lock=selection['lock'],
                  test_cache_hash=digest(source / 'experiment/predictions/test.npz'),
                  original_test_metrics_hash=digest(original_metrics) if original_metrics.exists() else None,
                  results=results)
    path = output / 'metrics.json'
    if path.exists():
        if object_hash(read_json(path)) != object_hash(result):
            raise ValueError('Existing test result or inputs changed; preserve the original output')
        return result
    # Portable metadata, without machine-specific source paths in the report.
    write_json(output / 'run.json', dict(config=selection['config'], runtime=runtime(),
                                       scipy=scipy.__version__, source_artifacts=selection['lock']['inputs'],
                                       execution=dict(interface='python_api', selection='rule_suite.select',
                                                      evaluation='rule_suite.evaluate'),
                                       track=selection['source_track'], seed=selection['source_seed']))
    report = ['# DPE weighting-rule comparison', '',
              'Exploratory cached-prediction adaptation; no backbone or heads were retrained.', '',
              f"Track: {selection['source_track']}; seed: {selection['source_seed']}.", '',
              'Weights fit on B; family and overall winners selected on C before test was read.',
              'Complete C ties retain uniform, then declared candidate order. This table does not select by test.', '',
              '| Rule | C-selected candidate | Test accuracy | Test WGA | WGA delta (pp) | Effective heads |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for family, row in results.items():
        report.append(f"| {family} | {row['candidate']} | {100*row['metrics']['overall']:.4f}% | "
                      f"{100*row['metrics']['wga']:.4f}% | {row['delta_pp']['wga']:.4f} | {row['effective_heads']:.4f} |")
    report += ['', 'Per-group accuracies, counts, paired differences and changed error IDs: `metrics.json`.',
               'Candidate C metrics: `candidate_comparison.csv`; fitted weights/configuration: `selection.json`.',
               'Input/code identities: `run.json`. Retain the source run separately to reproduce from probabilities.',
               'Cached predictions do not independently verify image-to-feature inference. Reusing an already',
               'inspected test split is exploratory; lock the procedure before later seed evaluations.', '']
    (output / 'report.md').write_text('\n'.join(report))
    # Publish the completion marker only after auxiliary reports are complete.
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'evaluate'])
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/dpe_head_reweighting/weighting_rules.yaml'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    action = select if args.action == 'select' else evaluate
    print(json.dumps(action(args.source_run, args.config, args.output_dir), indent=2))
