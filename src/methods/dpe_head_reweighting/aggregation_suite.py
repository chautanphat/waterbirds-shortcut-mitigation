"""Frozen DPE heads -> global-rule/neural comparison -> held-out evaluation."""
import argparse
import csv
import json
from pathlib import Path

import scipy
import numpy as np
import torch

from . import neural_combiner as neural
from .experiment import digest, object_hash, read_json, runtime, write_json
from .rule_suite import output_path
from .aggregation_source import context as source_context, load_cache, COMBINER_TRAINING, PROTOCOL

DEFAULT_CONFIG = 'configs/dpe_head_reweighting/aggregation_comparison.yaml'


def context(source, config_path):
    source, c, manifest, heads, data, identity = source_context(source, config_path)
    identity.update(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, execution='cpu-float64-fixed-steps')
    return source, c, manifest, heads, data, identity


def compute_models(source, config_path):
    _, c, manifest, heads, data, identity = context(source, config_path)
    p, y, groups, ids = data[COMBINER_TRAINING]
    selection = neural.fit_all(p, y, groups, c, manifest['provenance']['seed'])
    selection['training_ids'] = ids.tolist()
    selection.update(version=1, config=c, lock=identity, head_ids=heads,
                     source_track=manifest['provenance']['track'], source_seed=manifest['provenance']['seed'])
    return selection


def write_comparison(output, selection):
    fields = ['id', 'family', 'parameters', 'training_role', 'training_wga', 'training_overall']
    with (output / 'candidate_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in selection['candidates']:
            m = row['training']['metrics']
            writer.writerow(dict(id=row['id'], family=row['family'], parameters=json.dumps(row['parameters']),
                                 training_role=COMBINER_TRAINING, training_wga=m['wga'], training_overall=m['overall']))


def validated_models(source, config_path, output):
    output = output_path(Path(source).resolve(), output)
    selection = read_json(output / 'models.json')
    # Deterministic replay checks checkpoint values as well as inputs and choices.
    expected = compute_models(source, config_path)
    if object_hash(selection) != object_hash(expected):
        raise ValueError('Frozen combiner models or inputs changed; use a fresh output directory')
    return selection


def fit(source, config_path, output):
    """This path never opens test.npz; JSON checkpoints are frozen here."""
    output = output_path(Path(source).resolve(), output)
    if (output / 'models.json').exists():
        selection = validated_models(source, config_path, output)
    else:
        if (output / 'metrics.json').exists():
            raise ValueError('Existing test result has no frozen model record; preserve it')
        selection = compute_models(source, config_path)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'models.json', selection)
    write_comparison(output, selection)
    return selection


def evaluate(source, config_path, output):
    selection = validated_models(source, config_path, output)
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = read_json(source / 'experiment/manifest.json')
    p, y, g, ids = load_cache(source, 'test', manifest, selection['head_ids'],
                              selection['lock']['inputs']['checkpoints/ensemble.pt'])
    candidates = {r['id']: r for r in selection['candidates']}
    uniform, _ = neural.candidate_prediction(candidates['uniform'], p)
    u = neural.metrics(uniform, y, g)
    chosen = {r['id']: r['id'] for r in selection['candidates']}
    results = {}
    for family, name in chosen.items():
        row = candidates[name]
        prediction, w = neural.candidate_prediction(row, p)
        d = neural.diagnostics(prediction, y, g, w)
        m = d['metrics']
        uc, wc = uniform.argmax(axis=1) == y, prediction.argmax(axis=1) == y
        results[family] = dict(candidate=name, parameters=row['parameters'], **d,
                               delta_pp={k: 100 * (m[k] - u[k]) for k in ('wga', 'overall')},
                               group_delta_pp={k: 100 * (m['groups'][k] - u['groups'][k]) for k in neural.GROUPS},
                               changed_predictions=int((uc != wc).sum()),
                               fixed_ids=ids[~uc & wc].tolist(), new_error_ids=ids[uc & ~wc].tolist())
    result = dict(version=1, status='exploratory; independent head-aggregation comparison',
                  source_track=selection['source_track'], source_seed=selection['source_seed'],
                  models_hash=digest(output / 'models.json'), lock=selection['lock'],
                  test_cache_hash=digest(source / 'experiment/predictions/test.npz'),
                  test_identity=object_hash(sorted(zip(ids.tolist(), y.tolist(), g.tolist()))),
                  results=results, protocol=PROTOCOL)
    marker = output / 'metrics.json'
    if marker.exists():
        if object_hash(read_json(marker)) != object_hash(result):
            raise ValueError('Existing neural test result or inputs changed; preserve the original output')
        return result
    write_json(output / 'run.json', dict(config=selection['config'], runtime=runtime(),
                                       source_artifacts=selection['lock']['inputs'],
                                       execution=dict(selection='aggregation_suite.fit', evaluation='aggregation_suite.evaluate'),
                                       track=selection['source_track'], seed=selection['source_seed']))
    lines = ['# DPE head-aggregation comparison', '',
             'All configurations fit on the combiner-training split, then evaluated independently on test.',
             'The frozen heads use a separate head-training split. No test-based tuning or candidate selection.', '',
             '| Configuration | Test accuracy (%) | Test WGA (%) | WGA delta (pp) |',
             '| --- | ---: | ---: | ---: |']
    for name, row in results.items():
        lines.append(f"| {name} | {100*row['metrics']['overall']:.4f} | "
                     f"{100*row['metrics']['wga']:.4f} | {row['delta_pp']['wga']:.4f} |")
    lines += ['', 'Frozen model states, global weights and training diagnostics: `models.json`.',
              'Test group metrics, NLL and paired error IDs: `metrics.json`.',
              'Training metrics are not held-out results. This comparison is exploratory because the test set',
              'has previously informed project research. Cached inference is not an independent raw-image audit.']
    (output / 'report.md').write_text('\n'.join(lines))
    write_json(marker, result)
    return result


def summarize(output, completed, planned, track):
    """Aggregate every predeclared procedure; never pick a family by test WGA."""
    if not completed or len({r['test_identity'] for r in completed.values()}) != 1:
        raise ValueError('Need completed seeds with identical test IDs/labels/groups')
    summary = dict(status='complete' if sorted(completed) == sorted(planned) else 'incomplete',
                   track=track, planned_seeds=list(planned), completed_seeds=sorted(completed), results={})
    rows = []
    protocols = {r.get('protocol') for r in completed.values()}
    candidate_sets = {tuple(sorted(r['results'])) for r in completed.values()}
    if protocols != {PROTOCOL} or len(candidate_sets) != 1:
        raise ValueError('Cannot aggregate different protocols or candidate grids')
    summary['protocol'] = PROTOCOL
    for family in next(iter(candidate_sets)):
        values = [completed[s]['results'][family] for s in sorted(completed)]
        summary['results'][family] = {}
        for key, a in [('overall', [r['metrics']['overall'] for r in values]),
                       ('wga', [r['metrics']['wga'] for r in values]),
                       ('overall_delta_pp', [r['delta_pp']['overall'] for r in values]),
                       ('wga_delta_pp', [r['delta_pp']['wga'] for r in values])]:
            summary['results'][family][key] = dict(mean=float(np.mean(a)), values=a,
                                                  sample_std=float(np.std(a, ddof=1)) if len(a) > 1 else None)
        for seed, r in zip(sorted(completed), values):
            rows.append(dict(seed=seed, family=family, candidate=r['candidate'],
                             overall=r['metrics']['overall'], wga=r['metrics']['wga'],
                             wga_delta_pp=r['delta_pp']['wga'], overall_delta_pp=r['delta_pp']['overall']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', summary)
    with (output / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def display(value, scale=1):
        sd = 'pending' if value['sample_std'] is None else f"{scale*value['sample_std']:.4f}"
        return f"{scale*value['mean']:.4f} ± {sd}"
    lines = ['# DPE head-aggregation study', '',
             f"Status: {summary['status']}; track: {track}; completed seeds: {summary['completed_seeds']}.", '',
             'All declared configurations fit on combiner-training data and are evaluated independently on test.',
             'Mean ± sample SD across training seeds, not confidence intervals. No family is chosen by test.', '',
             '| Procedure | Accuracy (%) | WGA (%) | Paired WGA change (pp) |',
             '| --- | ---: | ---: | ---: |']
    for family, r in summary['results'].items():
        lines.append(f"| {family} | {display(r['overall'],100)} | {display(r['wga'],100)} | {display(r['wga_delta_pp'])} |")
    lines += ['', 'Per-seed configurations: `per_seed.csv`. Model states and detailed group metrics: `comparisons/`.',
              'The study archive retains original head/prediction bundles in `source_archives/`.',
              'Large backbones, training states, images and raw features remain in WORK_DIR.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'evaluate'])
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = (fit if args.action == 'fit' else evaluate)(args.source_run, args.config, args.output_dir)
    print(json.dumps(dict(action=args.action, output=str(args.output_dir),
                          configurations=len(result.get('candidates', result.get('results', {}))), status=result.get('status')), indent=2))
