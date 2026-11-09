"""Frozen H2-cache -> B-only neural fits -> C selection -> held-out evaluation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import neural_combiner as neural
from .experiment import digest, object_hash, read_json, runtime, write_json
from .rule_suite import load_cache, output_path, same_metrics, validation_context

DEFAULT_CONFIG = 'configs/dpe_head_reweighting/neural_combiners.yaml'


def context(source, config_path):
    source, c, manifest, heads, data, identity = validation_context(
        source, config_path, config_validator=neural.validate_config,
        script_path='scripts/dpe_neural_combiners.py', notebook_path='notebooks/dpe_neural_combiners.ipynb')
    identity.update(torch=torch.__version__, execution='cpu-float64-fixed-steps')
    return source, c, manifest, heads, data, identity


def compute_selection(source, config_path):
    _, c, manifest, heads, data, identity = context(source, config_path)
    selection = neural.fit_select(*data['B'][:3], *data['C'][:3], c, manifest['provenance']['seed'])
    selection.update(version=1, config=c, lock=identity, head_ids=heads,
                     source_track=manifest['provenance']['track'], source_seed=manifest['provenance']['seed'])
    return selection


def write_comparison(output, selection):
    fields = ['id', 'family', 'parameters', 'selected_for_family', 'selected_overall',
              'b_wga', 'c_wga', 'c_overall', 'c_prediction_changes', 'c_probability_max_abs_delta']
    with (output / 'candidate_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in selection['candidates']:
            writer.writerow(dict(id=r['id'], family=r['family'], parameters=json.dumps(r['parameters']),
                                 selected_for_family=r['id'] in selection['per_family_selected'].values(),
                                 selected_overall=r['id'] == selection['selected'], b_wga=r['b']['metrics']['wga'],
                                 c_wga=r['metrics']['wga'], c_overall=r['metrics']['overall'],
                                 c_prediction_changes=r['c_prediction_changes'],
                                 c_probability_max_abs_delta=r['c_probability_max_abs_delta']))


def validated_selection(source, config_path, output):
    output = output_path(Path(source).resolve(), output)
    selection = read_json(output / 'selection.json')
    # Deterministic replay checks checkpoint values as well as inputs and choices.
    expected = compute_selection(source, config_path)
    if object_hash(selection) != object_hash(expected):
        raise ValueError('Frozen neural selection or inputs changed; use a fresh output directory')
    return selection


def select(source, config_path, output):
    """This path never opens test.npz; JSON checkpoints are frozen here."""
    output = output_path(Path(source).resolve(), output)
    if (output / 'selection.json').exists():
        selection = validated_selection(source, config_path, output)
    else:
        if (output / 'metrics.json').exists():
            raise ValueError('Existing test result has no frozen selection; preserve it')
        selection = compute_selection(source, config_path)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'selection.json', selection)
    write_comparison(output, selection)
    return selection


def evaluate(source, config_path, output):
    selection = validated_selection(source, config_path, output)
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = read_json(source / 'experiment/manifest.json')
    p, y, g, ids = load_cache(source, 'test', manifest, selection['head_ids'],
                              selection['lock']['inputs']['checkpoints/ensemble.pt'])
    candidates = {r['id']: r for r in selection['candidates']}
    uniform, _ = neural.candidate_prediction(candidates['uniform'], p)
    u = neural.metrics(uniform, y, g)
    original = source / 'results/metrics.json'
    if original.exists():
        old = read_json(original)
        if (old['lock']['ensemble'] != selection['lock']['inputs']['checkpoints/ensemble.pt'] or
                old['lock']['manifest'] != selection['lock']['inputs']['experiment/manifest.json'] or
                not same_metrics(old['uniform'], u)):
            raise ValueError('Original uniform test metrics disagree with test cache')
    chosen = dict(uniform='uniform', **selection['per_family_selected'], overall_selected=selection['selected'])
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
    result = dict(version=1, status='exploratory; neural cached-prediction adaptation',
                  source_track=selection['source_track'], source_seed=selection['source_seed'],
                  selection_hash=digest(output / 'selection.json'), lock=selection['lock'],
                  test_cache_hash=digest(source / 'experiment/predictions/test.npz'),
                  test_identity=object_hash(sorted(zip(ids.tolist(), y.tolist(), g.tolist()))),
                  original_test_metrics_hash=digest(original) if original.exists() else None, results=results)
    marker = output / 'metrics.json'
    if marker.exists():
        if object_hash(read_json(marker)) != object_hash(result):
            raise ValueError('Existing neural test result or inputs changed; preserve the original output')
        return result
    write_json(output / 'run.json', dict(config=selection['config'], runtime=runtime(),
                                       source_artifacts=selection['lock']['inputs'],
                                       execution=dict(selection='neural_suite.select', evaluation='neural_suite.evaluate'),
                                       track=selection['source_track'], seed=selection['source_seed']))
    lines = ['# DPE neural-combiner comparison', '',
             'Exploratory project adaptation inspired by Regularized Neural Ensemblers (AutoML 2025).',
             'Backbone/heads are frozen. All fits use B only, with fixed training steps.',
             'C WGA, then C accuracy select each family and overall winner. Uniform wins complete ties.',
             'Only those frozen winners are evaluated on test. No B+C refit or test-driven sweep.', '',
             '| Procedure | C-selected candidate | Test accuracy (%) | Test WGA (%) | WGA delta (pp) |',
             '| --- | --- | ---: | ---: | ---: |']
    for family, r in results.items():
        lines.append(f"| {family} | {r['candidate']} | {100*r['metrics']['overall']:.4f} | "
                     f"{100*r['metrics']['wga']:.4f} | {r['delta_pp']['wga']:.4f} |")
    lines += ['', 'Every family includes uniform as a fallback: a family result can therefore be uniform.',
              'B/C complementarity and exact CPU float64 model states: `selection.json`.',
              'All C candidates: `candidate_comparison.csv`. Group accuracies/counts, NLL, sample-dependent',
              'gate concentration and paired error IDs: `metrics.json`. Stackers have no convex head weights.',
              'Selection is CPU deterministic within recorded software; cross-version equivalence is not claimed.',
              'Previously inspected test data remains exploratory. Cached inference is not raw-image verification.', '']
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
    for family in ('uniform', *neural.FAMILIES, 'overall_selected'):
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
    lines = ['# DPE neural-combiner study', '',
             f"Status: {summary['status']}; track: {track}; completed seeds: {summary['completed_seeds']}.", '',
             'Exploratory cached-prediction adaptation. Every family was selected on C with uniform fallback.',
             'Mean ± sample SD across training seeds, not confidence intervals. No family is chosen by test.', '',
             '| Procedure | Accuracy (%) | WGA (%) | Paired WGA change (pp) |',
             '| --- | ---: | ---: | ---: |']
    for family, r in summary['results'].items():
        lines.append(f"| {family} | {display(r['overall'],100)} | {display(r['wga'],100)} | {display(r['wga_delta_pp'])} |")
    lines += ['', 'Per-seed choices: `per_seed.csv`. C candidates, model states and detailed group metrics: `comparisons/`.',
              'The study archive retains original head/prediction bundles in `source_archives/`.',
              'Large backbones, training states, images and raw features remain in WORK_DIR.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'evaluate'])
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = (select if args.action == 'select' else evaluate)(args.source_run, args.config, args.output_dir)
    print(json.dumps(dict(action=args.action, output=str(args.output_dir),
                          selected=result.get('selected'), status=result.get('status')), indent=2))
