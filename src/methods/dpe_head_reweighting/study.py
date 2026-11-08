"""One-call Waterbirds -> ERM -> DPE -> all weighting rules -> seed summary."""
import csv
import gc
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

from . import experiment as ex
from .notebook_setup import extract_source, find_sources
from . import rule_suite


def ensure_dataset(work, roots, explicit=None, download=True):
    from .stage0 import validate_dataset
    work = Path(work)
    candidates = [Path(explicit).expanduser()] if explicit else [work / 'data/waterbirds', Path('data/waterbirds')]
    if not explicit:
        for root in roots:
            root = Path(root)
            if root.is_dir():
                candidates.extend(p.parent for p in root.rglob('metadata*.csv'))
    for path in dict.fromkeys(p.resolve() for p in candidates):
        try:
            validate_dataset(path)
            print(f'Using Waterbirds: {path}', flush=True)
            return path
        except (FileNotFoundError, ValueError, KeyError):
            if explicit:
                raise
    if not download:
        raise FileNotFoundError('No complete Waterbirds dataset found; attach it or enable download_data')
    path = work / 'data/waterbirds'
    print('Downloading/preparing the official Waterbirds dataset once.', flush=True)
    subprocess.run([sys.executable, 'scripts/prepare_waterbirds.py', '--data_dir', str(work / 'data')], check=True)
    validate_dataset(path)
    return path


def build_source(work, data, track, seed, device, workers, resume):
    """Stop after B/C selection: all new rule choices precede test inference."""
    from .stage0 import backbone_config, train_or_reuse, export_inputs
    backbone = work / 'backbones' / track / f'seed{seed}'
    c0 = backbone_config(data, backbone, track, seed, device, workers)
    train_or_reuse(c0, resume=resume)
    # Changed extractor code gets a fresh feature directory, preserving old data.
    extraction = [Path('src/methods/dpe_head_reweighting/stage0.py'), Path('src/evaluation/features.py'),
                  Path('src/models/resnet.py'), Path('src/datasets/waterbirds.py'), Path('src/datasets/loaders.py')]
    tag = ex.object_hash({str(p): ex.digest(p) for p in extraction})[:10]
    inputs = export_inputs(c0, work / 'features' / track / f'seed{seed}-{tag}')
    c = yaml.safe_load(Path(f'configs/dpe_head_reweighting/waterbirds_{track}.yaml').read_text())
    c.update(track=track, seed=seed, device=device, **inputs)
    identity = dict(track=track, seed=seed, device=device, training=c['training'],
                    implementation=ex.runtime()['implementation'],
                    inputs={k: ex.digest(inputs[k]) for k in ('metadata', 'backbone', 'val_features')})
    root = work / 'h2_sources' / f'{track}-seed{seed}-{ex.object_hash(identity)[:10]}'
    c.update(experiment_dir=str(root / 'experiment'), ensemble_path=str(root / 'checkpoints/ensemble.pt'),
             results_dir=str(root / 'results'))
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != c:
        raise ValueError('Saved H2 configuration changed')
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    c = ex.config(path)
    ex.preflight(c)
    if (root / 'experiment/manifest.json').exists():
        ex.context(c)
    else:
        ex.prepare(c)
    if (root / 'checkpoints/ensemble.pt').exists():
        ex.get_ensemble(c)
    else:
        ex.train_action(c)
    if (root / 'experiment/selection.json').exists():
        ex.validated_selection(c)
    else:
        ex.select_action(c)
    records = root / 'stage0_records'
    records.mkdir(exist_ok=True)
    for name in ('config.yaml', 'run.json', 'summary.json'):
        shutil.copyfile(backbone / name, records / name)
    return root, c


def ensure_test_cache(c):
    path = Path(c['results_dir']) / 'metrics.json'
    if path.exists():
        r = ex.read_json(path)
        selection = Path(c['experiment_dir']) / 'selection.json'
        if (r.get('selection_hash') != ex.digest(selection) or r['lock'] != ex.lock(c) or
                r['test_features_hash'] != ex.digest(c['test_features']) or r['metadata_hash'] != ex.digest(c['metadata'])):
            raise ValueError('Saved original H2 test result inputs changed')
    else:
        ex.evaluate_action(c, plots=False)


def summarize(study_dir, completed, planned, track):
    """Paired per-seed comparisons; never choose a family using test metrics."""
    families = ('uniform', 'power', 'softmax', 'top_k', 'joint', 'overall_selected')
    summary = dict(track=track, planned_seeds=planned, completed_seeds=sorted(completed),
                   status='complete' if sorted(completed) == sorted(planned) else 'incomplete',
                   units='accuracies are fractions; delta_pp is percentage points', results={})
    rows = []
    for family in families:
        values = [completed[seed]['results'][family] for seed in sorted(completed)]
        summary['results'][family] = {}
        for name, array in [('overall', [v['metrics']['overall'] for v in values]),
                            ('wga', [v['metrics']['wga'] for v in values]),
                            ('overall_delta_pp', [v['delta_pp']['overall'] for v in values]),
                            ('wga_delta_pp', [v['delta_pp']['wga'] for v in values])]:
            summary['results'][family][name] = dict(values=array, mean=float(np.mean(array)),
                                                   sample_std=float(np.std(array, ddof=1)) if len(array) > 1 else None)
        for seed, value in zip(sorted(completed), values):
            rows.append(dict(seed=seed, family=family, candidate=value['candidate'],
                             accuracy_percent=100*value['metrics']['overall'], wga_percent=100*value['metrics']['wga'],
                             accuracy_delta_pp=value['delta_pp']['overall'], wga_delta_pp=value['delta_pp']['wga']))
    ex.write_json(study_dir / 'summary.json', summary)
    with (study_dir / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# DPE weighting study', '', f"Status: {summary['status']}; track: {track}; completed seeds: {sorted(completed)}.", '',
             'Mean and sample SD across training seeds. No test-based family selection is performed.', '',
             '| Procedure | Accuracy mean ± SD (%) | WGA mean ± SD (%) | Paired WGA change mean ± SD (pp) |',
             '| --- | ---: | ---: | ---: |']
    def display(v, scale=1):
        sd = 'pending' if v['sample_std'] is None else f"{scale*v['sample_std']:.4f}"
        return f"{scale*v['mean']:.4f} ± {sd}"
    for family, value in summary['results'].items():
        lines.append(f"| {family} | {display(value['overall'],100)} | {display(value['wga'],100)} | {display(value['wga_delta_pp'])} |")
    lines += ['', 'Per-seed choices and paired differences: `per_seed.csv`. Detailed weights, solver diagnostics,',
              'group metrics/counts and source provenance: `comparisons/`. Full prediction bundles:',
              '`source_archives/`. Backbone/resume checkpoints and raw features remain in WORK_DIR,',
              'outside this compact archive. Save WORK_DIR persistently to resume after VM deletion.', '']
    (study_dir / 'report.md').write_text('\n'.join(lines))
    return summary


def run_study(work_dir, *, track=None, seeds=None, source=None, search_roots=(), data_dir=None,
              mode='auto', device='auto', workers=None, config_path='configs/dpe_head_reweighting/study.yaml'):
    settings = yaml.safe_load(Path(config_path).read_text())
    if (set(settings) != {'version','track','seeds','workers','download_data','resume_backbone','rules'} or
            type(settings['version']) is not int or settings['version'] != 1 or
            any(type(settings[k]) is not bool for k in ('download_data','resume_backbone')) or
            not isinstance(settings['rules'], str)):
        raise ValueError('Invalid study configuration')
    track = settings['track'] if track is None else track
    workers = settings['workers'] if workers is None else workers
    if track not in ('standard','enhanced') or mode not in ('auto','cached') or type(workers) is not int or workers < 0:
        raise ValueError('Choose standard/enhanced, auto/cached mode and nonnegative workers')
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    config_path = Path(settings['rules']).resolve()
    choices = find_sources([*search_roots, work / 'source-inputs']) if source is None else (
        list(source) if isinstance(source, (list,tuple)) else [source])
    cached = {}
    for item in choices:
        root = extract_source(item, work / 'source-inputs')
        _, _, manifest, _, _, identity = rule_suite.validation_context(root, config_path)
        provenance = manifest['provenance']
        if provenance['track'] != track:
            continue
        seed = provenance['seed']
        if seed in cached and cached[seed][1] != identity['inputs']:
            raise ValueError(f'Multiple different source runs for seed {seed}; specify SOURCE explicitly')
        cached[seed] = (root, identity['inputs'])
    if seeds is None:
        seeds = sorted(cached) if mode == 'cached' else settings['seeds']
    if mode == 'cached' and not seeds:
        raise ValueError('No H2 bundles found for cached mode; use MODE="auto" to train missing seeds')
    if not isinstance(seeds, (list,tuple)) or not seeds or any(type(s) is not int or s not in (0,1,2) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('SEEDS must contain distinct values from 0, 1, 2')
    seeds = list(seeds)
    missing = [s for s in seeds if s not in cached]
    if mode == 'cached' and missing:
        raise ValueError(f'Cached-only mode is missing H2 bundles for seeds {missing}')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.empty(1, device=device)
    if missing and device == 'cpu':
        print('CPU training is supported but slow. A GPU runtime is recommended for new backbones.', flush=True)
    plan = dict(version=1, track=track, seeds=seeds, device=device, workers=workers, mode=mode,
                settings=settings, rules=yaml.safe_load(config_path.read_text()),
                cached_sources={str(s): cached[s][1] for s in seeds if s in cached},
                implementation=ex.runtime()['implementation'],
                notebook=rule_suite.notebook_code_hash('notebooks/dpe_weighting_rules.ipynb'))
    study_dir = work / 'studies' / f"{track}-seeds{''.join(map(str,seeds))}-{ex.object_hash(plan)[:10]}"
    study_dir.mkdir(parents=True, exist_ok=True)
    ex.write_json(study_dir / 'plan.json', plan)
    completed, test_identity, dataset = {}, None, None
    for seed in seeds:
        progress = dict(planned_seeds=seeds, completed_seeds=sorted(completed), current_seed=seed, stage='source')
        ex.write_json(study_dir / 'progress.json', progress)
        print(f'\n=== {track}: seed {seed} ({seeds.index(seed)+1}/{len(seeds)}) ===', flush=True)
        c = None
        if seed in cached:
            root = cached[seed][0]
            print(f'Reusing supplied H2 prediction bundle: {root}', flush=True)
        else:
            if dataset is None:
                dataset = ensure_dataset(work, search_roots, data_dir, settings['download_data'])
            root, c = build_source(work, dataset, track, seed, device, workers, settings['resume_backbone'])
        _, _, _, _, _, identity = rule_suite.validation_context(root, config_path)
        output = work / 'results' / f'h2-{track}-seed{seed}-rules-{ex.object_hash(identity)[:10]}'
        # Freeze all families before generating any new test predictions.
        rule_suite.select(root, config_path, output)
        progress['stage'] = 'test_evaluation'
        ex.write_json(study_dir / 'progress.json', progress)
        if c is not None:
            ensure_test_cache(c)
        result = rule_suite.evaluate(root, config_path, output)
        with np.load(root / 'experiment/predictions/test.npz', allow_pickle=False) as f:
            rows = sorted(zip(f['ids'].astype(str).tolist(), f['labels'].tolist(), f['groups'].astype(str).tolist()))
        fingerprint = ex.object_hash(rows)
        if test_identity is not None and fingerprint != test_identity:
            raise ValueError('Seeds do not use identical test IDs/labels/groups')
        test_identity = fingerprint
        completed[seed] = result
        shutil.copytree(output, study_dir / 'comparisons' / f'seed{seed}', dirs_exist_ok=True)
        archives = study_dir / 'source_archives'
        archives.mkdir(exist_ok=True)
        shutil.make_archive(str(archives / f'h2-{track}-seed{seed}'), 'zip', root_dir=root)
        summary = summarize(study_dir, completed, seeds, track)
        progress.update(completed_seeds=sorted(completed), stage='complete' if summary['status']=='complete' else 'seed_complete')
        ex.write_json(study_dir / 'progress.json', progress)
        # Update the downloadable artifact after each seed, even before all finish.
        archive = shutil.make_archive(str(study_dir), 'zip', root_dir=study_dir)
        print(f'Seed {seed} complete. Current results ZIP: {archive}', flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dict(summary=summary, output_dir=str(study_dir), archive=archive)
