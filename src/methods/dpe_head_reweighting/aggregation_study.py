"""Waterbirds -> frozen DPE heads -> global-rule/neural comparison."""
import gc
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

import torch
import yaml

from . import aggregation_suite as suite
from .experiment import digest, object_hash, write_json
from .aggregation_source import (extract_source, find_sources, zip_prefix, recover_test_cache,
                                 verified_test_features, build_source, ensure_test_cache, head_settings)
from .study import ensure_dataset, publish_zip
from .rule_suite import notebook_code_hash

AGGREGATION_STUDY_RUNTIME_VERSION = 6
DEFAULT_STUDY_CONFIG = 'configs/dpe_head_reweighting/aggregation_study.yaml'


def source_choices(paths, work):
    """Unwrap only nested source ZIP bytes, never archive-supplied filesystem paths."""
    choices = find_sources(paths, require_test=False)
    for root in map(Path, paths):
        archives = [root] if root.is_file() else root.rglob('*.zip') if root.exists() else []
        for path in archives:
            if zip_prefix(path, require_test=False) is not None:
                continue
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        # Previous run_study exports original runs in source_archives/.
                        if 'source_archives' not in Path(member.filename).parts or not member.filename.endswith('.zip'):
                            continue
                        payload = archive.read(member)
                        destination = Path(work) / 'source-archives' / (hashlib.sha256(payload).hexdigest() + '.zip')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.exists():
                            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                                staged = Path(temporary) / 'source.zip'
                                staged.write_bytes(payload)
                                os.replace(staged, destination)
                        if digest(destination) != hashlib.sha256(payload).hexdigest():
                            raise ValueError('Nested source archive changed')
                        if zip_prefix(destination, require_test=False) is not None:
                            choices.append(destination)
            except zipfile.BadZipFile:
                continue
    return sorted(set(map(Path, choices)), key=str)


def run_study(work_dir, *, source=None, search_roots=(), track=None, seeds=None,
              config_path=suite.DEFAULT_CONFIG, evaluate_test=False, mode='auto',
              data_dir=None, device='auto', workers=None, study_config_path=DEFAULT_STUDY_CONFIG):
    """Reuse compatible sources; auto mode downloads data and trains missing seeds."""
    settings = yaml.safe_load(Path(study_config_path).read_text())
    if (not isinstance(settings, dict) or
            set(settings) != {'version', 'track', 'seeds', 'workers', 'download_data', 'resume_backbone'} or
            type(settings['version']) is not int or settings['version'] != 1 or
            any(type(settings[k]) is not bool for k in ('download_data', 'resume_backbone'))):
        raise ValueError('Invalid head-aggregation study configuration')
    track = settings['track'] if track is None else track
    seeds = settings['seeds'] if seeds is None else seeds
    workers = settings['workers'] if workers is None else workers
    if (track not in ('standard', 'enhanced') or type(evaluate_test) is not bool or mode not in ('auto', 'cached') or
            type(workers) is not int or workers < 0 or
            not isinstance(seeds, (list, tuple)) or not seeds or
            any(type(s) is not int or s not in (0, 1, 2) for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('Choose a track, distinct seeds 0/1/2, auto/cached mode, nonnegative workers and boolean evaluate_test')
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    # Fail before dataset downloads or expensive training for an invalid grid.
    suite.neural.validate_config(config, 15)
    source_settings = head_settings()
    work = Path(work_dir).expanduser().resolve()
    paths = list(source) if isinstance(source, (list, tuple)) else [source] if source is not None else list(search_roots)
    # Managed head-training sources survive notebook interruption before test inference.
    managed_roots = (work / 'head_sources', work / 'recovered_sources')
    choices = source_choices([*paths, *managed_roots, work / 'source-inputs'], work)
    selected, external = {}, {}
    for item in choices:
        root = extract_source(item, work / 'source-inputs', require_test=False)
        _, _, manifest, _, _, identity = suite.context(root, config_path)
        seed = manifest['provenance']['seed']
        if manifest['provenance']['track'] != track or seed not in seeds:
            continue
        if seed in selected and selected[seed][1] != identity:
            raise ValueError(f'Multiple different source bundles for seed {seed}; set source explicitly')
        selected[seed] = (root, identity)
        item = Path(item).resolve()
        if not any(base == item or base in item.parents for base in managed_roots):
            external[str(seed)] = identity['inputs']
    missing = sorted(set(seeds) - set(selected))
    if mode == 'cached' and missing:
        raise ValueError(f'Missing frozen head bundles for seeds {missing}. Attach the previous study ZIP '
                         'with source_archives/, individual source ZIPs, or extracted original runs. '
                         'Or use mode="auto" to prepare and train missing seeds.')
    # Reject unrecoverable cached sources before training other missing seeds.
    if evaluate_test:
        for root, _ in selected.values():
            if not (root / 'experiment/predictions/test.npz').is_file():
                verified_test_features(root, [work, *search_roots, *paths])
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.empty(1, device=device)
    if missing and device == 'cpu':
        print('Missing seeds require 100-epoch backbone training. Select a GPU runtime for practical run times.', flush=True)
    # Exclude managed outputs so a rerun reuses the same plan as seeds finish.
    runtime = suite.runtime()
    implementation = runtime['implementation']
    implementation['notebooks/dpe_head_aggregation.ipynb'] = notebook_code_hash('notebooks/dpe_head_aggregation.ipynb')
    implementation['scripts/dpe_head_aggregation.py'] = digest('scripts/dpe_head_aggregation.py')
    plan = dict(version=2, track=track, seeds=list(seeds), config=config, settings=settings,
                mode=mode, device=device, workers=workers, cached_sources=external, head_training=source_settings,
                implementation=implementation, software=dict(scipy=suite.scipy.__version__, **{k: runtime[k] for k in ('python', 'torch', 'numpy')}),
                status='exploratory adaptation')
    output = work / 'studies' / f"aggregation-{track}-seeds{''.join(map(str,seeds))}-{object_hash(plan)[:12]}"
    # Reject overlap before creating any result files, including plan/progress.
    for root, _ in selected.values():
        suite.output_path(root, output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    frozen, completed, generated, dataset = [], {}, {}, None
    previous = [s for s in seeds if (output / 'comparisons' / f'seed{s}' / 'metrics.json').exists()]
    archive = output.with_suffix('.zip')
    for seed in seeds:
        write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=[], current_seed=seed,
                                                 previously_evaluated_seeds=previous, stage='source'))
        if seed not in selected:
            if dataset is None:
                dataset = ensure_dataset(work, search_roots, data_dir, settings['download_data'])
            print(f'Preparing frozen DPE source: {track} seed {seed}', flush=True)
            root, c = build_source(work, dataset, track, seed, device, workers, settings['resume_backbone'])
            _, _, _, _, _, identity = suite.context(root, config_path)
            selected[seed] = (root, identity)
            generated[seed] = c
        else:
            print(f'Reusing frozen DPE source: {selected[seed][0]}', flush=True)
        print(f'Fitting/freezing head aggregation configurations: {track} seed {seed}', flush=True)
        comparison = output / 'comparisons' / f'seed{seed}'
        suite.fit(selected[seed][0], config_path, comparison)
        frozen.append(seed)
        write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=[],
                                                 previously_evaluated_seeds=previous, stage='models_frozen'))
        sources = output / 'source_archives'
        sources.mkdir(exist_ok=True)
        publish_zip(selected[seed][0], sources / f'heads-{track}-seed{seed}.zip')
        publish_zip(output, archive)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Freeze ALL planned seeds before opening ANY test cache.
    if evaluate_test:
        for seed in seeds:
            root, identity = selected[seed]
            if seed in generated:
                ensure_test_cache(generated[seed])
            elif not (root / 'experiment/predictions/test.npz').is_file():
                root = recover_test_cache(root, [work, *search_roots, *paths], work, device=device)
                selected[seed] = (root, identity)
            print(f'Evaluating frozen configurations: {track} seed {seed}', flush=True)
            completed[seed] = suite.evaluate(root, config_path, output / 'comparisons' / f'seed{seed}')
            suite.summarize(output, completed, seeds, track)
            write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=sorted(completed),
                                                       stage='complete' if len(completed) == len(seeds) else 'evaluation'))
            publish_zip(root, output / 'source_archives' / f'heads-{track}-seed{seed}.zip')
            publish_zip(output, archive)
    else:
        write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=[],
                                                 previously_evaluated_seeds=previous, stage='models_frozen'))
        publish_zip(output, archive)
    return dict(output_dir=str(output), archive=str(archive),
                summary=suite.read_json(output / 'summary.json') if evaluate_test else None)
