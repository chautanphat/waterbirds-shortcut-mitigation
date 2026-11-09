"""Cached-only neural comparison, including original multi-seed H2 study ZIPs."""
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

from . import neural_suite as suite
from .experiment import digest, object_hash, write_json
from .notebook_setup import extract_source, find_sources, zip_prefix
from .study import publish_zip


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


def run_study(work_dir, *, source=None, search_roots=(), track='standard', seeds=(0, 1, 2),
              config_path=suite.DEFAULT_CONFIG, evaluate_test=False):
    """Reuse frozen sources; no new backbone, heads, feature extraction or download."""
    if (track not in ('standard', 'enhanced') or type(evaluate_test) is not bool or
            not isinstance(seeds, (list, tuple)) or not seeds or
            any(type(s) is not int or s not in (0, 1, 2) for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('Choose a track, distinct seeds 0/1/2 and a boolean evaluate_test')
    work = Path(work_dir).expanduser().resolve()
    paths = list(source) if isinstance(source, (list, tuple)) else [source] if source is not None else list(search_roots)
    choices = source_choices(paths, work)
    selected = {}
    for item in choices:
        root = extract_source(item, work / 'source-inputs', require_test=False)
        _, config, manifest, _, _, identity = suite.context(root, config_path)
        seed = manifest['provenance']['seed']
        if manifest['provenance']['track'] != track or seed not in seeds:
            continue
        if seed in selected and selected[seed][1] != identity:
            raise ValueError(f'Multiple different source bundles for seed {seed}; set source explicitly')
        selected[seed] = (root, identity)
    missing = sorted(set(seeds) - set(selected))
    if missing:
        raise ValueError(f'Missing frozen H2 bundles for seeds {missing}. Attach the previous study ZIP '
                         'with source_archives/, individual H2 ZIPs, or extracted original runs. '
                         'This comparison does not train missing backbones or heads.')
    plan = dict(version=1, track=track, seeds=list(seeds), config=config,
                sources={str(s): selected[s][1] for s in seeds}, status='exploratory adaptation')
    output = work / 'studies' / f"neural-{track}-seeds{''.join(map(str,seeds))}-{object_hash(plan)[:12]}"
    # Reject overlap before creating any result files, including plan/progress.
    for root, _ in selected.values():
        suite.output_path(root, output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    frozen, completed = [], {}
    previous = [s for s in seeds if (output / 'comparisons' / f'seed{s}' / 'metrics.json').exists()]
    archive = output.with_suffix('.zip')
    for seed in seeds:
        print(f'Fitting/freezing neural combiners: {track} seed {seed}', flush=True)
        comparison = output / 'comparisons' / f'seed{seed}'
        suite.select(selected[seed][0], config_path, comparison)
        frozen.append(seed)
        write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=[],
                                                 previously_evaluated_seeds=previous, stage='selection'))
        publish_zip(output, archive)
    # Freeze ALL planned seeds before opening ANY test cache.
    if evaluate_test:
        missing_test = [s for s in seeds if not (selected[s][0] / 'experiment/predictions/test.npz').is_file()]
        if missing_test:
            raise ValueError(f'Frozen selections saved, but test prediction caches are missing for seeds {missing_test}')
        for seed in seeds:
            print(f'Evaluating frozen choices: {track} seed {seed}', flush=True)
            completed[seed] = suite.evaluate(selected[seed][0], config_path, output / 'comparisons' / f'seed{seed}')
            suite.summarize(output, completed, seeds, track)
            write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=sorted(completed),
                                                       stage='complete' if len(completed) == len(seeds) else 'evaluation'))
            publish_zip(output, archive)
    else:
        write_json(output / 'progress.json', dict(frozen_seeds=frozen, evaluated_seeds=[],
                                                 previously_evaluated_seeds=previous, stage='selection_complete'))
        publish_zip(output, archive)
    return dict(output_dir=str(output), archive=str(archive),
                summary=suite.read_json(output / 'summary.json') if evaluate_test else None)
