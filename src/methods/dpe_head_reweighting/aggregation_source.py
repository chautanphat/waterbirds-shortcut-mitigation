"""Native two-split sources for independent frozen-head comparisons."""
from functools import partial
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import yaml

from . import ensemble, experiment as ex, notebook_setup as storage
from .reweighting import GROUPS, metrics, probabilities
from .rule_suite import notebook_code_hash
from .train_heads import train

PROTOCOL = 'independent_comparison'
HEAD_TRAINING = 'head_training'
COMBINER_TRAINING = 'combiner_training'
REQUIRED = ('experiment/manifest.json', 'experiment/heads.json', 'checkpoints/ensemble.pt',
            'experiment/predictions/combiner_training.npz', 'experiment/predictions/test.npz')
zip_prefix = partial(storage.zip_prefix, required_files=REQUIRED)
find_sources = partial(storage.find_sources, required_files=REQUIRED)
extract_source = partial(storage.extract_source, required_files=REQUIRED)


def partition(rows):
    """Direct group-stratified 60/40 allocation, fixed split seed 0."""
    rng = np.random.RandomState(0)
    parts = {HEAD_TRAINING: [], COMBINER_TRAINING: []}
    for group in GROUPS:
        members = sorted([r for r in rows if r['group'] == group], key=lambda r: r['id'])
        quotas = np.array([.6, .4]) * len(members)
        counts = np.floor(quotas).astype(int)
        for i in np.argsort(-(quotas - counts), kind='stable')[:len(members) - counts.sum()]:
            counts[i] += 1
        if (counts == 0).any():
            raise ValueError('Each training split must contain all groups')
        order = rng.permutation(len(members))
        start = 0
        for role, count in zip(parts, counts):
            parts[role].extend(members[i] for i in order[start:start + count])
            start += count
    for role, values in parts.items():
        wanted = {r['id'] for r in values}
        parts[role] = [r for r in rows if r['id'] in wanted]
    return parts


def load_cache(source, role, manifest, head_ids, ensemble_hash):
    with np.load(Path(source) / 'experiment/predictions' / f'{role}.npz', allow_pickle=False) as f:
        p = probabilities(f['probabilities'])
        y, g, ids = f['labels'], f['groups'].astype(str), f['ids'].astype(str)
        if (str(f['role']) != role or str(f['ensemble_hash']) != ensemble_hash or
                str(f['preprocessing']) != ensemble.PREPROCESSING):
            raise ValueError('Cache identity differs')
    if (p.shape != (len(head_ids), len(y), 2) or y.shape != g.shape or y.shape != ids.shape or
            len(set(ids)) != len(ids) or any(not value for value in ids)):
        raise ValueError('Misaligned cache or duplicate/empty IDs')
    for head in p:
        metrics(head, y, g)
    if role == COMBINER_TRAINING:
        rows = manifest['parts'][role]
        if (ids.tolist() != [r['id'] for r in rows] or y.tolist() != [r['y'] for r in rows] or
                g.tolist() != [r['group'] for r in rows]):
            raise ValueError('Cache differs from training manifest')
    elif role == 'test':
        if {r['id'] for r in manifest['validation']}.intersection(ids):
            raise ValueError('Test overlaps training data')
    else:
        raise ValueError('Unknown prediction-cache role')
    return p, y, g, ids


def context(source, config_path):
    from .neural_combiner import validate_config
    source = Path(source).resolve()
    missing = [name for name in REQUIRED[:-1] if not (source / name).is_file()]
    if missing:
        raise ValueError('Incomplete head-aggregation source: ' + ', '.join(missing))
    c = yaml.safe_load(Path(config_path).read_text())
    m = ex.read_json(source / REQUIRED[0])
    heads = ex.read_json(source / REQUIRED[1])
    head_ids = heads['head_ids']
    validate_config(c, len(head_ids))
    if c.get('protocol') != PROTOCOL:
        raise ValueError('This comparison requires the independent-comparison protocol')
    if (m.get('version') != 2 or m.get('protocol') != PROTOCOL or m.get('split_seed') != 0 or
            m.get('ratios') != [.6, .4] or m['parts'] != partition(m['validation']) or
            len({r['id'] for r in m['validation']}) != len(m['validation'])):
        raise ValueError('Invalid two-split training manifest')
    if (m['provenance']['track'] not in ('standard', 'enhanced') or
            type(m['provenance']['seed']) is not int or m['provenance']['seed'] not in (0, 1, 2) or
            len(head_ids) != 15 or len(set(head_ids)) != 15):
        raise ValueError('Invalid source seed, track or head IDs')
    model_hash = ex.digest(source / 'checkpoints/ensemble.pt')
    if heads['lock'] != dict(manifest=ex.digest(source / REQUIRED[0]), ensemble=model_hash):
        raise ValueError('Frozen head identities changed')
    data = {COMBINER_TRAINING: load_cache(source, COMBINER_TRAINING, m, head_ids, model_hash)}
    identity = dict(inputs={name: ex.digest(source / name) for name in REQUIRED[:-1]},
                    config=ex.object_hash(c), implementation=ex.runtime()['implementation'])
    identity['implementation']['scripts/dpe_head_aggregation.py'] = ex.digest('scripts/dpe_head_aggregation.py')
    identity['implementation']['notebooks/dpe_head_aggregation.ipynb'] = notebook_code_hash('notebooks/dpe_head_aggregation.ipynb')
    return source, c, m, head_ids, data, identity


def head_settings():
    settings = yaml.safe_load(Path('configs/dpe_head_reweighting/head_training.yaml').read_text())
    ex.check_keys(settings, ex.TRAIN_SCHEMA)
    if (settings['heads'] != 15 or settings['epochs'] < 1 or settings['batch_size'] < 1 or
            any(settings[k] <= 0 for k in ('lr', 'entropic_scale')) or
            any(settings[k] < 0 for k in ('momentum', 'cov_reg', 'wd_weight'))):
        raise ValueError('Invalid head-training settings')
    return settings


def build_source(work, data, track, seed, device, workers, resume):
    from .stage0 import backbone_config, train_or_reuse, export_inputs
    backbone = work / 'backbones' / track / f'seed{seed}'
    c0 = backbone_config(data, backbone, track, seed, device, workers)
    train_or_reuse(c0, resume=resume)
    extraction = [Path('src/methods/dpe_head_reweighting/stage0.py'), Path('src/evaluation/features.py'),
                  Path('src/models/resnet.py'), Path('src/datasets/waterbirds.py'), Path('src/datasets/loaders.py')]
    tag = ex.object_hash({str(p): ex.digest(p) for p in extraction})[:10]
    inputs = export_inputs(c0, work / 'features' / track / f'seed{seed}-{tag}')
    settings = head_settings()
    rows = ex.metadata(inputs['metadata'])
    if [sum(r['split'] == i for r in rows) for i in range(3)] != [4795, 1199, 5794]:
        raise ValueError('Expected official Waterbirds split counts')
    validation = [r for r in rows if r['split'] == 1]
    raw = ex.feature_file(inputs['val_features'], validation, 2048)
    provenance = ex.read_json(inputs['provenance'])
    ex.check_keys(provenance, ex.PROVENANCE)
    if (provenance['track'] != track or provenance['seed'] != seed or
            provenance['checkpoint_sha256'] != ex.digest(inputs['backbone']) or
            provenance['val_features_sha256'] != ex.digest(inputs['val_features']) or
            provenance['checkpoint_selection'] != 'final_epoch_no_validation_selection' or
            provenance['feature_format'] != 'raw'):
        raise ValueError('Invalid frozen-backbone feature provenance')
    manifest = dict(version=2, protocol=PROTOCOL, provenance=provenance,
                    metadata_hash=ex.digest(inputs['metadata']), backbone_hash=ex.digest(inputs['backbone']),
                    val_features_hash=ex.digest(inputs['val_features']), validation=validation,
                    parts=partition(validation), split_seed=0, ratios=[.6, .4],
                    training=settings, implementation=ex.runtime()['implementation'])
    root = work / 'head_sources' / f'{track}-seed{seed}-{ex.object_hash(manifest)[:12]}'
    marker = root / REQUIRED[0]
    if marker.exists() and ex.read_json(marker) != manifest:
        raise ValueError('Existing source manifest changed')
    ex.write_json(marker, manifest)
    model = root / 'checkpoints/ensemble.pt'
    if model.exists():
        heads = ensemble.load(model, 2048, 15)
        if heads['provenance'] != dict(manifest=ex.digest(marker)):
            raise ValueError('Heads belong to a different training manifest')
    else:
        head_raw, y, g = ex.subset(manifest, raw, HEAD_TRAINING)
        heads = train(head_raw, y, g, settings, seed, dict(manifest=ex.digest(marker)), device, progress=True)
        ensemble.save(model, heads)
    head_record = dict(head_ids=heads['head_ids'], selected_epochs=heads['selected_epochs'],
                       lock=dict(manifest=ex.digest(marker), ensemble=ex.digest(model)))
    head_path = root / REQUIRED[1]
    if head_path.exists() and ex.read_json(head_path) != head_record:
        raise ValueError('Frozen head record changed')
    ex.write_json(head_path, head_record)
    c = dict(**inputs, experiment_dir=str(root / 'experiment'), ensemble_path=str(model), device=device)
    target = root / REQUIRED[3]
    if not target.exists():
        combiner_raw, _, _ = ex.subset(manifest, raw, COMBINER_TRAINING)
        predictions = ensemble.predict(heads, combiner_raw, device=device)
        ex.cache(c, COMBINER_TRAINING, predictions, manifest['parts'][COMBINER_TRAINING], heads)
    else:
        load_cache(root, COMBINER_TRAINING, manifest, heads['head_ids'], ex.digest(model))
    # Local feature handoff stays outside the portable prediction bundle.
    ex.write_json(root / 'feature_inputs.json', dict(inputs=inputs, hashes={k: ex.digest(v) for k, v in inputs.items()}))
    return root, c


def verified_test_features(source, roots):
    """Find exactly one raw-feature handoff bound to this source's backbone."""
    m = ex.read_json(Path(source) / REQUIRED[0])
    found = {}
    for root in dict.fromkeys(Path(p).resolve() for p in roots):
        if not root.is_dir():
            continue
        for marker in root.rglob('handoff.json'):
            handoff = ex.read_json(marker)
            identity, artifacts = handoff.get('identity', {}), handoff.get('artifacts', {})
            if (identity.get('checkpoint_sha256') != m['backbone_hash'] or
                    artifacts.get('metadata.csv') != m['metadata_hash'] or
                    artifacts.get('val_raw.npz') != m['val_features_hash']):
                continue
            for name in ('metadata.csv', 'val_raw.npz', 'test_raw.npz', 'provenance.json'):
                if name not in artifacts or ex.digest(marker.parent / name) != artifacts[name]:
                    raise ValueError('Feature handoff changed')
            if ex.read_json(marker.parent / 'provenance.json') != m['provenance']:
                raise ValueError('Feature provenance changed')
            found[artifacts['test_raw.npz']] = marker.parent
    if len(found) != 1:
        raise ValueError('No unique verified test-feature handoff; preserve the workspace or supply a complete source bundle')
    return next(iter(found.values()))


def ensure_test_cache(c):
    root = Path(c['experiment_dir']).parent
    m = ex.read_json(root / REQUIRED[0])
    head_record = ex.read_json(root / REQUIRED[1])
    target = root / REQUIRED[-1]
    local_inputs = root / 'feature_inputs.json'
    if local_inputs.exists():
        saved = ex.read_json(local_inputs)
        for key in ('metadata', 'test_features'):
            if ex.digest(c[key]) != saved['hashes'][key]:
                raise ValueError('Test feature inputs changed')
    if target.exists():
        load_cache(root, 'test', m, head_record['head_ids'], ex.digest(c['ensemble_path']))
        return
    rows = ex.metadata(c['metadata'])
    if [r for r in rows if r['split'] == 1] != m['validation'] or ex.digest(c['metadata']) != m['metadata_hash']:
        raise ValueError('Test metadata differs from source')
    test_rows = [r for r in rows if r['split'] == 2]
    raw = ex.feature_file(c['test_features'], test_rows, 2048)
    heads = ensemble.load(c['ensemble_path'], 2048, 15)
    if heads['provenance'] != dict(manifest=ex.digest(root / REQUIRED[0])):
        raise ValueError('Frozen head provenance changed')
    p = ensemble.predict(heads, raw, device=c['device'])
    ex.cache(c, 'test', p, test_rows, heads)


def recover_test_cache(source, roots, work, *, device='cpu'):
    source = Path(source).resolve()
    features = verified_test_features(source, roots)
    identity = {name: ex.digest(source / name) for name in REQUIRED[:-1]}
    target = Path(work) / 'recovered_sources' / ex.object_hash(identity)[:16]
    for name in REQUIRED[:-1]:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if ex.digest(destination) != identity[name]:
                raise ValueError('Recovered source changed')
        else:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
                temporary = Path(f.name)
            try:
                shutil.copyfile(source / name, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    ensure_test_cache(dict(experiment_dir=str(target / 'experiment'), ensemble_path=str(target / REQUIRED[2]),
                           metadata=str(features / 'metadata.csv'), test_features=str(features / 'test_raw.npz'), device=device))
    return target
