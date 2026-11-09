"""Recover frozen heads/B/C caches after a weighting failure, without retraining."""
from pathlib import Path
import shutil

from . import ensemble
from . import experiment as ex
from .notebook_setup import REQUIRED


def verified_test_features(source, roots):
    """Find a feature handoff bound to this exact backbone and validation data."""
    manifest = ex.read_json(source / 'experiment/manifest.json')
    found = {}
    for root in dict.fromkeys(Path(p).expanduser().resolve() for p in roots):
        if not root.is_dir():
            continue
        for path in root.rglob('handoff.json'):
            record = ex.read_json(path)
            identity, artifacts = record.get('identity', {}), record.get('artifacts', {})
            if (identity.get('checkpoint_sha256') != manifest.get('backbone_hash') or
                    artifacts.get('metadata.csv') != manifest.get('metadata_hash') or
                    artifacts.get('val_raw.npz') != manifest.get('val_features_hash')):
                continue
            bundle = path.parent
            for name in ('metadata.csv','val_raw.npz','test_raw.npz','provenance.json'):
                if name not in artifacts or not (bundle / name).is_file() or ex.digest(bundle / name) != artifacts[name]:
                    raise ValueError(f'Recovery feature artifact changed: {name}')
            provenance = ex.read_json(bundle / 'provenance.json')
            if provenance != manifest['provenance']:
                raise ValueError('Recovery feature provenance differs from frozen heads')
            found.setdefault(artifacts['test_raw.npz'], bundle)
    if len(found) != 1:
        raise ValueError('No unique verified test-feature handoff for the completed heads. '
                         'Attach the failed notebook outputs including features/ and h2_sources/; '
                         'do not discard its completed backbone or heads.')
    return next(iter(found.values()))


def recover_test_cache(source, roots, work, *, device='cpu'):
    """Called only AFTER new family selections have been frozen on B/C."""
    source, work = Path(source).resolve(), Path(work).resolve()
    bundle = verified_test_features(source, roots)
    manifest = ex.read_json(source / 'experiment/manifest.json')
    metadata = ex.metadata(bundle / 'metadata.csv')
    val = [r for r in metadata if r['split'] == 1]
    if val != manifest['validation']:
        raise ValueError('Recovery metadata differs from the frozen validation rows')
    rows = [r for r in metadata if r['split'] == 2]
    identity = {name: ex.digest(source / name) for name in REQUIRED[:-1]}
    root = work / 'recovered_sources' / ex.object_hash(identity)[:16]
    # Keep original failed/readonly artifacts intact, including their old locks.
    for name in (*REQUIRED[:-1], 'config.yaml', 'results/metrics.json'):
        original, copy = source / name, root / name
        if not original.is_file():
            continue
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.exists():
            if ex.digest(copy) != ex.digest(original):
                raise ValueError('Existing recovery source copy changed')
        else:
            shutil.copyfile(original, copy)
    target = root / 'experiment/predictions/test.npz'
    origin = root / 'test_cache_recovery.json'
    expected = dict(source_inputs=identity, test_features_hash=ex.digest(bundle / 'test_raw.npz'),
                    metadata_hash=ex.digest(bundle / 'metadata.csv'))
    if target.exists() and origin.is_file():
        saved = ex.read_json(origin)
        if saved['inputs'] != expected or saved['test_cache_hash'] != ex.digest(target):
            raise ValueError('Recovered test-cache identity changed')
        return root
    # An interruption between cache publication and its recovery record is safe:
    # regenerate the managed copy from verified inputs rather than trusting it.
    print('Reusing completed backbone features and all 15 frozen heads; no training is repeated.', flush=True)
    a = ensemble.load(root / 'checkpoints/ensemble.pt', dim=2048, count=15)
    if a['provenance'] != dict(manifest=ex.digest(root / 'experiment/manifest.json')):
        raise ValueError('Recovered ensemble belongs to a different validation partition')
    raw = ex.feature_file(bundle / 'test_raw.npz', rows, 2048)
    p = ensemble.predict(a, raw, device=device)
    c = dict(experiment_dir=str(root / 'experiment'), ensemble_path=str(root / 'checkpoints/ensemble.pt'))
    ex.cache(c, 'test', p, rows, a)
    ex.write_json(origin, dict(inputs=expected, test_cache_hash=ex.digest(target),
                               note='Test inference only; original frozen heads and B/C selection preserved',
                               runtime=ex.runtime()))
    return root
