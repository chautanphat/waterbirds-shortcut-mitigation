"""Content-addressed identities and strict sample-major prediction caches."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate(p, ids=None):
    p = np.asarray(p)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[-1] < 2:
        raise ValueError('Expected [samples, members, classes]')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=2e-6):
        raise ValueError('Invalid probabilities or softmax axis')
    if ids is not None and (len(ids) != len(p) or len(set(ids)) != len(ids)):
        raise ValueError('Duplicate or misaligned sample IDs')
    return p


def save_predictions(path, *, probs, sample_id, y, split, provenance, **optional):
    validate(probs, sample_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, probs=probs, sample_id=np.asarray(sample_id), y=y,
                        split=np.full(len(y), split), provenance=json.dumps(provenance, sort_keys=True),
                        **optional)
    return digest(path)


def load_predictions(path, provenance):
    with np.load(path, allow_pickle=False) as f:
        data = dict(f)
    if json.loads(str(data.pop('provenance'))) != provenance:
        raise ValueError('Prediction cache provenance changed')
    validate(data['probs'], data['sample_id'])
    return data
