"""Pooled, inference-safe signed teacher lookup; all fitting is development-only."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

MAPS = {
    'R1': [[.5, 0], [.515, .25], [.530, .5], [.540, .75], [.550, 1]],
    'R2': [[.5, 0], [.520, .5], [.540, 1]],
    'R3': [[.5, 0], [.600, .5], [.750, 1]],
    'R4': [[.5, 0], [.750, .5], [.950, 1]],
    'R6': [[.5, 1], [1., 1]],
}


def validate(p):
    p = np.asarray(p, dtype=float)
    assert p.ndim == 3 and p.shape[1:] == (15, 2)
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(-1), 1, atol=2e-6)
    return p


def observable(p):
    p = validate(p)
    pred = p.argmax(-1); conf = p.max(-1)
    count = np.stack([(pred == k).sum(1) for k in (0, 1)], -1)
    loo = (np.take_along_axis(count, pred, axis=1) - 1) / 14
    return pred, conf, loo


def fit_representation(map_id, logits, y):
    """Only shared NLL temperature uses labels; no group/holdout inputs."""
    temperature = .25 if map_id == 'R3' else 1.
    audit = {'method': 'fixed', 'success': True}
    if map_id == 'R4':
        def loss(logt):
            p = softmax(logits / np.exp(logt), axis=-1)
            return float(-np.log(p[np.arange(len(y))[:, None], np.arange(15), y[:, None]].clip(1e-15)).mean())
        result = minimize_scalar(loss, bounds=(np.log(.001), np.log(10.)), method='bounded', options={'xatol': 1e-10})
        if not result.success: raise RuntimeError(result.message)
        temperature = float(np.exp(result.x))
        audit = dict(method='shared_member_NLL', success=True, objective=float(result.fun), bounds=[.001, 10.])
    return dict(map_id=map_id, temperature=temperature, audit=audit)


def representation(rep, probs, logits):
    # Raw remains byte-for-byte as exported; temperatures act on ORIGINAL logits.
    return validate(softmax(np.asarray(logits, float) / rep['temperature'], axis=-1) if rep['map_id'] in ('R3', 'R4') else probs)


def fit_knots(map_id, p):
    pred, conf, _ = observable(p)
    if map_id != 'R5': return {'by_class': [MAPS[map_id], MAPS[map_id]], 'fallback': []}
    tables = []; fallback = []
    for k in (0, 1):
        values = conf[pred == k]
        q = np.quantile(values, [.5, .75, .9]) if len(values) else np.array([.515, .530, .550])
        knots = np.array([[.5, 0], [q[0], .5], [q[1], .75], [q[2], 1.]])
        if not len(values) or np.any(np.diff(knots[:, 0]) <= 0):
            knots = np.array(MAPS['R1']); fallback.append({'class': k, 'reason': 'empty/tied/non-increasing quantiles', 'to': 'R1'})
        tables.append(knots.tolist())
    return dict(by_class=tables, fallback=fallback)


def magnitude(p, knots):
    pred, conf, _ = observable(p); h = np.zeros_like(conf)
    for k in (0, 1):
        table = np.asarray(knots['by_class'][k]); mask = pred == k
        h[mask] = np.interp(conf[mask], table[:, 0], table[:, 1])
    return np.clip(h, 0, 1)


def teacher(p, y, knots):
    return magnitude(p, knots) * np.where(p.argmax(-1) == np.asarray(y)[:, None], 1., -1.)


def cell_indices(p, edges):
    pred, conf, loo = observable(p); cb = np.zeros_like(pred)
    for k in (0, 1):
        mask = pred == k; cb[mask] = np.searchsorted(edges[k], conf[mask], side='right')
    ab = np.searchsorted([1/3, 2/3], loo, side='right')
    return pred, cb, ab


def fit_lookup(p, y, sample_id, knots, kappa=50):
    pred, conf, _ = observable(p)
    edges = [np.quantile(conf[pred == k], [1/3, 2/3]).tolist() if np.any(pred == k) else [.515, .540] for k in (0, 1)]
    signed = teacher(p, y, knots); global_parent = float(signed.mean())
    parents = [float(signed[pred == k].mean()) if np.any(pred == k) else global_parent for k in (0, 1)]
    pc, cb, ab = cell_indices(p, edges); table = np.zeros((2, 3, 3)); rows = []
    correct = pred == np.asarray(y)[:, None]
    for k in (0, 1):
        for c in range(3):
            for a in range(3):
                mask = (pc == k) & (cb == c) & (ab == a); n = int(mask.sum())
                value = float((signed[mask].sum() + kappa * parents[k]) / (n + kappa))
                table[k, c, a] = value; unique = int(mask.any(1).sum())
                rows.append(dict(predicted_class=k, confidence_bin=c, agreement_bin=a, count=n,
                                 distinct_images=unique, low_support=unique < 30, score=value,
                                 empirical_correctness=float(correct[mask].mean()) if n else None,
                                 parent=parents[k], backoff='predicted_class' if not n else 'smoothed_cell'))
    return dict(edges=edges, table=table.tolist(), parents=parents, global_parent=global_parent,
                kappa=kappa, fit_ids=list(map(str, sample_id)), cells=rows, knots=knots)


def infer(p, lookup, alpha=.1, beta=1.):
    """Deployment API deliberately accepts no labels, correctness or groups."""
    p = validate(p); pc, cb, ab = cell_indices(p, lookup['edges'])
    scores = np.asarray(lookup['table'])[pc, cb, ab]
    q = softmax(beta * scores, axis=1)
    weights = (1-alpha) / 15 + alpha*q
    mix = (p * weights[..., None]).sum(1)
    assert np.all(weights > 0) and np.allclose(weights.sum(1), 1)
    assert np.allclose(mix.sum(1), 1, atol=2e-6)
    return mix, weights, scores
