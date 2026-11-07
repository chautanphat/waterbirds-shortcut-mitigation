"""Validation-only H2 weights and deterministic model selection."""
import numpy as np

GROUPS = ('00', '01', '10', '11')


def probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim not in (2, 3) or p.shape[-1] != 2 or not p.size:
        raise ValueError('Expected nonempty probabilities [..., N, 2]')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise ValueError('Invalid probabilities')
    return p


def metrics(p, y, groups):
    p = probabilities(p)
    y, groups = np.asarray(y), np.asarray(groups)
    if p.ndim != 2 or y.shape != (len(p),) or groups.shape != y.shape:
        raise ValueError('Misaligned metric inputs')
    if not np.isin(y, [0, 1]).all() or set(groups) != set(GROUPS):
        raise ValueError('Binary labels and all four groups are required')
    if not np.array_equal(y.astype(str), np.array([g[0] for g in groups])):
        raise ValueError('Group labels disagree with classes')
    correct = p.argmax(-1) == y
    per_group = {g: float(correct[groups == g].mean()) for g in GROUPS}
    return dict(overall=float(correct.mean()), wga=min(per_group.values()),
                groups=per_group, counts={g: int(sum(groups == g)) for g in GROUPS})


def weights(scores, alpha, gamma):
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not s.size or not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError('Scores must be finite accuracy fractions')
    if not 0 <= alpha <= 1 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Invalid alpha/gamma')
    fallback = bool(np.all(s == 0))
    q = np.ones_like(s) / len(s) if fallback else (s / s.max()) ** gamma
    q /= q.sum()
    return (1 - alpha) / len(s) + alpha * q, fallback


def combine(p, w):
    p, w = probabilities(p), np.asarray(w, dtype=np.float64)
    if p.ndim != 3 or w.shape != (len(p),) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1, atol=1e-12):
        raise ValueError('Invalid ensemble weights')
    return np.einsum('h,hnc->nc', w, p)


def fit_select(p_b, y_b, g_b, p_c, y_c, g_c):
    p_b, p_c = probabilities(p_b), probabilities(p_c)
    if p_b.ndim != 3 or p_c.ndim != 3 or len(p_b) != len(p_c):
        raise ValueError('Head counts differ')
    b = [metrics(p, y_b, g_b) for p in p_b]
    c = [metrics(p, y_c, g_c) for p in p_c]
    scores = [m['wga'] for m in b]
    grid = [(0, 1)] + [(a, g) for a in (.25, .5, .75, 1) for g in (1, 2, 4)]
    candidates, best = [], None
    for alpha, gamma in grid:
        w, fallback = weights(scores, alpha, gamma)
        m = metrics(combine(p_c, w), y_c, g_c)
        row = dict(alpha=alpha, gamma=gamma, weights=w.tolist(), fallback=fallback, metrics=m)
        candidates.append(row)
        # Grid order supplies the smaller-alpha/smaller-gamma tie breakers.
        if best is None or m['wga'] > best['metrics']['wga'] + 1e-12 or (
            abs(m['wga'] - best['metrics']['wga']) <= 1e-12 and
            m['overall'] > best['metrics']['overall'] + 1e-12
        ):
            best = row
    return dict(selected=best, candidates=candidates, per_head_B=b, per_head_C=c)
