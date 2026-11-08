"""Additional B-only weight fits; C is used exclusively for selection."""
import numpy as np
from scipy.optimize import minimize

from .reweighting import GROUPS, combine, metrics, probabilities, weights


def joint_weights(p, y, groups, penalty, solver):
    """Convex epigraph fit: worst B-group NLL + squared distance to uniform."""
    p = probabilities(p)
    # Validate alignment, binary labels, group coverage and class/group mapping.
    for head in p:
        metrics(head, y, groups)
    if p.ndim != 3 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid joint fit inputs')
    y, groups = np.asarray(y, dtype=int), np.asarray(groups)
    h, n, _ = p.shape
    u = np.full(h, 1 / h)
    # A declared numerical floor makes log loss finite even for exact zeros.
    correct = np.maximum(p[:, np.arange(n), y], solver['probability_floor'])
    grouped = [correct[:, groups == g] for g in GROUPS]

    def losses(w):
        return np.array([-np.log(w @ q).mean() for q in grouped])

    def objective(z):
        return z[-1] + penalty * np.square(z[:-1] - u).sum()

    def gradient(z):
        return np.r_[2 * penalty * (z[:-1] - u), 1.]

    def constraints(z):
        return z[-1] - losses(z[:-1])

    def jacobian(z):
        return np.array([np.r_[(q / (z[:-1] @ q)).mean(axis=1), 1.] for q in grouped])

    initial = np.r_[u, losses(u).max()]
    result = minimize(objective, initial, jac=gradient, method='SLSQP',
                      bounds=[(0., 1.)] * h + [(0., None)],
                      constraints=[
                          dict(type='eq', fun=lambda z: z[:-1].sum() - 1,
                               jac=lambda z: np.r_[np.ones(h), 0.]),
                          dict(type='ineq', fun=constraints, jac=jacobian),
                      ], options=dict(maxiter=solver['maxiter'], ftol=solver['ftol']))
    tol = solver['feasibility_tolerance']
    if (not result.success or not np.isfinite(result.x).all() or
            abs(result.x[:-1].sum() - 1) > tol or
            result.x[:-1].min() < -tol or constraints(result.x).min() < -tol):
        raise ValueError(f'Joint solver failed; no candidate accepted: {result.message}')
    w = np.clip(result.x[:-1], 0, 1)
    w /= w.sum()
    # Normalization must not hide a failure to improve on the feasible start.
    value = float(losses(w).max() + penalty * np.square(w - u).sum())
    if value > objective(initial) + tol:
        raise ValueError('Joint solver returned an objective worse than uniform')
    return w, dict(success=True, iterations=int(result.nit), message=str(result.message),
                   objective=value, max_group_nll=float(losses(w).max()),
                   group_nll=dict(zip(GROUPS, losses(w).tolist())))


def validate_config(c, heads):
    if set(c) != {'version', 'softmax_betas', 'top_k', 'joint_lambdas', 'solver'} or c['version'] != 1:
        raise ValueError('Unsupported weighting-rule configuration')
    for name in ('softmax_betas', 'joint_lambdas'):
        values = c[name]
        if (not isinstance(values, list) or not values or
                any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f'Invalid {name}')
    if (not isinstance(c['top_k'], list) or not c['top_k'] or
            any(type(k) is not int or not 1 <= k <= heads for k in c['top_k']) or
            len(set(c['top_k'])) != len(c['top_k'])):
        raise ValueError('Invalid top_k')
    solver = c['solver']
    if set(solver) != {'maxiter', 'ftol', 'feasibility_tolerance', 'probability_floor'}:
        raise ValueError('Invalid solver fields')
    if type(solver['maxiter']) is not int or solver['maxiter'] < 1:
        raise ValueError('Invalid solver iteration limit')
    for key in ('ftol', 'feasibility_tolerance', 'probability_floor'):
        if type(solver[key]) not in (float, int) or not np.isfinite(solver[key]) or not 0 < solver[key] < 1:
            raise ValueError(f'Invalid solver {key}')


def best_candidate(rows):
    """Uniform first, then declaration order, on a complete C-accuracy tie."""
    best = rows[0]
    for row in rows[1:]:
        m, b = row['metrics'], best['metrics']
        if m['wga'] > b['wga'] + 1e-12 or (
                abs(m['wga'] - b['wga']) <= 1e-12 and m['overall'] > b['overall'] + 1e-12):
            best = row
    return best


def fit_rules(p_b, y_b, g_b, p_c, y_c, g_c, config):
    p_b, p_c = probabilities(p_b), probabilities(p_c)
    if p_b.ndim != 3 or p_c.ndim != 3 or len(p_b) != len(p_c):
        raise ValueError('Expected matching head axes')
    validate_config(config, len(p_b))
    b = [metrics(p, y_b, g_b) for p in p_b]
    for p in p_c:
        metrics(p, y_c, g_c)
    scores = np.array([row['wga'] for row in b])
    h = len(scores)
    u = np.full(h, 1 / h)
    uniform = combine(p_c, u)
    candidates = []

    def add(name, family, parameters, w, fit=None):
        prediction = combine(p_c, w)
        candidates.append(dict(id=name, family=family, parameters=parameters, weights=w.tolist(),
                               fit=fit, metrics=metrics(prediction, y_c, g_c),
                               effective_heads=float(1 / np.square(w).sum()),
                               c_prediction_changes=int(np.sum(prediction.argmax(1) != uniform.argmax(1))),
                               c_probability_max_abs_delta=float(np.max(np.abs(prediction - uniform)))))

    add('uniform', 'uniform', {}, u)
    # Preserve the original power-rule candidate set and declaration order.
    for alpha in (.25, .5, .75, 1):
        for gamma in (1, 2, 4):
            w, fallback = weights(scores, alpha, gamma)
            add(f'power-a{alpha}-g{gamma}', 'power', dict(alpha=alpha, gamma=gamma),
                w, dict(fallback=fallback))
    for beta in config['softmax_betas']:
        q = np.exp(beta * (scores - scores.max()))
        add(f'softmax-b{beta}', 'softmax', dict(beta=beta), q / q.sum())
    # Stable ties use original head order, not C or test performance.
    order = np.argsort(-scores, kind='stable')
    for k in config['top_k']:
        w = np.zeros(h)
        w[order[:k]] = 1 / k
        add(f'top-k{k}', 'top_k', dict(k=k), w)
    for penalty in config['joint_lambdas']:
        w, fit = joint_weights(p_b, y_b, g_b, penalty, config['solver'])
        add(f'joint-l{penalty}', 'joint', dict(penalty=penalty), w, fit)
    # Each family includes an exact uniform fallback. No test input is accepted.
    families = {family: best_candidate([r for r in candidates if r['family'] in ('uniform', family)])['id']
                for family in ('power', 'softmax', 'top_k', 'joint')}
    return dict(candidates=candidates, per_family_selected=families,
                selected=best_candidate(candidates)['id'], b_scores=scores.tolist())
