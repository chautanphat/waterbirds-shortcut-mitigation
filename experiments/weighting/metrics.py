"""Classification, calibration, paired uncertainty and post-hoc weight diagnostics."""
import numpy as np


def metrics(p, y, group=None, threshold=.5):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim != 2 or len(p) != len(y) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid ensemble output')
    if not np.allclose(p.sum(1), 1, atol=2e-6):
        raise ValueError('Unnormalized ensemble output')
    pred = (p[:, 1] > threshold).astype(int) if p.shape[1] == 2 else p.argmax(1)
    correct = pred == y
    per_class = {str(c): float(correct[y == c].mean()) for c in np.unique(y)}
    confidence = p[np.arange(len(y)), pred]
    ece = 0.
    for i in range(15):
        mask = (confidence >= i / 15) & (confidence < (i + 1) / 15 if i < 14 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    result = dict(accuracy=float(correct.mean()), balanced_accuracy=float(np.mean(list(per_class.values()))),
                  nll=float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
                  brier=float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
                  ece=float(ece), per_class=per_class, count=len(y))
    if group is not None:
        result['per_group'] = {str(g): dict(accuracy=float(correct[group == g].mean()),
                                count=int((group == g).sum())) for g in np.unique(group)}
        result['wga'] = min(v['accuracy'] for v in result['per_group'].values())
    return result


def paired_ci(p, base, y, repeats=2000, seed=0, threshold=.5):
    pred = (p[:, 1] > threshold).astype(int) if p.shape[-1] == 2 else p.argmax(1)
    delta = (pred == y).astype(float) - (base.argmax(1) == y)
    rng = np.random.default_rng(seed)
    values = [delta[rng.integers(0, len(y), len(y))].mean() * 100 for _ in range(repeats)]
    return [float(v) for v in np.quantile(values, [.025, .975])]


def weight_diagnostics(w, expert_p, y, group=None):
    if w is None:
        return dict(reason='Predictor is not a convex expert mixture')
    w = np.asarray(w)
    entropy = -(w * np.log(w.clip(1e-12))).sum(1)
    correct = expert_p.argmax(-1) == y[:, None]
    result = dict(entropy_mean=float(entropy.mean()), effective_members=float(np.exp(entropy).mean()),
                  mean_weights=w.mean(0).tolist(),
                  weight_on_correct_experts=float((w * correct).sum(1).mean()),
                  weight_on_wrong_experts=float((w * ~correct).sum(1).mean()))
    if group is not None:
        result['mean_weights_by_group'] = {str(g): w[group == g].mean(0).tolist() for g in np.unique(group)}
    return result
