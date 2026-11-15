"""Binary calibrators, metrics and the validation-only selection rules.

Ported from experiments/dpe_nested_corrected/code/{nested_selection,corrected_nested}.py and
src/methods/dpe_independent/fitting.py (ResidualModel, fit_adam, predict_adam). Group means
run over the groups present, in sorted order (the Waterbirds order for Waterbirds data).
"""
import contextlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SETTINGS = [("average", 0.1), ("maximum", 0.1), ("average", 0.01), ("maximum", 0.01)]
FIXED = ("average", 0.1)
RULES = ("nll", "wga")


def name(setting):
    return f"{setting[0]}-l{setting[1]:g}"


def names_of(groups):
    return tuple(sorted(set(np.asarray(groups).tolist())))


# ----------------------------------------------------------------------------- metrics

def group_accuracy(pred, y, groups):
    groups = np.asarray(groups)
    correct = np.asarray(pred) == np.asarray(y)
    return {g: float(correct[groups == g].mean()) for g in names_of(groups)}


def record(pred, y, groups, **extra):
    acc = group_accuracy(pred, y, groups)
    groups = np.asarray(groups)
    out = dict(wga=min(acc.values()) * 100, overall=float((np.asarray(pred) == y).mean()) * 100,
               groups={g: v * 100 for g, v in acc.items()},
               counts={g: int((groups == g).sum()) for g in acc})
    out.update(extra)
    return out


def balanced_nll(prob1, y, groups):
    p = np.clip(np.asarray(prob1, dtype=np.float64), 1e-12, 1 - 1e-12)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return float(np.mean([loss[groups == g].mean() for g in names_of(groups)]))


def inner_score(prob1, y, groups):
    acc = group_accuracy(prob1 > .5, y, groups)
    return dict(wga=min(acc.values()) * 100, nll=balanced_nll(prob1, y, groups))


def choose(scores, rule):
    """scores: {setting_name: {wga, nll}}. Deterministic, validation-only."""
    order = [name(s) for s in SETTINGS]
    if rule == "nll":
        key = lambda n: (round(scores[n]["nll"], 9), -round(scores[n]["wga"], 6), order.index(n))
    elif rule == "wga":
        key = lambda n: (-round(scores[n]["wga"], 6), round(scores[n]["nll"], 9), order.index(n))
    else:
        raise ValueError(rule)
    return min((n for n in order if n in scores), key=key)


# ----------------------------------------------------------------------------- calibrators

def logit_clip(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def fit_bias(x, y, g, setting):
    """Offset-only recalibration sigmoid(logit(p) + b), group-balanced loss, no scale."""
    from scipy.optimize import minimize_scalar
    z, yy = logit_clip(x[:, 0]), np.asarray(y, dtype=np.float64)
    loss_name, penalty = setting
    names = names_of(g)

    def objective(b):
        margin = (2 * yy - 1) * (z + b)
        per = np.logaddexp(0.0, -margin)
        groups = [per[g == k].mean() for k in names]
        return (np.mean(groups) if loss_name == "average" else np.max(groups)) + penalty * b * b

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded", options=dict(xatol=1e-6))
    return dict(kind="bias", b=float(result.x))


@contextlib.contextmanager
def cpu_randomness(seed):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)


class ResidualModel(nn.Module):
    def __init__(self, kind, heads=15, hidden=8):
        super().__init__()
        if kind not in ("affine", "linear"):
            raise ValueError("Unknown residual model")
        self.kind, self.heads, self.hidden = kind, heads, hidden
        self.a = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float64))
        if kind == "linear":
            self.correction = nn.Linear(heads, 1, bias=False).double()
            nn.init.zeros_(self.correction.weight)

    def logits(self, q):
        mean = q.mean(1).clamp(1e-12, 1 - 1e-12)
        result = self.a * torch.logit(mean) + self.b
        if hasattr(self, "correction"):
            disagreement = q - q.mean(1, keepdim=True)
            result = result + (self.correction(disagreement) - self.correction(torch.zeros_like(disagreement))).squeeze(1)
        return result

    def regularizer(self):
        result = (self.a - 1).square() + self.b.square()
        if hasattr(self, "correction"):
            result = result + torch.cat([p.reshape(-1) for p in self.correction.parameters()]).square().mean()
        return result


def group_losses(values, index, count):
    return torch.stack([values[index == i].mean() for i in range(count)])


def fit_adam(inputs, y, groups, kind, setting, cfg, seed):
    """Fixed 400-step Adam fit of the affine or linear-residual map."""
    y, groups = np.asarray(y), np.asarray(groups)
    names = names_of(groups)
    x = torch.tensor(np.asarray(inputs), dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    g = torch.tensor([names.index(value) for value in groups])
    if not torch.isfinite(x).all():
        raise ValueError("Invalid fitting inputs")
    loss_name, penalty = setting
    with cpu_randomness(seed):
        model = ResidualModel(kind, x.shape[1], cfg["hidden_units"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        for step in range(cfg["steps"] + 1):
            logits = model.logits(x)
            per_example = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            losses = group_losses(per_example, g, len(names))
            data_loss = losses.mean() if loss_name == "average" else losses.max()
            objective = data_loss + penalty * model.regularizer()
            if not torch.isfinite(objective) or any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite fit; no candidate accepted")
            if step == cfg["steps"]:
                break
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        state = {k: v.detach().tolist() for k, v in model.state_dict().items()}
    return dict(kind=kind, state=state, heads=x.shape[1], hidden=cfg["hidden_units"], seed=seed, steps=cfg["steps"])


def predict_adam(model, inputs, cfg):
    with cpu_randomness(0):
        m = ResidualModel(model["kind"], model["heads"], model["hidden"])
        m.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in model["state"].items()}, strict=True)
        with torch.inference_mode():
            q = m.logits(torch.tensor(np.asarray(inputs), dtype=torch.float64)).sigmoid().numpy()
    # The original returned probabilities([1-q, q])[:, 1] after validation; that is q.
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite calibrated probabilities")
    return q


def fit_model(x, y, g, kind, setting, cfg, seed):
    if kind == "bias":
        return fit_bias(x, y, g, setting)
    return fit_adam(x, y, g, kind, setting, cfg, seed)


def predict_model(model, x, cfg):
    if model.get("kind") == "bias":
        return 1.0 / (1.0 + np.exp(-(logit_clip(x[:, 0]) + model["b"])))
    return predict_adam(model, x, cfg)


# ----------------------------------------------------------------------------- thresholds

def fit_threshold_wga(x, y, g):
    """Legacy coarse logit-offset grid (diagnostic only)."""
    z, yy = logit_clip(x[:, 0]), np.asarray(y)
    best = None
    for b in np.arange(-4.0, 4.0001, 0.05):
        pred = (z + b) > 0
        wga = min((pred[g == k] == yy[g == k]).mean() for k in names_of(g))
        key = (-round(wga, 9), abs(b))
        if best is None or key < best[0]:
            best = (key, float(b))
    return dict(kind="bias", b=best[1])


def fit_threshold_exact(x, y, g):
    """Exact empirical WGA maximiser for p > t over every distinct prediction partition."""
    p = np.asarray(x, dtype=np.float64).reshape(-1)
    y, g = np.asarray(y), np.asarray(g)
    if p.shape != y.shape or y.shape != g.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid threshold inputs")
    if (p < 0).any() or (p > 1).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected probabilities and binary labels")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    unique = np.unique(sorted_p)
    candidates = np.unique(np.r_[np.nextafter(unique, -np.inf), unique, .5])
    cut = np.searchsorted(sorted_p, candidates, side="right")
    group_acc = []
    for group in names_of(g):
        mask = g[order] == group
        changes = np.where(mask, 1 - 2 * y[order], 0)
        prefix = np.r_[0, np.cumsum(changes)]
        correct = np.sum(mask & (y[order] == 1)) + prefix[cut]
        group_acc.append(correct / mask.sum())
    wga = np.min(group_acc, axis=0)
    best = min(range(len(candidates)), key=lambda i: (-round(float(wga[i]), 12),
                                                     abs(float(candidates[i]) - .5), float(candidates[i])))
    t = float(candidates[best])
    default = min(group_accuracy(p > .5, y, g).values()) * 100
    coarse = fit_threshold_wga(p[:, None], y, g)
    coarse_wga = min(group_accuracy(logit_clip(p) + coarse["b"] > 0, y, g).values()) * 100
    return dict(kind="exact_threshold", threshold=t, fit_wga=float(wga[best] * 100),
                default_fit_wga=default, legacy_grid_fit_wga=coarse_wga,
                fit_prediction_changes=int(np.sum((p > t) != (p > .5))))


def apply_threshold(model, p):
    return np.asarray(p) > model["threshold"]
