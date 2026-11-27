"""K-class (K >= 3) versions of the calibrators, as written in PLAN.md Sect. 9.

Inputs are mean head probabilities [N, K] (bias, affine) or per-head probabilities
[N, 15, K] (linear residual). With s = log(clip(mean probability)):

  bias    : logits = s + b                        penalty  lambda * ||b||^2
  affine  : logits = a * s + b                    penalty  lambda * ((a-1)^2 + ||b||^2)
  linear  : affine + W vec(q_j - mean q)          penalty  + lambda * mean(W^2), W starts at 0

All fitted by the same 400 full-batch Adam steps (lr 0.01) on group-balanced ("average") or
worst-group ("maximum") cross-entropy. The binary offset-only map uses a bounded scalar
search instead; with K >= 3 there is no scalar threshold, so the exact-WGA threshold arms are
not defined. Decisions are the argmax.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .calibrate import cpu_randomness, group_losses, names_of


def log_mean(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0))


class MulticlassMap(nn.Module):
    def __init__(self, kind, classes, heads=15):
        super().__init__()
        if kind not in ("bias", "affine", "linear"):
            raise ValueError(kind)
        self.kind = kind
        self.b = nn.Parameter(torch.zeros(classes, dtype=torch.float64))
        if kind != "bias":
            self.a = nn.Parameter(torch.ones((), dtype=torch.float64))
        if kind == "linear":
            self.correction = nn.Linear(heads * classes, classes, bias=False).double()
            nn.init.zeros_(self.correction.weight)

    def logits(self, x):
        if self.kind == "linear":
            mean = x.mean(1)
            s = torch.log(mean.clamp(1e-12, 1.0))
            disagreement = (x - mean[:, None, :]).reshape(len(x), -1)
            return self.a * s + self.b + self.correction(disagreement)
        s = torch.log(x.clamp(1e-12, 1.0))
        return (self.a * s if hasattr(self, "a") else s) + self.b

    def regularizer(self):
        result = self.b.square().sum()
        if hasattr(self, "a"):
            result = result + (self.a - 1).square()
        if hasattr(self, "correction"):
            result = result + self.correction.weight.square().mean()
        return result


def fit(x, y, groups, kind, setting, cfg, seed):
    y, groups = np.asarray(y), np.asarray(groups)
    names = names_of(groups)
    xt = torch.tensor(np.asarray(x), dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.long)
    g = torch.tensor([names.index(v) for v in groups])
    classes = xt.shape[-1]
    loss_name, penalty = setting
    with cpu_randomness(seed):
        model = MulticlassMap(kind, classes, xt.shape[1] if kind == "linear" else 15)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        for step in range(cfg["steps"] + 1):
            per_example = F.cross_entropy(model.logits(xt), target, reduction="none")
            losses = group_losses(per_example, g, len(names))
            objective = (losses.mean() if loss_name == "average" else losses.max()) + penalty * model.regularizer()
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite multiclass fit")
            if step == cfg["steps"]:
                break
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
    return dict(kind=kind, classes=classes, heads=xt.shape[1] if kind == "linear" else 15,
                state={k: v.detach().tolist() for k, v in model.state_dict().items()}, seed=seed)


def predict(model, x):
    with cpu_randomness(0):
        m = MulticlassMap(model["kind"], model["classes"], model["heads"])
        m.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in model["state"].items()}, strict=True)
        with torch.inference_mode():
            p = m.logits(torch.tensor(np.asarray(x), dtype=torch.float64)).softmax(-1).numpy()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite multiclass probabilities")
    return p


def balanced_nll(prob, y, groups):
    p = np.clip(np.asarray(prob, dtype=np.float64)[np.arange(len(y)), y], 1e-12, 1.0)
    loss = -np.log(p)
    return float(np.mean([loss[groups == k].mean() for k in names_of(groups)]))


def inner_score(prob, y, groups):
    from .calibrate import group_accuracy
    acc = group_accuracy(np.asarray(prob).argmax(-1), y, groups)
    return dict(wga=min(acc.values()) * 100, nll=balanced_nll(prob, y, groups))
