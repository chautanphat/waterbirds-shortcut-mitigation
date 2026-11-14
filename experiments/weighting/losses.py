"""Differentiable objectives. Label-conditioned teacher is training-only."""
import torch
from torch.nn import functional as F


def signed_scores(p, y):
    true = p.gather(2, y[:, None, None].expand(-1, p.shape[1], 1)).squeeze(-1)
    if p.shape[-1] == 2:
        return 2 * true - 1
    wrong = p.clone()
    wrong.scatter_(2, y[:, None, None].expand(-1, p.shape[1], 1), -torch.inf)
    return true - wrong.max(-1).values


def objective(mix, w, p, y, kind='ce', kl=0., eta=0., gamma=2., tail=0.):
    true = mix.gather(1, y[:, None]).squeeze(1).clamp_min(1e-9)
    losses = -true.log()
    if kind == 'balanced':
        loss = torch.stack([losses[y == c].mean() for c in y.unique()]).mean()
    elif kind == 'brier':
        loss = (mix - F.one_hot(y, mix.shape[-1])).square().sum(1).mean()
    elif kind == 'focal':
        loss = ((1 - true) ** gamma * losses).mean()
    else:
        loss = losses.mean()
    if tail:
        loss = loss + tail * torch.stack([v.topk(max(1, int(.2 * len(v)))).values.mean()
                    for c in y.unique() for v in [losses[y == c]]]).mean()
    loss = loss + kl * (w * (w.clamp_min(1e-9).log() + __import__('math').log(w.shape[1]))).sum(1).mean()
    if eta:
        q = (5 * signed_scores(p, y)).softmax(1).detach()
        loss = loss + eta * (q * (q.clamp_min(1e-9).log() - w.clamp_min(1e-9).log())).sum(1).mean()
    return loss
