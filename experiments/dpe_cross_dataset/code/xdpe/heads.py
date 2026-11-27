"""DPE prototype heads on frozen features: training and probabilities.

Ported from src/methods/dpe_head_reweighting/{train_heads,ensemble}.py and
src/methods/dpe_independent/geometry.py with K classes and G groups. One prototype per class
per head; heads trained in sequence on group-balanced subsets (n = smallest group, groups
interleaved in decreasing size order), covariance penalty decaying by stage, and for each head
the epoch with the highest WGA on its own training pool.
"""
import random

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class Head(nn.Module):
    def __init__(self, dim, classes=2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(classes, dim).normal_(0, .01))
        self.distance_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return -self.distance_scale.abs() * torch.cdist(
            F.normalize(x, dim=-1), F.normalize(self.prototypes, dim=-1),
            p=2, compute_mode='donot_use_mm_for_euclid_dist')


def standardize(raw):
    """Per-row z-score in float32 (ddof 0), as the preserved notebook."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or not len(raw) or raw.shape[1] < 2 or raw.dtype.kind != 'f' or not np.isfinite(raw).all():
        raise ValueError('Invalid raw features')
    if raw.dtype.itemsize < 4:
        raw = raw.astype(np.float32)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        sd = raw.std(axis=1, keepdims=True)
        if (sd == 0).any():
            raise ValueError('Zero-variance feature row')
        normalized = np.asarray((raw - raw.mean(axis=1, keepdims=True)) / sd, dtype=np.float32)
    if not np.isfinite(sd).all() or not np.isfinite(normalized).all():
        raise ValueError('Feature standardization overflow')
    return torch.from_numpy(normalized)


def pool_wga(p, y, groups, names):
    correct = p.argmax(-1) == y
    return min(float(correct[groups == g].mean()) for g in names)


def train(head_raw, head_labels, head_groups, settings, seed, classes=2, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = standardize(head_raw)
    y, groups = np.asarray(head_labels), np.asarray(head_groups)
    names = tuple(sorted(set(groups)))
    if set(y) != set(range(classes)):
        raise ValueError('Head-training pool lacks a class')
    pools = [np.flatnonzero(groups == g) for g in names]
    order = sorted(range(len(names)), key=lambda i: (-len(pools[i]), i))
    n = min(map(len, pools))
    if len(names) * n < settings['batch_size']:
        raise ValueError(f'Balanced pool of {len(names)}x{n} rows gives zero batches of {settings["batch_size"]}')
    prototypes, scales, epochs = [], [], []
    for stage in range(1, settings['heads'] + 1):
        idx = np.stack([np.random.permutation(pools[i])[:n] for i in order], axis=1).ravel()
        loader = DataLoader(TensorDataset(x[idx], torch.tensor(y[idx], dtype=torch.long)),
                            batch_size=settings['batch_size'], shuffle=True, drop_last=True)
        head = Head(x.shape[1], classes).to(device)
        opt = torch.optim.SGD(head.parameters(), lr=settings['lr'], momentum=settings['momentum'])
        cov_reg = settings['cov_reg'] / (1 + .05 * max(0, stage - 2))
        best_wga, best = -1., None
        for epoch in range(1, settings['epochs'] + 1):
            for feats, labels in loader:
                opt.zero_grad()
                logits = head(feats.to(device))
                log_probs = F.log_softmax(settings['entropic_scale'] * logits, dim=1)
                loss = -log_probs[torch.arange(len(labels), device=device), labels.to(device)].mean()
                p = head.prototypes[:, None]
                loss = loss + torch.einsum('ijk,ilk->ijl', p, p).squeeze().mean() * settings['wd_weight']
                if prototypes and cov_reg > 0:
                    ps = torch.cat([p] + [v.to(device)[:, None] for v in prototypes], dim=1)
                    cov = torch.einsum('ijk,ilk->ijl', ps, ps) / (x.shape[1] - 1)
                    loss = loss + cov[:, 0, 1:].sum(1).div(ps.shape[1]).mean().abs() * cov_reg
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                opt.step()
            with torch.no_grad():
                probabilities = torch.cat([head(b.to(device)).softmax(-1).cpu() for b in x.split(256)]).numpy()
            wga = pool_wga(probabilities, y, groups, names)
            if wga >= best_wga:
                best_wga = wga
                best = (head.prototypes.detach().cpu().clone(), head.distance_scale.detach().cpu().clone(), epoch)
        prototypes.append(best[0])
        scales.append(best[1].squeeze())
        epochs.append(best[2])
    return dict(prototypes=torch.stack(prototypes), distance_scales=torch.stack(scales), selected_epochs=epochs)


def probabilities(heads, raw, batch_size=256):
    """[H, N, K] float64 probabilities, rounded through float32 like the saved caches.

    Distances in float64 on the float32 row z-score (geometry.distances), scores
    -|scale| * distance, softmax over classes."""
    x = standardize(raw).double()
    distances = []
    with torch.inference_mode():
        for prototypes in heads["prototypes"]:
            p = F.normalize(prototypes.to(dtype=torch.float64), dim=-1)
            distances.append(torch.cat([
                torch.cdist(F.normalize(b, dim=-1), p, p=2, compute_mode="donot_use_mm_for_euclid_dist")
                for b in x.split(batch_size)]).numpy())
    d = np.stack(distances)
    s = np.abs(heads["distance_scales"].numpy().astype(np.float64))
    if not np.isfinite(d).all() or (d < 0).any() or not np.isfinite(s).all():
        raise ValueError("Nonfinite distances or scales")
    z = -s[:, None, None] * d
    return softmax(z, axis=-1).astype(np.float32).astype(np.float64)
