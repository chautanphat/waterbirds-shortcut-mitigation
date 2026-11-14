"""Inference-only gate features, with explicit feature-family removal."""
import numpy as np
import torch
from torch import nn


def features(data, *, context=False, distance=False, embedding=False, remove=(), projection=None):
    p = data['probs']
    n, m, c = p.shape
    ordered = np.sort(p, axis=-1)
    parts, names = [], []
    def add(name, value):
        if name not in remove:
            parts.append(value)
            names.extend([name] * value.shape[-1])
    add('probabilities', p)
    add('confidence', ordered[..., -1:])
    add('margin', ordered[..., -1:] - ordered[..., -2:-1])
    add('entropy', -(p * np.log(p.clip(1e-9))).sum(-1, keepdims=True))
    add('gap', p - p.mean(1, keepdims=True))
    if context:
        mean = p.mean(1)
        stats = np.concatenate([p.var(1), p.max(1) - p.min(1),
                   -(mean * np.log(mean.clip(1e-9))).sum(-1, keepdims=True),
                   np.stack([(p.argmax(-1) == k).mean(1) for k in range(c)], -1)], -1)
        add('context', np.repeat(stats[:, None], m, axis=1))
    if distance:
        d = np.sort(data['distances'], axis=-1)
        add('distance', np.concatenate([d[..., :1], d[..., 1:2] - d[..., :1]], -1))
    if embedding:
        z = data['embedding'] @ projection
        add('embedding', np.repeat(z[:, None], m, axis=1))
    return np.concatenate(parts, -1).astype('float32'), names


class Gate(nn.Module):
    def __init__(self, members, dim, capacity='32x16', joint=False, dropout=0.):
        super().__init__()
        self.joint = joint
        widths = {'linear': [], '16': [16], '32x16': [32, 16], '64x32': [64, 32]}[capacity]
        sizes = [members * dim if joint else dim] + widths + [members if joint else 1]
        layers = []
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.bias = nn.Parameter(torch.zeros(members))

    def forward(self, x):
        v = self.net(x.flatten(1)) if self.joint else self.net(x).squeeze(-1)
        return (v + self.bias).softmax(1)
