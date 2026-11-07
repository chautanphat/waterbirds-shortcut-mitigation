"""Frozen notebook-compatible IsoMax+ heads; raw features are standardized once."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PREPROCESSING = 'raw-resnet50;row-zscore-ddof0;isomax-l2;temperature1'


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(2, dim).normal_(0, .01))
        self.distance_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return -self.distance_scale.abs() * torch.cdist(
            F.normalize(x, dim=-1), F.normalize(self.prototypes, dim=-1),
            p=2, compute_mode='donot_use_mm_for_euclid_dist')


def standardize(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or not len(raw) or raw.shape[1] < 2 or raw.dtype.kind != 'f' or not np.isfinite(raw).all():
        raise ValueError('Invalid raw features')
    sd = raw.std(axis=1, keepdims=True)
    if (sd == 0).any():
        raise ValueError('Zero-variance feature row')
    return torch.from_numpy(np.asarray((raw - raw.mean(axis=1, keepdims=True)) / sd, dtype=np.float32))


def validate(a, dim=None, count=None):
    if set(a) != {'version', 'prototypes', 'distance_scales', 'temperatures', 'head_ids',
                  'selected_epochs', 'classes', 'groups', 'preprocessing', 'provenance'}:
        raise ValueError('Unexpected ensemble fields')
    p, s, t = a['prototypes'], a['distance_scales'], a['temperatures']
    if a['version'] != 1 or a['classes'] != [0, 1] or a['groups'] != ['00', '01', '10', '11'] or a['preprocessing'] != PREPROCESSING:
        raise ValueError('Incompatible ensemble contract')
    if p.ndim != 3 or p.shape[1] != 2 or p.shape[0] < 1 or (dim is not None and p.shape[2] != dim) or (count is not None and len(p) != count):
        raise ValueError('Invalid prototype shape')
    h = len(p)
    if s.shape != (h,) or t.shape != (h,) or len(a['head_ids']) != h or len(set(a['head_ids'])) != h or len(a['selected_epochs']) != h:
        raise ValueError('Misaligned head metadata')
    if any(not torch.isfinite(x).all() for x in (p, s, t)) or not (t == 1).all() or not a['provenance']:
        raise ValueError('Invalid head parameters/provenance')
    return a


def save(path, a):
    validate(a)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in a.items()}
    torch.save(a, path)


def load(path, dim=None, count=None):
    return validate(torch.load(path, map_location='cpu', weights_only=True), dim, count)


def predict(a, raw, batch_size=256, device='cpu', logits=False):
    validate(a)
    x = standardize(raw)
    if x.shape[1] != a['prototypes'].shape[2] or batch_size < 1:
        raise ValueError('Feature dimension/batch size mismatch')
    result = []
    with torch.no_grad():
        for p, s in zip(a['prototypes'], a['distance_scales']):
            head = Head(x.shape[1]).to(device)
            head.prototypes.copy_(p.to(device))
            head.distance_scale.copy_(s.reshape(1).to(device))
            batches = [head(b.to(device)).cpu() for b in x.split(batch_size)]
            z = torch.cat(batches)
            result.append(z if logits else z.softmax(-1))
    return torch.stack(result).numpy()
