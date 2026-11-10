"""Feature-only Stage 1 adapted from notebook cells 13/14 (see method README)."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from .ensemble import Head, PREPROCESSING, standardize
from .reweighting import GROUPS, metrics


def train(head_raw, head_labels, head_groups, settings, seed, provenance, device='cpu', progress=False):
    """Only head-training examples are accepted; epoch selection reuses those examples."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = standardize(head_raw)
    y, groups = np.asarray(head_labels), np.asarray(head_groups)
    metrics(np.full((len(y), 2), .5), y, groups)
    pools = [np.flatnonzero(groups == g) for g in GROUPS]
    # Upstream samples groups in decreasing group-size order, interleaving rows.
    order = sorted(range(4), key=lambda i: (-len(pools[i]), i))
    n = min(map(len, pools))
    if 4 * n < settings['batch_size']:
        raise ValueError('Balanced head-training data produces zero optimizer batches with drop_last=True')
    prototypes, scales, epochs = [], [], []
    for stage in range(1, settings['heads'] + 1):
        if progress:
            print(f'Head {stage}/{settings["heads"]}: training on {4*n} balanced head-training examples', flush=True)
        idx = np.stack([np.random.permutation(pools[i])[:n] for i in order], axis=1).ravel()
        loader = DataLoader(TensorDataset(x[idx], torch.tensor(y[idx], dtype=torch.long)),
                            batch_size=settings['batch_size'], shuffle=True, drop_last=True)
        head = Head(x.shape[1]).to(device)
        opt = torch.optim.SGD(head.parameters(), lr=settings['lr'], momentum=settings['momentum'])
        # Notebook scheduler duplicates the initial coefficient for stages 1 and 2.
        cov_reg = settings['cov_reg'] / (1 + .05 * max(0, stage - 2))
        best_wga, best = -1., None
        for epoch in range(1, settings['epochs'] + 1):
            for feats, labels in loader:
                opt.zero_grad()
                logits = head(feats.to(device))
                # Same entropic objective as log(softmax), without softmax underflow.
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
                head_probabilities = torch.cat([head(b.to(device)).softmax(-1).cpu() for b in x.split(256)]).numpy()
            wga = metrics(head_probabilities, y, groups)['wga']
            if wga >= best_wga:
                best_wga = wga
                best = (head.prototypes.detach().cpu().clone(), head.distance_scale.detach().cpu().clone(), epoch)
        prototypes.append(best[0])
        scales.append(best[1].squeeze())
        epochs.append(best[2])
        if progress:
            print(f'Head {stage}: selected epoch {best[2]}, head-training WGA {best_wga:.4f}', flush=True)
    return dict(version=1, prototypes=torch.stack(prototypes), distance_scales=torch.stack(scales),
                temperatures=torch.ones(len(scales)), head_ids=[f'head-{i:02d}' for i in range(len(scales))],
                selected_epochs=epochs, classes=[0, 1], groups=list(GROUPS),
                preprocessing=PREPROCESSING, provenance=provenance)
