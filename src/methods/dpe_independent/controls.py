"""Threshold and offset controls for the five-fold cross-fit ensemble.

The affine calibrator of the cross-fit study fits sigma(a*s + b) on out-of-fold scores
s = logit(mean head probability) and averages the calibrated fold ensembles at test time.
These controls ask whether the slope a matters, or whether the same decisions come from
moving the decision threshold:

* offset: a fixed at 1, b fitted with the same group loss, penalty (b^2) and optimiser;
  applied to each fold ensemble before averaging (same inference rule as affine).
* fold threshold: one threshold on the fold-average score, chosen to maximise
  out-of-fold worst-group accuracy.
* transferred offset / threshold: the offset b or threshold t learnt on the out-of-fold
  scores of the fold models, applied to the score of full-validation DPE.

Only validation labels enter any fit. Test arrays are used for scoring alone.
"""
import numpy as np
import torch
from torch.nn import functional as F

from src.methods.dpe_head_reweighting.neural_combiner import cpu_randomness
from src.methods.dpe_head_reweighting.reweighting import GROUPS
from .fitting import reduce_groups

EPS = 1e-12


def score(p1):
    """logit of a waterbird probability, clipped like the affine combiner."""
    q = np.clip(np.asarray(p1, dtype=np.float64), EPS, 1-EPS)
    return np.log(q) - np.log1p(-q)


def group_index(groups):
    groups = np.asarray(groups)
    index = np.array([GROUPS.index(g) for g in groups])
    if set(index) != {0, 1, 2, 3}:
        raise ValueError("All four groups are required")
    return index


def fit_offset(s, y, groups, loss, penalty, config, seed):
    """Fit b in sigma(s + b) with the affine combiner's objective and optimiser, a fixed at 1."""
    x = torch.tensor(np.asarray(s, dtype=np.float64))
    target = torch.tensor(np.asarray(y, dtype=np.float64))
    g = torch.tensor(group_index(groups))
    with cpu_randomness(seed):
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([b], lr=config["learning_rate"])
        for _ in range(config["steps"]):
            per_example = F.binary_cross_entropy_with_logits(x + b, target, reduction="none")
            objective = reduce_groups(per_example, g, loss) + penalty * b.square()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
    value = float(b.detach())
    if not np.isfinite(value):
        raise ValueError("Nonfinite offset fit")
    return value


def worst_group_accuracy(decisions, y, groups):
    decisions, y = np.asarray(decisions), np.asarray(y)
    index = group_index(groups)
    correct = decisions == y
    return min(float(correct[index == k].mean()) for k in range(4))


def wga_threshold(s, y, groups):
    """Threshold t for the rule s > t that maximises worst-group accuracy.

    Candidates are 0 and the midpoints between consecutive distinct scores. Ties go to the
    candidate with the smallest |t| (closest to the uncalibrated rule), then the smaller t.
    """
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
    index = group_index(groups)
    values = np.unique(s)
    candidates = np.unique(np.r_[0.0, (values[1:] + values[:-1]) / 2])
    best = None
    for t in candidates:
        correct = (s > t).astype(int) == y
        wga = min(float(correct[index == k].mean()) for k in range(4))
        key = (-wga, abs(t), t)
        if best is None or key < best[0]:
            best = (key, float(t), wga)
    return best[1], best[2]


def offset_average(fold_scores, b):
    """Mean over fold ensembles of sigma(s_k + b); returns waterbird probabilities."""
    return np.mean([1/(1+np.exp(-(np.asarray(s) + b))) for s in fold_scores], axis=0)
