"""Stable prototype competition in canonical [head, example, class] order."""
import numpy as np
import torch
from scipy.special import logsumexp, softmax
from torch.nn import functional as F

from src.methods.dpe_head_reweighting import ensemble


def distances(heads, raw, device="cpu", batch_size=256):
    ensemble.validate(heads, count=15)
    # The preserved row z-score remains float32. Normalize and accumulate L2
    # distances in float64: CPU float32 cdist drifts from the saved GPU outputs.
    x = ensemble.standardize(raw).double()
    result = []
    with torch.inference_mode():
        for prototypes in heads["prototypes"]:
            p = F.normalize(prototypes.to(device=device, dtype=torch.float64), dim=-1)
            result.append(torch.cat([
                torch.cdist(F.normalize(b.to(device), dim=-1), p, p=2,
                            compute_mode="donot_use_mm_for_euclid_dist").cpu()
                for b in x.split(batch_size)
            ]).numpy())
    return np.stack(result)


def scores(distance, scales, scale="native"):
    d, s = np.asarray(distance, dtype=np.float64), np.abs(np.asarray(scales, dtype=np.float64))
    if d.ndim != 3 or d.shape[0] != 15 or d.shape[2] != 2 or s.shape != (15,):
        raise ValueError("Expected distances [15,N,2] and 15 scales")
    if not np.isfinite(d).all() or not np.isfinite(s).all() or (d < 0).any():
        raise ValueError("Nonfinite or negative distances")
    if scale == "shared":
        s = np.full(15, np.median(s))
    elif scale != "native":
        raise ValueError("Unknown scale construction")
    return -s[:, None, None] * d


def validate_scores(z, temperature):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 3 or z.shape[0] != 15 or z.shape[-1] != 2 or not z.shape[1]:
        raise ValueError("Expected scores [15,N,2]")
    if not np.isfinite(z).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid scores or temperature")
    return z / temperature


def memberships(k, membership="cyclic", permutation_seed=0):
    if type(k) is not int or not 1 <= k <= 15:
        raise ValueError("k must be between 1 and 15")
    order = np.arange(15)
    if membership == "permuted":
        order = np.random.RandomState(permutation_seed).permutation(15)
    elif membership != "cyclic":
        raise ValueError("Unknown membership")
    return order[(np.arange(15)[:, None] + np.arange(k)) % 15]


def subset_heads(z, k, temperature=1., membership="cyclic"):
    z = validate_scores(z, temperature)
    members = memberships(k, membership)
    # [receiving head, selected pair, example, class]. Same cardinality per class.
    evidence = logsumexp(z[members], axis=1) - np.log(k)
    return softmax(evidence, axis=-1)


def global_pool(z, temperature=1., alpha=None):
    z = validate_scores(z, temperature)
    if alpha is None:
        alpha = np.full((15, 2), 1 / 15)
    alpha = np.asarray(alpha, dtype=np.float64)
    if (alpha.shape != (15, 2) or not np.isfinite(alpha).all() or (alpha < 0).any()
            or not np.allclose(alpha.sum(0), 1, atol=1e-12, rtol=0)):
        raise ValueError("Prototype weights must sum to one separately per class")
    with np.errstate(divide="ignore"):
        evidence = logsumexp(z + np.log(alpha)[:, None, :], axis=0)
    return softmax(evidence, axis=-1)


def local_topk(z, k, temperature=1.):
    z = validate_scores(z, temperature)
    if type(k) is not int or not 1 <= k <= 15:
        raise ValueError("Invalid local k")
    selected = np.partition(z, 15-k, axis=0)[15-k:]
    return softmax(logsumexp(selected, axis=0) - np.log(k), axis=-1)


def representation(z, spec):
    if spec["family"] == "original":
        return softmax(validate_scores(z, 1.), axis=-1)
    if spec["family"] == "h3":
        return subset_heads(z, spec["k"], spec["temperature"], spec["membership"])
    raise ValueError("Unknown head representation")


def usage(z, k, temperature, groups):
    """Label-free ranking; group annotations are used only to summarize usage."""
    z = validate_scores(z, temperature)
    ranks = np.argsort(-z, axis=0, kind="stable")[:k]
    result = {}
    for group in ("all", "00", "01", "10", "11"):
        mask = np.ones(z.shape[1], dtype=bool) if group == "all" else np.asarray(groups) == group
        result[group] = [np.bincount(ranks[:, mask, c].ravel(), minlength=15).tolist() for c in range(2)]
    return result
