"""Direction 2 · Phase C · Step 7d: Zero-GPU Statistical & Threshold-Sensitivity Post-Hoc Audit.

Runs 100% locally on CPU from the saved Step 7c probability caches (`step7c_*_probs.npz`)
and verified CSV summaries. Produces:
  1. Waterbirds threshold-sensitivity curves & post-hoc diagnostic representation ceiling
     (`WGA(t)` over `t in [0.05, 0.95]`, clearly separated from pre-registered val-calibrated
     operating points).
  2. Group-stratified paired bootstrap 95% confidence intervals (2,000 replicates) and
     exact seed win counts across key Waterbirds (ResNet-50, ViT-B/16) and Spawrious OOD
     method comparisons.
  3. Spawrious (`o2o_hard`) 4x4 confusion & spurious-background shortcut attribution rates
     quantifying how often OOD errors fall directly on the training background's distractor class.

Zero test-set adaptation or checkpoint modification is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.rep_repair_phase_c_step6 import write_csv
from scripts.analysis.rep_repair_step7c_waterbirds_test import (
    calibrate_val_threshold,
    load_waterbirds_test_metadata,
)
from src.datasets.spawrious_o2o import CLASSES, TEST_SHIFTED_BG, TRAIN_SPURIOUS_BG
from src.methods.rep_repair.phase_a_data import (
    GROUPS,
    apply_label_condition,
    file_sha256,
    group_metrics,
    load_val_metadata,
)

THRESH_GRID = np.linspace(0.05, 0.95, 181)

RESNET_STD_ARMS = ("erm_raw", "dfr_l2", "dar", "pre_l4_gate_only", "affine_dfr_linear", "pre_l4_repair_w005")
RESNET_STAR_ARMS = ("erm_star_raw", "erm_star_dfr_l2")
VIT_ARMS = ("vit_erm_raw", "vit_dfr_l2", "vit_ln_dfr_late", "vit_pre_b6_gate_only", "vit_pre_b6_repair_w005")
SPAWRIOUS_ARMS = ("erm_raw", "dfr_l2", "dar", "affine_dfr_bn_l4", "affine_dfr_bn_all", "pre_l4_repair_w005")

# Alphabetical order of location under each domain in load_o2o_hard_ood_test_rows:
# domain 0 -> desert (labrador=3), jungle (corgi=1), mountain (bulldog=0), snow (dachshund=2)
# domain 1 -> desert (labrador=3), jungle (corgi=1), mountain (bulldog=0), snow (dachshund=2)
SPAWRIOUS_OOD_BLOCK_CLASSES = (3, 1, 0, 2, 3, 1, 0, 2)
SPAWRIOUS_OOD_BLOCK_SIZE = 3168

# Map each OOD true class index -> distractor class index trained on that OOD background.
_BG_TO_TRAIN_CLASS = {bg: CLASSES.index(cls_name) for cls_name, bg in TRAIN_SPURIOUS_BG.items()}
SPAWRIOUS_SPUR_DISTRACTOR = {
    CLASSES.index(cls_name): _BG_TO_TRAIN_CLASS[TEST_SHIFTED_BG[cls_name]] for cls_name in CLASSES
}


def build_spawrious_ood_labels() -> np.ndarray:
    """Reconstruct exact 25,344-element OOD class labels matching sorted path order."""
    return np.concatenate([np.full(SPAWRIOUS_OOD_BLOCK_SIZE, c, dtype=int) for c in SPAWRIOUS_OOD_BLOCK_CLASSES])


def sweep_wga_curve(p1: np.ndarray, y: np.ndarray, g_str: np.ndarray, thresholds: np.ndarray = THRESH_GRID) -> np.ndarray:
    """Vectorized exact Waterbirds WGA (%) curve over threshold grid using float64 >= comparison."""
    p1 = np.asarray(p1, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    g_str = np.asarray(g_str).astype(str)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    # Map each example into one of the 4 canonical Waterbirds groups.
    order = np.argsort(p1, kind="mergesort")
    p_sorted = p1[order]
    y_sorted = y[order]
    g_sorted = g_str[order]

    # First index in sorted order with p_sorted[i] >= t -> examples [k:] are predicted 1, [:k] predicted 0.
    cut_idx = np.searchsorted(p_sorted, thresholds, side="left")
    n = len(p1)
    min_acc = np.full(len(thresholds), 100.0, dtype=np.float64)
    for grp in GROUPS:
        mask = g_sorted == grp
        cnt = int(mask.sum())
        if cnt == 0:
            min_acc[:] = float("nan")
            continue
        grp_y = int(grp[0])  # "00","01" -> y=0; "10","11" -> y=1 (always consistent in Waterbirds)
        if not np.all(y_sorted[mask] == grp_y):
            # Generic fallback if a synthetic group ever mixes labels.
            cum_y0 = np.concatenate([[0], np.cumsum((mask & (y_sorted == 0)).astype(np.int64))])
            total_y1 = int((mask & (y_sorted == 1)).sum())
            cum_y1 = np.concatenate([[0], np.cumsum((mask & (y_sorted == 1)).astype(np.int64))])
            correct = cum_y0[cut_idx] + (total_y1 - cum_y1[cut_idx])
        elif grp_y == 0:
            cum_m = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
            correct = cum_m[cut_idx]
        else:
            cum_m = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
            correct = cnt - cum_m[cut_idx]
        acc = (correct.astype(np.float64) / float(cnt)) * 100.0
        min_acc = np.minimum(min_acc, acc)
    return min_acc


def fast_binary_auc_pct(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Fast exact binary AUROC (%) equivalent to sklearn.metrics.roc_auc_score * 100 using C rankdata."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return auc * 100.0


def _prepare_sorted_auc_state(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """Precompute sort order and tie-group ids once for O(N) weighted bootstrap AUROC."""
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s_ord = scores[order]
    y_ord = y_true[order]
    diff = np.empty(len(s_ord), dtype=bool)
    diff[0] = True
    diff[1:] = s_ord[1:] != s_ord[:-1]
    has_ties = not bool(diff.all())
    g_id = (np.cumsum(diff) - 1) if has_ties else None
    n_u = int(g_id[-1]) + 1 if has_ties else len(s_ord)
    return order, y_ord, has_ties, g_id, n_u


def _weighted_auc_from_state(state: tuple, weights: np.ndarray) -> float:
    """Compute exact weighted AUROC (%) in O(N) from precomputed sort state."""
    order, y_ord, has_ties, g_id, n_u = state
    w_ord = weights[order]
    pos_w = w_ord * y_ord
    neg_w = w_ord - pos_w
    tot_pos = float(pos_w.sum())
    tot_neg = float(neg_w.sum())
    if tot_pos <= 0.0 or tot_neg <= 0.0:
        return float("nan")
    if not has_ties:
        cum_neg_inclusive = np.cumsum(neg_w)
        rank_pairs = float((pos_w * (cum_neg_inclusive - 0.5 * neg_w)).sum())
    else:
        upos = np.bincount(g_id, weights=pos_w, minlength=n_u)
        uneg = np.bincount(g_id, weights=neg_w, minlength=n_u)
        cum_neg_inclusive = np.cumsum(uneg)
        rank_pairs = float((upos * (cum_neg_inclusive - 0.5 * uneg)).sum())
    return (rank_pairs / (tot_pos * tot_neg)) * 100.0


def bootstrap_arm_metrics(
    arm_weight_fns: dict[str, object],
    strata_indices: list[np.ndarray],
    n_total: int,
    *,
    n_boot: int = 2000,
    seed: int = 20261124,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute observed [n_seeds] and bootstrap [n_boot, n_seeds] arrays once per arm from shared weights."""
    full_w = np.ones(n_total, dtype=np.float64)
    obs = {name: np.asarray(fn(full_w), dtype=np.float64) for name, fn in arm_weight_fns.items()}
    rng = np.random.default_rng(seed)
    boot = {name: np.empty((n_boot, len(obs[name])), dtype=np.float64) for name in arm_weight_fns}
    for b in range(n_boot):
        w = np.zeros(n_total, dtype=np.float64)
        for idx in strata_indices:
            drawn = idx[rng.integers(0, len(idx), size=len(idx))]
            w += np.bincount(drawn, minlength=n_total)
        for name, fn in arm_weight_fns.items():
            boot[name][b] = fn(w)
    return obs, boot


def summarize_paired_bootstrap(
    obs_a: np.ndarray,
    boot_a: np.ndarray,
    obs_b: np.ndarray,
    boot_b: np.ndarray,
) -> dict:
    """Summarize paired seed-averaged bootstrap distribution from precomputed arm bootstrap arrays."""
    obs_diff = np.asarray(obs_a, dtype=np.float64) - np.asarray(obs_b, dtype=np.float64)
    obs_delta = float(obs_diff.mean())
    boot_deltas = (np.asarray(boot_a, dtype=np.float64) - np.asarray(boot_b, dtype=np.float64)).mean(axis=1)
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    p_pos = float((boot_deltas <= 0.0).mean())
    p_neg = float((boot_deltas >= 0.0).mean())
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    wins = int((obs_diff > 0).sum())
    return {
        "mean_a": float(np.mean(obs_a)),
        "mean_b": float(np.mean(obs_b)),
        "delta_mean": obs_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "boot_se": float(boot_deltas.std(ddof=1)),
        "p_two_sided": float(p_two),
        "seed_wins": f"{wins}/{len(obs_diff)}",
        "per_seed_deltas": [round(float(x), 4) for x in obs_diff],
    }


def spawrious_confusion_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute OOD mean acc, worst-class acc, 4x4 confusion matrix (%), and spurious-BG error attribution."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    preds = probs.argmax(axis=1)

    per_class_acc = np.array([(preds[y_true == c] == c).mean() * 100.0 for c in range(4)], dtype=np.float64)
    conf_pct = np.zeros((4, 4), dtype=np.float64)
    for c in range(4):
        m = y_true == c
        counts = np.bincount(preds[m], minlength=4)
        conf_pct[c] = (counts / float(m.sum())) * 100.0

    spur_rates = {c: float(conf_pct[c, SPAWRIOUS_SPUR_DISTRACTOR[c]]) for c in range(4)}
    macro_spur = float(np.mean(list(spur_rates.values())))
    macro_err = 100.0 - float(per_class_acc.mean())
    macro_non_spur = macro_err - macro_spur
    spur_share_of_errors = (macro_spur / macro_err * 100.0) if macro_err > 0 else 0.0

    macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")) * 100.0
    return {
        "ood_mean_acc": float(per_class_acc.mean()),
        "ood_worst_class_acc": float(per_class_acc.min()),
        "ood_macro_auc": macro_auc,
        "conf_pct": conf_pct,
        "spur_rates": spur_rates,
        "ood_spur_confusion_mean": macro_spur,
        "ood_non_spur_err_mean": macro_non_spur,
        "spur_share_of_errors": spur_share_of_errors,
    }


def _calibrate_val_balanced_minority_threshold(val_p1: np.ndarray, y_val: np.ndarray, g_val: np.ndarray) -> float:
    """Post-hoc diagnostic validation threshold maximizing smooth mean minority accuracy 0.5*(g01 + g10)."""
    best_t, best_score = 0.5, -1.0
    for t in THRESH_GRID:
        m = group_metrics(y_val, (val_p1 >= t).astype(int), g_val)
        score = 0.5 * (m["groups"]["01"] + m["groups"]["10"])
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def _verify_npz_csv_alignment(
    wb_npz: np.lib.npyio.NpzFile,
    wb_resnet_csv: Path,
    wb_vit_csv: Path,
    y_test_rel: np.ndarray,
    g_test_rel: np.ndarray,
    sp_npz: np.lib.npyio.NpzFile,
    sp_csv: Path,
    y_sp_ood: np.ndarray,
) -> dict[str, float]:
    """Fail-fast check that NPZ arrays match recorded Step 7c CSV summaries to < 1e-3."""
    max_wb_resnet_err = 0.0
    with Path(wb_resnet_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            prefix = "resnet_std" if r["cohort"] == "standard_erm" else "resnet_star"
            p1 = wb_npz[f"{prefix}_s{r['seed']}__{r['arm']}__test"][:, 1]
            t = float(r["val_thresh"])
            m = group_metrics(y_test_rel, (p1 >= t).astype(int), g_test_rel)
            max_wb_resnet_err = max(max_wb_resnet_err, abs(m["wga"] - float(r["rel_val_thr_wga"])))

    max_wb_vit_err = 0.0
    with Path(wb_vit_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p1 = wb_npz[f"vit_s{r['seed']}__{r['arm']}__test"][:, 1]
            t = float(r["val_thresh"])
            m = group_metrics(y_test_rel, (p1 >= t).astype(int), g_test_rel)
            max_wb_vit_err = max(max_wb_vit_err, abs(m["wga"] - float(r["rel_val_thr_wga"])))

    max_sp_err = 0.0
    with Path(sp_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            probs = sp_npz[f"spawrious_s{r['seed']}__{r['arm']}__ood"]
            res = spawrious_confusion_metrics(probs, y_sp_ood)
            preds = probs.argmax(axis=1)
            accs = [(preds[y_sp_ood == c] == c).mean() * 100.0 for c in range(4)]
            exp = [
                float(r["ood_bulldog_mountain"]),
                float(r["ood_corgi_jungle"]),
                float(r["ood_dachshund_snow"]),
                float(r["ood_labrador_desert"]),
            ]
            max_sp_err = max(
                max_sp_err,
                abs(res["ood_mean_acc"] - float(r["ood_mean_acc"])),
                *[abs(a - e) for a, e in zip(accs, exp)],
            )

    if max_wb_resnet_err >= 1e-3 or max_wb_vit_err >= 1e-3 or max_sp_err >= 1e-3:
        raise AssertionError(
            f"Alignment assertion failed: wb_resnet={max_wb_resnet_err}, wb_vit={max_wb_vit_err}, sp={max_sp_err}"
        )
    return {
        "max_wb_resnet_wga_diff": float(max_wb_resnet_err),
        "max_wb_vit_wga_diff": float(max_wb_vit_err),
        "max_spawrious_acc_diff": float(max_sp_err),
    }


def paired_stratified_bootstrap(
    values_a_per_seed_fn,
    values_b_per_seed_fn,
    strata_indices: list[np.ndarray],
    *,
    n_boot: int = 2000,
    seed: int = 20261124,
) -> dict:
    """Compute paired group-stratified bootstrap CI and two-sided p-value over seed-averaged deltas.

    `values_*_per_seed_fn(boot_idx)` returns a 1D array of shape `[n_seeds]` for bootstrap sample `boot_idx`.
    """
    full_idx = np.concatenate(strata_indices)
    obs_a = np.asarray(values_a_per_seed_fn(full_idx), dtype=np.float64)
    obs_b = np.asarray(values_b_per_seed_fn(full_idx), dtype=np.float64)
    obs_diff_per_seed = obs_a - obs_b
    obs_delta = float(obs_diff_per_seed.mean())
    n_seeds = len(obs_diff_per_seed)
    wins = int((obs_diff_per_seed > 0).sum())

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = [idx[rng.integers(0, len(idx), size=len(idx))] for idx in strata_indices]
        boot_idx = np.concatenate(sampled)
        da = np.asarray(values_a_per_seed_fn(boot_idx), dtype=np.float64)
        db = np.asarray(values_b_per_seed_fn(boot_idx), dtype=np.float64)
        boot_deltas[b] = float((da - db).mean())

    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    p_pos = float((boot_deltas <= 0.0).mean())
    p_neg = float((boot_deltas >= 0.0).mean())
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "mean_a": float(obs_a.mean()),
        "mean_b": float(obs_b.mean()),
        "delta_mean": obs_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "boot_se": float(boot_deltas.std(ddof=1)),
        "p_two_sided": float(p_two),
        "seed_wins": f"{wins}/{n_seeds}",
        "per_seed_deltas": [round(float(x), 4) for x in obs_diff_per_seed],
    }


def _waterbirds_weight_evaluator(
    p1_test_list: list[np.ndarray],
    val_thrs: list[float],
    y_rel: np.ndarray,
    g_rel: np.ndarray,
    y_off: np.ndarray,
    g_off: np.ndarray,
    metric: str,
):
    """Return fast weighted closure `fn(weights) -> np.ndarray[n_seeds]` for a Waterbirds test metric."""
    min_rel_mask = np.isin(g_rel, ("01", "10")).astype(np.float64)
    g_rel_id = np.array([GROUPS.index(x) for x in g_rel], dtype=int)
    g_off_id = np.array([GROUPS.index(x) for x in g_off], dtype=int)

    if metric == "rel_min_auc":
        sub_idx = np.flatnonzero(min_rel_mask > 0.5)
        y_sub = y_rel[sub_idx]
        states = [_prepare_sorted_auc_state(y_sub, p1[sub_idx]) for p1 in p1_test_list]

        def eval_w(w: np.ndarray) -> np.ndarray:
            w_sub = w[sub_idx]
            return np.array([_weighted_auc_from_state(st, w_sub) for st in states], dtype=np.float64)

        return eval_w

    if metric == "rel_all_auc":
        states = [_prepare_sorted_auc_state(y_rel, p1) for p1 in p1_test_list]

        def eval_w(w: np.ndarray) -> np.ndarray:
            return np.array([_weighted_auc_from_state(st, w) for st in states], dtype=np.float64)

        return eval_w

    if metric in ("rel_raw_wga", "rel_val_thr_wga", "off_val_thr_wga"):
        is_off = metric == "off_val_thr_wga"
        y_tgt = y_off if is_off else y_rel
        g_tgt_id = g_off_id if is_off else g_rel_id
        thrs = [0.5] * len(p1_test_list) if metric == "rel_raw_wga" else val_thrs
        correct_masks = [((p1 >= t).astype(int) == y_tgt).astype(np.float64) for p1, t in zip(p1_test_list, thrs)]

        def eval_w(w: np.ndarray) -> np.ndarray:
            denom = np.bincount(g_tgt_id, weights=w, minlength=4)
            out = np.empty(len(correct_masks), dtype=np.float64)
            for s, cm in enumerate(correct_masks):
                num = np.bincount(g_tgt_id, weights=w * cm, minlength=4)
                out[s] = float((num / denom).min() * 100.0)
            return out

        return eval_w

    if metric == "rel_diag_ceil_wga":
        curve_states = []
        for p1 in p1_test_list:
            order = np.argsort(p1, kind="mergesort")
            p_ord = p1[order]
            cut_idx = np.searchsorted(p_ord, THRESH_GRID, side="left")
            g_ord = g_rel_id[order]
            grp_masks = [(g_ord == gi).astype(np.float64) for gi in range(4)]
            curve_states.append((order, cut_idx, grp_masks))

        def eval_w(w: np.ndarray) -> np.ndarray:
            out = np.empty(len(curve_states), dtype=np.float64)
            for s, (order, cut_idx, grp_masks) in enumerate(curve_states):
                w_ord = w[order]
                min_acc = np.full(len(THRESH_GRID), 100.0, dtype=np.float64)
                for gi, gm in enumerate(grp_masks):
                    wm = w_ord * gm
                    tot = float(wm.sum())
                    cum = np.concatenate([[0.0], np.cumsum(wm)])
                    corr = cum[cut_idx] if gi < 2 else (tot - cum[cut_idx])
                    min_acc = np.minimum(min_acc, (corr / tot) * 100.0)
                out[s] = float(min_acc.max())
            return out

        return eval_w

    raise ValueError(f"Unsupported Waterbirds metric: {metric}")


def _spawrious_weight_evaluator(probs_list: list[np.ndarray], y_ood: np.ndarray, metric: str):
    """Return fast weighted closure `fn(weights) -> np.ndarray[n_seeds]` for a Spawrious OOD metric."""
    if metric in ("ood_mean_acc", "ood_worst_class_acc", "ood_spur_confusion_mean"):
        preds_list = [probs.argmax(axis=1) for probs in probs_list]
        if metric == "ood_spur_confusion_mean":
            distractors = np.array([SPAWRIOUS_SPUR_DISTRACTOR[c] for c in y_ood], dtype=int)
            masks = [(pr == distractors).astype(np.float64) for pr in preds_list]
        else:
            masks = [(pr == y_ood).astype(np.float64) for pr in preds_list]

        def eval_w(w: np.ndarray) -> np.ndarray:
            denom = np.bincount(y_ood, weights=w, minlength=4)
            out = np.empty(len(masks), dtype=np.float64)
            for s, mk in enumerate(masks):
                rates = (np.bincount(y_ood, weights=w * mk, minlength=4) / denom) * 100.0
                out[s] = float(rates.min() if metric == "ood_worst_class_acc" else rates.mean())
            return out

        return eval_w

    if metric == "ood_macro_auc":
        class_states = [
            [_prepare_sorted_auc_state((y_ood == c).astype(int), probs[:, c]) for c in range(4)]
            for probs in probs_list
        ]

        def eval_w(w: np.ndarray) -> np.ndarray:
            out = np.empty(len(class_states), dtype=np.float64)
            for s, states4 in enumerate(class_states):
                out[s] = float(np.mean([_weighted_auc_from_state(st, w) for st in states4]))
            return out

        return eval_w

    raise ValueError(f"Unsupported Spawrious metric: {metric}")


def _render_step7d_figure(
    out_png: Path,
    curves_by_arm: dict[str, np.ndarray],
    erm_star_ceil: float,
    ci_rows: list[dict],
    sp_agg: dict[str, dict],
) -> None:
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2), dpi=220)

    # Panel A: Waterbirds ResNet-50 threshold sensitivity curves WGA(t)
    ax = axes[0]
    curve_styles = [
        ("pre_l4_repair_w005", "Pre-L4 Repair (ours)", "#1f5a99", "-", 2.3),
        ("affine_dfr_linear", "Affine-DFR (bn_all)", "#d95f02", "--", 2.0),
        ("pre_l4_gate_only", "Pre-L4 Gate Only", "#2a7b4c", "-.", 1.8),
        ("dfr_l2", "DFR-L2 (GAP)", "#6a51a3", ":", 1.8),
        ("dar", "DAR (layer4.2)", "#7f7f7f", (0, (3, 1, 1, 1)), 1.6),
    ]
    for arm, label, color, ls, lw in curve_styles:
        mat = curves_by_arm[arm]  # [5, 181]
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1)
        ax.plot(THRESH_GRID, mu, label=f"{label} (max {mat.max(axis=1).mean():.2f}%)", color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(THRESH_GRID, mu - sd, mu + sd, color=color, alpha=0.12)
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.515, 92.6, "t = 0.50\n(ours #1:\n91.22%)", fontsize=8, color="#222222", bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5))
    ax.axhline(erm_star_ceil, color="#111111", linestyle=":", linewidth=1.2)
    ax.text(0.07, erm_star_ceil + 0.25, f"400-ep ERM* ceiling ({erm_star_ceil:.2f}%)", fontsize=8, color="#111111")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(82.5, 96.3)
    ax.set_xlabel("Decision Threshold t on P(waterbird)")
    ax.set_ylabel("Cleaned Held-Out Test WGA (%)")
    ax.set_title("A. Waterbirds Threshold Curve (5 Seeds ± 1 SD)")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.92)

    # Panel B: Paired Stratified Bootstrap 95% CI Forest Plot
    ax = axes[1]
    forest_specs = [
        ("resnet_std", "pre_l4_repair_w005", "affine_dfr_linear", "rel_min_auc", "WB Min-AUC: Repair − Affine"),
        ("resnet_std", "pre_l4_repair_w005", "affine_dfr_linear", "rel_raw_wga", "WB Raw WGA: Repair − Affine"),
        ("resnet_std", "pre_l4_repair_w005", "affine_dfr_linear", "rel_diag_ceil_wga", "WB Ceil WGA: Repair − Affine"),
        ("resnet_std", "pre_l4_repair_w005", "affine_dfr_linear", "rel_val_thr_wga", "WB ValThr WGA: Repair − Affine"),
        ("resnet_std", "pre_l4_gate_only", "dar", "rel_raw_wga", "WB Raw WGA: PreL4Gate − DAR"),
        ("spawrious", "pre_l4_repair_w005", "affine_dfr_bn_all", "ood_mean_acc", "Spawrious Mean: Repair − AffineAll"),
        ("spawrious", "pre_l4_repair_w005", "affine_dfr_bn_all", "ood_worst_class_acc", "Spawrious Worst: Repair − AffineAll"),
        ("spawrious", "pre_l4_repair_w005", "dfr_l2", "ood_mean_acc", "Spawrious Mean: Repair − DFR-L2"),
        ("spawrious", "pre_l4_repair_w005", "dfr_l2", "ood_worst_class_acc", "Spawrious Worst: Repair − DFR-L2"),
    ]
    lookup = {(r["cohort"], r["arm_a"], r["arm_b"], r["metric"]): r for r in ci_rows}
    y_pos = np.arange(len(forest_specs))[::-1]
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    for y_i, (coh, a, b, met, lbl) in zip(y_pos, forest_specs):
        row = lookup[(coh, a, b, met)]
        delta = float(row["delta_mean"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        color = "#1f5a99" if lo > 0 else ("#c44e52" if hi < 0 else "#666666")
        ax.errorbar(delta, y_i, xerr=[[delta - lo], [hi - delta]], fmt="o", color=color, ecolor=color, capsize=3, markersize=5.5, elinewidth=1.6)
        label_x = (hi + 0.18) if hi > 0 else (lo - 0.18)
        ha = "left" if hi > 0 else "right"
        ax.text(label_x, y_i, f"{delta:+.2f} [{row['seed_wins']}]", va="center", ha=ha, fontsize=7.6, color="#222222", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[4] for s in forest_specs], fontsize=8.0)
    ax.set_xlim(-4.6, 8.6)
    ax.set_xlabel("Paired Delta (pp) with 95% Stratified Bootstrap CI")
    ax.set_title("B. Paired 95% Bootstrap CIs [Seed Wins]")
    ax.grid(axis="x", alpha=0.22)

    # Panel C: Spawrious OOD Spurious-Background Confusion Rates
    ax = axes[2]
    arms_c = ("erm_raw", "dfr_l2", "dar", "affine_dfr_bn_l4", "affine_dfr_bn_all", "pre_l4_repair_w005")
    arm_labels = ("ERM", "DFR-L2", "DAR", "Affine\nBN-L4", "Affine\nBN-All", "Pre-L4\nRepair")
    x = np.arange(len(arms_c))
    w = 0.25
    dach_vals = [sp_agg[a]["spur_dach_snow_mean"] for a in arms_c]
    bull_vals = [sp_agg[a]["spur_bull_mtn_mean"] for a in arms_c]
    macro_vals = [sp_agg[a]["ood_spur_confusion_mean"] for a in arms_c]
    b1 = ax.bar(x - w, dach_vals, width=w, color="#d95f02", label="dach@snow -> lab(snow)")
    b2 = ax.bar(x, bull_vals, width=w, color="#7570b3", label="bull@mtn -> dach(mtn)")
    b3 = ax.bar(x + w, macro_vals, width=w, color="#1f5a99", label="Macro Spur-BG Error")
    for bars in (b1, b3):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(arm_labels, fontsize=8.2)
    ax.set_ylim(0.0, 51.0)
    ax.set_ylabel("Spurious Distractor Error Rate (%)")
    ax.set_title("C. Spawrious OOD Spurious-BG Confusion")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.92)

    fig.suptitle("Direction 2 · Phase C · Step 7d Zero-GPU Statistical & Threshold-Sensitivity Audit", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run_audit(
    wb_metadata: Path,
    wb_npz_path: Path,
    wb_resnet_csv: Path,
    wb_vit_csv: Path,
    sp_npz_path: Path,
    sp_csv: Path,
    output_dir: Path,
    *,
    n_boot: int = 2000,
    seed: int = 20261124,
) -> dict:
    """Execute complete zero-GPU Step 7d audit and write CSV/JSON/PNG/README artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_rel = apply_label_condition(load_val_metadata(wb_metadata), "relabel")
    wb_test_rows = load_waterbirds_test_metadata(wb_metadata)
    _, test_off = apply_label_condition(wb_test_rows, "official")
    _, test_rel = apply_label_condition(wb_test_rows, "relabel")

    y_val_rel = np.array([r["y"] for r in val_rel], dtype=int)
    g_val_rel = np.array([r["group"] for r in val_rel], dtype=str)
    y_test_off = np.array([r["y"] for r in test_off], dtype=int)
    g_test_off = np.array([r["group"] for r in test_off], dtype=str)
    y_test_rel = np.array([r["y"] for r in test_rel], dtype=int)
    g_test_rel = np.array([r["group"] for r in test_rel], dtype=str)
    y_sp_ood = build_spawrious_ood_labels()

    wb_npz = np.load(wb_npz_path)
    sp_npz = np.load(sp_npz_path)

    alignment = _verify_npz_csv_alignment(
        wb_npz, wb_resnet_csv, wb_vit_csv, y_test_rel, g_test_rel, sp_npz, sp_csv, y_sp_ood
    )
    print(f"Alignment verified: {alignment}", flush=True)

    # 1. Waterbirds threshold-sensitivity & post-hoc representation ceiling
    wb_per_seed_rows: list[dict] = []
    wb_agg_rows: list[dict] = []
    curves_by_arm: dict[str, np.ndarray] = {}
    val_thrs_cache: dict[tuple[str, str], list[float]] = {}
    p1_test_cache: dict[tuple[str, str], list[np.ndarray]] = {}

    cohorts = (
        ("resnet_std", 5, RESNET_STD_ARMS),
        ("resnet_star", 2, RESNET_STAR_ARMS),
        ("vit", 3, VIT_ARMS),
    )
    min_rel_mask = np.isin(g_test_rel, ("01", "10"))
    for cohort, n_seeds, arms in cohorts:
        for arm in arms:
            val_thrs, p1_tests = [], []
            arm_curves = []
            arm_seed_rows = []
            for s in range(n_seeds):
                pv = wb_npz[f"{cohort}_s{s}__{arm}__val"][:, 1]
                pt = wb_npz[f"{cohort}_s{s}__{arm}__test"][:, 1]
                t_wga = calibrate_val_threshold(pv, y_val_rel, g_val_rel)
                t_bal = _calibrate_val_balanced_minority_threshold(pv, y_val_rel, g_val_rel)

                curve = sweep_wga_curve(pt, y_test_rel, g_test_rel, THRESH_GRID)
                best_idx = int(np.argmax(curve))
                ceil_wga = float(curve[best_idx])
                best_t = float(THRESH_GRID[best_idx])

                off_thr_wga = group_metrics(y_test_off, (pt >= t_wga).astype(int), g_test_off)["wga"]
                rel_raw_wga = group_metrics(y_test_rel, (pt >= 0.5).astype(int), g_test_rel)["wga"]
                rel_thr_wga = group_metrics(y_test_rel, (pt >= t_wga).astype(int), g_test_rel)["wga"]
                rel_bal_wga = group_metrics(y_test_rel, (pt >= t_bal).astype(int), g_test_rel)["wga"]
                min_auc = fast_binary_auc_pct(y_test_rel[min_rel_mask], pt[min_rel_mask])
                all_auc = fast_binary_auc_pct(y_test_rel, pt)

                band92 = float((curve >= 92.0).mean() * 0.90)
                band93 = float((curve >= 93.0).mean() * 0.90)
                band94 = float((curve >= 94.0).mean() * 0.90)

                val_thrs.append(t_wga)
                p1_tests.append(pt)
                arm_curves.append(curve)
                row = {
                    "cohort": cohort,
                    "seed": s,
                    "arm": arm,
                    "val_thresh": round(t_wga, 4),
                    "val_bal_thresh": round(t_bal, 4),
                    "off_val_thr_wga": round(off_thr_wga, 4),
                    "rel_raw_wga": round(rel_raw_wga, 4),
                    "rel_val_thr_wga": round(rel_thr_wga, 4),
                    "rel_val_bal_wga": round(rel_bal_wga, 4),
                    "rel_diag_ceil_wga": round(ceil_wga, 4),
                    "rel_diag_best_t": round(best_t, 4),
                    "band92_width": round(band92, 4),
                    "band93_width": round(band93, 4),
                    "band94_width": round(band94, 4),
                    "rel_min_auc": round(min_auc, 4),
                    "rel_all_auc": round(all_auc, 4),
                }
                wb_per_seed_rows.append(row)
                arm_seed_rows.append(row)

            val_thrs_cache[(cohort, arm)] = val_thrs
            p1_test_cache[(cohort, arm)] = p1_tests
            curves_by_arm[f"{cohort}:{arm}"] = np.stack(arm_curves, axis=0)
            if cohort == "resnet_std":
                curves_by_arm[arm] = curves_by_arm[f"{cohort}:{arm}"]

            def _mean_sd(col: str) -> tuple[float, float]:
                vals = np.array([r[col] for r in arm_seed_rows], dtype=np.float64)
                return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

            off_m, off_s = _mean_sd("off_val_thr_wga")
            raw_m, raw_s = _mean_sd("rel_raw_wga")
            val_m, val_s = _mean_sd("rel_val_thr_wga")
            bal_m, bal_s = _mean_sd("rel_val_bal_wga")
            ceil_m, ceil_s = _mean_sd("rel_diag_ceil_wga")
            b92_m, _ = _mean_sd("band92_width")
            b93_m, _ = _mean_sd("band93_width")
            b94_m, _ = _mean_sd("band94_width")
            mauc_m, _ = _mean_sd("rel_min_auc")
            aauc_m, _ = _mean_sd("rel_all_auc")
            wb_agg_rows.append({
                "cohort": cohort,
                "arm": arm,
                "n_seeds": n_seeds,
                "off_val_thr_wga_mean": round(off_m, 2),
                "off_val_thr_wga_sd": round(off_s, 2),
                "rel_raw_wga_mean": round(raw_m, 2),
                "rel_raw_wga_sd": round(raw_s, 2),
                "rel_val_thr_wga_mean": round(val_m, 2),
                "rel_val_thr_wga_sd": round(val_s, 2),
                "rel_val_bal_wga_mean": round(bal_m, 2),
                "rel_val_bal_wga_sd": round(bal_s, 2),
                "rel_diag_ceil_wga_mean": round(ceil_m, 2),
                "rel_diag_ceil_wga_sd": round(ceil_s, 2),
                "band92_width_mean": round(b92_m, 3),
                "band93_width_mean": round(b93_m, 3),
                "band94_width_mean": round(b94_m, 3),
                "rel_min_auc_mean": round(mauc_m, 2),
                "rel_all_auc_mean": round(aauc_m, 2),
            })

    # 2. Spawrious 4x4 confusion & spurious-background attribution
    sp_probs_cache: dict[str, list[np.ndarray]] = {}
    sp_conf_rows: list[dict] = []
    sp_agg_lookup: dict[str, dict] = {}
    for arm in SPAWRIOUS_ARMS:
        probs_list = [sp_npz[f"spawrious_s{s}__{arm}__ood"] for s in range(3)]
        sp_probs_cache[arm] = probs_list
        seed_res = [spawrious_confusion_metrics(p, y_sp_ood) for p in probs_list]

        def _sp_ms(fn) -> tuple[float, float]:
            vals = np.array([fn(r) for r in seed_res], dtype=np.float64)
            return float(vals.mean()), float(vals.std(ddof=1))

        mean_m, mean_s = _sp_ms(lambda r: r["ood_mean_acc"])
        worst_m, worst_s = _sp_ms(lambda r: r["ood_worst_class_acc"])
        auc_m, _ = _sp_ms(lambda r: r["ood_macro_auc"])
        bull_m, bull_s = _sp_ms(lambda r: r["spur_rates"][0])
        corgi_m, corgi_s = _sp_ms(lambda r: r["spur_rates"][1])
        dach_m, dach_s = _sp_ms(lambda r: r["spur_rates"][2])
        lab_m, lab_s = _sp_ms(lambda r: r["spur_rates"][3])
        spur_m, spur_s = _sp_ms(lambda r: r["ood_spur_confusion_mean"])
        non_spur_m, _ = _sp_ms(lambda r: r["ood_non_spur_err_mean"])
        share_m, _ = _sp_ms(lambda r: r["spur_share_of_errors"])

        row = {
            "arm": arm,
            "n_seeds": 3,
            "ood_mean_acc_mean": round(mean_m, 2),
            "ood_mean_acc_sd": round(mean_s, 2),
            "ood_worst_class_acc_mean": round(worst_m, 2),
            "ood_worst_class_acc_sd": round(worst_s, 2),
            "ood_macro_auc_mean": round(auc_m, 2),
            "spur_bull_mtn_mean": round(bull_m, 2),
            "spur_bull_mtn_sd": round(bull_s, 2),
            "spur_corgi_jungle_mean": round(corgi_m, 2),
            "spur_corgi_jungle_sd": round(corgi_s, 2),
            "spur_dach_snow_mean": round(dach_m, 2),
            "spur_dach_snow_sd": round(dach_s, 2),
            "spur_lab_desert_mean": round(lab_m, 2),
            "spur_lab_desert_sd": round(lab_s, 2),
            "ood_spur_confusion_mean": round(spur_m, 2),
            "ood_spur_confusion_sd": round(spur_s, 2),
            "ood_non_spur_err_mean": round(non_spur_m, 2),
            "spur_share_of_errors_pct": round(share_m, 1),
        }
        sp_conf_rows.append(row)
        sp_agg_lookup[arm] = row

    # 3. Paired group-stratified bootstrap 95% CIs (cached per unique arm to avoid redundant resamples)
    wb_strata = [np.flatnonzero(g_test_rel == grp) for grp in GROUPS]
    sp_strata = [np.flatnonzero(y_sp_ood == c) for c in range(4)]

    def _run_cohort_pairs(
        cohort_label: str,
        arms_needed: tuple[str, ...],
        metrics: tuple[str, ...],
        pairs: tuple[tuple[str, str], ...],
        make_fn,
        strata: list[np.ndarray],
        n_total: int,
        base_seed: int,
    ) -> list[dict]:
        out_rows: list[dict] = []
        for idx_m, met in enumerate(metrics):
            fns = {arm: make_fn(arm, met) for arm in arms_needed}
            obs_map, boot_map = bootstrap_arm_metrics(fns, strata, n_total, n_boot=n_boot, seed=base_seed + idx_m)
            for arm_a, arm_b in pairs:
                res = summarize_paired_bootstrap(obs_map[arm_a], boot_map[arm_a], obs_map[arm_b], boot_map[arm_b])
                out_rows.append({
                    "cohort": cohort_label,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "metric": met,
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
                })
        print(f"  Bootstrapped cohort={cohort_label} ({len(out_rows)} pair-metric rows)", flush=True)
        return out_rows

    ci_rows: list[dict] = []
    wb_resnet_pairs = (
        ("pre_l4_repair_w005", "affine_dfr_linear"),
        ("pre_l4_gate_only", "dar"),
        ("pre_l4_repair_w005", "dfr_l2"),
        ("pre_l4_repair_w005", "dar"),
    )
    wb_resnet_metrics = ("rel_min_auc", "rel_all_auc", "rel_raw_wga", "off_val_thr_wga", "rel_val_thr_wga", "rel_diag_ceil_wga")
    ci_rows.extend(_run_cohort_pairs(
        "resnet_std",
        ("dfr_l2", "dar", "pre_l4_gate_only", "affine_dfr_linear", "pre_l4_repair_w005"),
        wb_resnet_metrics,
        wb_resnet_pairs,
        lambda arm, met: _waterbirds_weight_evaluator(
            p1_test_cache[("resnet_std", arm)], val_thrs_cache[("resnet_std", arm)],
            y_test_rel, g_test_rel, y_test_off, g_test_off, met,
        ),
        wb_strata,
        len(y_test_rel),
        seed,
    ))

    wb_vit_pairs = (
        ("vit_ln_dfr_late", "vit_dfr_l2"),
        ("vit_pre_b6_repair_w005", "vit_ln_dfr_late"),
    )
    wb_vit_metrics = ("rel_min_auc", "rel_raw_wga", "rel_val_thr_wga", "rel_diag_ceil_wga")
    ci_rows.extend(_run_cohort_pairs(
        "vit",
        ("vit_dfr_l2", "vit_ln_dfr_late", "vit_pre_b6_repair_w005"),
        wb_vit_metrics,
        wb_vit_pairs,
        lambda arm, met: _waterbirds_weight_evaluator(
            p1_test_cache[("vit", arm)], val_thrs_cache[("vit", arm)],
            y_test_rel, g_test_rel, y_test_off, g_test_off, met,
        ),
        wb_strata,
        len(y_test_rel),
        seed + 100,
    ))

    sp_pairs = (
        ("pre_l4_repair_w005", "affine_dfr_bn_all"),
        ("pre_l4_repair_w005", "affine_dfr_bn_l4"),
        ("pre_l4_repair_w005", "dfr_l2"),
        ("pre_l4_repair_w005", "dar"),
        ("affine_dfr_bn_all", "dfr_l2"),
    )
    sp_metrics = ("ood_mean_acc", "ood_worst_class_acc", "ood_macro_auc", "ood_spur_confusion_mean")
    ci_rows.extend(_run_cohort_pairs(
        "spawrious",
        ("dfr_l2", "dar", "affine_dfr_bn_l4", "affine_dfr_bn_all", "pre_l4_repair_w005"),
        sp_metrics,
        sp_pairs,
        lambda arm, met: _spawrious_weight_evaluator(sp_probs_cache[arm], y_sp_ood, met),
        sp_strata,
        len(y_sp_ood),
        seed + 200,
    ))

    # 4. Write CSV, JSON, PNG, and README artifacts
    write_csv(output_dir / "step7d_waterbirds_threshold_per_seed.csv", wb_per_seed_rows)
    write_csv(output_dir / "step7d_waterbirds_threshold_sensitivity.csv", wb_agg_rows)
    write_csv(output_dir / "step7d_paired_bootstrap_cis.csv", ci_rows)
    write_csv(output_dir / "step7d_spawrious_confusion_rates.csv", sp_conf_rows)

    erm_star_ceil = float(curves_by_arm["resnet_star:erm_star_dfr_l2"].max(axis=1).mean())
    _render_step7d_figure(
        output_dir / "step7d_posthoc_audit.png",
        curves_by_arm,
        erm_star_ceil,
        ci_rows,
        sp_agg_lookup,
    )

    summary = {
        "step": "phase_c_step7d_zero_gpu_posthoc_audit",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "inputs_sha256": {
            "wb_npz": file_sha256(wb_npz_path),
            "sp_npz": file_sha256(sp_npz_path),
        },
        "alignment_check": alignment,
        "waterbirds_threshold_sensitivity": wb_agg_rows,
        "paired_bootstrap_cis": ci_rows,
        "spawrious_confusion_rates": sp_conf_rows,
    }
    (output_dir / "step7d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Build concise step README.md
    wb_resnet_agg = [r for r in wb_agg_rows if r["cohort"] in ("resnet_std", "resnet_star")]
    lines = [
        "# Direction 2 · Phase C · Step 7d Zero-GPU Statistical & Threshold-Sensitivity Audit",
        "",
        "Generated 100% locally on CPU from the saved Step 7c held-out test probability caches (`step7c_waterbirds_test_probs.npz` and `step7c_spawrious_ood_probs.npz`) with zero test-set tuning or checkpoint changes.",
        "",
        "> **Protocol note:** Pre-registered validation-calibrated operating points (`off_val_thr_wga`, `rel_val_thr_wga`, `rel_raw_wga` at `p = 0.5`) remain the primary benchmark claims. Post-hoc threshold sweeps (`rel_diag_ceil_wga = max_t WGA_test(t)`) and `rel_val_bal_wga` are explicitly labelled diagnostic upper-bound / sensitivity checks separating representation quality from small-validation threshold variance.",
        "",
        "## 1. Waterbirds ResNet-50 Threshold Sensitivity & Representation Ceiling (Cleaned Held-Out Test)",
        "",
        "| Cohort | Arm | Off ValThr WGA (%) | Raw `p=0.5` WGA (%) | Pre-Reg ValThr WGA (%) | Smooth ValBal WGA (%) | Diag Ceiling `max_t WGA(t)` (%) | `>=93%` Span | Min-Grp AUC (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in wb_resnet_agg:
        lines.append(
            f"| `{r['cohort']}` | `{r['arm']}` | {r['off_val_thr_wga_mean']:.2f} ± {r['off_val_thr_wga_sd']:.2f} | "
            f"{r['rel_raw_wga_mean']:.2f} ± {r['rel_raw_wga_sd']:.2f} | **{r['rel_val_thr_wga_mean']:.2f} ± {r['rel_val_thr_wga_sd']:.2f}** | "
            f"{r['rel_val_bal_wga_mean']:.2f} | **{r['rel_diag_ceil_wga_mean']:.2f} ± {r['rel_diag_ceil_wga_sd']:.2f}** | "
            f"{r['band93_width_mean']:.3f} | **{r['rel_min_auc_mean']:.2f}** |"
        )

    lines.extend([
        "",
        "## 2. Paired Group-Stratified Bootstrap 95% CIs (2,000 Replicates)",
        "",
        "| Cohort | Comparison (`A − B`) | Metric | Δ Mean (pp) | 95% Bootstrap CI | 2-Sided `p` | Seed Wins |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for r in ci_rows:
        lines.append(
            f"| `{r['cohort']}` | `{r['arm_a']}` vs `{r['arm_b']}` | `{r['metric']}` | "
            f"{r['delta_mean']:+.2f} | [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['p_two_sided']:.4f} | {r['seed_wins']} |"
        )

    lines.extend([
        "",
        "## 3. Spawrious (`o2o_hard`) OOD Spurious-Background Confusion Attribution (25,344 OOD Images, 3 Seeds)",
        "",
        "Each breed is paired on OOD test with a background that was 87–97% correlated with a **different** breed during training:",
        "- `dachshund @ snow` → distractor `labrador` (trained on snow); `bulldog @ mountain` → distractor `dachshund`; `labrador @ desert` → distractor `corgi`; `corgi @ jungle` → distractor `bulldog`.",
        "",
        "| Arm | OOD Mean Acc (%) | OOD Worst-Cls (%) | `dach@snow→lab` (%) | `bull@mtn→dach` (%) | `lab@des→cor` (%) | Macro Spur-BG Err (%) | Share of All Errors (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in sp_conf_rows:
        lines.append(
            f"| `{r['arm']}` | {r['ood_mean_acc_mean']:.2f} ± {r['ood_mean_acc_sd']:.2f} | "
            f"{r['ood_worst_class_acc_mean']:.2f} ± {r['ood_worst_class_acc_sd']:.2f} | "
            f"{r['spur_dach_snow_mean']:.2f} | {r['spur_bull_mtn_mean']:.2f} | {r['spur_lab_desert_mean']:.2f} | "
            f"**{r['ood_spur_confusion_mean']:.2f}** | {r['spur_share_of_errors_pct']:.1f}% |"
        )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7d zero-GPU statistical & threshold-sensitivity audit.")
    parser.add_argument(
        "--wb-metadata",
        type=Path,
        default=ROOT / "baselines/autobackswap/data/waterbird_complete95_forest2water2/metadata.csv",
    )
    parser.add_argument(
        "--wb-npz",
        type=Path,
        default=ROOT / "results/rep_repair/step7c_waterbirds_test_20261124/step7c_waterbirds_test_probs.npz",
    )
    parser.add_argument(
        "--wb-resnet-csv",
        type=Path,
        default=ROOT / "results/rep_repair/step7c_waterbirds_test_20261124/step7c_resnet_test_per_seed.csv",
    )
    parser.add_argument(
        "--wb-vit-csv",
        type=Path,
        default=ROOT / "results/rep_repair/step7c_waterbirds_test_20261124/step7c_vit_test_per_seed.csv",
    )
    parser.add_argument(
        "--sp-npz",
        type=Path,
        default=ROOT / "results/rep_repair/step7c_spawrious_test_20261124/step7c_spawrious_ood_probs.npz",
    )
    parser.add_argument(
        "--sp-csv",
        type=Path,
        default=ROOT / "results/rep_repair/step7c_spawrious_test_20261124/step7c_spawrious_ood_per_seed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/rep_repair/step7d_posthoc_audit_20261124",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    run_audit(
        args.wb_metadata,
        args.wb_npz,
        args.wb_resnet_csv,
        args.wb_vit_csv,
        args.sp_npz,
        args.sp_csv,
        args.output_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"Step 7d audit written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
