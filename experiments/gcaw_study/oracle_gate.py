"""
Giai đoạn 1 (Pre-Flight Gate):
Giải bài toán Test-Optimal Oracle và trích xuất Heatmap trên đúng lưới đặc trưng của GCAW:
  (Calibrated Distance Quantile x Agreement Bin x Predicted Class)
cho cả Full Set và Flip-able Margin Subset (p_bar_max in [0.50, 0.62]).
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import WATERMARK
from .calibration import (
    compute_head_predictions_and_agreement,
    evaluate_ensemble_metrics,
)


def assign_gcaw_grid_cells(
    d_tilde: np.ndarray,
    probs: np.ndarray,
    agree_thresholds: Tuple[float, float] = (0.35, 0.70),
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Gán mỗi mẫu ảnh i vào 1 trong 18 ô đặc trưng của GCAW dựa trên trung bình mẫu
    hoặc trạng thái của phe thiểu số:
      - Trục 1: Calibrated Distance Quantile [Near (Q1), Medium (Q2), Far (Q3)]
      - Trục 2: Ensemble Dispersion / Minority Agreement [Low, Mid, High]
      - Trục 3: Uniform Predicted Class [Class 0, Class 1]
    """
    _, agreement = compute_head_predictions_and_agreement(probs)
    sample_d = np.mean(d_tilde, axis=1)
    sample_agree = np.mean(agreement, axis=1)
    pred_cls = np.argmax(np.mean(probs, axis=1), axis=1)

    d_edges = np.quantile(sample_d, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    d_bin = np.clip(np.digitize(sample_d, d_edges[1:-1], right=False), 0, 2)
    a_bin = np.clip(np.digitize(sample_agree, agree_thresholds, right=True), 0, 2)

    cell_ids = pred_cls * 9 + d_bin * 3 + a_bin

    d_names = ["Near (Q1)", "Med (Q2)", "Far (Q3)"]
    a_names = ["LowAgree", "MidAgree", "HighAgree"]
    cell_meta = {}
    for c_idx in (0, 1):
        for q_idx in range(3):
            for a_idx in range(3):
                cid = c_idx * 9 + q_idx * 3 + a_idx
                cell_meta[cid] = {
                    "cell_id": cid,
                    "pred_class": c_idx,
                    "dist_label": d_names[q_idx],
                    "agree_label": a_names[a_idx],
                    "cell_name": f"C{c_idx}_{d_names[q_idx][:4]}_{a_names[a_idx]}",
                    "count": int(np.sum(cell_ids == cid)),
                }
    return cell_ids, cell_meta


def solve_gcaw_grid_oracle(
    probs: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    cell_ids: np.ndarray,
    cfg_oracle: Dict[str, Any],
    n_cells: int = 18,
) -> np.ndarray:
    """
    Tìm ma trận trọng số W* in R^{18 x 15} tối đa hóa WGA trên lưới đặc trưng GCAW.
    """
    N, K, _ = probs.shape
    max_cap = float(cfg_oracle["max_weight_cap"])
    aa_tol = float(cfg_oracle["aa_drop_tolerance"])
    tau = float(cfg_oracle["sigmoid_temp"])
    l2_reg = float(cfg_oracle["l2_uniform_reg"])

    p_true = probs[np.arange(N), :, y]
    margins = 2.0 * p_true - 1.0

    w_unif = np.full((N, K), 1.0 / K, dtype=np.float64)
    unif_metrics = evaluate_ensemble_metrics(w_unif, probs, y, group)
    aa_floor = unif_metrics["AA"] - aa_tol

    group_masks = [group == g for g in range(4)]
    group_counts = [max(1, int(np.sum(m))) for m in group_masks]

    cell_ind = np.zeros((N, n_cells), dtype=np.float64)
    cell_ind[np.arange(N), cell_ids] = 1.0
    unif_mat = np.full((n_cells, K), 1.0 / K, dtype=np.float64)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, float]:
        return x[:-1].reshape(n_cells, K), float(x[-1])

    def smooth_pred(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.sum(W[cell_ids] * margins, axis=1) / tau
        s = expit(z)
        return s, s * (1.0 - s) / tau

    def obj(x: np.ndarray) -> float:
        W, t = unpack(x)
        return -t + 0.5 * l2_reg * float(np.sum((W - unif_mat) ** 2))

    def obj_jac(x: np.ndarray) -> np.ndarray:
        W, _ = unpack(x)
        g = np.zeros_like(x)
        g[:-1] = (l2_reg * (W - unif_mat)).ravel()
        g[-1] = -1.0
        return g

    constraints = []
    for g_idx in range(4):
        gm = group_masks[g_idx]
        gc = group_counts[g_idx]
        Sg = cell_ind[gm]
        Mg = margins[gm]

        def make_c(gm=gm, gc=gc, Sg=Sg, Mg=Mg):
            def f(x: np.ndarray) -> float:
                W, t = unpack(x)
                s, _ = smooth_pred(W)
                return float(np.sum(s[gm]) / gc - t)

            def j(x: np.ndarray) -> np.ndarray:
                W, _ = unpack(x)
                _, ds = smooth_pred(W)
                grad_W = (Sg.T @ (ds[gm][:, None] * Mg)) / gc
                out = np.zeros_like(x)
                out[:-1] = grad_W.ravel()
                out[-1] = -1.0
                return out

            return {"type": "ineq", "fun": f, "jac": j}

        constraints.append(make_c())

    def aa_f(x: np.ndarray) -> float:
        W, _ = unpack(x)
        s, _ = smooth_pred(W)
        return float(np.mean(s) - aa_floor)

    def aa_j(x: np.ndarray) -> np.ndarray:
        W, _ = unpack(x)
        _, ds = smooth_pred(W)
        grad_W = (cell_ind.T @ (ds[:, None] * margins)) / N
        out = np.zeros_like(x)
        out[:-1] = grad_W.ravel()
        return out

    def eq_f(x: np.ndarray) -> np.ndarray:
        W, _ = unpack(x)
        return np.sum(W, axis=1) - 1.0

    def eq_j(x: np.ndarray) -> np.ndarray:
        J = np.zeros((n_cells, n_cells * K + 1), dtype=np.float64)
        for c in range(n_cells):
            J[c, c * K : (c + 1) * K] = 1.0
        return J

    constraints.append({"type": "ineq", "fun": aa_f, "jac": aa_j})
    constraints.append({"type": "eq", "fun": eq_f, "jac": eq_j})

    bounds = [(0.0, max_cap) for _ in range(n_cells * K)] + [(0.0, 1.0)]
    x0 = np.concatenate([unif_mat.ravel(), [unif_metrics["WGA"]]])

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        jac=obj_jac,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(cfg_oracle["max_iter"]), "ftol": 1e-6, "disp": False},
    )
    W_opt, _ = unpack(res.x)
    W_opt = np.clip(W_opt, 0.0, max_cap)
    W_opt /= np.sum(W_opt, axis=1, keepdims=True)
    return W_opt


def render_gcaw_pre_flight_heatmaps(
    emp_acc_table: np.ndarray,
    lift_table: np.ndarray,
    oracle_W: np.ndarray,
    cell_meta: Dict[int, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Vẽ 2 biểu đồ Heatmap cốt lõi cho Giai đoạn 1 (Pre-flight Gate):
      1. gcaw_preflight_01_reliability_surface.png: Đối chiếu Empirical Accuracy P(C=y)
         và Marginal-Free Lift g_j^{lift}(x) trên lưới (Calibrated Distance x Agreement).
      2. gcaw_preflight_02_oracle_weights.png: Phân phối trọng số lý tưởng W* trên 18 ô GCAW.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    d_labels = ["Near (Q1)", "Medium (Q2)", "Far (Q3)"]
    a_labels = ["Low Agree (Minority)", "Mid Agree", "High Agree (Majority)"]

    # --- Biểu đồ 1: Reliability Surface & Marginal-Free Lift (3x3x2 panels) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig1.suptitle(
        f"{WATERMARK}\nStage 1 Gate: Empirical Head Accuracy vs. Marginal-Free Reliability Lift",
        fontsize=12,
        fontweight="bold",
    )

    for c_idx in range(2):
        cls_name = "Class 0 (Landbird)" if c_idx == 0 else "Class 1 (Waterbird)"
        # Panel trái: Empirical Accuracy
        ax_acc = axes1[c_idx, 0]
        mat_acc = emp_acc_table[:, :, c_idx]
        im1 = ax_acc.imshow(mat_acc, cmap="viridis", vmin=0.45, vmax=0.98)
        ax_acc.set_title(f"{cls_name} — OOF Shrunk Accuracy P(C=y)", fontsize=10)
        ax_acc.set_xticks(range(3))
        ax_acc.set_xticklabels(a_labels, fontsize=8.5)
        ax_acc.set_yticks(range(3))
        ax_acc.set_yticklabels(d_labels, fontsize=8.5)
        ax_acc.set_ylabel("Calibrated Distance Quantile", fontsize=9)
        for r in range(3):
            for c in range(3):
                ax_acc.text(c, r, f"{mat_acc[r, c]:.3f}", ha="center", va="center", color="white", fontweight="bold")
        fig1.colorbar(im1, ax=ax_acc, shrink=0.8)

        # Panel phải: Marginal-Free Lift g_j^{lift}
        ax_lift = axes1[c_idx, 1]
        mat_lift = lift_table[:, :, c_idx]
        im2 = ax_lift.imshow(mat_lift, cmap="RdBu", vmin=-0.25, vmax=0.25)
        ax_lift.set_title(f"{cls_name} — Marginal-Free Lift g_j(x) (Agreement De-biased)", fontsize=10)
        ax_lift.set_xticks(range(3))
        ax_lift.set_xticklabels(a_labels, fontsize=8.5)
        ax_lift.set_yticks(range(3))
        ax_lift.set_yticklabels(d_labels, fontsize=8.5)
        for r in range(3):
            for c in range(3):
                ax_lift.text(c, r, f"{mat_lift[r, c]:+.3f}", ha="center", va="center", color="black", fontweight="bold")
        fig1.colorbar(im2, ax=ax_lift, shrink=0.8)

    fd1, tmp1 = tempfile.mkstemp(dir=output_dir, suffix=".tmp.png")
    os.close(fd1)
    fig1.savefig(tmp1, dpi=150)
    plt.close(fig1)
    os.replace(tmp1, output_dir / "gcaw_preflight_01_reliability_surface.png")

    # --- Biểu đồ 2: Oracle Weights trên 18 ô GCAW ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    fig2.suptitle(
        f"{WATERMARK}\nStage 1 Gate: Test-Optimal Oracle Weights on GCAW Feature Grid",
        fontsize=12,
        fontweight="bold",
    )
    for c_idx, ax in enumerate(axes2):
        rows = [c_idx * 9 + r for r in range(9)]
        sub_W = oracle_W[rows, :]
        y_ticks = [f"{cell_meta[idx]['dist_label']} | {cell_meta[idx]['agree_label']}" for idx in rows]
        im = ax.imshow(sub_W, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=0.40)
        ax.set_title(f"Predicted Class {c_idx} (9 GCAW Cells x 15 Heads)", fontsize=10)
        ax.set_xticks(range(sub_W.shape[1]))
        ax.set_xticklabels([f"H{j}" for j in range(sub_W.shape[1])], fontsize=8)
        ax.set_yticks(range(9))
        ax.set_yticklabels(y_ticks, fontsize=8)
        for b in (2.5, 5.5):
            ax.axhline(b, color="white", linestyle="--", linewidth=1.5)
    fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.85, label="Oracle Head Weight W*")

    fd2, tmp2 = tempfile.mkstemp(dir=output_dir, suffix=".tmp.png")
    os.close(fd2)
    fig2.savefig(tmp2, dpi=150)
    plt.close(fig2)
    os.replace(tmp2, output_dir / "gcaw_preflight_02_oracle_weights.png")
