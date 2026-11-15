"""
Giai đoạn 1 (Pre-Flight Gate):
Sửa lỗi gom trung bình mẫu của v7.0: Phân loại mỗi mẫu theo trạng thái hình học của
nhóm head thiểu số (Minority Prototype Distance Quantile x Ensemble Agreement Regime x Predicted Class)
để đảm bảo các ô trên lưới 18 cells được kích hoạt đầy đủ.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
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
    Phân chia mỗi mẫu ảnh i vào 18 ô đặc trưng GCAW:
      - Trục 1: Khoảng cách chuẩn hóa tại nhóm head gần tâm nhất (Top-3 Nearest Calibrated Distance)
        chia theo 3 phân vị [Near (Q1), Med (Q2), Far (Q3)].
      - Trục 2: Mức độ phân tán đồng thuận của ensemble (Ensemble Agreement Bin: Low / Mid / High)
        chia theo phân vị thực nghiệm hoặc ngưỡng để cả 3 cột đều có mẫu thực tế.
      - Trục 3: Nhãn dự đoán của Uniform Ensemble [Class 0, Class 1].
    """
    _, agreement = compute_head_predictions_and_agreement(probs)
    # Lấy trung bình 3 head có khoảng cách chuẩn hóa gần nhất trên mỗi ảnh
    sorted_d = np.sort(d_tilde, axis=1)
    sample_d = np.mean(sorted_d[:, :3], axis=1)

    # Mức độ đồng thuận nhỏ nhất (đại diện cho phe thiểu số trong bức ảnh đó)
    min_agree = np.min(agreement, axis=1)
    pred_cls = np.argmax(np.mean(probs, axis=1), axis=1)

    d_edges = np.quantile(sample_d, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    d_bin = np.clip(np.digitize(sample_d, d_edges[1:-1], right=False), 0, 2)

    # Nếu ngưỡng cố định làm trống ô, tự động dùng tertiles thực nghiệm trên min_agree
    a_bin = np.clip(np.digitize(min_agree, agree_thresholds, right=True), 0, 2)
    if len(np.unique(a_bin)) < 3:
        a_edges = np.quantile(np.mean(agreement, axis=1), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        a_bin = np.clip(
            np.digitize(np.mean(agreement, axis=1), a_edges[1:-1], right=False), 0, 2
        )

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
    output_dir.mkdir(parents=True, exist_ok=True)
    d_labels = ["Near (Q1)", "Medium (Q2)", "Far (Q3)"]
    a_labels = ["Low Agree (Minority)", "Mid Agree", "High Agree (Majority)"]

    # --- Heatmap 1: OOF Shrunk Accuracy vs. Marginal-Free Lift ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig1.suptitle(
        f"{WATERMARK}\nStage 1 Gate: Empirical Head Accuracy vs. Marginal-Free Reliability Lift",
        fontsize=11.5,
        fontweight="bold",
    )

    max_lift = max(0.05, float(np.max(np.abs(lift_table))))
    for c_idx in range(2):
        cls_name = "Class 0 (Landbird)" if c_idx == 0 else "Class 1 (Waterbird)"
        ax_acc = axes1[c_idx, 0]
        mat_acc = emp_acc_table[:, :, c_idx]
        im1 = ax_acc.imshow(mat_acc, cmap="viridis", vmin=0.50, vmax=1.00)
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

        ax_lift = axes1[c_idx, 1]
        mat_lift = lift_table[:, :, c_idx]
        im2 = ax_lift.imshow(mat_lift, cmap="RdBu_r", vmin=-max_lift, vmax=max_lift)
        ax_lift.set_title(f"{cls_name} — Marginal-Free Lift (Positive = Red, Negative = Blue)", fontsize=10)
        ax_lift.set_xticks(range(3))
        ax_lift.set_xticklabels(a_labels, fontsize=8.5)
        ax_lift.set_yticks(range(3))
        ax_lift.set_yticklabels(d_labels, fontsize=8.5)
        for r in range(3):
            for c in range(3):
                ax_lift.text(c, r, f"{mat_lift[r, c]:+.4f}", ha="center", va="center", color="black", fontweight="bold")
        fig1.colorbar(im2, ax=ax_lift, shrink=0.8)

    fd1, tmp1 = tempfile.mkstemp(dir=output_dir, suffix=".tmp.png")
    os.close(fd1)
    fig1.savefig(tmp1, dpi=150)
    plt.close(fig1)
    os.replace(tmp1, output_dir / "gcaw_preflight_01_reliability_surface.png")

    # --- Heatmap 2: Oracle Weights trên 18 ô GCAW (hiển thị cả giá trị số và cỡ mẫu n) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7.5), layout="constrained")
    fig2.suptitle(
        f"{WATERMARK}\nStage 1 Gate: Test-Optimal Oracle Weights on GCAW Feature Grid",
        fontsize=11.5,
        fontweight="bold",
    )
    for c_idx, ax in enumerate(axes2):
        rows = [c_idx * 9 + r for r in range(9)]
        sub_W = oracle_W[rows, :]
        y_ticks = [
            f"{cell_meta[idx]['dist_label']} | {cell_meta[idx]['agree_label']} (n={cell_meta[idx]['count']})"
            for idx in rows
        ]
        im = ax.imshow(sub_W, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=0.40)
        ax.set_title(f"Predicted Class {c_idx} (9 GCAW Cells x 15 Heads)", fontsize=10)
        ax.set_xticks(range(sub_W.shape[1]))
        ax.set_xticklabels([f"H{j}" for j in range(sub_W.shape[1])], fontsize=8)
        ax.set_yticks(range(9))
        ax.set_yticklabels(y_ticks, fontsize=8)
        for b in (2.5, 5.5):
            ax.axhline(b, color="white", linestyle="--", linewidth=1.5)
        for r in range(9):
            for c in range(sub_W.shape[1]):
                val = sub_W[r, c]
                col_txt = "white" if val > 0.22 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=col_txt)
    fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.85, label="Oracle Head Weight W*")

    fd2, tmp2 = tempfile.mkstemp(dir=output_dir, suffix=".tmp.png")
    os.close(fd2)
    fig2.savefig(tmp2, dpi=150)
    plt.close(fig2)
    os.replace(tmp2, output_dir / "gcaw_preflight_02_oracle_weights.png")
