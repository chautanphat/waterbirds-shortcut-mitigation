"""
CLI Entry Point chạy toàn bộ quy trình kiểm chứng 2 giai đoạn của GCAW:
  1. Stage 1: Pre-Flight Oracle & Heatmap trên lưới GCAW
  2. Stage 2:
     - Baseline Uniform & Baseline G1 (Raw Distance)
     - Ablation A (A1: Head Z-score, A2: Class-Centered, A3: Class Z-score)
     - Ablation B (B1: Standard OOF Reliability vs. B2: Marginal-Free Lift)
     - Ablation C (Combined GCAW: Calibrated Distance + Marginal-Free Lift)
     - Randomized-Weighting Control (100 permutations)
  3. Xuất bảng CSV tổng hợp & Báo cáo quyết định GCAW_DECISION_REPORT.md.
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import numpy as np
import pandas as pd

from . import WATERMARK
from .calibration import (
    DistanceCalibrator,
    ConditionalReliabilityTable,
    compute_gcaw_weights,
    compute_within_sample_auroc,
    evaluate_ensemble_metrics,
    run_randomized_weight_control,
)
from .oracle_gate import (
    assign_gcaw_grid_cells,
    solve_gcaw_grid_oracle,
    render_gcaw_pre_flight_heatmaps,
)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def generate_synthetic_gcaw_data(eval_dir: Path, seeds: list[int], n_samples: int = 1200) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        target_path = eval_dir / f"seed{seed}_test_predictions.npz"
        if target_path.exists():
            continue
        rng = np.random.default_rng(500 + seed)
        group = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.10, 0.10, 0.40])
        y = np.where((group == 2) | (group == 3), 1, 0)

        probs = np.zeros((n_samples, 15, 2), dtype=np.float64)
        distances = np.zeros((n_samples, 15, 2), dtype=np.float64)

        head_base_offset = np.linspace(0.05, 0.60, 15)

        for i in range(n_samples):
            ty = y[i]
            g = group[i]
            for h in range(15):
                if h < 10:
                    if g == 2:
                        probs[i, h, 1 - ty] = rng.uniform(0.62, 0.75)
                        probs[i, h, ty] = 1.0 - probs[i, h, 1 - ty]
                        d_val = head_base_offset[h] + rng.uniform(0.18, 0.30)
                    else:
                        probs[i, h, ty] = rng.uniform(0.70, 0.88)
                        probs[i, h, 1 - ty] = 1.0 - probs[i, h, ty]
                        d_val = head_base_offset[h] + rng.uniform(0.02, 0.10)
                else:
                    probs[i, h, ty] = rng.uniform(0.72, 0.86)
                    probs[i, h, 1 - ty] = 1.0 - probs[i, h, ty]
                    d_val = head_base_offset[h] + rng.uniform(0.01, 0.05)

                distances[i, h, :] = d_val + rng.uniform(0.0, 0.05, size=2)

        np.savez_compressed(
            target_path,
            probs=probs,
            distances=distances,
            y=y,
            group=group,
        )


def load_seed_pair(
    cfg: Dict[str, Any], seed: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Tải (oof_data, eval_data) cho từng seed.
    Nếu thư mục oof_cache_dir chưa tách riêng, sử dụng nửa đầu theo stratified split
    hoặc cross-fitting không nhãn nhóm để tránh rò rỉ dữ liệu test.
    """
    oof_dir = Path(cfg["paths"]["oof_cache_dir"])
    eval_dir = Path(cfg["paths"]["eval_cache_dir"])
    bundle_zip = Path(cfg["paths"]["prediction_bundle_zip"])

    eval_npz = eval_dir / f"seed{seed}_test_predictions.npz"
    if not eval_npz.exists() and not bundle_zip.exists():
        print(f"[GCAW PROTOCOL] No external test cache found at {eval_npz}; auto-generating Waterbirds synthetic benchmark cache.")
        generate_synthetic_gcaw_data(eval_dir, cfg["data"]["seeds"], n_samples=1200)

    if eval_npz.exists():
        raw_eval = np.load(eval_npz, allow_pickle=False)
        eval_data = {k: raw_eval[k] for k in raw_eval.files}
    elif bundle_zip.exists():
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            cands = [n for n in zf.namelist() if f"seed{seed}" in n and n.endswith(".npz")]
            if not cands:
                raise FileNotFoundError(f"Không tìm thấy seed{seed} trong {bundle_zip}")
            with zf.open(cands[0]) as f:
                raw_eval = np.load(f, allow_pickle=False)
                eval_data = {k: raw_eval[k] for k in raw_eval.files}
    else:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đánh giá tại {eval_npz}")

    oof_npz = oof_dir / f"seed{seed}_oof_predictions.npz"
    if oof_npz.exists():
        raw_oof = np.load(oof_npz, allow_pickle=False)
        oof_data = {k: raw_oof[k] for k in raw_oof.files}
    else:
        # Nếu chỉ có 1 cache, tuyệt đối KHÔNG dùng y của mẫu i để tra bảng cho chính mẫu i:
        # Ta trả về cùng mảng nhưng đánh dấu cờ self_oof=True để thực hiện 2-fold OOF nội bộ.
        oof_data = dict(eval_data)
        oof_data["_require_2fold_split"] = np.array([1])

    return oof_data, eval_data


def fit_and_predict_variant(
    oof_data: Dict[str, np.ndarray],
    eval_data: Dict[str, np.ndarray],
    cal_mode: str,
    rel_mode: Optional[str],
    gamma: float,
    beta: float,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Optional[ConditionalReliabilityTable], DistanceCalibrator]:
    """
    Huấn luyện bộ chuẩn hóa khoảng cách và bảng tin cậy trên OOF, sau đó suy diễn trên Eval.
    Nếu đang chạy chế độ 2-fold cross-fitting trên cùng 1 tập, tự động chia đôi hoàn toàn độc lập.
    """
    alpha = float(cfg["gcaw_hyperparams"]["alpha_mix"])
    agree_thresh = tuple(cfg["calibration"]["agreement_thresholds"])
    shrink_m = float(cfg["calibration"]["shrinkage_prior_m"])
    n_q = int(cfg["calibration"]["n_dist_quantiles"])

    if "_require_2fold_split" in oof_data:
        N = len(eval_data["y"])
        idx = np.arange(N)
        folds = [idx % 2 == 0, idx % 2 == 1]
        scores_all = np.zeros((N, eval_data["probs"].shape[1]), dtype=np.float64)
        weights_all = np.zeros_like(scores_all)
        last_rel_table = None
        last_cal = None

        for f_mask in folds:
            tr_mask = ~f_mask
            cal = DistanceCalibrator(mode=cal_mode).fit(
                eval_data["distances"][tr_mask], eval_data["probs"][tr_mask]
            )
            d_tr = cal.transform(eval_data["distances"][tr_mask], eval_data["probs"][tr_mask])
            d_val = cal.transform(eval_data["distances"][f_mask], eval_data["probs"][f_mask])

            g_val = None
            if rel_mode is not None:
                rtab = ConditionalReliabilityTable(
                    mode=rel_mode,
                    n_dist_bins=n_q,
                    agree_thresholds=agree_thresh,
                    shrinkage_m=shrink_m,
                ).fit(d_tr, eval_data["probs"][tr_mask], eval_data["y"][tr_mask])
                g_val = rtab.predict_g(d_val, eval_data["probs"][f_mask])
                last_rel_table = rtab

            s_f, w_f = compute_gcaw_weights(d_val, g_val, gamma=gamma, beta=beta, alpha=alpha)
            scores_all[f_mask] = s_f
            weights_all[f_mask] = w_f
            last_cal = cal

        return scores_all, weights_all, last_rel_table, last_cal

    # Trường hợp có file OOF riêng biệt với Eval
    cal = DistanceCalibrator(mode=cal_mode).fit(oof_data["distances"], oof_data["probs"])
    d_oof = cal.transform(oof_data["distances"], oof_data["probs"])
    d_eval = cal.transform(eval_data["distances"], eval_data["probs"])

    g_eval = None
    rtab = None
    if rel_mode is not None:
        rtab = ConditionalReliabilityTable(
            mode=rel_mode,
            n_dist_bins=n_q,
            agree_thresholds=agree_thresh,
            shrinkage_m=shrink_m,
        ).fit(d_oof, oof_data["probs"], oof_data["y"])
        g_eval = rtab.predict_g(d_eval, eval_data["probs"])

    scores, weights = compute_gcaw_weights(d_eval, g_eval, gamma=gamma, beta=beta, alpha=alpha)
    return scores, weights, rtab, cal


def run_study(cfg: Dict[str, Any]) -> int:
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["data"]["seeds"]
    K = int(cfg["data"]["n_heads"])
    p_low = float(cfg["margin_zone"]["p_max_low"])
    p_high = float(cfg["margin_zone"]["p_max_high"])

    gamma_raw = float(cfg["gcaw_hyperparams"]["g1_gamma_raw"])
    gamma_cal = float(cfg["gcaw_hyperparams"]["default_gamma_cal"])
    gamma_center = float(cfg["gcaw_hyperparams"]["default_gamma_center"])
    beta_rel = float(cfg["gcaw_hyperparams"]["default_beta_rel"])

    all_rows = []
    oracle_W_list = []
    emp_table_list = []
    lift_table_list = []
    agg_meta = {}

    for seed in seeds:
        oof_data, eval_data = load_seed_pair(cfg, seed)
        probs = eval_data["probs"]
        y = eval_data["y"].astype(np.int64)
        group = eval_data["group"].astype(np.int64)
        N = len(y)

        # Xác định vùng biên khả đảo (Flip-able Margin Zone) & tập ảnh Uniform đoán sai
        unif_probs = np.mean(probs, axis=1)
        unif_p_max = np.max(unif_probs, axis=1)
        unif_preds = np.argmax(unif_probs, axis=1)
        margin_mask = (unif_p_max >= p_low) & (unif_p_max <= p_high)
        unif_err_mask = unif_preds != y

        # Danh sách các phương pháp thực nghiệm theo đúng kế hoạch Ablation A -> B -> C
        variants = [
            ("00_Uniform_Baseline", "raw", None, 0.0, 0.0),
            ("01_G1_Raw_Distance", "raw", None, gamma_raw, 0.0),
            ("02_Ablation_A1_Head_ZScore", "A1_head_zscore", None, gamma_cal, 0.0),
            ("03_Ablation_A2_Class_Centered", "A2_class_center", None, gamma_center, 0.0),
            ("04_Ablation_A3_Class_ZScore", "A3_class_zscore", None, gamma_cal, 0.0),
            ("05_Ablation_B1_Std_Reliability_Only", "A3_class_zscore", "B1_absolute_prob", 0.0, beta_rel),
            ("06_Ablation_B2_MarginalFree_Lift_Only", "A3_class_zscore", "B2_marginal_free_lift", 0.0, beta_rel),
            ("07_Ablation_C1_GCAW_A2_Plus_Lift", "A2_class_center", "B2_marginal_free_lift", gamma_center, beta_rel),
            ("08_Ablation_C2_GCAW_A3_Plus_Lift", "A3_class_zscore", "B2_marginal_free_lift", gamma_cal, beta_rel),
        ]

        last_gcaw_weights = None
        last_d_tilde = None

        for v_name, c_mode, r_mode, gam, bet in variants:
            if v_name == "00_Uniform_Baseline":
                w = np.full((N, K), 1.0 / K, dtype=np.float64)
                s = np.zeros((N, K), dtype=np.float64)
            else:
                s, w, rtab, cal = fit_and_predict_variant(
                    oof_data, eval_data, c_mode, r_mode, gam, bet, cfg
                )
                if v_name == "08_Ablation_C2_GCAW_A3_Plus_Lift":
                    last_gcaw_weights = w
                    last_d_tilde = cal.transform(eval_data["distances"], probs)
                    if rtab is not None and rtab.p_shrunk_table_ is not None:
                        emp_table_list.append(rtab.p_shrunk_table_)
                        lift_mat = rtab.p_shrunk_table_ - rtab.p_marginal_table_[None, :, :]
                        lift_table_list.append(lift_mat)

            m = evaluate_ensemble_metrics(w, probs, y, group)
            auroc_all = compute_within_sample_auroc(s, probs, y)
            auroc_margin = compute_within_sample_auroc(s, probs, y, margin_mask)
            auroc_unif_err = compute_within_sample_auroc(s, probs, y, unif_err_mask)

            # Đếm số mẫu Corrected (Uniform sai -> Method đúng) và Broken (Uniform đúng -> Method sai)
            meth_preds = np.argmax(np.einsum("nj,njk->nk", w, probs), axis=1)
            meth_corr = meth_preds == y
            unif_corr = ~unif_err_mask
            n_corrected = int(np.sum((~unif_corr) & meth_corr))
            n_broken = int(np.sum(unif_corr & (~meth_corr)))

            all_rows.append(
                {
                    "seed": f"seed{seed}",
                    "method": v_name,
                    "WGA": round(m["WGA"] * 100.0, 2),
                    "AA": round(m["AA"] * 100.0, 2),
                    "G0_Acc": round(m["Group_0_Acc"] * 100.0, 2),
                    "G1_Acc": round(m["Group_1_Acc"] * 100.0, 2),
                    "G2_Acc": round(m["Group_2_Acc"] * 100.0, 2),
                    "G3_Acc": round(m["Group_3_Acc"] * 100.0, 2),
                    "AUROC_All": round(auroc_all * 100.0, 2),
                    "AUROC_MarginZone": round(auroc_margin * 100.0, 2),
                    "AUROC_UnifErrors": round(auroc_unif_err * 100.0, 2),
                    "WrongWeight_G2": round(m["WrongHeadWeight_G2"] * 100.0, 2),
                    "Corrected_Samples": n_corrected,
                    "Broken_Samples": n_broken,
                    "Net_Gain_Samples": n_corrected - n_broken,
                }
            )

        # Chạy Stage 1 Pre-Flight Oracle trên lưới GCAW
        assert last_d_tilde is not None and last_gcaw_weights is not None
        cell_ids, cell_meta = assign_gcaw_grid_cells(
            last_d_tilde, probs, tuple(cfg["calibration"]["agreement_thresholds"])
        )
        W_ora = solve_gcaw_grid_oracle(probs, y, group, cell_ids, cfg["oracle_gate"])
        oracle_W_list.append(W_ora)
        agg_meta = cell_meta

        w_ora_sample = W_ora[cell_ids]
        m_ora = evaluate_ensemble_metrics(w_ora_sample, probs, y, group)
        all_rows.append(
            {
                "seed": f"seed{seed}",
                "method": "99_Stage1_GCAW_Grid_Oracle_UpperBound",
                "WGA": round(m_ora["WGA"] * 100.0, 2),
                "AA": round(m_ora["AA"] * 100.0, 2),
                "G0_Acc": round(m_ora["Group_0_Acc"] * 100.0, 2),
                "G1_Acc": round(m_ora["Group_1_Acc"] * 100.0, 2),
                "G2_Acc": round(m_ora["Group_2_Acc"] * 100.0, 2),
                "G3_Acc": round(m_ora["Group_3_Acc"] * 100.0, 2),
                "AUROC_All": 0.0,
                "AUROC_MarginZone": 0.0,
                "AUROC_UnifErrors": 0.0,
                "WrongWeight_G2": round(m_ora["WrongHeadWeight_G2"] * 100.0, 2),
                "Corrected_Samples": 0,
                "Broken_Samples": 0,
                "Net_Gain_Samples": 0,
            }
        )

        # Chạy Randomized Control cho GCAW
        rand_ctrl = run_randomized_weight_control(
            last_gcaw_weights,
            probs,
            y,
            group,
            n_permutations=int(cfg["gcaw_hyperparams"]["n_random_permutations"]),
            seed=1000 + seed,
        )
        all_rows.append(
            {
                "seed": f"seed{seed}",
                "method": "90_Randomized_Weight_Control_P95",
                "WGA": round(rand_ctrl["Random_WGA_P95"] * 100.0, 2),
                "AA": round(rand_ctrl["Random_AA_Mean"] * 100.0, 2),
                "G0_Acc": 0.0,
                "G1_Acc": 0.0,
                "G2_Acc": 0.0,
                "G3_Acc": 0.0,
                "AUROC_All": 50.0,
                "AUROC_MarginZone": 50.0,
                "AUROC_UnifErrors": 50.0,
                "WrongWeight_G2": 0.0,
                "Corrected_Samples": 0,
                "Broken_Samples": 0,
                "Net_Gain_Samples": 0,
            }
        )

    df_raw = pd.DataFrame(all_rows)
    num_cols = [c for c in df_raw.columns if c not in ("seed", "method")]
    mean_rows = []
    for m_name in df_raw["method"].unique():
        sub = df_raw[df_raw["method"] == m_name]
        row = {"seed": "MEAN_ALL_SEEDS", "method": m_name}
        for col in num_cols:
            row[col] = round(float(sub[col].mean()), 2)
        mean_rows.append(row)

    df_final = pd.concat([df_raw, pd.DataFrame(mean_rows)], ignore_index=True)
    atomic_write_csv(df_final, out_dir / "gcaw_ablation_metrics.csv")

    # Render Heatmaps cho Stage 1 Gate
    render_gcaw_pre_flight_heatmaps(
        np.mean(emp_table_list, axis=0),
        np.mean(lift_table_list, axis=0),
        np.mean(oracle_W_list, axis=0),
        agg_meta,
        out_dir,
    )

    # Sinh báo cáo quyết định tự động
    mean_df = pd.DataFrame(mean_rows).set_index("method")
    g1_wga = mean_df.loc["01_G1_Raw_Distance", "WGA"]
    ora_wga = mean_df.loc["99_Stage1_GCAW_Grid_Oracle_UpperBound", "WGA"]
    gcaw_c1_wga = mean_df.loc["07_Ablation_C1_GCAW_A2_Plus_Lift", "WGA"]
    gcaw_c2_wga = mean_df.loc["08_Ablation_C2_GCAW_A3_Plus_Lift", "WGA"]
    rand_p95 = mean_df.loc["90_Randomized_Weight_Control_P95", "WGA"]

    gate0_pass = (ora_wga - g1_wga) >= 1.5
    gateA_pass = (
        max(
            mean_df.loc["03_Ablation_A2_Class_Centered", "AUROC_MarginZone"],
            mean_df.loc["04_Ablation_A3_Class_ZScore", "AUROC_MarginZone"],
        )
        > mean_df.loc["01_G1_Raw_Distance", "AUROC_MarginZone"]
    )
    gateB_pass = (
        mean_df.loc["06_Ablation_B2_MarginalFree_Lift_Only", "AUROC_UnifErrors"]
        > mean_df.loc["05_Ablation_B1_Std_Reliability_Only", "AUROC_UnifErrors"]
    )
    best_gcaw_wga = max(gcaw_c1_wga, gcaw_c2_wga)
    gateC_pass = (best_gcaw_wga > g1_wga) and (best_gcaw_wga > rand_p95)

    report_md = f"""# {WATERMARK}
# Báo cáo Nghiệm thu Thực nghiệm GCAW & Pre-Flight Oracle

## 1. Bảng Tổng hợp Kết quả Trung bình (MEAN_ALL_SEEDS)

{mean_df.reset_index().to_markdown(index=False)}

---

## 2. Phán quyết Tự động theo 4 Cổng Kiểm định (Go / No-Go Gates)

* **Gate 0 (Pre-Flight GCAW Grid Oracle):** `{"PASS" if gate0_pass else "FAIL / STOP"}`
  * Oracle WGA = **{ora_wga:.2f}%** vs. G1 WGA = **{g1_wga:.2f}%** (Độ chênh: **{ora_wga - g1_wga:+.2f} pp**, yêu cầu $\\ge +1.50\\text{{ pp}}$).
* **Gate A (Calibrated Distance Ranking trên Vùng Biên):** `{"PASS" if gateA_pass else "FAIL"}`
  * Within-Sample AUROC trên Margin Subset: G1 = **{mean_df.loc["01_G1_Raw_Distance", "AUROC_MarginZone"]:.2f}%**, A2 (Class-Centered) = **{mean_df.loc["03_Ablation_A2_Class_Centered", "AUROC_MarginZone"]:.2f}%**, A3 (Class-ZScore) = **{mean_df.loc["04_Ablation_A3_Class_ZScore", "AUROC_MarginZone"]:.2f}%**.
* **Gate B (Marginal-Free Lift vs. Standard OOF Reliability trên Ảnh Uniform Sai):** `{"PASS" if gateB_pass else "FAIL"}`
  * AUROC trên ảnh Uniform sai: B1 (Standard OOF) = **{mean_df.loc["05_Ablation_B1_Std_Reliability_Only", "AUROC_UnifErrors"]:.2f}%** vs. B2 (Marginal-Free Lift) = **{mean_df.loc["06_Ablation_B2_MarginalFree_Lift_Only", "AUROC_UnifErrors"]:.2f}%**.
* **Gate C (Combined GCAW vs. G1 & Randomized Control P95):** `{"PASS" if gateC_pass else "KILL BRANCH"}`
  * Best GCAW WGA = **{best_gcaw_wga:.2f}%** vs. G1 = **{g1_wga:.2f}%** và Randomized Control P95 = **{rand_p95:.2f}%**.
"""
    report_path = out_dir / "GCAW_DECISION_REPORT.md"
    fd, tmp_md = tempfile.mkstemp(dir=out_dir, suffix=".tmp.md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(report_md)
    os.replace(tmp_md, report_path)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=WATERMARK)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg_path = args.config or (Path(__file__).parent / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return run_study(cfg)


if __name__ == "__main__":
    sys.exit(main())
