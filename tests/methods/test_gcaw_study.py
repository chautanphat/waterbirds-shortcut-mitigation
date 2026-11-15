"""
Unit test kiểm định toàn bộ pipeline GCAW (Stage 1 Pre-Flight Oracle + Stage 2 Ablation A/B/C).
Chạy lệnh: pytest tests/methods/test_gcaw_study.py -v
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from experiments.gcaw_study import WATERMARK
from experiments.gcaw_study.run_gcaw import main as run_gcaw_cli


def _generate_synthetic_gcaw_data(eval_dir: Path, seeds: list[int], n_samples: int = 420) -> None:
    """
    Sinh dữ liệu mô phỏng cho unit test:
    - Các head 0..10 bị lệch thang đo khoảng cách và đồng thuận sai theo background trên Group 2.
    - Các head 11..14 là minority experts đoán đúng trên Group 2 và có khoảng cách chuẩn hóa theo lớp rất gần tâm.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        rng = np.random.default_rng(500 + seed)
        group = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.10, 0.10, 0.40])
        y = np.where((group == 2) | (group == 3), 1, 0)

        probs = np.zeros((n_samples, 15, 2), dtype=np.float64)
        distances = np.zeros((n_samples, 15, 2), dtype=np.float64)

        # Tạo độ lệch thang đo cơ sở giữa các head (để kiểm chứng Ablation A)
        head_base_offset = np.linspace(0.05, 0.60, 15)

        for i in range(n_samples):
            ty = y[i]
            g = group[i]
            for h in range(15):
                if h < 10:
                    # Majority shortcut heads: đúng trên G0, G3 nhưng sai trên G2
                    if g == 2:
                        probs[i, h, 1 - ty] = rng.uniform(0.62, 0.75)
                        probs[i, h, ty] = 1.0 - probs[i, h, 1 - ty]
                        d_val = head_base_offset[h] + rng.uniform(0.18, 0.30)
                    else:
                        probs[i, h, ty] = rng.uniform(0.70, 0.88)
                        probs[i, h, 1 - ty] = 1.0 - probs[i, h, ty]
                        d_val = head_base_offset[h] + rng.uniform(0.02, 0.10)
                else:
                    # Minority invariant heads (h = 10..14): đúng trên cả G2 với khoảng cách gần tâm của chính nó
                    probs[i, h, ty] = rng.uniform(0.72, 0.86)
                    probs[i, h, 1 - ty] = 1.0 - probs[i, h, ty]
                    d_val = head_base_offset[h] + rng.uniform(0.01, 0.05)

                distances[i, h, :] = d_val + rng.uniform(0.0, 0.05, size=2)

        np.savez_compressed(
            eval_dir / f"seed{seed}_test_predictions.npz",
            probs=probs,
            distances=distances,
            y=y,
            group=group,
        )


def test_gcaw_full_pipeline_execution(tmp_path: Path):
    eval_dir = tmp_path / "fits"
    out_dir = tmp_path / "results_gcaw"
    _generate_synthetic_gcaw_data(eval_dir, seeds=[0, 1], n_samples=360)

    cfg = {
        "study_name": "GCAW Test",
        "watermark": WATERMARK,
        "paths": {
            "oof_cache_dir": str(tmp_path / "nonexistent_oof"),
            "eval_cache_dir": str(eval_dir),
            "prediction_bundle_zip": str(tmp_path / "nonexistent.zip"),
            "output_dir": str(out_dir),
        },
        "data": {"seeds": [0, 1], "n_heads": 15, "n_classes": 2, "expected_min_samples": 100},
        "margin_zone": {"p_max_low": 0.50, "p_max_high": 0.65},
        "calibration": {
            "eps": 1e-6,
            "n_dist_quantiles": 3,
            "agreement_thresholds": [0.35, 0.70],
            "shrinkage_prior_m": 20.0,
        },
        "gcaw_hyperparams": {
            "alpha_mix": 0.25,
            "g1_gamma_raw": 50.0,
            "default_gamma_zscore": 2.0,
            "default_beta_rel": 1.5,
            "n_random_permutations": 25,
        },
        "oracle_gate": {
            "max_weight_cap": 0.40,
            "aa_drop_tolerance": 0.01,
            "sigmoid_temp": 0.02,
            "l2_uniform_reg": 0.005,
            "max_iter": 150,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Chạy CLI với cờ --allow-synthetic-pytest cho unit test
    rc = run_gcaw_cli(["--config", str(cfg_file), "--allow-synthetic-pytest"])
    assert rc == 0

    # Kiểm tra các deliverables được sinh đầy đủ
    assert (out_dir / "gcaw_ablation_metrics.csv").exists()
    assert (out_dir / "gcaw_preflight_01_reliability_surface.png").exists()
    assert (out_dir / "gcaw_preflight_02_oracle_weights.png").exists()
    assert (out_dir / "GCAW_DECISION_REPORT.md").exists()
    assert (out_dir / "data_lineage_manifest.json").exists()

    df = pd.read_csv(out_dir / "gcaw_ablation_metrics.csv")
    mean_sub = df[df["seed"] == "MEAN_ALL_SEEDS"].set_index("method")

    # Kiểm chứng Marginal-Free Lift (B2) đạt AUROC trên ảnh Uniform sai cao hơn B1 (Standard OOF)
    assert (
        mean_sub.loc["06_Ablation_B2_MarginalFree_Lift_Only", "AUROC_UnifErrors"]
        >= mean_sub.loc["05_Ablation_B1_Std_Reliability_Only", "AUROC_UnifErrors"] - 1e-4
    )
    # Kiểm chứng Calibrated Distance (A2/A3) cải thiện AUROC so với G1 khi các head lệch thang đo
    assert (
        mean_sub.loc["03_Ablation_A2_Class_Centered", "AUROC_All"]
        > mean_sub.loc["01_G1_Raw_Distance", "AUROC_All"]
    )
