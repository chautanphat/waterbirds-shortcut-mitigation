"""Direction 2 · Phase A · Step 2: Frozen Post-GAP Readout Control Ladder & Gate 1.

Evaluates 6 frozen post-GAP readout families on identical 5-fold group-stratified
validation splits (zero test access, inner 4-fold hyperparameter selection):
  - Rung 0: dfr_l2, dfr_l1, dpe_oof (linear & prototype baselines)
  - Rung 1: frozen_mlp (2-layer LayerNorm+GELU MLP, 3-seed ensemble),
            frozen_rbf_svm (RBF-kernel SVM with col_zscore / l2_unit),
            frozen_cosine_knn (group-weighted cosine kNN)

Reports cleaned OOF WGA (relabel & drop) and exact rescue rate on the persistent
ERM-only minority failures from Step 1, then evaluates Pre-Registered Gate 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.methods.dpe_head_reweighting import ensemble, train_heads
from src.methods.dpe_head_reweighting.reweighting import GROUPS
from src.methods.rep_repair.phase_a_data import (
    apply_label_condition,
    balanced_group_weights,
    file_sha256,
    group_metrics,
    load_all_sources,
    stratified_group_folds,
)

ROOT = Path(__file__).resolve().parents[2]
DPE_HEAD_SETTINGS = {
    "heads": 15,
    "epochs": 20,
    "batch_size": 256,
    "lr": 1e-3,
    "momentum": 0.9,
    "entropic_scale": 30.0,
    "wd_weight": 10.0,
    "cov_reg": 5e5,
}
METHODS_ORDER = (
    "dfr_l2",
    "dfr_l1",
    "dpe_oof",
    "frozen_rbf_svm",
    "frozen_mlp",
    "frozen_cosine_knn",
)


def row_l2_unit(X: np.ndarray) -> np.ndarray:
    """Row-wise z-score followed by L2 unit normalization (DPE feature geometry)."""
    X = np.asarray(X, dtype=np.float64)
    z = (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-8)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-12)


def transform_features(X_tr: np.ndarray, X_te: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "col_zscore":
        scaler = StandardScaler().fit(X_tr)
        return scaler.transform(X_tr), scaler.transform(X_te)
    if mode == "l2_unit":
        return row_l2_unit(X_tr), row_l2_unit(X_te)
    raise ValueError(f"Unknown feature normalization mode: {mode}")


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(0.0, a2 + b2 - 2.0 * (A @ B.T))


def fit_predict_logistic(X_tr, y_tr, g_tr, X_te, *, penalty: str, c_val: float, is_inner: bool = False) -> np.ndarray:
    Z_tr, Z_te = transform_features(X_tr, X_te, "col_zscore")
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(penalty=penalty, C=c_val, solver=solver, max_iter=1000, random_state=0)
    clf.fit(Z_tr, y_tr, sample_weight=balanced_group_weights(g_tr))
    return clf.predict_proba(Z_te)


def fit_predict_rbf_svm(
    X_tr, y_tr, g_tr, X_te, *, c_val: float, gamma: str, norm: str, is_inner: bool = False
) -> np.ndarray:
    Z_tr, Z_te = transform_features(X_tr, X_te, norm)
    clf = SVC(C=c_val, kernel="rbf", gamma=gamma, probability=not is_inner, random_state=0)
    clf.fit(Z_tr, y_tr, sample_weight=balanced_group_weights(g_tr))
    if not is_inner:
        return clf.predict_proba(Z_te)
    dec = clf.decision_function(Z_te)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(dec, -40.0, 40.0)))
    return np.stack([1.0 - p1, p1], axis=1)


def nested_cv_rbf_svm_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    inner_splits: int,
    seed: int,
    c_values: list[float],
    gammas: list[str],
    norms: list[str],
) -> np.ndarray:
    """Exact precomputed-kernel nested CV for RBF-SVM (~8x faster, bitwise-equivalent)."""
    probs = np.zeros((len(y), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te = X[te]

        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=inner_splits, seed=seed + 100 + fold)

        # Precompute strictly per-inner-fold scaled distance matrices once per norm.
        inner_cache = {}
        for norm in norms:
            fold_mats = []
            for infold in range(inner_splits):
                itr = in_folds != infold
                iva = in_folds == infold
                Z_itr, Z_iva = transform_features(X_tr[itr], X_tr[iva], norm)
                d_tt = _sq_dists(Z_itr, Z_itr)
                d_vt = _sq_dists(Z_iva, Z_itr)
                g_scale = 1.0 / (Z_itr.shape[1] * float(Z_itr.var()))
                g_auto = 1.0 / float(Z_itr.shape[1])
                fold_mats.append((itr, iva, d_tt, d_vt, {"scale": g_scale, "auto": g_auto}))
            inner_cache[norm] = fold_mats

        best_cfg = (norms[0], gammas[0], c_values[0])
        best_key = (-1.0, -1.0)
        for norm in norms:
            for gamma_name in gammas:
                for c_val in c_values:
                    in_preds = np.zeros(len(y_tr), dtype=int)
                    for itr, iva, d_tt, d_vt, gmap in inner_cache[norm]:
                        g_val = gmap[gamma_name]
                        clf = SVC(C=c_val, kernel="precomputed", probability=False, random_state=0)
                        clf.fit(np.exp(-g_val * d_tt), y_tr[itr], sample_weight=balanced_group_weights(g_tr[itr]))
                        in_preds[iva] = clf.predict(np.exp(-g_val * d_vt))
                    m = group_metrics(y_tr, in_preds, g_tr)
                    key = (m["wga"], m["overall"])
                    if key > best_key:
                        best_key = key
                        best_cfg = (norm, gamma_name, c_val)

        b_norm, b_gamma, b_c = best_cfg
        Z_tr, Z_te = transform_features(X_tr, X_te, b_norm)
        g_val = 1.0 / (Z_tr.shape[1] * float(Z_tr.var())) if b_gamma == "scale" else 1.0 / float(Z_tr.shape[1])
        clf = SVC(C=b_c, kernel="precomputed", probability=True, random_state=0)
        clf.fit(np.exp(-g_val * _sq_dists(Z_tr, Z_tr)), y_tr, sample_weight=balanced_group_weights(g_tr))
        probs[te] = clf.predict_proba(np.exp(-g_val * _sq_dists(Z_te, Z_tr)))
    return probs


def fit_predict_cosine_knn(X_tr, y_tr, g_tr, X_te, *, k: int, is_inner: bool = False) -> np.ndarray:
    Z_tr, Z_te = transform_features(X_tr, X_te, "l2_unit")
    w_tr = balanced_group_weights(g_tr)
    sim = Z_te @ Z_tr.T
    k_eff = min(k, len(y_tr))
    top_idx = np.argpartition(sim, -k_eff, axis=1)[:, -k_eff:]
    rows = np.arange(len(Z_te))[:, None]
    top_sim = np.clip(sim[rows, top_idx], 0.0, None) + 1e-6
    top_w = top_sim * w_tr[top_idx]
    top_y = y_tr[top_idx]
    score_1 = (top_w * (top_y == 1)).sum(axis=1)
    score_0 = (top_w * (top_y == 0)).sum(axis=1)
    total = np.maximum(score_0 + score_1, 1e-12)
    p1 = score_1 / total
    return np.stack([1.0 - p1, p1], axis=1)


class _TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def fit_predict_mlp(
    X_tr,
    y_tr,
    g_tr,
    X_te,
    *,
    hidden: int,
    dropout: float,
    weight_decay: float,
    lr: float,
    epochs: int,
    seeds: list[int],
    is_inner: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Group-weighted 2-layer MLP ensemble (full-batch AdamW on CPU or CUDA)."""
    Z_tr, Z_te = transform_features(X_tr, X_te, "col_zscore")
    xt = torch.tensor(Z_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    wt = torch.tensor(balanced_group_weights(g_tr), dtype=torch.float32, device=device)
    xte = torch.tensor(Z_te, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(reduction="none")

    active_seeds = [seeds[0]] if is_inner else seeds
    active_epochs = min(epochs, 100) if is_inner else epochs
    prob_sum = np.zeros((len(X_te), 2), dtype=np.float64)
    for s in active_seeds:
        torch.manual_seed(s)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(s)
        model = _TwoLayerMLP(Z_tr.shape[1], hidden, dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(active_epochs):
            model.train()
            opt.zero_grad()
            loss = (ce(model(xt), yt) * wt).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            prob_sum += model(xte).softmax(dim=1).cpu().numpy()
    return prob_sum / len(active_seeds)


def nested_cv_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    inner_splits: int,
    seed: int,
    candidates: list[dict],
    fit_fn,
) -> tuple[np.ndarray, list[dict]]:
    """Run strict nested CV: fit scalers & select best candidate inside each outer training fold."""
    probs = np.zeros((len(y), 2), dtype=np.float64)
    chosen = []
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te = X[te]

        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=inner_splits, seed=seed + 100 + fold)
        best_cfg = candidates[0]
        best_key = (-1.0, -1.0)
        for cfg in candidates:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(inner_splits):
                itr = in_folds != infold
                iva = in_folds == infold
                p_va = fit_fn(X_tr[itr], y_tr[itr], g_tr[itr], X_tr[iva], is_inner=True, **cfg)
                in_preds[iva] = p_va.argmax(axis=-1)
            m = group_metrics(y_tr, in_preds, g_tr)
            key = (m["wga"], m["overall"])
            if key > best_key:
                best_key = key
                best_cfg = cfg

        chosen.append({"outer_fold": fold, "inner_wga": round(best_key[0], 4), **best_cfg})
        probs[te] = fit_fn(X_tr, y_tr, g_tr, X_te, is_inner=False, **best_cfg)
    return probs, chosen


def dpe_5fold_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    seed: int,
    device: str = "cpu",
) -> np.ndarray:
    """Train exact 15-head DPE per outer fold on raw features and predict held-out fold."""
    probs = np.zeros((len(y), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        state = train_heads.train(
            X[tr], y[tr], groups[tr], DPE_HEAD_SETTINGS,
            seed=seed + 1000 * fold,
            provenance={"phase": "rep_repair_phase_a", "outer_fold": fold},
            device=device,
            progress=False,
        )
        fold_scores = ensemble.predict(state, X[te], device=device)
        probs[te] = fold_scores.mean(axis=0)
    return probs


def _run_source_task(payload: dict) -> dict:
    """Worker task for one (condition, source) pair."""
    device = payload.get("device", "cpu")
    if not str(device).startswith("cuda"):
        torch.set_num_threads(1)
    t0 = time.perf_counter()
    cond = payload["cond"]
    cohort = payload["cohort"]
    seed = payload["seed"]
    sid = payload["source_id"]
    X = payload["X"]
    y = payload["y"]
    g = payload["g"]
    folds = payload["folds"]
    pers_mask = payload["pers_mask"]
    inner_k = payload["inner_k"]
    base_seed = payload["base_seed"]
    rcfg = payload["rcfg"]

    l2_cands = [{"penalty": "l2", "c_val": c} for c in rcfg["dfr_l2_C"]]
    l1_cands = [{"penalty": "l1", "c_val": c} for c in rcfg["dfr_l1_C"]]
    mlp_cands = [
        {
            "hidden": h,
            "dropout": d,
            "weight_decay": wd,
            "lr": float(rcfg["mlp_lr"]),
            "epochs": int(rcfg["mlp_epochs"]),
            "seeds": list(rcfg["mlp_seeds"]),
            "device": device,
        }
        for h in rcfg["mlp_hidden"]
        for d in rcfg["mlp_dropout"]
        for wd in rcfg["mlp_weight_decay"]
    ]
    knn_cands = [{"k": int(k)} for k in rcfg["knn_k"]]

    runs = [
        ("dfr_l2", nested_cv_oof(X, y, g, folds, inner_k, base_seed, l2_cands, fit_predict_logistic)[0]),
        ("dfr_l1", nested_cv_oof(X, y, g, folds, inner_k, base_seed, l1_cands, fit_predict_logistic)[0]),
        (
            "frozen_rbf_svm",
            nested_cv_rbf_svm_oof(
                X, y, g, folds, inner_k, base_seed,
                list(rcfg["rbf_svm_C"]), list(rcfg["rbf_svm_gamma"]), list(rcfg["rbf_svm_norm"]),
            ),
        ),
        ("frozen_mlp", nested_cv_oof(X, y, g, folds, inner_k, base_seed, mlp_cands, fit_predict_mlp)[0]),
        ("frozen_cosine_knn", nested_cv_oof(X, y, g, folds, inner_k, base_seed, knn_cands, fit_predict_cosine_knn)[0]),
    ]
    if rcfg.get("run_dpe_oof", True):
        runs.insert(2, ("dpe_oof", dpe_5fold_oof(X, y, g, folds, seed=base_seed, device=device)))

    rows = []
    oof_out = {}
    for method_name, probs in runs:
        oof_out[f"{cond}__{sid}__{method_name}"] = probs.astype(np.float32)
        preds = probs.argmax(axis=-1)
        m = group_metrics(y, preds, g)
        rescue_n = int((preds[pers_mask] == y[pers_mask]).sum()) if pers_mask.any() else 0
        rescue_pct = (100.0 * rescue_n / int(pers_mask.sum())) if pers_mask.any() else float("nan")
        rows.append(
            {
                "condition": cond,
                "cohort": cohort,
                "seed": seed,
                "method": method_name,
                "n_rows": len(y),
                "wga": round(m["wga"], 4),
                "overall": round(m["overall"], 4),
                **{f"acc_{grp}": round(m["groups"][grp], 4) for grp in GROUPS},
                "persistent_total": int(pers_mask.sum()),
                "persistent_rescued": rescue_n,
                "persistent_rescue_pct": round(rescue_pct, 2) if not np.isnan(rescue_pct) else "",
            }
        )
    elapsed = time.perf_counter() - t0
    return {"cond": cond, "sid": sid, "elapsed": elapsed, "rows": rows, "oof": oof_out}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_ladder(agg_rows: list[dict], output_path: Path):
    rel_rows = [r for r in agg_rows if r["condition"] == "relabel"]
    methods = [m for m in METHODS_ORDER if any(r["method"] == m for r in rel_rows)]
    labels = {
        "dfr_l2": "DFR-L2",
        "dfr_l1": "DFR-L1",
        "dpe_oof": "DPE (15-head)",
        "frozen_rbf_svm": "Frozen RBF-SVM",
        "frozen_mlp": "Frozen MLP",
        "frozen_cosine_knn": "Cosine kNN",
    }
    x = np.arange(len(methods))
    width = 0.36
    std_wga = [next(r["wga_mean"] for r in rel_rows if r["cohort"] == "standard_erm" and r["method"] == m) for m in methods]
    std_sd = [next(r["wga_sd"] for r in rel_rows if r["cohort"] == "standard_erm" and r["method"] == m) for m in methods]
    es_wga = [next(r["wga_mean"] for r in rel_rows if r["cohort"] == "erm_star" and r["method"] == m) for m in methods]
    std_res = [next(r["persistent_rescue_pct_mean"] for r in rel_rows if r["cohort"] == "standard_erm" and r["method"] == m) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
    ax1.bar(x - width / 2, std_wga, width, yerr=std_sd, capsize=3, color="#245da6", label="Standard ERM (5 seeds ± SD)")
    ax1.bar(x + width / 2, es_wga, width, color="#bc5100", label="ERM* (2 seeds mean)")
    ax1.set_xticks(x, [labels[m] for m in methods], rotation=20, ha="right")
    ax1.set_ylabel("Cleaned 5-Fold OOF WGA (%)")
    ax1.set_title("Frozen Post-GAP Readout Ladder (Relabelled Val)")
    ax1.set_ylim(max(70.0, min(std_wga) - 4.0), 100.0)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5)

    bars = ax2.bar(x, std_res, color="#4a7c59")
    ax2.axhline(50.0, color="#8b0000", ls="--", lw=1.1, label="Gate 1 cutoff (50% max rescue)")
    for b, val in zip(bars, std_res):
        ax2.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x, [labels[m] for m in methods], rotation=20, ha="right")
    ax2.set_ylabel("Rescue Rate on Persistent ERM-Only Failures (%)")
    ax2.set_title("Standard ERM Failure Rescue by Stronger Frozen Readouts")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Direction 2 · Phase A · Frozen Post-GAP Readout Control Ladder (Validation Only)", fontsize=12)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--failure-stats", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, official_val_rows, sources = load_all_sources(config, ROOT, asset_root_override=args.asset_root)
    species = tuple(config["tern_species"])
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_candidates = [
        p for p in (
            args.failure_stats,
            out_dir / "failure_audit_stats.json",
            ROOT / "results/rep_repair/phase_a_20261123/failure_audit_stats.json",
        )
        if p is not None and Path(p).is_file()
    ]
    persistent_val_indices = (
        set(json.loads(Path(stats_candidates[0]).read_text(encoding="utf-8"))["persistent_val_indices"])
        if stats_candidates
        else set()
    )

    outer_k = int(config["cv"]["outer_folds"])
    inner_k = int(config["cv"]["inner_folds"])
    split_seed = int(config["cv"]["split_seed"])

    tasks = []
    for cond in config["cv"]["conditions"]:
        keep_idx, cond_rows = apply_label_condition(official_val_rows, cond, species=species)
        y = np.array([r["y"] for r in cond_rows], dtype=int)
        g = np.array([r["group"] for r in cond_rows], dtype=str)
        folds = stratified_group_folds(cond_rows, n_splits=outer_k, seed=split_seed)
        pers_mask = np.array([int(orig_i) in persistent_val_indices for orig_i in keep_idx], dtype=bool)
        for src in sources:
            tasks.append(
                {
                    "cond": cond,
                    "cohort": src["cohort"],
                    "seed": src["seed"],
                    "source_id": src["source_id"],
                    "X": src["features"][keep_idx],
                    "y": y,
                    "g": g,
                    "folds": folds,
                    "pers_mask": pers_mask,
                    "inner_k": inner_k,
                    "base_seed": split_seed + 17 * src["seed"],
                    "rcfg": config["readouts"],
                    "device": device,
                }
            )

    summary_rows = []
    oof_store = {}
    # On CUDA, run tasks sequentially on GPU (or 1 process per GPU) to avoid CUDA fork issues.
    workers = 1 if device == "cuda" else max(1, int(args.workers))
    print(f"Launching {len(tasks)} (condition, backbone) tasks on device={device} (workers={workers})...", flush=True)
    if workers == 1:
        for task in tasks:
            res = _run_source_task(task)
            summary_rows.extend(res["rows"])
            oof_store.update(res["oof"])
            print(f"  done [{res['cond']}] {res['sid']} in {res['elapsed']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_source_task, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                summary_rows.extend(res["rows"])
                oof_store.update(res["oof"])
                print(f"  done [{res['cond']}] {res['sid']} in {res['elapsed']:.1f}s", flush=True)

    # Canonical sort for deterministic CSV output regardless of worker finish order.
    cond_ord = {c: i for i, c in enumerate(config["cv"]["conditions"])}
    cohort_ord = {"standard_erm": 0, "erm_star": 1}
    method_ord = {m: i for i, m in enumerate(METHODS_ORDER)}
    summary_rows.sort(
        key=lambda r: (cond_ord[r["condition"]], cohort_ord[r["cohort"]], r["seed"], method_ord[r["method"]])
    )

    agg_rows = []
    for cond in config["cv"]["conditions"]:
        for cohort in ("standard_erm", "erm_star"):
            for method_name in METHODS_ORDER:
                sub = [
                    r for r in summary_rows
                    if r["condition"] == cond and r["cohort"] == cohort and r["method"] == method_name
                ]
                if not sub:
                    continue
                wgas = np.array([r["wga"] for r in sub], dtype=float)
                accs = np.array([r["overall"] for r in sub], dtype=float)
                resc = np.array(
                    [float(r["persistent_rescue_pct"]) for r in sub if r["persistent_rescue_pct"] != ""], dtype=float
                )
                agg_rows.append(
                    {
                        "condition": cond,
                        "cohort": cohort,
                        "method": method_name,
                        "n_seeds": len(sub),
                        "wga_mean": round(float(wgas.mean()), 4),
                        "wga_sd": round(float(wgas.std(ddof=1)), 4) if len(wgas) > 1 else 0.0,
                        "overall_mean": round(float(accs.mean()), 4),
                        "g01_mean": round(float(np.mean([r["acc_01"] for r in sub])), 4),
                        "g10_mean": round(float(np.mean([r["acc_10"] for r in sub])), 4),
                        "persistent_rescue_pct_mean": round(float(resc.mean()), 2) if len(resc) else 0.0,
                    }
                )

    gate_cfg = config["gate1"]
    es_rel_ref = max(
        r["wga_mean"] for r in agg_rows
        if r["condition"] == "relabel" and r["cohort"] == "erm_star" and r["method"] in ("dfr_l2", "dpe_oof")
    )
    keep_rel, rel_rows = apply_label_condition(official_val_rows, "relabel", species=species)
    y_rel = np.array([r["y"] for r in rel_rows], dtype=int)
    pers_rel_mask = np.array([int(orig_i) in persistent_val_indices for orig_i in keep_rel], dtype=bool)

    per_seed_gate = []
    passing_seeds = 0
    for seed in range(5):
        sid = f"standard_erm_s{seed}"
        seed_rows = {
            r["method"]: r for r in summary_rows
            if r["condition"] == "relabel" and r["cohort"] == "standard_erm" and r["seed"] == seed
        }
        best_nonlin_method = max(("frozen_mlp", "frozen_rbf_svm"), key=lambda m: seed_rows[m]["wga"])
        best_nonlin_wga = seed_rows[best_nonlin_method]["wga"]
        gap_pp = float(es_rel_ref - best_nonlin_wga)

        mlp_pred = oof_store[f"relabel__{sid}__frozen_mlp"].argmax(axis=-1)
        svm_pred = oof_store[f"relabel__{sid}__frozen_rbf_svm"].argmax(axis=-1)
        either_rescued = ((mlp_pred == y_rel) | (svm_pred == y_rel)) & pers_rel_mask
        unresolved_frac = (
            1.0 - float(either_rescued.sum()) / float(pers_rel_mask.sum())
            if pers_rel_mask.any() else 0.0
        )
        qualifies = (
            gap_pp >= float(gate_cfg["min_wga_gap_pp"])
            and unresolved_frac >= float(gate_cfg["min_unresolved_fraction"])
        )
        passing_seeds += int(qualifies)
        per_seed_gate.append(
            {
                "seed": seed,
                "best_nonlinear_method": best_nonlin_method,
                "best_nonlinear_wga": round(best_nonlin_wga, 4),
                "ermstar_reference_wga": round(es_rel_ref, 4),
                "wga_gap_pp": round(gap_pp, 4),
                "persistent_count": int(pers_rel_mask.sum()),
                "unresolved_by_both_mlp_and_svm": int(pers_rel_mask.sum() - either_rescued.sum()),
                "unresolved_fraction": round(unresolved_frac, 4),
                "qualifies": qualifies,
            }
        )

    gate1_decision = {
        "study_id": config["study_id"],
        "decision": "GO" if passing_seeds >= int(gate_cfg["min_standard_seeds"]) else "NO-GO",
        "qualifying_standard_seeds": passing_seeds,
        "required_standard_seeds": int(gate_cfg["min_standard_seeds"]),
        "criteria": gate_cfg,
        "ermstar_reference_wga_relabel": round(es_rel_ref, 4),
        "per_seed": per_seed_gate,
    }

    write_csv(out_dir / "readout_ladder_summary.csv", summary_rows)
    write_csv(out_dir / "readout_ladder_aggregate.csv", agg_rows)
    np.savez_compressed(out_dir / "readout_oof_predictions.npz", **oof_store)
    plot_ladder(agg_rows, out_dir / "readout_ladder.png")
    (out_dir / "gate1_decision.json").write_text(json.dumps(gate1_decision, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Direction 2 · Phase A Results (Validation Only)",
        "",
        f"**Pre-Registered Gate 1 Decision:** **{gate1_decision['decision']}** "
        f"({passing_seeds}/{gate1_decision['required_standard_seeds']} qualifying standard-ERM seeds).",
        "",
        "## Frozen Post-GAP Readout Ladder (5-Fold OOF, Relabelled Validation = 1,199)",
        "",
        "| Cohort | Method | OOF WGA (%) | Overall (%) | G01 (%) | G10 (%) | Persistent Failure Rescue (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        if r["condition"] != "relabel":
            continue
        wga_str = f"{r['wga_mean']:.2f} ± {r['wga_sd']:.2f}" if r["n_seeds"] > 1 else f"{r['wga_mean']:.2f}"
        lines.append(
            f"| {r['cohort']} | `{r['method']}` | {wga_str} | {r['overall_mean']:.2f} | "
            f"{r['g01_mean']:.2f} | {r['g10_mean']:.2f} | {r['persistent_rescue_pct_mean']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Gate 1 Per-Seed Breakdown (Standard ERM)",
            "",
            "| Seed | Best Nonlinear Readout | Nonlinear WGA (%) | ERM* Ref WGA (%) | Gap (pp) | Unresolved by Both MLP & SVM | Qualifies |",
            "|---:|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in per_seed_gate:
        lines.append(
            f"| {row['seed']} | `{row['best_nonlinear_method']}` | {row['best_nonlinear_wga']:.2f} | "
            f"{row['ermstar_reference_wga']:.2f} | {row['wga_gap_pp']:.2f} | "
            f"{row['unresolved_by_both_mlp_and_svm']}/{row['persistent_count']} ({row['unresolved_fraction']*100:.1f}%) | "
            f"{'YES' if row['qualifies'] else 'NO'} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "study_id": config["study_id"],
        "step": "phase_a_step2_frozen_readouts",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "device": device,
        "workers": workers,
        "command": ["python", "-B", "-m", "scripts.analysis.rep_repair_frozen_readouts", *(argv if argv is not None else sys.argv[1:])],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "config_sha256": file_sha256(args.config),
        "gate1_decision": gate1_decision["decision"],
    }
    (out_dir / "readout_ladder_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Phase A Step 2 complete. Gate 1 Decision: {gate1_decision['decision']} ({passing_seeds}/5 seeds).", flush=True)


if __name__ == "__main__":
    main()
