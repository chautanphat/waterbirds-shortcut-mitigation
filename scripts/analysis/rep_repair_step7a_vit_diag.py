"""Direction 2 · Phase C · Step 7a: Waterbirds ViT-B/16 ERM Training & Validation Mechanism Diagnostic.

Runs strictly without test access (split == 0 for ERM training; split == 1 with cleaned "relabel"
labels for 5-fold validation controls, counterfactuals, block drift, and spatial token patching).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.rep_repair_phase_c_step6 import CONDITIONS, eval_oof_metrics, write_csv
from src.datasets.loaders import training_loader
from src.datasets.waterbirds import Waterbirds
from src.methods.rep_repair.counterfactuals import (
    build_clean_bg_donors,
    composite_foreground,
    make_neutral_gray_bg,
    make_ood_backgrounds,
    rgb_to_model_tensor,
)
from src.methods.rep_repair.phase_a_data import (
    TERN_SPECIES,
    apply_label_condition,
    balanced_group_weights,
    file_sha256,
    group_metrics,
    load_val_metadata,
    stratified_group_folds,
)
from src.methods.rep_repair.phase_b_assets import (
    downsample_mask_to_grid,
    ensure_cub_segmentations,
    ensure_waterbirds_images,
    load_val_mask_224,
    pil_eval_crop,
)
from src.methods.rep_repair.controls import ValImageDataset
from src.models.vit import (
    VIT_BLOCK_SITES,
    VIT_PATCH_SITES,
    ViTClassifier,
    configure_vit_ln_params,
    load_vit_checkpoint,
    pool_vit_fg_tokens,
    vit_foreground_token_drift,
)
from src.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[2]
ARM_ORDER = ("erm_raw", "dfr_l2", "oracle_fg_token_pool", "ln_dfr_late", "ln_dfr_all")


def oof_threshold_preds(p1: np.ndarray, y: np.ndarray, g: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Strictly outer-fold thresholded binary predictions mirroring eval_oof_metrics."""
    preds = np.zeros(len(y), dtype=int)
    grid = np.linspace(0.05, 0.95, 181)
    for f in range(int(folds.max()) + 1):
        tr, te = folds != f, folds == f
        best_t, best_k = 0.5, (-1.0, -1.0)
        for t in grid:
            m = group_metrics(y[tr], (p1[tr] >= t).astype(int), g[tr])
            k = (m["wga"], m["overall"])
            if k > best_k:
                best_k, best_t = k, t
        preds[te] = (p1[te] >= best_t).astype(int)
    return preds


def _eval_vit_loader(model: ViTClassifier, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs = []
    with torch.inference_mode():
        for _, xb, _, _ in loader:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(xb.to(device))
            probs.append(logits.float().softmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_vit_erm_backbone(
    image_dir: Path,
    val_rows: list[dict],
    ckpt_path: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    workers: int,
) -> tuple[Path, list[dict]]:
    """Train standard-ERM ViT-B/16 on Waterbirds split == 0 only, or reuse existing checkpoint."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.is_file():
        print(f"=== [vit_s{seed}] Reusing saved checkpoint {ckpt_path.name} ===", flush=True)
        return ckpt_path, []

    seed_everything(seed)
    train_ds = Waterbirds(image_dir, "train")
    train_loader = training_loader(train_ds, batch_size=batch_size, workers=workers, sampling="class_balanced")
    val_loader = DataLoader(ValImageDataset(image_dir, val_rows), batch_size=128, shuffle=False, num_workers=workers)
    val_y = np.array([int(r["y"]) for r in val_rows], dtype=int)
    val_g = np.array([str(r["group"]) for r in val_rows], dtype=str)

    model = ViTClassifier(weights="IMAGENET1K_V1", num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, len(train_loader))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * (float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    eval_epochs = {1, 5, 10, 15, epochs}
    history: list[dict] = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for _, xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(yb)
            count += len(yb)

        row = {"epoch": ep, "train_loss": round(loss_sum / max(1, count), 4)}
        if ep in eval_epochs:
            val_probs = _eval_vit_loader(model, val_loader, device)
            m = group_metrics(val_y, (val_probs[:, 1] >= 0.5).astype(int), val_g)
            row["val_raw_wga"] = round(m["wga"], 2)
            row["val_raw_acc"] = round(m["overall"], 2)
            print(
                f"    [vit_s{seed}] ep {ep:02d}/{epochs} | loss={row['train_loss']:.4f} | "
                f"val raw WGA={row['val_raw_wga']:.2f}% | Acc={row['val_raw_acc']:.2f}% ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
        history.append(row)

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(cpu_state, ckpt_path)
    return ckpt_path, history


@torch.inference_mode()
def extract_vit_val_cache(
    model: ViTClassifier,
    image_dir: Path,
    val_rows: list[dict],
    device: str,
    workers: int,
) -> dict[str, np.ndarray]:
    """Extract frozen [CLS] embeddings, 14x14 patch maps, and head probabilities on split == 1."""
    model.eval()
    loader = DataLoader(ValImageDataset(image_dir, val_rows), batch_size=128, shuffle=False, num_workers=workers)
    probs, cls_feats, patch_maps = [], [], []
    for _, xb, _, _ in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            toks = model.extract_block_tokens(xb.to(device))
        probs.append(toks["logits"].float().softmax(dim=1).cpu().numpy().astype(np.float32))
        cls_feats.append(toks["pooled"].float().cpu().numpy().astype(np.float32))
        patch_maps.append(toks["patch_map"].float().cpu().numpy().astype(np.float32))
    return {
        "erm_probs": np.concatenate(probs, axis=0),
        "cls_feats": np.concatenate(cls_feats, axis=0),
        "patch_maps": np.concatenate(patch_maps, axis=0),
    }


def fit_dfr_fold_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    *,
    inner_splits: int = 4,
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    seed: int = 0,
) -> tuple[np.ndarray, dict[int, tuple[StandardScaler, LogisticRegression]]]:
    """Nested-CV group-weighted L2 logistic regression returning (oof_probs, {fold: (scaler, clf)})."""
    probs = np.zeros((len(y), 2), dtype=np.float64)
    fold_models: dict[int, tuple[StandardScaler, LogisticRegression]] = {}
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=inner_splits, seed=seed + 100 + fold)
        best_c, best_key = c_grid[0], (-1.0, -1.0)
        for c_val in c_grid:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(inner_splits):
                itr = in_folds != infold
                iva = in_folds == infold
                sc = StandardScaler().fit(X_tr[itr])
                clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
                clf.fit(sc.transform(X_tr[itr]), y_tr[itr], sample_weight=balanced_group_weights(g_tr[itr]))
                in_preds[iva] = clf.predict(sc.transform(X_tr[iva]))
            m = group_metrics(y_tr, in_preds, g_tr)
            key = (m["wga"], m["overall"])
            if key > best_key:
                best_key, best_c = key, c_val
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X_tr), y_tr, sample_weight=balanced_group_weights(g_tr))
        probs[te] = clf.predict_proba(sc.transform(X[te]))
        fold_models[fold] = (sc, clf)
    return probs, fold_models


def train_vit_ln_5fold(
    base_cpu: ViTClassifier,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    *,
    scope: str,
    seed: int,
    epochs: int,
    device: str,
    workers: int,
) -> tuple[np.ndarray, dict[int, dict], int]:
    """5-fold LayerNorm-affine + linear-head adaptation (strictly zero outer leakage)."""
    probs = np.zeros((len(val_rows), 2), dtype=np.float64)
    fold_states: dict[int, dict] = {}
    ce = nn.CrossEntropyLoss()
    n_trainable = 0

    for fold in range(int(outer_folds.max()) + 1):
        te_idx = np.flatnonzero(outer_folds == fold)
        tr_idx = np.flatnonzero(outer_folds != fold)
        tr_rows = [val_rows[i] for i in tr_idx]
        te_rows = [val_rows[i] for i in te_idx]

        inner_folds = stratified_group_folds(tr_rows, n_splits=4, seed=seed + 300 + fold)
        fit_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds != 0)]
        sel_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds == 0)]
        sel_y = np.array([int(r["y"]) for r in sel_sub], dtype=int)
        sel_g = np.array([str(r["group"]) for r in sel_sub], dtype=str)

        torch.manual_seed(seed + fold)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed + fold)

        model = copy.deepcopy(base_cpu).to(device)
        params = configure_vit_ln_params(model, scope)
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-2)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        w_fit = balanced_group_weights([str(r["group"]) for r in fit_sub])
        gen = torch.Generator().manual_seed(seed + 50 + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_fit).double(), num_samples=len(fit_sub), replacement=True, generator=gen)
        fit_loader = DataLoader(ValImageDataset(image_dir, fit_sub), batch_size=64, sampler=sampler, num_workers=workers)
        sel_loader = DataLoader(ValImageDataset(image_dir, sel_sub), batch_size=128, shuffle=False, num_workers=workers)
        te_loader = DataLoader(ValImageDataset(image_dir, te_rows), batch_size=128, shuffle=False, num_workers=workers)

        best_key = (-1.0, -1.0)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for _ in range(epochs):
            model.eval()
            for _, xb, yb, _ in fit_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss = ce(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            sel_p1 = _eval_vit_loader(model, sel_loader, device)[:, 1]
            sel_min = np.isin(sel_g, ("01", "10"))
            sel_auc = float(roc_auc_score(sel_y[sel_min], sel_p1[sel_min])) * 100.0
            sel_m = group_metrics(sel_y, (sel_p1 >= 0.5).astype(int), sel_g)
            key = (round(sel_auc, 3), round(sel_m["wga"], 3))
            if key >= best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs[te_idx] = _eval_vit_loader(model, te_loader, device)
        fold_states[fold] = best_state
    return probs, fold_states, n_trainable


def select_vit_failure_and_control_indices(
    dfr_probs_by_seed: dict[int, np.ndarray],
    y: np.ndarray,
    g: np.ndarray,
    outer_folds: np.ndarray,
    seeds: list[int],
) -> tuple[list[int], list[int], str]:
    """Identify ViT DFR-L2 persistent minority failures + matched passing minority controls."""
    min_indices = np.flatnonzero(np.isin(g, ("01", "10")))
    miss_counts = np.zeros(len(y), dtype=int)
    p_true_sum = np.zeros(len(y), dtype=np.float64)
    for s in seeds:
        probs = dfr_probs_by_seed[s]
        preds = oof_threshold_preds(probs[:, 1], y, g, outer_folds)
        miss_counts += (preds != y).astype(int)
        p_true_sum += probs[np.arange(len(y)), y]

    mean_p_true = p_true_sum / float(len(seeds))
    req_misses = min(2, len(seeds))
    pers = [int(i) for i in min_indices if miss_counts[i] >= req_misses]
    if len(pers) >= 15:
        criterion = f"missed_by_>={req_misses}/{len(seeds)}_dfr_l2_seeds"
    else:
        ranked = sorted(min_indices.tolist(), key=lambda i: float(mean_p_true[i]))
        pers = sorted(ranked[:25])
        criterion = f"lowest_mean_p_true_top25_fallback (only {len([i for i in min_indices if miss_counts[i] >= req_misses])} missed)"

    pers_set = set(pers)
    controls: list[int] = []
    for grp in ("01", "10"):
        cand = [int(i) for i in min_indices if g[i] == grp and miss_counts[i] == 0 and i not in pers_set]
        cand_sorted = sorted(cand, key=lambda i: float(mean_p_true[i]), reverse=True)
        controls.extend(cand_sorted[:13])
    return sorted(pers), sorted(controls), criterion


def plot_step7a_vit(agg_rows: list[dict], cf_agg: list[dict], drift_agg: list[dict], patch_agg: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), layout="constrained")
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in agg_rows)]
    pretty = {
        "erm_raw": "ViT ERM\n(Raw)",
        "dfr_l2": "DFR-L2\n([CLS])",
        "oracle_fg_token_pool": "Oracle FG\nToken Pool",
        "ln_dfr_late": "LN-DFR\n(Late 8-11)",
        "ln_dfr_all": "LN-DFR\n(All LN)",
    }
    x = np.arange(len(arms))

    # Panel A: Val OOF WGA & Minority AUROC
    ax = axes[0, 0]
    w = 0.36
    wgas = [next(r["oof_thr_wga_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    wsds = [next(r["oof_thr_wga_sd"] for r in agg_rows if r["arm"] == a) for a in arms]
    aucs = [next(r["min_grp_auc_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    ax.bar(x - w / 2, wgas, w, yerr=wsds, capsize=3, color="#245da6", label="OOF-Thr WGA (%)")
    ax.bar(x + w / 2, aucs, w, color="#4a7c59", label="Minority-Grp AUROC (%)")
    for i, (wv, av) in enumerate(zip(wgas, aucs)):
        ax.text(i - w / 2, wv + 0.3, f"{wv:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w / 2, av + 0.3, f"{av:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("A. Waterbirds ViT-B/16 Val Control Ladder (3 Seeds)")
    ax.set_ylim(max(50.0, min(wgas) - 6.0), 100.8)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    # Panel B: Counterfactual Robustness
    ax = axes[0, 1]
    cx = np.arange(len(CONDITIONS))
    cf_arms = ["dfr_l2", "oracle_fg_token_pool", "ln_dfr_all"]
    cf_cols = {"dfr_l2": "#245da6", "oracle_fg_token_pool": "#bc5100", "ln_dfr_all": "#2e7d32"}
    for arm in cf_arms:
        sub = [next(r["acc_pct"] for r in cf_agg if r["arm"] == arm and r["condition"] == c) for c in CONDITIONS]
        ax.plot(cx, sub, marker="o", lw=2.0, color=cf_cols[arm], label=arm)
    ax.set_xticks(cx, ["Orig", "Matched\nBG", "Flipped\nBG", "Neutral\nGray", "OOD\nSynth"], fontsize=8.5)
    ax.set_ylabel("Accuracy on ViT Failures (%)")
    ax.set_title("B. Counterfactual Robustness Across 5 Background Views")
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    # Panel C: Block-wise Foreground Token Drift (orig -> neutral_gray)
    ax = axes[1, 0]
    sites = list(VIT_BLOCK_SITES) + ["final_cls"]
    sx = np.arange(len(sites))
    for cohort_name, label, col, ls in (
        ("frozen_persistent", "Frozen ViT (Failures)", "#c62828", "-"),
        ("frozen_control", "Frozen ViT (Passing Ctrls)", "#245da6", "--"),
        ("ln_all_persistent", "LN-DFR All (Failures)", "#2e7d32", "-."),
    ):
        vals = [next(r["drift_neutral_mean"] for r in drift_agg if r["cohort"] == cohort_name and r["site"] == s) for s in sites]
        ax.plot(sx, vals, marker="o", ms=4, lw=1.8, color=col, ls=ls, label=label)
    ax.set_xticks(sx, ["emb"] + [f"b{i}" for i in range(12)] + ["CLS"], fontsize=7.5)
    ax.set_ylabel("FG-Token Relative L2 Drift (orig -> neutral)")
    ax.set_title("C. Depth Profile of Background Bleed Into Bird Tokens")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    # Panel D: Spatial Token Patching (orig <-> neutral_gray)
    ax = axes[1, 1]
    px = np.arange(len(VIT_PATCH_SITES))
    for patch_mode, scorer, label, col, ls in (
        ("bg_only", "dfr_l2", "BG-only -> Neutral (DFR [CLS])", "#245da6", "-"),
        ("fg_only", "dfr_l2", "FG-only -> Neutral (DFR [CLS])", "#bc5100", "-"),
        ("bg_only", "oracle_fg_token_pool", "BG-only -> Neutral (Oracle FG Pool)", "#2e7d32", "--"),
        ("fg_only", "oracle_fg_token_pool", "FG-only -> Neutral (Oracle FG Pool)", "#7b3294", "--"),
    ):
        vals = [
            next(r["acc_pct"] for r in patch_agg if r["patch_mode"] == patch_mode and r["scorer"] == scorer and r["site"] == s)
            for s in VIT_PATCH_SITES
        ]
        ax.plot(px, vals, marker="o", ms=4.5, lw=1.8, color=col, ls=ls, label=label)
    ax.set_xticks(px, ["emb" if s == "embed" else s.replace("block_", "b") for s in VIT_PATCH_SITES], fontsize=8)
    ax.set_ylabel("Patched Counterfactual Accuracy (%)")
    ax.set_title("D. Block-Wise Spatial Token Patching (orig <-> neutral_gray)")
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Direction 2 · Step 7a Waterbirds ViT-B/16 ERM Training & Mechanism Diagnostic", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ln-epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    work_dir = args.work_dir.resolve()
    ckpt_dir = args.ckpt_dir.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    image_dir = ensure_waterbirds_images(work_dir / "waterbirds")
    seg_root = ensure_cub_segmentations(work_dir / "cub_segmentations")

    official_rows = load_val_metadata(image_dir / "metadata.csv")
    _, rel_rows = apply_label_condition(official_rows, "relabel", species=TERN_SPECIES)
    y = np.array([int(r["y"]) for r in rel_rows], dtype=int)
    g = np.array([str(r["group"]) for r in rel_rows], dtype=str)
    outer_folds = stratified_group_folds(rel_rows, n_splits=5, seed=0)

    all_masks_224 = [load_val_mask_224(seg_root, r["id"]) for r in rel_rows]
    all_masks_14 = np.stack([downsample_mask_to_grid(m, grid=14) for m in all_masks_224], axis=0)

    # Stage 1 + Stage 2 per seed.
    ckpt_records = []
    per_seed_rows = []
    oof_store: dict[str, np.ndarray] = {}
    dfr_probs_by_seed: dict[int, np.ndarray] = {}
    dfr_models_by_seed: dict[int, dict] = {}
    oracle_models_by_seed: dict[int, dict] = {}
    ln_all_states_by_seed: dict[int, dict] = {}
    base_models_cpu: dict[int, ViTClassifier] = {}

    for seed in args.seeds:
        ckpt_path = ckpt_dir / f"vit_b16_s{seed}_ckpt_last.pt"
        _, history = train_vit_erm_backbone(
            image_dir, rel_rows, ckpt_path,
            seed=seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, workers=args.workers,
        )
        if history:
            (out_dir / f"train_history_s{seed}.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        ckpt_records.append({"seed": seed, "path": str(ckpt_path), "sha256": file_sha256(ckpt_path)})

        base_cpu = load_vit_checkpoint(ckpt_path, device="cpu")
        base_models_cpu[seed] = base_cpu
        cache = extract_vit_val_cache(copy.deepcopy(base_cpu).to(device), image_dir, rel_rows, device=device, workers=args.workers)
        fg_feats = pool_vit_fg_tokens(cache["patch_maps"], all_masks_14)

        dfr_probs, dfr_models = fit_dfr_fold_models(cache["cls_feats"], y, g, outer_folds, seed=seed)
        ora_probs, ora_models = fit_dfr_fold_models(fg_feats, y, g, outer_folds, seed=seed)
        ln_late_probs, _, ln_late_params = train_vit_ln_5fold(
            base_cpu, image_dir, rel_rows, outer_folds,
            scope="ln_late", seed=seed, epochs=args.ln_epochs, device=device, workers=args.workers,
        )
        ln_all_probs, ln_all_states, ln_all_params = train_vit_ln_5fold(
            base_cpu, image_dir, rel_rows, outer_folds,
            scope="ln_all", seed=seed, epochs=args.ln_epochs, device=device, workers=args.workers,
        )

        dfr_probs_by_seed[seed] = dfr_probs
        dfr_models_by_seed[seed] = dfr_models
        oracle_models_by_seed[seed] = ora_models
        ln_all_states_by_seed[seed] = ln_all_states

        arm_probs = {
            "erm_raw": (cache["erm_probs"], sum(p.numel() for p in base_cpu.parameters())),
            "dfr_l2": (dfr_probs, 768 * 2 + 2),
            "oracle_fg_token_pool": (ora_probs, 768 * 2 + 2),
            "ln_dfr_late": (ln_late_probs, ln_late_params),
            "ln_dfr_all": (ln_all_probs, ln_all_params),
        }
        for arm in ARM_ORDER:
            probs, n_params = arm_probs[arm]
            oof_store[f"vit_s{seed}__{arm}"] = probs.astype(np.float32)
            m = eval_oof_metrics(probs[:, 1], y, g, outer_folds)
            per_seed_rows.append({"seed": seed, "arm": arm, "trainable_params": n_params, **m})
            print(
                f"    [vit_s{seed}] {arm:21s}: OOF-Thr WGA={m['oof_thr_wga']:.2f}% | "
                f"Min-AUC={m['min_grp_auc']:.2f}% | Acc={m['oof_thr_acc']:.2f}%",
                flush=True,
            )

    # Stage 3: Select ViT persistent failures + passing controls, then run counterfactuals, drift & patching.
    pers_indices, ctrl_indices, failure_criterion = select_vit_failure_and_control_indices(
        dfr_probs_by_seed, y, g, outer_folds, list(args.seeds)
    )
    print(
        f"=== Stage 3 ViT Diagnostic: {len(pers_indices)} failures ({failure_criterion}), {len(ctrl_indices)} controls ===",
        flush=True,
    )

    donors = build_clean_bg_donors(image_dir, seg_root, rel_rows, per_place=15)
    neutral_bg = make_neutral_gray_bg()
    ood_bgs = make_ood_backgrounds()

    def _build_bundle(idx: int, include_views: bool):
        r = rel_rows[idx]
        with Image.open(image_dir / r["id"]) as im:
            orig_rgb = np.asarray(pil_eval_crop(im.convert("RGB")), dtype=np.float32) / 255.0
        mask_224 = all_masks_224[idx]
        mask_14 = all_masks_14[idx]
        y_true = int(r["y"])
        bundle = {
            "val_index": idx,
            "id": r["id"],
            "y": y_true,
            "group": str(r["group"]),
            "fold": int(outer_folds[idx]),
            "mask_224": mask_224,
            "mask_14": mask_14,
            "orig_rgb": orig_rgb,
            "neutral_rgb": composite_foreground(orig_rgb, mask_224, neutral_bg),
            "matched_rgb": composite_foreground(orig_rgb, mask_224, donors[y_true][idx % len(donors[y_true])]),
        }
        if include_views:
            d_idx = [(idx + k * 5) % len(donors[y_true]) for k in range(3)]
            bundle["views"] = {
                "orig": [orig_rgb],
                "matched_bg": [composite_foreground(orig_rgb, mask_224, donors[y_true][j]) for j in d_idx],
                "flipped_bg": [composite_foreground(orig_rgb, mask_224, donors[1 - y_true][j]) for j in d_idx],
                "neutral_gray": [bundle["neutral_rgb"]],
                "ood_bg": [composite_foreground(orig_rgb, mask_224, bg) for bg in ood_bgs],
            }
        return bundle

    pers_bundles = [_build_bundle(i, include_views=True) for i in pers_indices]
    ctrl_bundles = [_build_bundle(i, include_views=False) for i in ctrl_indices]

    cf_rows, drift_rows, patch_rows = [], [], []
    sites_all = (*VIT_BLOCK_SITES, "final_cls")

    for seed in args.seeds:
        frozen_dev = copy.deepcopy(base_models_cpu[seed]).to(device)
        frozen_dev.eval()
        dfr_models = dfr_models_by_seed[seed]
        ora_models = oracle_models_by_seed[seed]
        ln_models = {}
        for fid, st in ln_all_states_by_seed[seed].items():
            lm = ViTClassifier(weights=None)
            lm.load_state_dict(st)
            lm.eval()
            ln_models[fid] = lm.to(device)

        with torch.inference_mode():
            for b in pers_bundles:
                fid = b["fold"]
                sc_dfr, clf_dfr = dfr_models[fid]
                sc_ora, clf_ora = ora_models[fid]
                ln_m = ln_models[fid]

                # 1) Counterfactuals across 5 views
                orig_tok = None
                neu_tok = None
                for cond in CONDITIONS:
                    batch = torch.stack([rgb_to_model_tensor(im) for im in b["views"][cond]], dim=0).to(device)
                    tok_f = frozen_dev.extract_block_tokens(batch)
                    tok_ln = ln_m.extract_block_tokens(batch)
                    if cond == "orig":
                        orig_tok = {k: v[:1] for k, v in tok_f.items()}
                    elif cond == "neutral_gray":
                        neu_tok = {k: v[:1] for k, v in tok_f.items()}

                    cls_np = tok_f["pooled"].cpu().numpy()
                    fg_np = pool_vit_fg_tokens(tok_f["patch_map"].cpu().numpy(), np.repeat(b["mask_14"][None], len(cls_np), axis=0))
                    p_dfr = float(clf_dfr.predict_proba(sc_dfr.transform(cls_np))[:, b["y"]].mean())
                    p_ora = float(clf_ora.predict_proba(sc_ora.transform(fg_np))[:, b["y"]].mean())
                    p_ln = float(tok_ln["logits"].softmax(dim=1)[:, b["y"]].mean().item())
                    for arm_name, p_val in (("dfr_l2", p_dfr), ("oracle_fg_token_pool", p_ora), ("ln_dfr_all", p_ln)):
                        cf_rows.append({"seed": seed, "arm": arm_name, "condition": cond, "val_index": b["val_index"], "correct": int(p_val >= 0.5), "p_true": p_val})

                # 2) Foreground-token depth drift (frozen & ln_dfr_all on persistent failures)
                mat_t = rgb_to_model_tensor(b["matched_rgb"]).unsqueeze(0).to(device)
                neu_t = rgb_to_model_tensor(b["neutral_rgb"]).unsqueeze(0).to(device)
                ori_t = rgb_to_model_tensor(b["orig_rgb"]).unsqueeze(0).to(device)
                mat_tok = frozen_dev.extract_block_tokens(mat_t)
                d_frz_neu = vit_foreground_token_drift(orig_tok, neu_tok, b["mask_224"])
                d_frz_mat = vit_foreground_token_drift(orig_tok, mat_tok, b["mask_224"])
                d_ln_neu = vit_foreground_token_drift(ln_m.extract_block_tokens(ori_t), ln_m.extract_block_tokens(neu_t), b["mask_224"])
                for s in sites_all:
                    drift_rows.append({"seed": seed, "cohort": "frozen_persistent", "site": s, "val_index": b["val_index"], "drift_neutral": d_frz_neu[s], "drift_matched": d_frz_mat[s]})
                    drift_rows.append({"seed": seed, "cohort": "ln_all_persistent", "site": s, "val_index": b["val_index"], "drift_neutral": d_ln_neu[s], "drift_matched": float("nan")})

                # 3) Spatial token patching across ViT blocks (orig <-> neutral_gray)
                m_tok = torch.from_numpy(b["mask_14"].reshape(1, 196, 1)).to(device=device, dtype=orig_tok["embed"].dtype)
                for site in VIT_PATCH_SITES:
                    o_seq, n_seq = orig_tok[site], neu_tok[site]
                    o_p, n_p = o_seq[:, 1:, :], n_seq[:, 1:, :]
                    patched_bg = torch.cat([o_seq[:, :1, :], m_tok * o_p + (1.0 - m_tok) * n_p], dim=1)
                    patched_fg = torch.cat([o_seq[:, :1, :], (1.0 - m_tok) * o_p + m_tok * n_p], dim=1)
                    for pmode, ptok in (("bg_only", patched_bg), ("fg_only", patched_fg)):
                        res = frozen_dev.forward_from_site(ptok, site)
                        p_dfr = float(clf_dfr.predict_proba(sc_dfr.transform(res["pooled"].cpu().numpy()))[0, b["y"]])
                        fg_v = pool_vit_fg_tokens(res["patch_map"].cpu().numpy(), b["mask_14"][None])
                        p_ora = float(clf_ora.predict_proba(sc_ora.transform(fg_v))[0, b["y"]])
                        for scorer, pv in (("dfr_l2", p_dfr), ("oracle_fg_token_pool", p_ora)):
                            patch_rows.append({"seed": seed, "patch_mode": pmode, "scorer": scorer, "site": site, "val_index": b["val_index"], "correct": int(pv >= 0.5), "p_true": pv})

            # Frozen FG drift on passing controls
            for b in ctrl_bundles:
                t_o = frozen_dev.extract_block_tokens(rgb_to_model_tensor(b["orig_rgb"]).unsqueeze(0).to(device))
                t_n = frozen_dev.extract_block_tokens(rgb_to_model_tensor(b["neutral_rgb"]).unsqueeze(0).to(device))
                t_m = frozen_dev.extract_block_tokens(rgb_to_model_tensor(b["matched_rgb"]).unsqueeze(0).to(device))
                d_neu = vit_foreground_token_drift(t_o, t_n, b["mask_224"])
                d_mat = vit_foreground_token_drift(t_o, t_m, b["mask_224"])
                for s in sites_all:
                    drift_rows.append({"seed": seed, "cohort": "frozen_control", "site": s, "val_index": b["val_index"], "drift_neutral": d_neu[s], "drift_matched": d_mat[s]})

    # Aggregate all tables
    agg_rows = []
    for arm in ARM_ORDER:
        sub = [r for r in per_seed_rows if r["arm"] == arm]
        agg_rows.append({
            "arm": arm,
            "n_seeds": len(sub),
            "trainable_params": sub[0]["trainable_params"],
            "raw_wga_mean": round(float(np.mean([r["raw_wga"] for r in sub])), 2),
            "oof_thr_wga_mean": round(float(np.mean([r["oof_thr_wga"] for r in sub])), 2),
            "oof_thr_wga_sd": round(float(np.std([r["oof_thr_wga"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "oof_thr_acc_mean": round(float(np.mean([r["oof_thr_acc"] for r in sub])), 2),
            "g01_thr_mean": round(float(np.mean([r["g01_thr"] for r in sub])), 2),
            "g10_thr_mean": round(float(np.mean([r["g10_thr"] for r in sub])), 2),
            "min_grp_auc_mean": round(float(np.mean([r["min_grp_auc"] for r in sub])), 2),
            "all_auc_mean": round(float(np.mean([r["all_auc"] for r in sub])), 2),
        })

    cf_agg = []
    for arm in ("dfr_l2", "oracle_fg_token_pool", "ln_dfr_all"):
        for cond in CONDITIONS:
            sub = [r for r in cf_rows if r["arm"] == arm and r["condition"] == cond]
            cf_agg.append({"arm": arm, "condition": cond, "acc_pct": round(100.0 * float(np.mean([r["correct"] for r in sub])), 2), "mean_p_true": round(float(np.mean([r["p_true"] for r in sub])), 4)})

    drift_agg = []
    for cohort_name in ("frozen_persistent", "frozen_control", "ln_all_persistent"):
        for s in sites_all:
            sub = [r for r in drift_rows if r["cohort"] == cohort_name and r["site"] == s]
            drift_agg.append({
                "cohort": cohort_name,
                "site": s,
                "drift_neutral_mean": round(float(np.mean([r["drift_neutral"] for r in sub])), 4),
                "drift_matched_mean": round(float(np.nanmean([r["drift_matched"] for r in sub])), 4) if any(not math.isnan(r["drift_matched"]) for r in sub) else "",
            })

    patch_agg = []
    for pmode in ("bg_only", "fg_only"):
        for scorer in ("dfr_l2", "oracle_fg_token_pool"):
            for s in VIT_PATCH_SITES:
                sub = [r for r in patch_rows if r["patch_mode"] == pmode and r["scorer"] == scorer and r["site"] == s]
                patch_agg.append({"patch_mode": pmode, "scorer": scorer, "site": s, "acc_pct": round(100.0 * float(np.mean([r["correct"] for r in sub])), 2), "mean_p_true": round(float(np.mean([r["p_true"] for r in sub])), 4)})

    write_csv(out_dir / "step7a_vit_per_seed.csv", per_seed_rows)
    write_csv(out_dir / "step7a_vit_aggregate.csv", agg_rows)
    write_csv(out_dir / "step7a_vit_counterfactuals.csv", cf_agg)
    write_csv(out_dir / "step7a_vit_block_drift.csv", drift_agg)
    write_csv(out_dir / "step7a_vit_spatial_patching.csv", patch_agg)
    np.savez_compressed(out_dir / "vit_oof_predictions.npz", **oof_store)
    plot_step7a_vit(agg_rows, cf_agg, drift_agg, patch_agg, out_dir / "step7a_vit_mechanism.png")

    lines = [
        "# Direction 2 · Phase C · Step 7a Waterbirds ViT-B/16 ERM & Mechanism Diagnostic (Validation Only)",
        "",
        f"Failure subset: **{len(pers_indices)}** minority val images (`{failure_criterion}`); matched passing controls: **{len(ctrl_indices)}**.",
        "",
        "## 1. Cleaned 5-Fold Validation OOF Control Ladder (3 Standard-ERM ViT-B/16 Seeds)",
        "",
        "| Arm | Trainable Params | Raw WGA (%) | OOF-Thr WGA (%) | OOF-Thr Acc (%) | Min-Grp AUROC (%) | All AUROC (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        lines.append(
            f"| `{r['arm']}` | {r['trainable_params']:,} | {r['raw_wga_mean']:.2f} | "
            f"**{r['oof_thr_wga_mean']:.2f} ± {r['oof_thr_wga_sd']:.2f}** | {r['oof_thr_acc_mean']:.2f} | "
            f"**{r['min_grp_auc_mean']:.2f}** | {r['all_auc_mean']:.2f} |"
        )
    lines.extend(["", "## 2. Counterfactual Robustness on ViT Persistent Failures", "", "| Arm | `orig` (%) | `matched` (%) | `flipped` (%) | `neutral` (%) | `ood` (%) |", "|---|---:|---:|---:|---:|---:|"])
    for arm in ("dfr_l2", "oracle_fg_token_pool", "ln_dfr_all"):
        vals = {c: next(r["acc_pct"] for r in cf_agg if r["arm"] == arm and r["condition"] == c) for c in CONDITIONS}
        lines.append(f"| `{arm}` | {vals['orig']:.1f} | {vals['matched_bg']:.1f} | {vals['flipped_bg']:.1f} | {vals['neutral_gray']:.1f} | {vals['ood_bg']:.1f} |")

    lines.extend(["", "## 3. Block-Wise Foreground Token Drift (`orig -> neutral_gray`)", "", "| Site | Frozen Failures | Frozen Passing Ctrls | `ln_dfr_all` Failures |", "|---|---:|---:|---:|"])
    for s in sites_all:
        f_p = next(r["drift_neutral_mean"] for r in drift_agg if r["cohort"] == "frozen_persistent" and r["site"] == s)
        f_c = next(r["drift_neutral_mean"] for r in drift_agg if r["cohort"] == "frozen_control" and r["site"] == s)
        l_p = next(r["drift_neutral_mean"] for r in drift_agg if r["cohort"] == "ln_all_persistent" and r["site"] == s)
        lines.append(f"| `{s}` | {f_p:.4f} | {f_c:.4f} | {l_p:.4f} |")

    lines.extend(["", "## 4. Spatial Token Patching (`orig <-> neutral_gray`)", "", "| Site | BG-Only DFR (%) | FG-Only DFR (%) | BG-Only Oracle-FG (%) | FG-Only Oracle-FG (%) |", "|---|---:|---:|---:|---:|"])
    for s in VIT_PATCH_SITES:
        b_d = next(r["acc_pct"] for r in patch_agg if r["patch_mode"] == "bg_only" and r["scorer"] == "dfr_l2" and r["site"] == s)
        f_d = next(r["acc_pct"] for r in patch_agg if r["patch_mode"] == "fg_only" and r["scorer"] == "dfr_l2" and r["site"] == s)
        b_o = next(r["acc_pct"] for r in patch_agg if r["patch_mode"] == "bg_only" and r["scorer"] == "oracle_fg_token_pool" and r["site"] == s)
        f_o = next(r["acc_pct"] for r in patch_agg if r["patch_mode"] == "fg_only" and r["scorer"] == "oracle_fg_token_pool" and r["site"] == s)
        lines.append(f"| `{s}` | {b_d:.1f} | {f_d:.1f} | {b_o:.1f} | {f_o:.1f} |")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {
        "step": "phase_c_step7a_vit_b16_diagnostic",
        "split": "train_split0_for_erm__val_only_split1_for_all_diagnostics",
        "test_files_accessed": 0,
        "device": device,
        "seeds": list(args.seeds),
        "failure_criterion": failure_criterion,
        "persistent_val_indices": pers_indices,
        "control_val_indices": ctrl_indices,
        "checkpoints": ckpt_records,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Step 7a ViT diagnostic complete. Summary saved to:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
