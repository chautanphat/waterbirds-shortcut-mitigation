"""Direction 2 · Phase C · Step 5: Causal Block Patching, Branch Drift, Affine-DFR & Channel Audit.

Validation-only (split == 1, cleaned "relabel" labels, zero test access).
Runs four mechanistic probes across saved standard_erm (seeds 0-4) and erm_star (seeds 0-1):
  Part A: Spatially resolved block activation patching (patch_bg_only, patch_fg_only, patch_full)
          across layer1..layer4.2 on the 26 persistent ERM-only minority failures;
  Part B: Bottleneck skip-branch vs. 3x3 residual-conv drift & ReLU gate-flip decomposition;
  Part C: Stage-wise BN-affine ablation (bn_l1_l2, bn_l3, bn_l4, bn_l3_l4, bn_all) + Affine-DFR
          counterfactual accuracy & Layer-4 foreground-cell drift audit;
  Part D: Layer-4 channel bleed sensitivity S_c vs. core signal C_c vs. mask-free observational
          group-shift proxy G_c (Spearman rank correlation).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.resnet import load_resnet_checkpoint
from src.methods.rep_repair.controls import (
    ValImageDataset,
    extract_val_stage_cache,
    pool_with_spatial_masks,
)
from src.methods.rep_repair.counterfactuals import (
    build_clean_bg_donors,
    composite_foreground,
    foreground_feature_drift,
    make_neutral_gray_bg,
    make_ood_backgrounds,
    rgb_to_model_tensor,
)
from src.methods.rep_repair.patching import (
    PATCH_BLOCKS,
    bottleneck_branch_decomposition,
    capture_block_outputs,
    configure_bn_stage_params,
    forward_with_spatial_patch,
    summarize_bn_affine_shifts,
)
from src.methods.rep_repair.phase_a_data import (
    TERN_SPECIES,
    apply_label_condition,
    balanced_group_weights,
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
    recover_backbones,
)

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("orig", "matched_bg", "flipped_bg", "neutral_gray", "ood_bg")
BN_SCOPES = ("bn_l1_l2", "bn_l3", "bn_l4", "bn_l3_l4", "bn_all")
BLOCK_NAMES = tuple(name for name, _, _ in PATCH_BLOCKS)
BRANCH_BLOCKS = ("layer3.0", "layer3.5", "layer4.0", "layer4.1", "layer4.2")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def eval_oof_threshold_metrics(p1: np.ndarray, y: np.ndarray, g: np.ndarray, folds: np.ndarray) -> dict:
    """Compute raw WGA, strictly OOF-thresholded WGA/Acc, and threshold-free Minority-Group AUROC."""
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
    m_raw = group_metrics(y, (p1 >= 0.5).astype(int), g)
    m_thr = group_metrics(y, preds, g)
    min_mask = np.isin(g, ("01", "10"))
    return {
        "raw_wga": round(m_raw["wga"], 4),
        "oof_thr_wga": round(m_thr["wga"], 4),
        "oof_thr_acc": round(m_thr["overall"], 4),
        "min_grp_auc": round(float(roc_auc_score(y[min_mask], p1[min_mask])) * 100.0, 4),
        "all_auc": round(float(roc_auc_score(y, p1)) * 100.0, 4),
    }


def fit_fold_dfr_heads(X: np.ndarray, y: np.ndarray, g: np.ndarray, outer_folds: np.ndarray, c_val: float = 0.3):
    models = {}
    for f in range(int(outer_folds.max()) + 1):
        tr = outer_folds != f
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X[tr]), y[tr], sample_weight=balanced_group_weights(g[tr]))
        models[f] = (sc, clf)
    return models


def train_bn_scope_folds(
    base_model: nn.Module,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    *,
    bn_scope: str,
    lr: float = 3e-3,
    weight_decay: float = 1e-2,
    epochs: int = 20,
    seed: int = 0,
    device: str = "cuda",
    workers: int = 2,
    return_fold_states: bool = False,
):
    """5-fold OOF training for stage-scoped BN-affine adaptation (all convs frozen)."""
    g_all = np.array([str(r["group"]) for r in val_rows], dtype=str)
    probs = np.zeros((len(val_rows), 2), dtype=np.float64)
    fold_states = {}
    ce = nn.CrossEntropyLoss()

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
        model = copy.deepcopy(base_model).to(device)
        params = configure_bn_stage_params(model, bn_scope)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

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
            model[2].train()
            for _, x, yb, _ in fit_loader:
                opt.zero_grad()
                loss = ce(model(x.to(device)), yb.to(device))
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                sel_preds = np.concatenate(
                    [model(x.to(device)).argmax(dim=1).cpu().numpy() for _, x, _, _ in sel_loader], axis=0
                )
            m = group_metrics(sel_y, sel_preds, sel_g)
            key = (m["wga"], m["overall"])
            if key >= best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            probs[te_idx] = np.concatenate(
                [model(x.to(device)).softmax(dim=1).cpu().numpy() for _, x, _, _ in te_loader], axis=0
            )
        if return_fold_states:
            fold_states[fold] = best_state
    return probs, fold_states


def plot_step5_results(patch_agg: list[dict], branch_agg: list[dict], bn_agg: list[dict], cf_aff_agg: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), layout="constrained")

    # Panel A: Spatial Activation Patching on standard_erm (orig patched from neutral_gray)
    ax = axes[0, 0]
    bx = np.arange(len(BLOCK_NAMES))
    for mode, head, color, ls, label in [
        ("bg_only", "dfr_l2", "#245da6", "-", "Patch BG only -> DFR-L2"),
        ("fg_only", "dfr_l2", "#bc5100", "-", "Patch FG only -> DFR-L2"),
        ("bg_only", "oracle_mask", "#5b9bd5", "--", "Patch BG only -> Oracle-Mask"),
        ("fg_only", "oracle_mask", "#e8985e", "--", "Patch FG only -> Oracle-Mask"),
        ("full", "dfr_l2", "#4a7c59", ":", "Patch Full (upper bound)"),
    ]:
        vals = [
            next(
                r["acc_pct"] for r in patch_agg
                if r["clean_source"] == "neutral_gray" and r["patch_mode"] == mode
                and r["head"] == head and r["block"] == b
            )
            for b in BLOCK_NAMES
        ]
        ax.plot(bx, vals, marker="o", color=color, ls=ls, lw=1.8, label=label)
    ax.set_xticks(bx, BLOCK_NAMES, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy on 26 Persistent Failures (%)")
    ax.set_title("A. Spatial Causal Patching (orig patched from neutral_gray)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    # Panel B: Bottleneck Residual-Conv vs Skip Branch FG Drift
    ax = axes[0, 1]
    rx = np.arange(len(BRANCH_BLOCKS))
    for cohort, col, color, ls, label in [
        ("standard_erm", "residual_conv_fg_drift", "#245da6", "-", "Std ERM · 3x3 Residual Branch"),
        ("standard_erm", "skip_fg_drift", "#5b9bd5", "--", "Std ERM · Skip Branch"),
        ("erm_star", "residual_conv_fg_drift", "#bc5100", "-", "ERM* · 3x3 Residual Branch"),
        ("erm_star", "skip_fg_drift", "#e8985e", "--", "ERM* · Skip Branch"),
    ]:
        vals = [next(r[col] for r in branch_agg if r["cohort"] == cohort and r["block"] == b) for b in BRANCH_BLOCKS]
        ax.plot(rx, vals, marker="s", color=color, ls=ls, lw=1.8, label=label)
    ax.set_xticks(rx, BRANCH_BLOCKS, fontsize=8.5)
    ax.set_ylabel("Relative FG-Cell Feature Drift (orig -> neutral)")
    ax.set_title("B. Bottleneck Branch Drift Attribution (Skip vs. 3x3 Conv)")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.8)

    # Panel C: Stage-Wise BN-Affine Ablation (OOF-Thr WGA & Min-Grp AUROC)
    ax = axes[1, 0]
    sx = np.arange(len(BN_SCOPES))
    w = 0.36
    wgas = [next(r["oof_thr_wga_mean"] for r in bn_agg if r["bn_scope"] == s) for s in BN_SCOPES]
    aucs = [next(r["min_grp_auc_mean"] for r in bn_agg if r["bn_scope"] == s) for s in BN_SCOPES]
    ax.bar(sx - w / 2, wgas, w, color="#245da6", label="OOF-Thr WGA (%)")
    ax.bar(sx + w / 2, aucs, w, color="#4a7c59", label="Minority-Grp AUROC (%)")
    for i, (wv, av) in enumerate(zip(wgas, aucs)):
        ax.text(i - w / 2, wv + 0.3, f"{wv:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w / 2, av + 0.3, f"{av:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(sx, BN_SCOPES, fontsize=8.5)
    ax.set_ylabel("Score (%)")
    ax.set_title("C. Stage-Scoped BN-Affine Ablation (5 Std ERM Seeds)")
    ax.set_ylim(85, 100.5)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    # Panel D: Affine-DFR Counterfactual Accuracy vs Base ERM on 26 Persistent Failures
    ax = axes[1, 1]
    cx = np.arange(len(CONDITIONS))
    for model_tag, color, ls, label in [
        ("base_dfr_l2", "#245da6", "--", "Frozen Std ERM (DFR-L2)"),
        ("affine_dfr_bn_all", "#4a7c59", "-", "Affine-DFR (bn_all)"),
        ("affine_dfr_bn_l3_l4", "#bc5100", "-", "Affine-DFR (bn_l3_l4)"),
    ]:
        vals = [
            next(r["acc_pct"] for r in cf_aff_agg if r["model_tag"] == model_tag and r["condition"] == c)
            for c in CONDITIONS
        ]
        ax.plot(cx, vals, marker="o", color=color, ls=ls, lw=2.0, label=label)
    ax.set_xticks(cx, ["Orig", "Matched\nBG", "Flipped\nBG", "Neutral\nGray", "OOD\nSynth"], fontsize=8.5)
    ax.set_ylabel("Accuracy on 26 Persistent Failures (%)")
    ax.set_title("D. Affine-DFR Counterfactual Robustness Audit")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Direction 2 · Phase C · Step 5 Causal Localization & Affine-DFR Audit", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-stats", type=Path, default=ROOT / "results/rep_repair/phase_a_20261123/failure_audit_stats.json")
    parser.add_argument("--std-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ermstar-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    work_dir = args.work_dir.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = ensure_waterbirds_images(work_dir / "waterbirds")
    seg_root = ensure_cub_segmentations(work_dir / "cub_segmentations")
    backbones = recover_backbones(
        work_dir / "backbones",
        cohorts=("standard_erm", "erm_star"),
        seeds={"standard_erm": list(args.std_seeds), "erm_star": list(args.ermstar_seeds)},
    )

    official_rows = load_val_metadata(image_dir / "metadata.csv")
    _, rel_rows = apply_label_condition(official_rows, "relabel", species=TERN_SPECIES)
    y = np.array([int(r["y"]) for r in rel_rows], dtype=int)
    g = np.array([str(r["group"]) for r in rel_rows], dtype=str)
    outer_folds = stratified_group_folds(rel_rows, n_splits=5, seed=0)

    stats = json.loads(args.failure_stats.read_text(encoding="utf-8"))
    persistent_indices = sorted(stats["persistent_val_indices"])

    print("Preparing clean donor backgrounds & CUB masks for 26 persistent failures...", flush=True)
    donors = build_clean_bg_donors(image_dir, seg_root, rel_rows, per_place=15)
    neutral_bg = make_neutral_gray_bg()
    ood_bgs = make_ood_backgrounds()
    all_masks_7x7 = np.stack([downsample_mask_to_grid(load_val_mask_224(seg_root, r["id"]), grid=7) for r in rel_rows], axis=0)

    pers_bundles = []
    for idx in persistent_indices:
        r = rel_rows[idx]
        with Image.open(image_dir / r["id"]) as im:
            orig_rgb = np.asarray(pil_eval_crop(im.convert("RGB")), dtype=np.float32) / 255.0
        mask_224 = load_val_mask_224(seg_root, r["id"])
        y_true = int(r["y"])
        matched_place, flipped_place = y_true, 1 - y_true
        d_idx = [(idx + k * 5) % len(donors[matched_place]) for k in range(3)]
        views = {
            "orig": [orig_rgb],
            "matched_bg": [composite_foreground(orig_rgb, mask_224, donors[matched_place][j]) for j in d_idx],
            "flipped_bg": [composite_foreground(orig_rgb, mask_224, donors[flipped_place][j]) for j in d_idx],
            "neutral_gray": [composite_foreground(orig_rgb, mask_224, neutral_bg)],
            "ood_bg": [composite_foreground(orig_rgb, mask_224, bg) for bg in ood_bgs],
        }
        pers_bundles.append({
            "val_index": idx,
            "id": r["id"],
            "y": y_true,
            "group": str(r["group"]),
            "fold": int(outer_folds[idx]),
            "mask_224": mask_224,
            "mask_7x7": all_masks_7x7[idx],
            "views": views,
        })

    patch_rows = []
    branch_rows = []
    bn_ablation_rows = []
    bn_shift_rows = []
    cf_aff_rows = []
    aff_drift_rows = []
    channel_corr_rows = []

    for rec in backbones:
        t0 = time.perf_counter()
        cohort, seed, sid, ckpt = rec["cohort"], int(rec["seed"]), rec["source_id"], Path(rec["path"])
        print(f"=== [{sid}] Running Parts A, B, D ===", flush=True)
        base_model = load_resnet_checkpoint(ckpt, device=device)
        base_model.eval()

        cache = extract_val_stage_cache(ckpt, image_dir, rel_rows, device=device, workers=args.workers)
        oracle_all = pool_with_spatial_masks(cache["layer4_map"], all_masks_7x7)
        dfr_heads = fit_fold_dfr_heads(cache["pooled"], y, g, outer_folds)
        oracle_heads = fit_fold_dfr_heads(oracle_all, y, g, outer_folds)

        # Part D arrays: collect per-channel FG activations on orig vs neutral_gray across the 26 failures.
        fg_orig_vecs, fg_neu_vecs, fg_ys = [], [], []

        for b in pers_bundles:
            x_orig = rgb_to_model_tensor(b["views"]["orig"][0])[None].to(device)
            x_neu = rgb_to_model_tensor(b["views"]["neutral_gray"][0])[None].to(device)
            x_mat = rgb_to_model_tensor(b["views"]["matched_bg"][0])[None].to(device)
            y_true = b["y"]
            sc_dfr, clf_dfr = dfr_heads[b["fold"]]
            sc_ora, clf_ora = oracle_heads[b["fold"]]

            # Part B: Bottleneck branch decomposition (orig vs neutral_gray)
            for brow in bottleneck_branch_decomposition(base_model, x_orig, x_neu, b["mask_224"], BRANCH_BLOCKS):
                branch_rows.append({"cohort": cohort, "seed": seed, "val_index": b["val_index"], **brow})

            # Collect Layer-4 FG-only vectors for Part D channel bleed analysis.
            with torch.inference_mode():
                s_orig = base_model.extract_stage_maps(x_orig)
                s_neu = base_model.extract_stage_maps(x_neu)
            v_o = pool_with_spatial_masks(s_orig["layer4_map"].detach().cpu().numpy(), b["mask_7x7"][None])[0]
            v_n = pool_with_spatial_masks(s_neu["layer4_map"].detach().cpu().numpy(), b["mask_7x7"][None])[0]
            fg_orig_vecs.append(v_o)
            fg_neu_vecs.append(v_n)
            fg_ys.append(y_true)

            # Part A: Spatial causal patching (standard_erm only to save runtime)
            if cohort == "standard_erm":
                for clean_src, x_clean in (("neutral_gray", x_neu), ("matched_bg", x_mat)):
                    clean_acts = capture_block_outputs(base_model, x_clean)
                    for bname in BLOCK_NAMES:
                        for pmode in ("bg_only", "fg_only", "full"):
                            out = forward_with_spatial_patch(
                                base_model, x_orig, clean_acts[bname], bname, b["mask_224"], pmode
                            )
                            pooled_np = out["pooled"].cpu().numpy()
                            ora_np = pool_with_spatial_masks(out["layer4_map"].cpu().numpy(), b["mask_7x7"][None])
                            p_dfr = float(clf_dfr.predict_proba(sc_dfr.transform(pooled_np))[0, y_true])
                            p_ora = float(clf_ora.predict_proba(sc_ora.transform(ora_np))[0, y_true])
                            patch_rows.append({
                                "seed": seed,
                                "val_index": b["val_index"],
                                "clean_source": clean_src,
                                "block": bname,
                                "patch_mode": pmode,
                                "dfr_correct": int(p_dfr >= 0.5),
                                "dfr_p_true": p_dfr,
                                "oracle_correct": int(p_ora >= 0.5),
                                "oracle_p_true": p_ora,
                            })

        # Part D: Channel-level bleed sensitivity S_c vs Core C_c vs Observational Mask-Free G_c
        fg_o = np.stack(fg_orig_vecs, axis=0)  # [26, 2048]
        fg_n = np.stack(fg_neu_vecs, axis=0)
        fy = np.array(fg_ys)
        S_c = np.mean(np.abs(fg_o - fg_n), axis=0)
        C_c = np.abs(fg_n[fy == 1].mean(0) - fg_n[fy == 0].mean(0)) / np.maximum(fg_n.std(0), 1e-6)
        # Purely observational mask-free background shift across full 1,199 val pooled features!
        X_val = cache["pooled"]
        g00, g01, g10, g11 = (X_val[g == k].mean(0) for k in ("00", "01", "10", "11"))
        G_c = 0.5 * (np.abs(g01 - g00) + np.abs(g11 - g10))
        G_c_norm = G_c / np.maximum(X_val.std(0), 1e-6)
        top10_share = float(np.sort(S_c)[-205:].sum() / max(S_c.sum(), 1e-6))
        channel_corr_rows.append({
            "cohort": cohort,
            "seed": seed,
            "spearman_S_vs_G_raw": round(float(spearmanr(S_c, G_c).statistic), 4),
            "spearman_S_vs_G_std": round(float(spearmanr(S_c, G_c_norm).statistic), 4),
            "spearman_S_vs_CoreC": round(float(spearmanr(S_c, C_c).statistic), 4),
            "top10pct_channels_bleed_share": round(top10_share, 4),
            "mean_S_c": round(float(S_c.mean()), 4),
        })

        # Part C: Stage-wise BN-affine ablation + Affine-DFR counterfactual & FG-drift audit (standard_erm only)
        if cohort == "standard_erm":
            print(f"    [{sid}] Running Part C: Stage-scoped BN-affine ablations & Affine-DFR CF audit...", flush=True)
            base_cpu = load_resnet_checkpoint(ckpt, device="cpu")
            # Record base DFR-L2 counterfactual accuracy for direct comparison in Panel D.
            with torch.inference_mode():
                for b in pers_bundles:
                    sc_dfr, clf_dfr = dfr_heads[b["fold"]]
                    for cond in CONDITIONS:
                        batch = torch.stack([rgb_to_model_tensor(im) for im in b["views"][cond]], dim=0).to(device)
                        p = float(clf_dfr.predict_proba(sc_dfr.transform(base_model.extract_features(batch).cpu().numpy()))[:, b["y"]].mean())
                        cf_aff_rows.append({"seed": seed, "model_tag": "base_dfr_l2", "condition": cond, "val_index": b["val_index"], "correct": int(p >= 0.5), "p_true": p})

            for scope in BN_SCOPES:
                need_states = scope in ("bn_all", "bn_l3_l4")
                probs, fold_states = train_bn_scope_folds(
                    base_cpu, image_dir, rel_rows, outer_folds,
                    bn_scope=scope, seed=seed, device=device, workers=args.workers,
                    return_fold_states=need_states,
                )
                m = eval_oof_threshold_metrics(probs[:, 1], y, g, outer_folds)
                bn_ablation_rows.append({"seed": seed, "bn_scope": scope, **m})

                if need_states:
                    # Measure learned BN shifts on fold 0 (bn_all) + evaluate counterfactuals & FG drift strictly OOF!
                    if scope == "bn_all":
                        m0 = copy.deepcopy(base_cpu)
                        m0.load_state_dict(fold_states[0])
                        for srow in summarize_bn_affine_shifts(base_cpu, m0):
                            bn_shift_rows.append({"seed": seed, **srow})

                    tag = f"affine_dfr_{scope}"
                    fold_models = {}
                    for f_id, st in fold_states.items():
                        fm = copy.deepcopy(base_cpu).to(device)
                        fm.load_state_dict(st)
                        fm.eval()
                        fold_models[f_id] = fm

                    with torch.inference_mode():
                        for b in pers_bundles:
                            fm = fold_models[b["fold"]]
                            stages_by_cond = {}
                            for cond in CONDITIONS:
                                batch = torch.stack([rgb_to_model_tensor(im) for im in b["views"][cond]], dim=0).to(device)
                                logits = fm(batch)
                                p = float(logits.softmax(dim=1)[:, b["y"]].mean().item())
                                cf_aff_rows.append({"seed": seed, "model_tag": tag, "condition": cond, "val_index": b["val_index"], "correct": int(p >= 0.5), "p_true": p})
                                if cond in ("orig", "neutral_gray", "matched_bg"):
                                    stages_by_cond[cond] = fm.extract_stage_maps(batch[:1])
                            d_neu = foreground_feature_drift(stages_by_cond["orig"], stages_by_cond["neutral_gray"], b["mask_224"])
                            d_mat = foreground_feature_drift(stages_by_cond["orig"], stages_by_cond["matched_bg"], b["mask_224"])
                            for st_name in ("layer1", "layer2", "layer3", "layer4_map"):
                                aff_drift_rows.append({
                                    "seed": seed,
                                    "model_tag": tag,
                                    "val_index": b["val_index"],
                                    "stage": st_name,
                                    "drift_orig_to_neutral": d_neu[st_name],
                                    "drift_orig_to_matched": d_mat[st_name],
                                })
        print(f"    [{sid}] Completed in {time.perf_counter() - t0:.1f}s", flush=True)

    # Aggregate outputs cleanly.
    patch_agg = []
    for clean_src in ("neutral_gray", "matched_bg"):
        for bname in BLOCK_NAMES:
            for pmode in ("bg_only", "fg_only", "full"):
                sub = [r for r in patch_rows if r["clean_source"] == clean_src and r["block"] == bname and r["patch_mode"] == pmode]
                if not sub:
                    continue
                for head, col_c, col_p in (("dfr_l2", "dfr_correct", "dfr_p_true"), ("oracle_mask", "oracle_correct", "oracle_p_true")):
                    patch_agg.append({
                        "clean_source": clean_src,
                        "block": bname,
                        "patch_mode": pmode,
                        "head": head,
                        "acc_pct": round(100.0 * float(np.mean([r[col_c] for r in sub])), 2),
                        "mean_p_true": round(float(np.mean([r[col_p] for r in sub])), 4),
                    })

    branch_agg = []
    for cohort in ("standard_erm", "erm_star"):
        for bname in BRANCH_BLOCKS:
            sub = [r for r in branch_rows if r["cohort"] == cohort and r["block"] == bname]
            if not sub:
                continue
            branch_agg.append({
                "cohort": cohort,
                "block": bname,
                "skip_fg_drift": round(float(np.mean([r["skip_fg_drift"] for r in sub])), 4),
                "residual_conv_fg_drift": round(float(np.mean([r["residual_conv_fg_drift"] for r in sub])), 4),
                "post_relu_fg_drift": round(float(np.mean([r["post_relu_fg_drift"] for r in sub])), 4),
                "residual_delta_share": round(float(np.mean([r["residual_delta_share"] for r in sub])), 4),
                "fg_relu_flip_rate": round(float(np.mean([r["fg_relu_flip_rate"] for r in sub])), 4),
            })

    bn_agg = []
    for scope in BN_SCOPES:
        sub = [r for r in bn_ablation_rows if r["bn_scope"] == scope]
        if not sub:
            continue
        bn_agg.append({
            "bn_scope": scope,
            "n_seeds": len(sub),
            "raw_wga_mean": round(float(np.mean([r["raw_wga"] for r in sub])), 2),
            "oof_thr_wga_mean": round(float(np.mean([r["oof_thr_wga"] for r in sub])), 2),
            "oof_thr_wga_sd": round(float(np.std([r["oof_thr_wga"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "oof_thr_acc_mean": round(float(np.mean([r["oof_thr_acc"] for r in sub])), 2),
            "min_grp_auc_mean": round(float(np.mean([r["min_grp_auc"] for r in sub])), 2),
            "all_auc_mean": round(float(np.mean([r["all_auc"] for r in sub])), 2),
        })

    cf_aff_agg = []
    for tag in ("base_dfr_l2", "affine_dfr_bn_all", "affine_dfr_bn_l3_l4"):
        for cond in CONDITIONS:
            sub = [r for r in cf_aff_rows if r["model_tag"] == tag and r["condition"] == cond]
            if not sub:
                continue
            cf_aff_agg.append({
                "model_tag": tag,
                "condition": cond,
                "acc_pct": round(100.0 * float(np.mean([r["correct"] for r in sub])), 2),
                "mean_p_true": round(float(np.mean([r["p_true"] for r in sub])), 4),
            })

    aff_drift_agg = []
    for tag in ("affine_dfr_bn_all", "affine_dfr_bn_l3_l4"):
        for st in ("layer1", "layer2", "layer3", "layer4_map"):
            sub = [r for r in aff_drift_rows if r["model_tag"] == tag and r["stage"] == st]
            if not sub:
                continue
            aff_drift_agg.append({
                "model_tag": tag,
                "stage": st,
                "drift_orig_to_neutral_mean": round(float(np.mean([r["drift_orig_to_neutral"] for r in sub])), 4),
                "drift_orig_to_matched_mean": round(float(np.mean([r["drift_orig_to_matched"] for r in sub])), 4),
            })

    write_csv(out_dir / "patching_summary.csv", patch_agg)
    write_csv(out_dir / "branch_decomposition.csv", branch_agg)
    write_csv(out_dir / "bn_stage_ablation_per_seed.csv", bn_ablation_rows)
    write_csv(out_dir / "bn_stage_ablation_aggregate.csv", bn_agg)
    write_csv(out_dir / "bn_affine_shifts.csv", bn_shift_rows)
    write_csv(out_dir / "affine_dfr_counterfactuals.csv", cf_aff_agg)
    write_csv(out_dir / "affine_dfr_fg_drift.csv", aff_drift_agg)
    write_csv(out_dir / "channel_bleed_identifiability.csv", channel_corr_rows)
    plot_step5_results(patch_agg, branch_agg, bn_agg, cf_aff_agg, out_dir / "step5_causal_localization.png")

    # Build concise Markdown summary.
    lines = [
        "# Direction 2 · Phase C · Step 5 Causal Localization & Affine-DFR Audit (Validation Only)",
        "",
        "## A. Spatial Causal Patching on 26 Persistent Failures (`orig` patched from `neutral_gray`, Std ERM 5-seed mean)",
        "",
        "| Block | `bg_only` (DFR-L2) | `fg_only` (DFR-L2) | `bg_only` (Oracle-Mask) | `fg_only` (Oracle-Mask) | `full` (DFR-L2) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bname in BLOCK_NAMES:
        def _pa(m, h):
            return next(r["acc_pct"] for r in patch_agg if r["clean_source"] == "neutral_gray" and r["block"] == bname and r["patch_mode"] == m and r["head"] == h)
        lines.append(f"| `{bname}` | {_pa('bg_only','dfr_l2'):.1f}% | {_pa('fg_only','dfr_l2'):.1f}% | {_pa('bg_only','oracle_mask'):.1f}% | {_pa('fg_only','oracle_mask'):.1f}% | {_pa('full','dfr_l2'):.1f}% |")

    lines.extend([
        "",
        "## B. Bottleneck Branch Drift Attribution (Skip vs. 3x3 Residual Conv, `orig -> neutral_gray`)",
        "",
        "| Cohort | Block | Skip FG Drift | 3x3 Residual FG Drift | Post-ReLU FG Drift | Residual Delta Share | FG ReLU Flip Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for r in branch_agg:
        lines.append(
            f"| {r['cohort']} | `{r['block']}` | {r['skip_fg_drift']:.4f} | {r['residual_conv_fg_drift']:.4f} | "
            f"{r['post_relu_fg_drift']:.4f} | {r['residual_delta_share']*100:.1f}% | {r['fg_relu_flip_rate']*100:.1f}% |"
        )

    lines.extend([
        "",
        "## C1. Stage-Scoped BN-Affine Ablation (5 Std ERM Seeds, 5-Fold Cleaned Val OOF)",
        "",
        "| BN Scope | Raw WGA (%) | OOF-Thr WGA (%) | OOF-Thr Acc (%) | Minority-Grp AUROC (%) | All AUROC (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in bn_agg:
        lines.append(
            f"| `{r['bn_scope']}` | {r['raw_wga_mean']:.2f} | {r['oof_thr_wga_mean']:.2f} ± {r['oof_thr_wga_sd']:.2f} | "
            f"{r['oof_thr_acc_mean']:.2f} | {r['min_grp_auc_mean']:.2f} | {r['all_auc_mean']:.2f} |"
        )

    lines.extend([
        "",
        "## C2. Affine-DFR Layer-4 Foreground Drift (`orig -> neutral_gray` / `matched_bg`)",
        "",
        "| Model | Layer 1 | Layer 2 | Layer 3 | Layer 4 |",
        "|---|---:|---:|---:|---:|",
    ])
    for tag in ("affine_dfr_bn_all", "affine_dfr_bn_l3_l4"):
        vals = [next(r["drift_orig_to_neutral_mean"] for r in aff_drift_agg if r["model_tag"] == tag and r["stage"] == st)
                for st in ("layer1", "layer2", "layer3", "layer4_map")]
        lines.append(f"| `{tag}` | " + " | ".join(f"{v:.4f}" for v in vals) + " |")

    lines.extend([
        "",
        "## D. Channel-Level Bleed Identifiability (Spearman Correlation with Mask-Free Observational Group Shift `G_c`)",
        "",
        "| Cohort | Seed | Spearman(`S_c`, `G_c` raw) | Spearman(`S_c`, `G_c` std) | Spearman(`S_c`, Core `C_c`) | Top-10% Channels Bleed Share |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in channel_corr_rows:
        lines.append(
            f"| {r['cohort']} | {r['seed']} | {r['spearman_S_vs_G_raw']:.3f} | {r['spearman_S_vs_G_std']:.3f} | "
            f"{r['spearman_S_vs_CoreC']:.3f} | {r['top10pct_channels_bleed_share']*100:.1f}% |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_c_step5_causal_patching",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "device": device,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
        "backbones": backbones,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Phase C Step 5 complete. Summary saved to:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
