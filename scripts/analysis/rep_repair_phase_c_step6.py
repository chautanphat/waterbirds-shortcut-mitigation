"""Direction 2 · Phase C · Step 6: Pre-Layer4 Mechanism-Derived Repair & 6-Arm Ablation.

Validation-only (split == 1, cleaned "relabel" labels, zero test access).
Evaluates 6 strictly matched-budget arms across all 5 standard_erm seeds (5-fold CV):
  1. affine_dfr_linear      (bn_all + linear head, CE only)           -- Step 5 baseline
  2. affine_dfr_cosine      (bn_all + cosine head, CE only)           -- readout control
  3. pre_l4_gate_only       (layer3.5 gate + linear, frozen backbone) -- vs. DAR (layer4.2)
  4. pre_l4_gate_bn_ce      (layer3.5 gate + bn_all + cosine, CE only)
  5. pre_l4_repair_bn_l3l4  (layer3.5 gate + bn_l3_l4 + cosine + layer3.5 drift invariance)
  6. pre_l4_repair_bn_all   (layer3.5 gate + bn_all   + cosine + layer3.5 drift invariance)

Reports Raw WGA, strictly OOF-thresholded WGA/Acc, Minority-Group AUROC, counterfactual
accuracy across 5 views on the 26 persistent failures, Layer-4 FG drift, and post-hoc
alignment of the mask-free 14x14 Pre-Layer4 gate with ground-truth CUB bird masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.resnet import load_resnet_checkpoint
from src.methods.rep_repair.controls import ValImageDataset
from src.methods.rep_repair.counterfactuals import (
    build_clean_bg_donors,
    composite_foreground,
    foreground_feature_drift,
    make_neutral_gray_bg,
    make_ood_backgrounds,
    rgb_to_model_tensor,
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
from src.methods.rep_repair.repair import PreLayer4RepairModel, context_invariance_loss

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("orig", "matched_bg", "flipped_bg", "neutral_gray", "ood_bg")

ARM_SPECS = (
    {"name": "affine_dfr_linear",     "use_gate": False, "bn_scope": "bn_all",   "head_type": "linear", "use_inv": False, "lr": 3e-3, "epochs": 20},
    {"name": "affine_dfr_cosine",     "use_gate": False, "bn_scope": "bn_all",   "head_type": "cosine", "use_inv": False, "lr": 3e-3, "epochs": 20},
    {"name": "pre_l4_gate_only",      "use_gate": True,  "bn_scope": "none",     "head_type": "linear", "use_inv": False, "lr": 3e-3, "epochs": 20},
    {"name": "pre_l4_gate_bn_ce",     "use_gate": True,  "bn_scope": "bn_all",   "head_type": "cosine", "use_inv": False, "lr": 3e-3, "epochs": 20},
    {"name": "pre_l4_repair_bn_l3l4", "use_gate": True,  "bn_scope": "bn_l3_l4", "head_type": "cosine", "use_inv": True,  "lr": 3e-3, "epochs": 20},
    {"name": "pre_l4_repair_bn_all",  "use_gate": True,  "bn_scope": "bn_all",   "head_type": "cosine", "use_inv": True,  "lr": 3e-3, "epochs": 20},
)
ARM_ORDER = tuple(spec["name"] for spec in ARM_SPECS)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def eval_oof_metrics(p1: np.ndarray, y: np.ndarray, g: np.ndarray, folds: np.ndarray) -> dict:
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
        "g01_thr": round(m_thr["groups"]["01"], 4),
        "g10_thr": round(m_thr["groups"]["10"], 4),
        "min_grp_auc": round(float(roc_auc_score(y[min_mask], p1[min_mask])) * 100.0, 4),
        "all_auc": round(float(roc_auc_score(y, p1)) * 100.0, 4),
    }


def train_arm_5fold(
    base_cpu: nn.Module,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    spec: dict,
    *,
    seed: int,
    device: str,
    workers: int,
    drift_weight: float = 1.0,
    cons_weight: float = 0.5,
    gate_sparsity: float = 0.05,
):
    """Train one Step-6 arm across 5 outer folds with strict inner 1/4 validation checkpointing."""
    probs = np.zeros((len(val_rows), 2), dtype=np.float64)
    fold_states = {}
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

        model = PreLayer4RepairModel(
            base_cpu,
            use_gate=spec["use_gate"],
            head_type=spec["head_type"],
            bn_scope=spec["bn_scope"],
        ).to(device)
        params = model.trainable_parameters()
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=float(spec["lr"]), weight_decay=1e-2)

        w_fit = balanced_group_weights([str(r["group"]) for r in fit_sub])
        gen = torch.Generator().manual_seed(seed + 50 + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_fit).double(), num_samples=len(fit_sub), replacement=True, generator=gen)
        fit_loader = DataLoader(ValImageDataset(image_dir, fit_sub), batch_size=64, sampler=sampler, num_workers=workers)
        sel_loader = DataLoader(ValImageDataset(image_dir, sel_sub), batch_size=128, shuffle=False, num_workers=workers)
        te_loader = DataLoader(ValImageDataset(image_dir, te_rows), batch_size=128, shuffle=False, num_workers=workers)

        best_key = (-1.0, -1.0)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for _ in range(int(spec["epochs"])):
            model.set_repair_train_mode()
            for _, xb, yb, gb in fit_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                if spec["use_inv"]:
                    ab = torch.tensor([int(str(s)[1]) for s in gb], dtype=torch.long, device=device)
                    out = model.forward_with_context_swap(xb, ab)
                    loss_ce = ce(out["logits"], yb)
                    inv = context_invariance_loss(
                        out["logits"], out["logits_swap"],
                        out["layer4_map"], out["layer4_map_swap"], out["gate_map"],
                        drift_weight=drift_weight, cons_weight=cons_weight, gate_sparsity=gate_sparsity,
                    )
                    loss = loss_ce + inv["total"]
                else:
                    loss = ce(model(xb), yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.inference_mode():
                sel_p1 = np.concatenate(
                    [model(xb.to(device)).softmax(dim=1)[:, 1].cpu().numpy() for _, xb, _, _ in sel_loader], axis=0
                )
            # Select checkpoint by threshold-free Minority-Group AUROC + raw WGA so class-bias cannot pick poor models.
            sel_min = np.isin(sel_g, ("01", "10"))
            sel_auc = float(roc_auc_score(sel_y[sel_min], sel_p1[sel_min])) * 100.0
            sel_m = group_metrics(sel_y, (sel_p1 >= 0.5).astype(int), sel_g)
            key = (round(sel_auc, 3), round(sel_m["wga"], 3))
            if key >= best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            probs[te_idx] = np.concatenate(
                [model(xb.to(device)).softmax(dim=1).cpu().numpy() for _, xb, _, _ in te_loader], axis=0
            )
        fold_states[fold] = best_state
    return probs, fold_states, n_trainable


def plot_step6(agg_rows: list[dict], cf_agg: list[dict], drift_agg: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6), layout="constrained")
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in agg_rows)]
    pretty = {
        "affine_dfr_linear": "Affine-DFR\n(Linear)",
        "affine_dfr_cosine": "Affine-DFR\n(Cosine)",
        "pre_l4_gate_only": "Pre-L4 Gate\nOnly",
        "pre_l4_gate_bn_ce": "Gate+BN\n(CE only)",
        "pre_l4_repair_bn_l3l4": "Pre-L4 Repair\n(BN L3-L4)",
        "pre_l4_repair_bn_all": "Pre-L4 Repair\n(BN All)",
    }
    x = np.arange(len(arms))

    # Panel A: OOF-Thr WGA and Minority-Group AUROC
    ax = axes[0, 0]
    w = 0.36
    wgas = [next(r["oof_thr_wga_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    wsds = [next(r["oof_thr_wga_sd"] for r in agg_rows if r["arm"] == a) for a in arms]
    aucs = [next(r["min_grp_auc_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    ax.bar(x - w / 2, wgas, w, yerr=wsds, capsize=3, color="#245da6", label="OOF-Thr WGA (%)")
    ax.bar(x + w / 2, aucs, w, color="#4a7c59", label="Minority-Grp AUROC (%)")
    ax.axhline(92.69, color="#8b0000", ls="--", lw=1.0, label="Step 5 Affine-DFR WGA (92.69%)")
    for i, (wv, av) in enumerate(zip(wgas, aucs)):
        ax.text(i - w / 2, wv + 0.35, f"{wv:.2f}", ha="center", va="bottom", fontsize=7.3)
        ax.text(i + w / 2, av + 0.35, f"{av:.2f}", ha="center", va="bottom", fontsize=7.3)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("A. Cleaned 5-Fold Val OOF WGA & Minority AUROC (5 Std ERM Seeds)")
    ax.set_ylim(87, 100.5)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    # Panel B: Layer-4 Foreground-Cell Drift (orig -> neutral_gray)
    ax = axes[0, 1]
    drifts = [next(r["l4_drift_neutral_mean"] for r in drift_agg if r["arm"] == a) for a in arms]
    bars = ax.bar(x, drifts, color="#bc5100")
    ax.axhline(0.8512, color="#444444", ls=":", lw=1.1, label="Frozen Std ERM (0.851)")
    ax.axhline(0.6862, color="#245da6", ls="--", lw=1.1, label="ERM* target (0.686)")
    for b, v in zip(bars, drifts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Layer-4 FG-Cell Drift (orig -> neutral)")
    ax.set_title("B. Layer-4 Background Interference Inside Bird Mask (Lower = Better)")
    ax.set_ylim(0.4, 0.95)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    # Panel C: Counterfactual Accuracy Across All 5 Views (26 Persistent Failures)
    ax = axes[1, 0]
    cx = np.arange(len(CONDITIONS))
    palette = ["#245da6", "#5b9bd5", "#888888", "#e8985e", "#bc5100", "#2e7d32"]
    for a, col in zip(arms, palette):
        vals = [next(r["acc_pct"] for r in cf_agg if r["arm"] == a and r["condition"] == c) for c in CONDITIONS]
        ax.plot(cx, vals, marker="o", lw=1.8, color=col, label=a)
    ax.set_xticks(cx, ["Orig", "Matched\nBG", "Flipped\nBG", "Neutral\nGray", "OOD\nSynth"], fontsize=8.5)
    ax.set_ylabel("Accuracy on 26 Persistent Failures (%)")
    ax.set_title("C. Counterfactual Robustness Across 5 Background Views")
    ax.set_ylim(35, 102)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.2)

    # Panel D: Overall Acc vs Original-Minority Counterfactual Recovery
    ax = axes[1, 1]
    accs = [next(r["oof_thr_acc_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    orig_cfs = [next(r["acc_pct"] for r in cf_agg if r["arm"] == a and r["condition"] == "orig") for a in arms]
    for i, a in enumerate(arms):
        ax.scatter(accs[i], orig_cfs[i], s=90, color=palette[i], label=a, zorder=3)
        ax.annotate(pretty[a].replace("\n", " "), (accs[i], orig_cfs[i]), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("Cleaned OOF-Thr Overall Accuracy (%)")
    ax.set_ylabel("Accuracy on 26 Persistent Failures (`orig` Minority BG, %)")
    ax.set_title("D. Overall Accuracy vs. Hard Minority Recovery Trade-off")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Direction 2 · Phase C · Step 6 Pre-Layer4 Mechanism-Derived Repair Ablation", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_gate_visuals(examples: list[dict], out_path: Path):
    if not examples:
        return
    n = min(8, len(examples))
    fig, axes = plt.subplots(n, 3, figsize=(7.8, n * 2.3), layout="constrained")
    axes = np.atleast_2d(axes)
    for idx in range(n):
        ex = examples[idx]
        axes[idx, 0].imshow(ex["rgb"])
        axes[idx, 0].set_title(f"{ex['species'][:18]} (g={ex['group']})", fontsize=8)
        axes[idx, 1].imshow(ex["cub_mask_14"], cmap="gray", vmin=0, vmax=1)
        axes[idx, 1].set_title("Ground-Truth CUB Mask (14x14)", fontsize=8)
        im = axes[idx, 2].imshow(ex["gate_14"], cmap="viridis", vmin=0, vmax=1)
        axes[idx, 2].set_title(f"Learned Mask-Free Gate (FG={ex['fg_g']:.2f}, BG={ex['bg_g']:.2f})", fontsize=8)
        for c in range(3):
            axes[idx, c].axis("off")
    fig.suptitle("Learned Mask-Free layer3.5 Spatial Gate (pre_l4_repair_bn_all) vs. Ground-Truth CUB Mask", fontsize=10.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-stats", type=Path, default=ROOT / "results/rep_repair/phase_a_20261123/failure_audit_stats.json")
    parser.add_argument("--std-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--drift-weight", type=float, default=1.0)
    parser.add_argument("--cons-weight", type=float, default=0.5)
    parser.add_argument("--gate-sparsity", type=float, default=0.05)
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
        cohorts=("standard_erm",),
        seeds={"standard_erm": list(args.std_seeds)},
    )

    official_rows = load_val_metadata(image_dir / "metadata.csv")
    _, rel_rows = apply_label_condition(official_rows, "relabel", species=TERN_SPECIES)
    y = np.array([int(r["y"]) for r in rel_rows], dtype=int)
    g = np.array([str(r["group"]) for r in rel_rows], dtype=str)
    outer_folds = stratified_group_folds(rel_rows, n_splits=5, seed=0)

    stats = json.loads(args.failure_stats.read_text(encoding="utf-8"))
    persistent_indices = sorted(stats["persistent_val_indices"])

    donors = build_clean_bg_donors(image_dir, seg_root, rel_rows, per_place=15)
    neutral_bg = make_neutral_gray_bg()
    ood_bgs = make_ood_backgrounds()

    pers_bundles = []
    for idx in persistent_indices:
        r = rel_rows[idx]
        with Image.open(image_dir / r["id"]) as im:
            orig_rgb = np.asarray(pil_eval_crop(im.convert("RGB")), dtype=np.float32) / 255.0
        mask_224 = load_val_mask_224(seg_root, r["id"])
        mask_14 = downsample_mask_to_grid(mask_224, grid=14)
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
            "species": r["id"].split("/")[0],
            "y": y_true,
            "group": str(r["group"]),
            "fold": int(outer_folds[idx]),
            "rgb": orig_rgb,
            "mask_224": mask_224,
            "mask_14": mask_14,
            "views": views,
        })

    per_seed_rows = []
    cf_rows = []
    drift_rows = []
    gate_align_rows = []
    oof_store = {}
    gate_visual_examples = []

    for rec in backbones:
        seed = int(rec["seed"])
        sid = rec["source_id"]
        base_cpu = load_resnet_checkpoint(Path(rec["path"]), device="cpu")
        print(f"=== [{sid}] Running 6-arm Step 6 ablation ===", flush=True)

        for spec in ARM_SPECS:
            t0 = time.perf_counter()
            arm = spec["name"]
            probs, fold_states, n_params = train_arm_5fold(
                base_cpu, image_dir, rel_rows, outer_folds, spec,
                seed=seed, device=device, workers=args.workers,
                drift_weight=args.drift_weight, cons_weight=args.cons_weight, gate_sparsity=args.gate_sparsity,
            )
            oof_store[f"{sid}__{arm}"] = probs.astype(np.float32)
            m = eval_oof_metrics(probs[:, 1], y, g, outer_folds)
            per_seed_rows.append({"seed": seed, "arm": arm, "trainable_params": n_params, **m})

            # Load fold-held-out models to evaluate counterfactuals, Layer-4 FG drift, & gate alignment!
            fold_models = {}
            for fid, st in fold_states.items():
                fm = PreLayer4RepairModel(
                    base_cpu, use_gate=spec["use_gate"], head_type=spec["head_type"], bn_scope=spec["bn_scope"]
                ).to(device)
                fm.load_state_dict(st)
                fm.eval()
                fold_models[fid] = fm

            with torch.inference_mode():
                for b_idx, b in enumerate(pers_bundles):
                    fm = fold_models[b["fold"]]
                    stages_by_cond = {}
                    for cond in CONDITIONS:
                        batch = torch.stack([rgb_to_model_tensor(im) for im in b["views"][cond]], dim=0).to(device)
                        out = fm.extract_repaired_stage_maps(batch)
                        p_true = float(out["logits"].softmax(dim=1)[:, b["y"]].mean().item())
                        cf_rows.append({
                            "seed": seed, "arm": arm, "condition": cond,
                            "val_index": b["val_index"], "correct": int(p_true >= 0.5), "p_true": p_true,
                        })
                        if cond in ("orig", "neutral_gray", "matched_bg"):
                            stages_by_cond[cond] = {k: v[:1] for k, v in out.items()}

                    d_neu = foreground_feature_drift(stages_by_cond["orig"], stages_by_cond["neutral_gray"], b["mask_224"])
                    d_mat = foreground_feature_drift(stages_by_cond["orig"], stages_by_cond["matched_bg"], b["mask_224"])
                    drift_rows.append({
                        "seed": seed, "arm": arm, "val_index": b["val_index"],
                        "l3_drift_neutral": d_neu["layer3"], "l4_drift_neutral": d_neu["layer4_map"],
                        "l4_drift_matched": d_mat["layer4_map"],
                    })

                    if spec["use_gate"]:
                        g14 = stages_by_cond["orig"]["gate_map"][0, 0].cpu().numpy()
                        fg_sel = b["mask_14"] > 0.25
                        bg_sel = ~fg_sel
                        fg_g = float(g14[fg_sel].mean()) if fg_sel.any() else float(g14.mean())
                        bg_g = float(g14[bg_sel].mean()) if bg_sel.any() else float(g14.mean())
                        gate_align_rows.append({
                            "seed": seed, "arm": arm, "val_index": b["val_index"],
                            "fg_gate_mean": fg_g, "bg_gate_mean": bg_g, "fg_minus_bg_gate": fg_g - bg_g,
                        })
                        if seed == 0 and arm == "pre_l4_repair_bn_all" and b_idx < 8:
                            gate_visual_examples.append({
                                "species": b["species"], "group": b["group"],
                                "rgb": b["rgb"], "cub_mask_14": b["mask_14"], "gate_14": g14,
                                "fg_g": fg_g, "bg_g": bg_g,
                            })
            print(
                f"    [{sid}] {arm:22s}: OOF-Thr WGA={m['oof_thr_wga']:.2f}% | "
                f"Min-AUC={m['min_grp_auc']:.2f}% | Acc={m['oof_thr_acc']:.2f}% ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    # Aggregate results across the 5 standard_erm seeds.
    agg_rows = []
    for arm in ARM_ORDER:
        sub = [r for r in per_seed_rows if r["arm"] == arm]
        if not sub:
            continue
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
    for arm in ARM_ORDER:
        for cond in CONDITIONS:
            sub = [r for r in cf_rows if r["arm"] == arm and r["condition"] == cond]
            if not sub:
                continue
            cf_agg.append({
                "arm": arm, "condition": cond,
                "acc_pct": round(100.0 * float(np.mean([r["correct"] for r in sub])), 2),
                "mean_p_true": round(float(np.mean([r["p_true"] for r in sub])), 4),
            })

    drift_agg = []
    for arm in ARM_ORDER:
        sub = [r for r in drift_rows if r["arm"] == arm]
        if not sub:
            continue
        gsub = [r for r in gate_align_rows if r["arm"] == arm]
        drift_agg.append({
            "arm": arm,
            "l3_drift_neutral_mean": round(float(np.mean([r["l3_drift_neutral"] for r in sub])), 4),
            "l4_drift_neutral_mean": round(float(np.mean([r["l4_drift_neutral"] for r in sub])), 4),
            "l4_drift_matched_mean": round(float(np.mean([r["l4_drift_matched"] for r in sub])), 4),
            "fg_gate_mean": round(float(np.mean([r["fg_gate_mean"] for r in gsub])), 4) if gsub else "",
            "bg_gate_mean": round(float(np.mean([r["bg_gate_mean"] for r in gsub])), 4) if gsub else "",
            "fg_minus_bg_gate": round(float(np.mean([r["fg_minus_bg_gate"] for r in gsub])), 4) if gsub else "",
        })

    write_csv(out_dir / "step6_per_seed.csv", per_seed_rows)
    write_csv(out_dir / "step6_aggregate.csv", agg_rows)
    write_csv(out_dir / "step6_counterfactuals.csv", cf_agg)
    write_csv(out_dir / "step6_drift_and_gate.csv", drift_agg)
    np.savez_compressed(out_dir / "step6_oof_predictions.npz", **oof_store)
    plot_step6(agg_rows, cf_agg, drift_agg, out_dir / "step6_repair_ablation.png")
    plot_gate_visuals(gate_visual_examples, out_dir / "gate_maps_visual.png")

    lines = [
        "# Direction 2 · Phase C · Step 6 Minimal Repair & 6-Arm Ablation (Validation Only)",
        "",
        "## 1. Cleaned 5-Fold Validation OOF Results (5 Standard-ERM Seeds)",
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

    lines.extend([
        "",
        "## 2. Counterfactual Robustness (26 Persistent Failures) & Layer-4 Foreground Drift",
        "",
        "| Arm | `orig` (%) | `matched` (%) | `flipped` (%) | `neutral` (%) | `ood` (%) | L4 FG Drift (neu) | Gate FG / BG (`Δ`) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ARM_ORDER:
        def _cf(c):
            return next(r["acc_pct"] for r in cf_agg if r["arm"] == arm and r["condition"] == c)
        dr = next(r for r in drift_agg if r["arm"] == arm)
        gstr = f"{dr['fg_gate_mean']:.2f} / {dr['bg_gate_mean']:.2f} ({dr['fg_minus_bg_gate']:+.2f})" if dr["fg_gate_mean"] != "" else "—"
        lines.append(
            f"| `{arm}` | {_cf('orig'):.1f} | {_cf('matched_bg'):.1f} | {_cf('flipped_bg'):.1f} | "
            f"{_cf('neutral_gray'):.1f} | {_cf('ood_bg'):.1f} | {dr['l4_drift_neutral_mean']:.4f} | {gstr} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_c_step6_repair_ablation",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "device": device,
        "hyperparams": {
            "drift_weight": args.drift_weight,
            "cons_weight": args.cons_weight,
            "gate_sparsity": args.gate_sparsity,
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
        "backbones": backbones,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Phase C Step 6 complete. Summary saved to:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
