"""Direction 2 · Phase C · Step 6b: Fixed Pre-Layer4 Repair + Layer4.conv2 LoRA Ablation.

Validation-only (split == 1, cleaned "relabel" labels, zero test access).
Addresses all three Step 6 failure modes and tests the Step 5 Panel B 3x3 conv LoRA hypothesis:
  Fix 1: Peak-normalized neutral-mean gate (`g_peak * c3 + (1 - g_peak) * neutral_c3`) so
         foreground bird amplitude stays ~1.0 and frozen layer4.0 BN stays in-distribution.
  Fix 2: Pretrained Linear classification head across all arms (removes -0.94 pp Cosine penalty).
  Fix 3: Gentle context-invariance weights (`drift_w=0.05/0.15`, `cons_w=0.10`, `sparsity=0.005`).
  New Control (Step 5 Panel B): Rank-8 LoRA strictly on `layer4.{0,1,2}.conv2` (the 3x3 convs
         responsible for ~75% of foreground bleed), alone and combined with the Pre-Layer4 gate.
"""

from __future__ import annotations

import argparse
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

from scripts.analysis.rep_repair_phase_c_step6 import CONDITIONS, eval_oof_metrics, write_csv
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
from src.methods.rep_repair.repair import (
    PreLayer4RepairModel,
    compute_group_balanced_layer3_mean,
    context_invariance_loss,
)
from src.models.resnet import load_resnet_checkpoint

ROOT = Path(__file__).resolve().parents[2]

ARM_SPECS = (
    {
        "name": "affine_dfr_linear",
        "use_gate": False, "gate_mode": "raw", "bn_scope": "bn_all", "head_type": "linear",
        "use_lora": False, "lora_rank": 8, "use_inv": False,
        "drift_w": 0.0, "cons_w": 0.0, "sparsity": 0.0, "lr": 3e-3, "lora_lr": None, "epochs": 20,
    },
    {
        "name": "pre_l4_gate_bn_ce",
        "use_gate": True, "gate_mode": "peak_neutral", "bn_scope": "bn_all", "head_type": "linear",
        "use_lora": False, "lora_rank": 8, "use_inv": False,
        "drift_w": 0.0, "cons_w": 0.0, "sparsity": 0.0, "lr": 3e-3, "lora_lr": None, "epochs": 20,
    },
    {
        "name": "pre_l4_repair_w005",
        "use_gate": True, "gate_mode": "peak_neutral", "bn_scope": "bn_all", "head_type": "linear",
        "use_lora": False, "lora_rank": 8, "use_inv": True,
        "drift_w": 0.05, "cons_w": 0.10, "sparsity": 0.005, "lr": 3e-3, "lora_lr": None, "epochs": 20,
    },
    {
        "name": "pre_l4_repair_w015",
        "use_gate": True, "gate_mode": "peak_neutral", "bn_scope": "bn_all", "head_type": "linear",
        "use_lora": False, "lora_rank": 8, "use_inv": True,
        "drift_w": 0.15, "cons_w": 0.10, "sparsity": 0.005, "lr": 3e-3, "lora_lr": None, "epochs": 20,
    },
    {
        "name": "l4_conv2_lora_bn",
        "use_gate": False, "gate_mode": "raw", "bn_scope": "bn_all", "head_type": "linear",
        "use_lora": True, "lora_rank": 8, "use_inv": False,
        "drift_w": 0.0, "cons_w": 0.0, "sparsity": 0.0, "lr": 3e-3, "lora_lr": 1e-3, "epochs": 20,
    },
    {
        "name": "l4_conv2_lora_repair_w005",
        "use_gate": True, "gate_mode": "peak_neutral", "bn_scope": "bn_all", "head_type": "linear",
        "use_lora": True, "lora_rank": 8, "use_inv": True,
        "drift_w": 0.05, "cons_w": 0.10, "sparsity": 0.005, "lr": 3e-3, "lora_lr": 1e-3, "epochs": 20,
    },
)
ARM_ORDER = tuple(spec["name"] for spec in ARM_SPECS)


def precompute_fold_neutral_c3(
    base_cpu: nn.Module,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    *,
    seed: int,
    device: str,
    workers: int,
) -> dict[int, torch.Tensor]:
    """Compute strictly fit-split group-balanced neutral layer3 channel means once per outer fold."""
    fold_means = {}
    for fold in range(int(outer_folds.max()) + 1):
        tr_idx = np.flatnonzero(outer_folds != fold)
        tr_rows = [val_rows[i] for i in tr_idx]
        inner_folds = stratified_group_folds(tr_rows, n_splits=4, seed=seed + 300 + fold)
        fit_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds != 0)]
        loader = DataLoader(ValImageDataset(image_dir, fit_sub), batch_size=128, shuffle=False, num_workers=workers)
        fold_means[fold] = compute_group_balanced_layer3_mean(base_cpu, loader, device=device)
    return fold_means


def train_arm_5fold_step6b(
    base_cpu: nn.Module,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    fold_neutral_c3: dict[int, torch.Tensor],
    spec: dict,
    *,
    seed: int,
    device: str,
    workers: int,
):
    """Train one Step-6b arm across 5 outer folds with strict inner 1/4 validation checkpointing."""
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
            gate_mode=spec["gate_mode"],
            head_type=spec["head_type"],
            bn_scope=spec["bn_scope"],
            use_l4_conv2_lora=spec["use_lora"],
            lora_rank=spec["lora_rank"],
        ).to(device)
        if spec["use_gate"] and spec["gate_mode"] == "peak_neutral":
            model.set_neutral_c3(fold_neutral_c3[fold])

        n_trainable = sum(p.numel() for p in model.trainable_parameters())
        opt = torch.optim.AdamW(
            model.parameter_groups(lr=spec["lr"], lora_lr=spec["lora_lr"]),
            weight_decay=1e-2,
        )

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
                        drift_weight=spec["drift_w"],
                        cons_weight=spec["cons_w"],
                        gate_sparsity=spec["sparsity"],
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


def plot_step6b(agg_rows: list[dict], cf_agg: list[dict], drift_agg: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), layout="constrained")
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in agg_rows)]
    pretty = {
        "affine_dfr_linear": "Affine-DFR\n(Linear)",
        "pre_l4_gate_bn_ce": "PeakGate+BN\n(CE only)",
        "pre_l4_repair_w005": "PeakGate\nRepair w=0.05",
        "pre_l4_repair_w015": "PeakGate\nRepair w=0.15",
        "l4_conv2_lora_bn": "L4.conv2 LoRA\n+BN (CE)",
        "l4_conv2_lora_repair_w005": "PeakGate+LoRA\nRepair w=0.05",
    }
    x = np.arange(len(arms))

    # Panel A: OOF-Thr WGA and Minority-Group AUROC
    ax = axes[0, 0]
    w = 0.36
    wgas = [next(r["oof_thr_wga_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    wsds = [next(r["oof_thr_wga_sd"] for r in agg_rows if r["arm"] == a) for a in arms]
    aucs = [next(r["min_grp_auc_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    base_wga = wgas[0]
    ax.bar(x - w / 2, wgas, w, yerr=wsds, capsize=3, color="#245da6", label="OOF-Thr WGA (%)")
    ax.bar(x + w / 2, aucs, w, color="#4a7c59", label="Minority-Grp AUROC (%)")
    ax.axhline(base_wga, color="#8b0000", ls="--", lw=1.0, label=f"Affine-DFR Linear ({base_wga:.2f}%)")
    for i, (wv, av) in enumerate(zip(wgas, aucs)):
        ax.text(i - w / 2, wv + 0.25, f"{wv:.2f}", ha="center", va="bottom", fontsize=7.3)
        ax.text(i + w / 2, av + 0.25, f"{av:.2f}", ha="center", va="bottom", fontsize=7.3)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("A. Cleaned 5-Fold Val OOF WGA & Minority AUROC (5 Std ERM Seeds)")
    ax.set_ylim(88, 100.5)
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
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
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
    palette = ["#245da6", "#5b9bd5", "#2e7d32", "#bc5100", "#7b3294", "#c2185b"]
    for a, col in zip(arms, palette):
        vals = [next(r["acc_pct"] for r in cf_agg if r["arm"] == a and r["condition"] == c) for c in CONDITIONS]
        ax.plot(cx, vals, marker="o", lw=1.8, color=col, label=a)
    ax.set_xticks(cx, ["Orig", "Matched\nBG", "Flipped\nBG", "Neutral\nGray", "OOD\nSynth"], fontsize=8.5)
    ax.set_ylabel("Accuracy on 26 Persistent Failures (%)")
    ax.set_title("C. Counterfactual Robustness Across 5 Background Views")
    ax.set_ylim(45, 102)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.0)

    # Panel D: Overall Acc vs Original-Minority Counterfactual Recovery
    ax = axes[1, 1]
    accs = [next(r["oof_thr_acc_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    orig_cfs = [next(r["acc_pct"] for r in cf_agg if r["arm"] == a and r["condition"] == "orig") for a in arms]
    for i, a in enumerate(arms):
        ax.scatter(accs[i], orig_cfs[i], s=90, color=palette[i], label=a, zorder=3)
        ax.annotate(pretty[a].replace("\n", " "), (accs[i], orig_cfs[i]), textcoords="offset points", xytext=(5, 4), fontsize=7.3)
    ax.set_xlabel("Cleaned OOF-Thr Overall Accuracy (%)")
    ax.set_ylabel("Accuracy on 26 Persistent Failures (`orig` Minority BG, %)")
    ax.set_title("D. Overall Accuracy vs. Hard Minority Recovery Trade-off")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Direction 2 · Phase C · Step 6b Peak-Neutral Gate + Layer4.conv2 LoRA Repair", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_gate_visuals_step6b(examples_by_arm: dict[str, list[dict]], out_path: Path):
    arm_a = "pre_l4_repair_w005"
    arm_b = "l4_conv2_lora_repair_w005"
    ex_a = examples_by_arm.get(arm_a, [])
    ex_b = examples_by_arm.get(arm_b, [])
    n = min(8, len(ex_a), len(ex_b))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(10.2, n * 2.2), layout="constrained")
    axes = np.atleast_2d(axes)
    for idx in range(n):
        a, b = ex_a[idx], ex_b[idx]
        axes[idx, 0].imshow(a["rgb"])
        axes[idx, 0].set_title(f"{a['species'][:18]} (g={a['group']})", fontsize=8)
        axes[idx, 1].imshow(a["cub_mask_14"], cmap="gray", vmin=0, vmax=1)
        axes[idx, 1].set_title("GT CUB Mask (14x14)", fontsize=8)
        axes[idx, 2].imshow(a["gate_14"], cmap="viridis", vmin=0, vmax=1)
        axes[idx, 2].set_title(f"PeakGate w005 (FG={a['fg_g']:.2f}, BG={a['bg_g']:.2f})", fontsize=7.8)
        axes[idx, 3].imshow(b["gate_14"], cmap="viridis", vmin=0, vmax=1)
        axes[idx, 3].set_title(f"Gate+LoRA w005 (FG={b['fg_g']:.2f}, BG={b['bg_g']:.2f})", fontsize=7.8)
        for c in range(4):
            axes[idx, c].axis("off")
    fig.suptitle("Step 6b Peak-Normalized Neutral-Mean Gate Maps (Seed 0, 8 Persistent Failures)", fontsize=10.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-stats", type=Path, default=ROOT / "results/rep_repair/phase_a_20261123/failure_audit_stats.json")
    parser.add_argument("--std-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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
    gate_visuals: dict[str, list[dict]] = {"pre_l4_repair_w005": [], "l4_conv2_lora_repair_w005": []}

    for rec in backbones:
        seed = int(rec["seed"])
        sid = rec["source_id"]
        base_cpu = load_resnet_checkpoint(Path(rec["path"]), device="cpu")
        print(f"=== [{sid}] Precomputing fold-training neutral layer3 baselines ===", flush=True)
        fold_neutral_c3 = precompute_fold_neutral_c3(
            base_cpu, image_dir, rel_rows, outer_folds, seed=seed, device=device, workers=args.workers
        )

        for spec in ARM_SPECS:
            t0 = time.perf_counter()
            arm = spec["name"]
            probs, fold_states, n_params = train_arm_5fold_step6b(
                base_cpu, image_dir, rel_rows, outer_folds, fold_neutral_c3, spec,
                seed=seed, device=device, workers=args.workers,
            )
            oof_store[f"{sid}__{arm}"] = probs.astype(np.float32)
            m = eval_oof_metrics(probs[:, 1], y, g, outer_folds)
            per_seed_rows.append({"seed": seed, "arm": arm, "trainable_params": n_params, **m})

            fold_models = {}
            for fid, st in fold_states.items():
                fm = PreLayer4RepairModel(
                    base_cpu,
                    use_gate=spec["use_gate"],
                    gate_mode=spec["gate_mode"],
                    head_type=spec["head_type"],
                    bn_scope=spec["bn_scope"],
                    use_l4_conv2_lora=spec["use_lora"],
                    lora_rank=spec["lora_rank"],
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
                        if seed == 0 and arm in gate_visuals and b_idx < 8:
                            gate_visuals[arm].append({
                                "species": b["species"], "group": b["group"],
                                "rgb": b["rgb"], "cub_mask_14": b["mask_14"], "gate_14": g14,
                                "fg_g": fg_g, "bg_g": bg_g,
                            })
            print(
                f"    [{sid}] {arm:25s}: OOF-Thr WGA={m['oof_thr_wga']:.2f}% | "
                f"Min-AUC={m['min_grp_auc']:.2f}% | Acc={m['oof_thr_acc']:.2f}% ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    base_by_seed = {r["seed"]: r for r in per_seed_rows if r["arm"] == "affine_dfr_linear"}
    agg_rows = []
    for arm in ARM_ORDER:
        sub = sorted([r for r in per_seed_rows if r["arm"] == arm], key=lambda r: r["seed"])
        if not sub:
            continue
        dwga = [r["oof_thr_wga"] - base_by_seed[r["seed"]]["oof_thr_wga"] for r in sub]
        dauc = [r["min_grp_auc"] - base_by_seed[r["seed"]]["min_grp_auc"] for r in sub]
        agg_rows.append({
            "arm": arm,
            "n_seeds": len(sub),
            "trainable_params": sub[0]["trainable_params"],
            "raw_wga_mean": round(float(np.mean([r["raw_wga"] for r in sub])), 2),
            "oof_thr_wga_mean": round(float(np.mean([r["oof_thr_wga"] for r in sub])), 2),
            "oof_thr_wga_sd": round(float(np.std([r["oof_thr_wga"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "delta_wga_vs_affine": round(float(np.mean(dwga)), 2),
            "wga_seed_wins": f"{sum(d > 0 for d in dwga)}/{len(sub)}",
            "oof_thr_acc_mean": round(float(np.mean([r["oof_thr_acc"] for r in sub])), 2),
            "g01_thr_mean": round(float(np.mean([r["g01_thr"] for r in sub])), 2),
            "g10_thr_mean": round(float(np.mean([r["g10_thr"] for r in sub])), 2),
            "min_grp_auc_mean": round(float(np.mean([r["min_grp_auc"] for r in sub])), 2),
            "delta_auc_vs_affine": round(float(np.mean(dauc)), 2),
            "auc_seed_wins": f"{sum(d > 0 for d in dauc)}/{len(sub)}",
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

    write_csv(out_dir / "step6b_per_seed.csv", per_seed_rows)
    write_csv(out_dir / "step6b_aggregate.csv", agg_rows)
    write_csv(out_dir / "step6b_counterfactuals.csv", cf_agg)
    write_csv(out_dir / "step6b_drift_and_gate.csv", drift_agg)
    np.savez_compressed(out_dir / "step6b_oof_predictions.npz", **oof_store)
    plot_step6b(agg_rows, cf_agg, drift_agg, out_dir / "step6b_ablation.png")
    plot_gate_visuals_step6b(gate_visuals, out_dir / "gate_maps_step6b.png")

    lines = [
        "# Direction 2 · Phase C · Step 6b Fixed Pre-Layer4 Repair & Layer4.conv2 LoRA Ablation (Validation Only)",
        "",
        "## 1. Cleaned 5-Fold Validation OOF Results (5 Standard-ERM Seeds)",
        "",
        "| Arm | Params | Raw WGA (%) | OOF-Thr WGA (%) | Δ WGA (Wins) | OOF-Thr Acc (%) | Min-Grp AUROC (%) | Δ AUC (Wins) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        lines.append(
            f"| `{r['arm']}` | {r['trainable_params']:,} | {r['raw_wga_mean']:.2f} | "
            f"**{r['oof_thr_wga_mean']:.2f} ± {r['oof_thr_wga_sd']:.2f}** | "
            f"{r['delta_wga_vs_affine']:+.2f} ({r['wga_seed_wins']}) | "
            f"{r['oof_thr_acc_mean']:.2f} | **{r['min_grp_auc_mean']:.2f}** | "
            f"{r['delta_auc_vs_affine']:+.2f} ({r['auc_seed_wins']}) |"
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
        "step": "phase_c_step6b_fixed_repair_and_lora",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "device": device,
        "arms": list(ARM_SPECS),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
        "backbones": backbones,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Phase C Step 6b complete. Summary saved to:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
