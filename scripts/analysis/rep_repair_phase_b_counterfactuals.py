"""Direction 2 · Phase B · Slot 2: Counterfactual Background-Swap Causality & Layer-Wise FG Drift.

Validation-only (split == 1, cleaned "relabel" labels, zero test access).
Evaluates all 26 persistent ERM-only failures + 13 shared failures + 80 matched minority controls
across 5 pixel-exact CUB-mask counterfactual views:
  orig | matched_bg | flipped_bg | neutral_gray | ood_bg

Measures:
  1) Out-of-fold DFR-L2 and Oracle-Mask-Pool accuracy & flip rates across all 5 views;
  2) Exact CUB pixel foreground area ratio alpha for failing vs. passing minority birds;
  3) Layer-wise foreground-region activation drift (layer1 -> layer4) when ONLY the background changes.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.resnet import load_resnet_checkpoint
from src.methods.rep_repair.controls import extract_val_stage_cache, pool_with_spatial_masks
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
    load_val_metadata,
    stratified_group_folds,
    tern_mask,
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
STAGES = ("layer1", "layer2", "layer3", "layer4_map")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_fold_heads(X: np.ndarray, y: np.ndarray, g: np.ndarray, outer_folds: np.ndarray, c_val: float = 0.3):
    """Fit one (StandardScaler, LogisticRegression) per outer fold strictly on the other 4 folds."""
    models = {}
    for fold in range(int(outer_folds.max()) + 1):
        tr = outer_folds != fold
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X[tr]), y[tr], sample_weight=balanced_group_weights(g[tr]))
        models[fold] = (sc, clf)
    return models


def plot_counterfactual_summary(summary_rows: list[dict], drift_rows: list[dict], output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), layout="constrained")
    x = np.arange(len(CONDITIONS))
    pretty_c = ["Original\n(Minority BG)", "Matched\nReal BG", "Flipped\nReal BG", "Neutral\nGray BG", "OOD\nSynth BG"]

    for cohort, head, color, ls, label in [
        ("standard_erm", "dfr_l2", "#245da6", "-", "Std ERM · DFR-L2 (GAP)"),
        ("standard_erm", "oracle_mask", "#5b9bd5", "--", "Std ERM · Oracle Mask Pool"),
        ("erm_star", "dfr_l2", "#bc5100", "-", "ERM* · DFR-L2 (GAP)"),
        ("erm_star", "oracle_mask", "#e8985e", "--", "ERM* · Oracle Mask Pool"),
    ]:
        vals = [
            next(
                r["acc_pct"] for r in summary_rows
                if r["subset"] == "persistent_erm_only" and r["cohort"] == cohort
                and r["head"] == head and r["condition"] == c
            )
            for c in CONDITIONS
        ]
        ax1.plot(x, vals, marker="o", color=color, ls=ls, lw=2.0, label=label)
    ax1.axhline(50.0, color="#666666", ls=":", lw=1.0)
    ax1.set_xticks(x, pretty_c, fontsize=8.5)
    ax1.set_ylabel("True-Class Accuracy (%)")
    ax1.set_title("26 Persistent ERM-Only Failures Under BG Swaps")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8)

    sx = np.arange(len(STAGES))
    s_labels = ["Layer 1\n(56x56)", "Layer 2\n(28x28)", "Layer 3\n(14x14)", "Layer 4\n(7x7)"]
    for cohort, subset, color, ls, label in [
        ("standard_erm", "persistent_erm_only", "#245da6", "-", "Std ERM · Persistent Failures"),
        ("standard_erm", "passing_control", "#5b9bd5", "--", "Std ERM · Passing Controls"),
        ("erm_star", "persistent_erm_only", "#bc5100", "-", "ERM* · Persistent Failures"),
        ("erm_star", "passing_control", "#e8985e", "--", "ERM* · Passing Controls"),
    ]:
        vals = [
            next(
                r["drift_orig_to_neutral_mean"] for r in drift_rows
                if r["cohort"] == cohort and r["subset"] == subset and r["stage"] == st
            )
            for st in STAGES
        ]
        ax2.plot(sx, vals, marker="s", color=color, ls=ls, lw=2.0, label=label)
    ax2.set_xticks(sx, s_labels, fontsize=8.5)
    ax2.set_ylabel("Relative Foreground-Cell Feature Drift (orig -> neutral)")
    ax2.set_title("Layer-Wise Background Interference Inside Bird Mask")
    ax2.grid(alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Direction 2 · Phase B · Counterfactual Causality & Internal FG-Cell Drift", fontsize=12)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_example_panel(examples: list[dict], output_path: Path):
    if not examples:
        return
    n_rows = min(8, len(examples))
    n_cols = len(CONDITIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.55), layout="constrained")
    axes = np.atleast_2d(axes)
    for r_idx in range(n_rows):
        ex = examples[r_idx]
        for c_idx, cond in enumerate(CONDITIONS):
            ax = axes[r_idx, c_idx]
            ax.imshow(ex["images"][cond])
            ax.axis("off")
            p_std = ex["std_p"][cond]
            p_es = ex["es_p"][cond]
            prefix = f"{ex['species'][:15]} (g={ex['group']})\n" if c_idx == 0 else ""
            ax.set_title(f"{prefix}{cond}\nERM p={p_std:.2f} | ERM* p={p_es:.2f}", fontsize=7.2)
    fig.suptitle("Counterfactual Background-Swap Visual Check on Persistent ERM-Only Failures", fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-stats", type=Path, default=ROOT / "results/rep_repair/phase_a_20261123/failure_audit_stats.json")
    parser.add_argument("--failure-cases", type=Path, default=ROOT / "results/rep_repair/phase_a_20261123/failure_cases.csv")
    parser.add_argument("--std-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ermstar-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--controls-per-group", type=int, default=40)
    parser.add_argument("--donors-per-swap", type=int, default=3)
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
    is_tern = tern_mask([r["id"] for r in official_rows], species=TERN_SPECIES)
    y = np.array([int(r["y"]) for r in rel_rows], dtype=int)
    g = np.array([str(r["group"]) for r in rel_rows], dtype=str)
    outer_folds = stratified_group_folds(rel_rows, n_splits=5, seed=0)

    stats = json.loads(args.failure_stats.read_text(encoding="utf-8"))
    persistent_set = set(stats["persistent_val_indices"])

    # Identify shared both-fail and passing minority controls from Step 1 failure_cases.csv.
    shared_set, pass_01, pass_10 = set(), [], []
    if args.failure_cases.is_file():
        with args.failure_cases.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                idx = int(row["val_index"])
                if row["is_shared_both_fail"] == "True":
                    shared_set.add(idx)
                elif (
                    row["is_tern"] == "False"
                    and int(row["oof_std_fail"]) == 0
                    and int(row["oof_ermstar_pass"]) == 2
                ):
                    if row["group_relabel"] == "01":
                        pass_01.append(idx)
                    elif row["group_relabel"] == "10":
                        pass_10.append(idx)
    rng = np.random.RandomState(0)
    ctrl_01 = rng.choice(pass_01, min(args.controls_per_group, len(pass_01)), replace=False).tolist() if pass_01 else []
    ctrl_10 = rng.choice(pass_10, min(args.controls_per_group, len(pass_10)), replace=False).tolist() if pass_10 else []

    target_specs = []
    for idx in sorted(persistent_set):
        target_specs.append((idx, "persistent_erm_only"))
    for idx in sorted(shared_set - persistent_set):
        target_specs.append((idx, "shared_both_fail"))
    for idx in sorted(set(ctrl_01 + ctrl_10) - persistent_set - shared_set):
        target_specs.append((idx, "passing_control"))

    print(
        f"Selected {len(target_specs)} validation target birds "
        f"(persistent={len(persistent_set)}, shared={len(shared_set)}, controls={len(ctrl_01)+len(ctrl_10)})",
        flush=True,
    )

    print("Building clean real land/water donor backgrounds (inpainting smallest-bird val images)...", flush=True)
    donors = build_clean_bg_donors(image_dir, seg_root, rel_rows, per_place=15)
    neutral_bg = make_neutral_gray_bg()
    ood_bgs = make_ood_backgrounds()

    # Pre-load all 1,199 CUB 7x7 masks for fitting Oracle-Mask-Pool heads cleanly.
    all_masks_7x7 = np.stack([downsample_mask_to_grid(load_val_mask_224(seg_root, r["id"]), grid=7) for r in rel_rows], axis=0)

    # Pre-construct the 5 counterfactual image sets for every target bird.
    target_bundles = []
    for idx, subset in target_specs:
        r = rel_rows[idx]
        with Image.open(image_dir / r["id"]) as im:
            orig_rgb = np.asarray(pil_eval_crop(im.convert("RGB")), dtype=np.float32) / 255.0
        mask_224 = load_val_mask_224(seg_root, r["id"])
        mask_7x7 = all_masks_7x7[idx]
        fg_area = float((mask_224 > 0.5).mean())

        y_true = int(r["y"])
        matched_place = y_true
        flipped_place = 1 - y_true
        d_idx = [(idx + k * 5) % len(donors[matched_place]) for k in range(args.donors_per_swap)]

        views = {
            "orig": [orig_rgb],
            "matched_bg": [composite_foreground(orig_rgb, mask_224, donors[matched_place][j]) for j in d_idx],
            "flipped_bg": [composite_foreground(orig_rgb, mask_224, donors[flipped_place][j]) for j in d_idx],
            "neutral_gray": [composite_foreground(orig_rgb, mask_224, neutral_bg)],
            "ood_bg": [composite_foreground(orig_rgb, mask_224, bg) for bg in ood_bgs],
        }
        target_bundles.append({
            "val_index": idx,
            "subset": subset,
            "id": r["id"],
            "species": r["id"].split("/")[0],
            "y": y_true,
            "group": str(r["group"]),
            "fold": int(outer_folds[idx]),
            "fg_area": fg_area,
            "mask_224": mask_224,
            "mask_7x7": mask_7x7,
            "views": views,
        })

    per_image_rows = []
    drift_records = []
    visual_accum = {
        b["val_index"]: {
            "species": b["species"],
            "group": b["group"],
            "images": {c: b["views"][c][0] for c in CONDITIONS},
            "std_p": {c: [] for c in CONDITIONS},
            "es_p": {c: [] for c in CONDITIONS},
        }
        for b in target_bundles[:8]
    }

    for rec in backbones:
        t0 = time.perf_counter()
        cohort, seed, sid, ckpt = rec["cohort"], int(rec["seed"]), rec["source_id"], Path(rec["path"])
        print(f"=== [{sid}] Fitting fold-held-out DFR-L2 & Oracle-Mask heads + probing counterfactuals ===", flush=True)
        cache = extract_val_stage_cache(ckpt, image_dir, rel_rows, device=device, workers=args.workers)
        oracle_all = pool_with_spatial_masks(cache["layer4_map"], all_masks_7x7)

        dfr_heads = fit_fold_heads(cache["pooled"], y, g, outer_folds)
        oracle_heads = fit_fold_heads(oracle_all, y, g, outer_folds)

        model = load_resnet_checkpoint(ckpt, device=device)
        model.eval()

        with torch.inference_mode():
            for b in target_bundles:
                fold = b["fold"]
                y_true = b["y"]
                sc_dfr, clf_dfr = dfr_heads[fold]
                sc_ora, clf_ora = oracle_heads[fold]

                stage_cache_by_cond = {}
                for cond in CONDITIONS:
                    batch = torch.stack([rgb_to_model_tensor(im) for im in b["views"][cond]], dim=0).to(device)
                    out = model.extract_stage_maps(batch)
                    stage_cache_by_cond[cond] = out

                    pooled_np = out["pooled"].cpu().numpy()
                    l4_np = out["layer4_map"].cpu().numpy()
                    ora_np = pool_with_spatial_masks(l4_np, np.repeat(b["mask_7x7"][None], len(l4_np), axis=0))

                    p_dfr = float(clf_dfr.predict_proba(sc_dfr.transform(pooled_np))[:, y_true].mean())
                    p_ora = float(clf_ora.predict_proba(sc_ora.transform(ora_np))[:, y_true].mean())

                    per_image_rows.append({
                        "cohort": cohort,
                        "seed": seed,
                        "subset": b["subset"],
                        "val_index": b["val_index"],
                        "id": b["id"],
                        "group": b["group"],
                        "fg_area": round(b["fg_area"], 4),
                        "condition": cond,
                        "dfr_p_true": round(p_dfr, 5),
                        "dfr_correct": int(p_dfr >= 0.5),
                        "oracle_p_true": round(p_ora, 5),
                        "oracle_correct": int(p_ora >= 0.5),
                    })
                    if b["val_index"] in visual_accum:
                        bucket = "std_p" if cohort == "standard_erm" else "es_p"
                        visual_accum[b["val_index"]][bucket][cond].append(p_dfr)

                # Measure foreground-only activation drift when ONLY background changes.
                d_neu = foreground_feature_drift(stage_cache_by_cond["orig"], stage_cache_by_cond["neutral_gray"], b["mask_224"])
                d_mat = foreground_feature_drift(stage_cache_by_cond["orig"], stage_cache_by_cond["matched_bg"], b["mask_224"])
                for st in STAGES:
                    drift_records.append({
                        "cohort": cohort,
                        "seed": seed,
                        "subset": b["subset"],
                        "group": b["group"],
                        "val_index": b["val_index"],
                        "stage": st,
                        "drift_orig_to_neutral": d_neu[st],
                        "drift_orig_to_matched": d_mat[st],
                    })
        print(f"    [{sid}] Done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Aggregate accuracy & counterfactual flip rates.
    summary_rows = []
    for subset in ("persistent_erm_only", "shared_both_fail", "passing_control"):
        for cohort in ("standard_erm", "erm_star"):
            for head, col_c, col_p in (("dfr_l2", "dfr_correct", "dfr_p_true"), ("oracle_mask", "oracle_correct", "oracle_p_true")):
                for cond in CONDITIONS:
                    sub = [
                        r for r in per_image_rows
                        if r["subset"] == subset and r["cohort"] == cohort and r["condition"] == cond
                    ]
                    if not sub:
                        continue
                    summary_rows.append({
                        "subset": subset,
                        "cohort": cohort,
                        "head": head,
                        "condition": cond,
                        "n_evals": len(sub),
                        "acc_pct": round(100.0 * float(np.mean([r[col_c] for r in sub])), 2),
                        "mean_p_true": round(float(np.mean([r[col_p] for r in sub])), 4),
                        "mean_fg_area": round(float(np.mean([r["fg_area"] for r in sub])), 4),
                    })

    drift_summary = []
    for subset in ("persistent_erm_only", "shared_both_fail", "passing_control"):
        for cohort in ("standard_erm", "erm_star"):
            for st in STAGES:
                sub = [r for r in drift_records if r["subset"] == subset and r["cohort"] == cohort and r["stage"] == st]
                if not sub:
                    continue
                drift_summary.append({
                    "subset": subset,
                    "cohort": cohort,
                    "stage": st,
                    "drift_orig_to_neutral_mean": round(float(np.mean([r["drift_orig_to_neutral"] for r in sub])), 4),
                    "drift_orig_to_matched_mean": round(float(np.mean([r["drift_orig_to_matched"] for r in sub])), 4),
                })

    def _acc(subset, cohort, head, cond):
        row = next((r for r in summary_rows if r["subset"] == subset and r["cohort"] == cohort and r["head"] == head and r["condition"] == cond), None)
        return row["acc_pct"] if row else None

    gate2_cf = {
        "persistent_std_dfr_orig_acc": _acc("persistent_erm_only", "standard_erm", "dfr_l2", "orig"),
        "persistent_std_dfr_matched_bg_acc": _acc("persistent_erm_only", "standard_erm", "dfr_l2", "matched_bg"),
        "persistent_std_dfr_neutral_gray_acc": _acc("persistent_erm_only", "standard_erm", "dfr_l2", "neutral_gray"),
        "persistent_std_dfr_ood_bg_acc": _acc("persistent_erm_only", "standard_erm", "dfr_l2", "ood_bg"),
        "persistent_std_oracle_orig_acc": _acc("persistent_erm_only", "standard_erm", "oracle_mask", "orig"),
        "persistent_std_oracle_neutral_gray_acc": _acc("persistent_erm_only", "standard_erm", "oracle_mask", "neutral_gray"),
        "persistent_ermstar_dfr_orig_acc": _acc("persistent_erm_only", "erm_star", "dfr_l2", "orig"),
        "persistent_ermstar_dfr_ood_bg_acc": _acc("persistent_erm_only", "erm_star", "dfr_l2", "ood_bg"),
        "causal_bg_flip_gate_pass": (
            max(
                _acc("persistent_erm_only", "standard_erm", "dfr_l2", "matched_bg") or 0.0,
                _acc("persistent_erm_only", "standard_erm", "dfr_l2", "neutral_gray") or 0.0,
            ) >= 50.0
        ),
    }

    write_csv(out_dir / "counterfactual_per_image.csv", per_image_rows)
    write_csv(out_dir / "counterfactual_summary.csv", summary_rows)
    write_csv(out_dir / "layerwise_fg_drift.csv", drift_summary)
    (out_dir / "gate2_counterfactual_summary.json").write_text(json.dumps(gate2_cf, indent=2) + "\n", encoding="utf-8")

    plot_counterfactual_summary(summary_rows, drift_summary, out_dir / "counterfactual_probe.png")
    examples = [
        {
            "species": v["species"],
            "group": v["group"],
            "images": v["images"],
            "std_p": {c: float(np.mean(v["std_p"][c])) for c in CONDITIONS},
            "es_p": {c: float(np.mean(v["es_p"][c])) for c in CONDITIONS},
        }
        for v in visual_accum.values()
    ]
    plot_example_panel(examples, out_dir / "counterfactual_examples.png")

    lines = [
        "# Direction 2 · Phase B · Slot 2 Counterfactual Causality Probe (Validation Only)",
        "",
        f"**Causal BG-Flip Gate ($\ge 50\%$ recovery on matched or neutral BG):** "
        f"**{'PASS' if gate2_cf['causal_bg_flip_gate_pass'] else 'FAIL'}**",
        "",
        "## Persistent ERM-Only Failures (26 Images): Accuracy Across Counterfactual Backgrounds (%)",
        "",
        "| Cohort | Head | Orig (Minority BG) | Matched Real BG | Flipped Real BG | Neutral Gray BG | OOD Synth BG |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cohort in ("standard_erm", "erm_star"):
        for head in ("dfr_l2", "oracle_mask"):
            vals = [_acc("persistent_erm_only", cohort, head, c) for c in CONDITIONS]
            lines.append(f"| {cohort} | `{head}` | " + " | ".join(f"{v:.1f}%" for v in vals) + " |")

    lines.extend([
        "",
        "## Layer-Wise Foreground-Cell Feature Drift (`orig -> neutral_gray`, Inside CUB Bird Mask)",
        "",
        "| Subset | Cohort | Layer 1 | Layer 2 | Layer 3 | Layer 4 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for subset in ("persistent_erm_only", "passing_control"):
        for cohort in ("standard_erm", "erm_star"):
            vals = [
                next(r["drift_orig_to_neutral_mean"] for r in drift_summary if r["subset"] == subset and r["cohort"] == cohort and r["stage"] == st)
                for st in STAGES
            ]
            lines.append(f"| {subset} | {cohort} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_b_slot2_counterfactuals",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "device": device,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
        "backbones": backbones,
        "gate2_counterfactuals": gate2_cf,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Phase B Slot 2 complete:", json.dumps(gate2_cf, indent=2), flush=True)


if __name__ == "__main__":
    main()
