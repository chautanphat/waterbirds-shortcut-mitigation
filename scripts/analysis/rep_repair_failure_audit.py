"""Direction 2 · Phase A · Step 1: Cleaned Waterbirds validation failure-case audit.

Inspects non-"tern" minority-group validation failures strictly on split == 1 (zero test access):
  1) Saved 479 held-out DPE head predictions (continuous with Pilot 1) -> head unanimity check;
  2) Full 1,199-image 5-fold group-weighted linear DFR-L2 OOF -> isolates persistent ERM-only
     minority failures (>=4/5 standard_erm fail while 2/2 erm_star succeed);
  3) Cheap raw-image visual traits (aspect ratio, center-vs-border color contrast, border water/
     foliage tint, border clutter) + annotated visual contact sheet for TA review.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.methods.dpe_head_reweighting.reweighting import GROUPS
from src.methods.rep_repair.phase_a_data import (
    apply_label_condition,
    balanced_group_weights,
    file_sha256,
    group_metrics,
    load_all_sources,
    stratified_group_folds,
    tern_mask,
)

ROOT = Path(__file__).resolve().parents[2]
MINORITY_GROUPS = ("01", "10")


def center_crop_224(image: Image.Image) -> Image.Image:
    """Exact deterministic Waterbirds eval transform: Resize(256, 256) -> CenterCrop(224)."""
    img = image.convert("RGB").resize((256, 256), Image.BILINEAR)
    left = (256 - 224) // 2
    return img.crop((left, left, left + 224, left + 224))


def extract_visual_traits(image_path: Path) -> dict:
    with Image.open(image_path) as raw_img:
        orig_w, orig_h = raw_img.size
        crop = center_crop_224(raw_img)
    arr = np.asarray(crop, dtype=np.float32) / 255.0  # [224, 224, 3]
    # Center 50% (56:168) vs outer 25% border ring.
    center_mask = np.zeros((224, 224), dtype=bool)
    center_mask[56:168, 56:168] = True
    border_mask = ~center_mask

    center_rgb = arr[center_mask].mean(axis=0)
    border_rgb = arr[border_mask].mean(axis=0)
    rgb_dist = float(np.linalg.norm(center_rgb - border_rgb))

    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    edge = gx + gy

    return {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "aspect_ratio": float(max(orig_w, orig_h) / max(1, min(orig_w, orig_h))),
        "center_border_rgb_dist": rgb_dist,
        "border_blue_minus_green": float(border_rgb[2] - border_rgb[1]),
        "center_blue_minus_green": float(center_rgb[2] - center_rgb[1]),
        "border_edge_energy": float(edge[border_mask].mean()),
        "center_edge_energy": float(edge[center_mask].mean()),
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
    }


def oof_dfr_l2_probs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    c_grid: list[float],
    inner_splits: int,
    seed: int,
) -> np.ndarray:
    """Zero-leakage nested-CV group-weighted L2 logistic regression OOF probabilities [N, 2]."""
    probs = np.zeros((len(y), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te = X[te]

        # Inner CV to choose C strictly on training fold.
        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=inner_splits, seed=seed + 100 + fold)
        best_c, best_key = c_grid[0], (-1.0, -1.0)
        for c_val in c_grid:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(inner_splits):
                itr = in_folds != infold
                iva = in_folds == infold
                scaler = StandardScaler().fit(X_tr[itr])
                clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
                clf.fit(scaler.transform(X_tr[itr]), y_tr[itr], sample_weight=balanced_group_weights(g_tr[itr]))
                in_preds[iva] = clf.predict(scaler.transform(X_tr[iva]))
            m = group_metrics(y_tr, in_preds, g_tr)
            key = (m["wga"], m["overall"])
            if key > best_key:
                best_key = key
                best_c = c_val

        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs")
        clf.fit(scaler.transform(X_tr), y_tr, sample_weight=balanced_group_weights(g_tr))
        probs[te] = clf.predict_proba(scaler.transform(X_te))
    return probs


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_contact_sheet(selected_rows: list[dict], image_dir: Path, output_path: Path):
    if not selected_rows:
        return
    cols = 6
    n = len(selected_rows)
    grid_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 2.7, grid_rows * 3.1), layout="constrained")
    axes_arr = np.atleast_2d(axes)
    for idx in range(grid_rows * cols):
        ax = axes_arr[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        row = selected_rows[idx]
        img_path = image_dir / row["id"]
        with Image.open(img_path) as im:
            ax.imshow(center_crop_224(im))
        species = row["id"].split("/")[0].replace("_", " ")
        if len(species) > 22:
            species = species[:21] + "…"
        held_tag = ""
        if row["in_held479"]:
            held_tag = f"\nDPE479: {row['held479_std_fail']}/5 fail ({row['held479_std_unanimous_wrong']}x 0/15)"
        ax.set_title(
            f"g={row['group_relabel']} | {species}\n"
            f"ERM {row['oof_std_fail']}/5 fail (p={row['oof_std_mean_p_true']:.2f})\n"
            f"ERM* {row['oof_ermstar_pass']}/2 pass (p={row['oof_ermstar_mean_p_true']:.2f}){held_tag}",
            fontsize=7.3,
            color="#8b0000" if row["group_relabel"] == "10" else "#1c4966",
        )
    fig.suptitle(
        "Persistent Non-'Tern' Minority Validation Failures (Standard ERM fails >=4/5, ERM* passes 2/2)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    asset_root, official_val_rows, sources = load_all_sources(config, ROOT)
    image_dir = (asset_root / config["image_dir"]).resolve(strict=True)
    species = tuple(config["tern_species"])

    _, relabel_rows = apply_label_condition(official_val_rows, "relabel", species=species)
    is_tern = tern_mask([r["id"] for r in official_val_rows], species=species)
    y_rel = np.array([r["y"] for r in relabel_rows], dtype=int)
    g_rel = np.array([r["group"] for r in relabel_rows], dtype=str)
    y_off = np.array([r["y"] for r in official_val_rows], dtype=int)
    g_off = np.array([r["group"] for r in official_val_rows], dtype=str)

    outer_folds = stratified_group_folds(
        relabel_rows, n_splits=config["cv"]["outer_folds"], seed=config["cv"]["split_seed"]
    )
    c_grid = list(config["readouts"]["dfr_l2_C"])
    inner_splits = int(config["cv"]["inner_folds"])
    split_seed = int(config["cv"]["split_seed"])

    std_sources = [s for s in sources if s["cohort"] == "standard_erm"]
    es_sources = [s for s in sources if s["cohort"] == "erm_star"]

    # 1. Compute full 1,199-image cleaned OOF DFR-L2 predictions per backbone.
    oof_probs = {}
    summary_rows = []
    held_idx = std_sources[0]["held_val_indices"]
    for src in sources:
        if not np.array_equal(src["held_val_indices"], held_idx):
            raise ValueError(f"Held-out 479 indices differ for {src['source_id']}")
        probs = oof_dfr_l2_probs(
            src["features"], y_rel, g_rel, outer_folds, c_grid, inner_splits, split_seed + src["seed"]
        )
        oof_probs[src["source_id"]] = probs
        pred_oof = probs.argmax(axis=-1)
        m_oof_rel = group_metrics(y_rel, pred_oof, g_rel)
        m_oof_drop = group_metrics(y_rel[~is_tern], pred_oof[~is_tern], g_rel[~is_tern])

        # Saved 479 held-out DPE ensemble metrics on official vs relabel vs drop.
        held_ens_pred = src["held_scores"].mean(axis=0).argmax(axis=-1)
        held_tern = is_tern[held_idx]
        m_h_off = group_metrics(y_off[held_idx], held_ens_pred, g_off[held_idx])
        m_h_rel = group_metrics(y_rel[held_idx], held_ens_pred, g_rel[held_idx])
        m_h_drop = group_metrics(y_rel[held_idx][~held_tern], held_ens_pred[~held_tern], g_rel[held_idx][~held_tern])

        # Unanimous 0/15 head errors on non-tern minority held-479 rows.
        head_preds = src["held_scores"].argmax(axis=-1)  # [15, 479]
        head_correct = head_preds == y_rel[held_idx][None, :]
        ens_wrong_nt = (held_ens_pred != y_rel[held_idx]) & (~held_tern)
        min_mask_h = np.isin(g_rel[held_idx], MINORITY_GROUPS) & (~held_tern)
        ens_wrong_min = ens_wrong_nt & min_mask_h
        unanimous_min = int(np.sum(head_correct[:, ens_wrong_min].sum(axis=0) == 0))

        summary_rows.append(
            {
                "cohort": src["cohort"],
                "seed": src["seed"],
                "oof_dfr_l2_relabel_wga": round(m_oof_rel["wga"], 4),
                "oof_dfr_l2_relabel_acc": round(m_oof_rel["overall"], 4),
                "oof_dfr_l2_drop_wga": round(m_oof_drop["wga"], 4),
                "oof_dfr_l2_g01": round(m_oof_rel["groups"]["01"], 4),
                "oof_dfr_l2_g10": round(m_oof_rel["groups"]["10"], 4),
                "held479_dpe_official_wga": round(m_h_off["wga"], 4),
                "held479_dpe_relabel_wga": round(m_h_rel["wga"], 4),
                "held479_dpe_drop_wga": round(m_h_drop["wga"], 4),
                "held479_nontern_total_errors": int(ens_wrong_nt.sum()),
                "held479_nontern_minority_errors": int(ens_wrong_min.sum()),
                "held479_nontern_minority_unanimous_0of15": unanimous_min,
            }
        )

    # 2. Build per-image failure table across all 1,199 validation images.
    held_pos = {int(val_i): pos for pos, val_i in enumerate(held_idx)}
    min_std_fail = int(config["failure_audit"]["min_std_fail_count"])
    min_es_pass = int(config["failure_audit"]["min_ermstar_pass_count"])

    case_rows = []
    for i, r_rel in enumerate(relabel_rows):
        y_true = int(r_rel["y"])
        std_p_true = [float(oof_probs[s["source_id"]][i, y_true]) for s in std_sources]
        es_p_true = [float(oof_probs[s["source_id"]][i, y_true]) for s in es_sources]
        std_fail = sum(p < 0.5 for p in std_p_true)
        es_pass = sum(p >= 0.5 for p in es_p_true)

        in_held = i in held_pos
        h_std_fail = h_es_pass = h_std_unanimous = -1
        h_std_mean_p = h_es_mean_p = float("nan")
        if in_held:
            pos = held_pos[i]
            h_std_probs = [float(s["held_scores"][:, pos, y_true].mean()) for s in std_sources]
            h_es_probs = [float(s["held_scores"][:, pos, y_true].mean()) for s in es_sources]
            h_std_fail = sum(p < 0.5 for p in h_std_probs)
            h_es_pass = sum(p >= 0.5 for p in h_es_probs)
            h_std_unanimous = sum(int((s["held_scores"][:, pos].argmax(-1) == y_true).sum() == 0) for s in std_sources)
            h_std_mean_p = float(np.mean(h_std_probs))
            h_es_mean_p = float(np.mean(h_es_probs))

        is_minority = r_rel["group"] in MINORITY_GROUPS
        is_persistent = (
            (not bool(is_tern[i]))
            and is_minority
            and (std_fail >= min_std_fail)
            and (es_pass >= min_es_pass)
        )
        is_both_fail = (
            (not bool(is_tern[i]))
            and is_minority
            and (std_fail >= min_std_fail)
            and (es_pass == 0)
        )

        # Compute visual traits for all non-tern minority rows (fast: ~599 images, ~2s).
        traits = (
            extract_visual_traits(image_dir / r_rel["id"])
            if (is_minority and not bool(is_tern[i]))
            else {}
        )

        case_rows.append(
            {
                "val_index": i,
                "img_id": r_rel["img_id"],
                "id": r_rel["id"],
                "species": r_rel["id"].split("/")[0],
                "is_tern": bool(is_tern[i]),
                "y_official": int(official_val_rows[i]["y"]),
                "group_official": official_val_rows[i]["group"],
                "y_relabel": y_true,
                "group_relabel": r_rel["group"],
                "outer_fold": int(outer_folds[i]),
                "oof_std_fail": std_fail,
                "oof_ermstar_pass": es_pass,
                "oof_std_mean_p_true": round(float(np.mean(std_p_true)), 5),
                "oof_ermstar_mean_p_true": round(float(np.mean(es_p_true)), 5),
                "oof_margin_gap_ermstar_minus_std": round(float(np.mean(es_p_true) - np.mean(std_p_true)), 5),
                "in_held479": in_held,
                "held479_std_fail": h_std_fail,
                "held479_ermstar_pass": h_es_pass,
                "held479_std_unanimous_wrong": h_std_unanimous,
                "held479_std_mean_p_true": "" if np.isnan(h_std_mean_p) else round(h_std_mean_p, 5),
                "held479_ermstar_mean_p_true": "" if np.isnan(h_es_mean_p) else round(h_es_mean_p, 5),
                "is_persistent_erm_only": is_persistent,
                "is_shared_both_fail": is_both_fail,
                **{
                    k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in traits.items()
                },
            }
        )

    # Ensure consistent keys across all rows for CSV output.
    trait_keys = [
        "orig_w", "orig_h", "aspect_ratio", "center_border_rgb_dist",
        "border_blue_minus_green", "center_blue_minus_green",
        "border_edge_energy", "center_edge_energy", "luminance_mean", "luminance_std",
    ]
    for row in case_rows:
        for k in trait_keys:
            row.setdefault(k, "")

    persistent = sorted(
        [r for r in case_rows if r["is_persistent_erm_only"]],
        key=lambda r: (-r["oof_std_fail"], -r["oof_margin_gap_ermstar_minus_std"]),
    )
    both_fail = [r for r in case_rows if r["is_shared_both_fail"]]
    nontern_min_pass = [
        r for r in case_rows
        if (not r["is_tern"]) and (r["group_relabel"] in MINORITY_GROUPS)
        and (r["oof_std_fail"] == 0) and (r["oof_ermstar_pass"] == len(es_sources))
    ]

    # Compare visual traits: persistent ERM-only failures vs passing minority controls.
    trait_comparison = []
    for g in MINORITY_GROUPS:
        fail_g = [r for r in persistent if r["group_relabel"] == g]
        pass_g = [r for r in nontern_min_pass if r["group_relabel"] == g]
        if not fail_g or not pass_g:
            continue
        for k in ("aspect_ratio", "center_border_rgb_dist", "border_blue_minus_green",
                  "center_blue_minus_green", "border_edge_energy", "luminance_mean"):
            f_vals = np.array([float(r[k]) for r in fail_g])
            p_vals = np.array([float(r[k]) for r in pass_g])
            trait_comparison.append(
                {
                    "group": g,
                    "trait": k,
                    "n_persistent_fail": len(fail_g),
                    "n_both_pass": len(pass_g),
                    "fail_median": round(float(np.median(f_vals)), 4),
                    "pass_median": round(float(np.median(p_vals)), 4),
                    "fail_mean": round(float(f_vals.mean()), 4),
                    "pass_mean": round(float(p_vals.mean()), 4),
                }
            )

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "failure_summary.csv", summary_rows)
    write_csv(out_dir / "failure_cases.csv", case_rows)
    write_csv(out_dir / "failure_traits.csv", trait_comparison)

    max_sheet = int(config["failure_audit"]["contact_sheet_max_images"])
    render_contact_sheet(persistent[:max_sheet], image_dir, out_dir / "failure_contact_sheet.png")

    audit_stats = {
        "total_val_rows": len(relabel_rows),
        "tern_val_rows": int(is_tern.sum()),
        "persistent_erm_only_minority_count": len(persistent),
        "persistent_erm_only_by_group": {
            g: sum(1 for r in persistent if r["group_relabel"] == g) for g in MINORITY_GROUPS
        },
        "persistent_erm_only_in_held479": sum(1 for r in persistent if r["in_held479"]),
        "shared_both_fail_minority_count": len(both_fail),
        "shared_both_fail_by_group": {
            g: sum(1 for r in both_fail if r["group_relabel"] == g) for g in MINORITY_GROUPS
        },
        "persistent_val_indices": [r["val_index"] for r in persistent],
        "persistent_ids": [r["id"] for r in persistent],
    }
    (out_dir / "failure_audit_stats.json").write_text(
        json.dumps(audit_stats, indent=2) + "\n", encoding="utf-8"
    )

    provenance = {
        "study_id": config["study_id"],
        "step": "phase_a_step1_failure_audit",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "command": ["python", "-B", "-m", "scripts.analysis.rep_repair_failure_audit", *(argv if argv is not None else sys.argv[1:])],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "config_sha256": file_sha256(args.config),
        "sources": [
            {
                "source_id": s["source_id"],
                "inputs": [
                    {"path": str(p.relative_to(asset_root).as_posix()), "sha256": file_sha256(p)}
                    for p in s["inputs"]
                ],
            }
            for s in sources
        ],
        "audit_stats": audit_stats,
    }
    (out_dir / "failure_audit_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Phase A Step 1 complete: {len(persistent)} persistent ERM-only minority failures "
        f"(g01={audit_stats['persistent_erm_only_by_group']['01']}, "
        f"g10={audit_stats['persistent_erm_only_by_group']['10']}); "
        f"{len(both_fail)} shared ERM+ERM* minority failures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
