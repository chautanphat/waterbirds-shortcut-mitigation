"""Direction 2 · Phase B · Slot 1: Pre-GAP (DAR / Oracle-Mask) & Internal Adaptation Control Ladder.

Validation-only (split == 1, cleaned "relabel" labels, zero test access).
Evaluates 7 matched-budget controls across saved standard_erm (seeds 0-4) and erm_star (seeds 0-1)
ResNet-50 backbones on identical 5-fold group-stratified splits:
  dfr_l2 -> dar -> oracle_mask_pool -> h2t_dfr -> bn_stat_only -> affine_dfr -> layer4_ft
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.methods.dpe_head_reweighting.reweighting import GROUPS
from src.methods.rep_repair.controls import (
    bn_stat_only_oof,
    extract_val_stage_cache,
    nested_dar_oof,
    nested_dfr_l2_oof,
    pool_with_spatial_masks,
    selective_adaptation_oof,
)
from src.methods.rep_repair.phase_a_data import (
    TERN_SPECIES,
    apply_label_condition,
    group_metrics,
    load_val_metadata,
    stratified_group_folds,
)
from src.methods.rep_repair.phase_b_assets import (
    downsample_mask_to_grid,
    ensure_cub_segmentations,
    ensure_waterbirds_images,
    load_val_mask_224,
    recover_backbones,
)

ROOT = Path(__file__).resolve().parents[2]
CONTROL_ORDER = (
    "dfr_l2",
    "dar",
    "oracle_mask_pool",
    "h2t_dfr",
    "bn_stat_only",
    "affine_dfr",
    "layer4_ft",
)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_controls(agg_rows: list[dict], output_path: Path):
    methods = [m for m in CONTROL_ORDER if any(r["method"] == m for r in agg_rows)]
    pretty = {
        "dfr_l2": "DFR-L2\n(GAP)",
        "dar": "DAR\n(Pre-GAP Attn)",
        "oracle_mask_pool": "Oracle Mask\nPool (7x7)",
        "h2t_dfr": "H2T-DFR\n(L1-L4)",
        "bn_stat_only": "BN-Stat\nOnly",
        "affine_dfr": "Affine-DFR\n(BN γ,β)",
        "layer4_ft": "Layer4\nFine-Tune",
    }
    x = np.arange(len(methods))
    width = 0.36
    std_wga = [next(r["wga_mean"] for r in agg_rows if r["cohort"] == "standard_erm" and r["method"] == m) for m in methods]
    std_sd = [next(r["wga_sd"] for r in agg_rows if r["cohort"] == "standard_erm" and r["method"] == m) for m in methods]
    es_wga = [
        next((r["wga_mean"] for r in agg_rows if r["cohort"] == "erm_star" and r["method"] == m), np.nan)
        for m in methods
    ]
    std_res = [next(r["persistent_rescue_pct_mean"] for r in agg_rows if r["cohort"] == "standard_erm" and r["method"] == m) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), layout="constrained")
    ax1.bar(x - width / 2, std_wga, width, yerr=std_sd, capsize=3, color="#245da6", label="Standard ERM (mean ± SD)")
    valid_es = ~np.isnan(es_wga)
    if valid_es.any():
        ax1.bar(x[valid_es] + width / 2, np.array(es_wga)[valid_es], width, color="#bc5100", label="ERM* (mean)")
    ax1.set_xticks(x, [pretty[m] for m in methods], fontsize=8.5)
    ax1.set_ylabel("Cleaned 5-Fold OOF WGA (%)")
    ax1.set_title("Phase B Control Ladder (Relabelled Validation)")
    ax1.set_ylim(max(70.0, min(std_wga) - 4.0), 100.0)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5)

    bars = ax2.bar(x, std_res, color="#4a7c59")
    for b, val in zip(bars, std_res):
        ax2.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x, [pretty[m] for m in methods], fontsize=8.5)
    ax2.set_ylabel("Rescue Rate on 26 Persistent ERM-Only Failures (%)")
    ax2.set_title("Persistent Minority Failure Rescue Across Elimination Rungs")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Direction 2 · Phase B · Pre-GAP & Internal Representation Control Ladder", fontsize=12)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory for downloaded backbones/images/masks")
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

    persistent_indices = (
        set(json.loads(args.failure_stats.read_text(encoding="utf-8"))["persistent_val_indices"])
        if args.failure_stats.is_file()
        else set()
    )
    pers_mask = np.array([i in persistent_indices for i in range(len(rel_rows))], dtype=bool)

    print("Loading 1,199 validation CUB pixel masks -> 7x7 grids...", flush=True)
    masks_224 = np.stack([load_val_mask_224(seg_root, r["id"]) for r in rel_rows], axis=0)
    masks_7x7 = np.stack([downsample_mask_to_grid(m, grid=7) for m in masks_224], axis=0)
    fg_area = (masks_224 > 0.5).mean(axis=(1, 2))

    summary_rows = []
    dar_align_rows = []
    oof_store = {}

    for rec in backbones:
        t0 = time.perf_counter()
        cohort = rec["cohort"]
        seed = int(rec["seed"])
        sid = rec["source_id"]
        ckpt = Path(rec["path"])
        print(f"=== [{sid}] Extracting frozen stage maps (val only) ===", flush=True)
        cache = extract_val_stage_cache(ckpt, image_dir, rel_rows, device=device, workers=args.workers)

        # Rung 0: DFR-L2 (GAP pooled)
        p_dfr = nested_dfr_l2_oof(cache["pooled"], y, g, outer_folds, inner_splits=4, seed=seed)
        # Rung 3a: DAR (Pre-GAP spatial attention on frozen 7x7x2048 layer4_map)
        p_dar, dar_attn = nested_dar_oof(cache["layer4_map"], y, g, outer_folds, inner_splits=4, seed=seed, device=device)
        # Rung 3b: Oracle foreground-only 7x7 pooling + DFR-L2
        oracle_feats = pool_with_spatial_masks(cache["layer4_map"], masks_7x7)
        p_oracle = nested_dfr_l2_oof(oracle_feats, y, g, outer_folds, inner_splits=4, seed=seed)
        # Rung 4a: H2T-DFR (all-layer pooled layer1..layer4)
        h2t_feats = np.concatenate(
            [cache["layer1_pool"], cache["layer2_pool"], cache["layer3_pool"], cache["pooled"]], axis=1
        )
        p_h2t = nested_dfr_l2_oof(h2t_feats, y, g, outer_folds, inner_splits=4, seed=seed)

        # Measure DAR attention mass landing inside true CUB foreground mask.
        fg_bin_7x7 = (masks_7x7 > 0.25).astype(np.float32)
        dar_fg_mass = (dar_attn * fg_bin_7x7).sum(axis=(1, 2))
        unif_fg_mass = fg_bin_7x7.mean(axis=(1, 2))
        dar_align_rows.append(
            {
                "cohort": cohort,
                "seed": seed,
                "dar_fg_mass_all_mean": round(float(dar_fg_mass.mean()), 4),
                "uniform_gap_fg_mass_all_mean": round(float(unif_fg_mass.mean()), 4),
                "dar_fg_mass_persistent_mean": round(float(dar_fg_mass[pers_mask].mean()), 4) if pers_mask.any() else "",
                "uniform_gap_fg_mass_persistent_mean": round(float(unif_fg_mass[pers_mask].mean()), 4) if pers_mask.any() else "",
                "mean_fg_pixel_area_persistent": round(float(fg_area[pers_mask].mean()), 4) if pers_mask.any() else "",
                "mean_fg_pixel_area_all": round(float(fg_area.mean()), 4),
            }
        )

        runs = [
            ("dfr_l2", p_dfr),
            ("dar", p_dar),
            ("oracle_mask_pool", p_oracle),
            ("h2t_dfr", p_h2t),
        ]

        # Run internal adaptation controls (BN-Stat-Only, Affine-DFR, Layer4-FT) on all standard_erm
        # backbones and erm_star seed 0 reference.
        if cohort == "standard_erm" or seed == 0:
            print(f"    [{sid}] Running BN-Stat-Only, Affine-DFR, Layer4-FT...", flush=True)
            p_bn = bn_stat_only_oof(ckpt, image_dir, rel_rows, outer_folds, seed=seed, device=device, workers=args.workers)
            p_aff = selective_adaptation_oof(
                ckpt, image_dir, rel_rows, outer_folds,
                mode="affine_dfr", lr=3e-3, weight_decay=1e-2, epochs=20,
                seed=seed, device=device, workers=args.workers,
            )
            p_l4 = selective_adaptation_oof(
                ckpt, image_dir, rel_rows, outer_folds,
                mode="layer4_ft", lr=2e-4, weight_decay=1e-2, epochs=15,
                seed=seed, device=device, workers=args.workers,
            )
            runs.extend([("bn_stat_only", p_bn), ("affine_dfr", p_aff), ("layer4_ft", p_l4)])

        for method_name, probs in runs:
            oof_store[f"{sid}__{method_name}"] = probs.astype(np.float32)
            preds = probs.argmax(axis=-1)
            m = group_metrics(y, preds, g)
            rescue_n = int((preds[pers_mask] == y[pers_mask]).sum()) if pers_mask.any() else 0
            rescue_pct = (100.0 * rescue_n / int(pers_mask.sum())) if pers_mask.any() else float("nan")
            summary_rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "method": method_name,
                    "wga": round(m["wga"], 4),
                    "overall": round(m["overall"], 4),
                    **{f"acc_{grp}": round(m["groups"][grp], 4) for grp in GROUPS},
                    "persistent_total": int(pers_mask.sum()),
                    "persistent_rescued": rescue_n,
                    "persistent_rescue_pct": round(rescue_pct, 2) if not np.isnan(rescue_pct) else "",
                }
            )
        print(f"    [{sid}] Completed in {time.perf_counter() - t0:.1f}s", flush=True)

    agg_rows = []
    for cohort in ("standard_erm", "erm_star"):
        for method_name in CONTROL_ORDER:
            sub = [r for r in summary_rows if r["cohort"] == cohort and r["method"] == method_name]
            if not sub:
                continue
            wgas = np.array([r["wga"] for r in sub], dtype=float)
            accs = np.array([r["overall"] for r in sub], dtype=float)
            resc = np.array([float(r["persistent_rescue_pct"]) for r in sub if r["persistent_rescue_pct"] != ""], dtype=float)
            agg_rows.append(
                {
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

    # Gate 2 controls check: compare DAR & Oracle-Mask-Pool vs ERM* and Layer4-FT on standard_erm.
    def _mean(cohort, method):
        row = next((r for r in agg_rows if r["cohort"] == cohort and r["method"] == method), None)
        return row["wga_mean"] if row else None

    gate2_controls = {
        "standard_erm_dfr_l2_wga": _mean("standard_erm", "dfr_l2"),
        "standard_erm_dar_wga": _mean("standard_erm", "dar"),
        "standard_erm_oracle_mask_pool_wga": _mean("standard_erm", "oracle_mask_pool"),
        "standard_erm_h2t_dfr_wga": _mean("standard_erm", "h2t_dfr"),
        "standard_erm_bn_stat_only_wga": _mean("standard_erm", "bn_stat_only"),
        "standard_erm_affine_dfr_wga": _mean("standard_erm", "affine_dfr"),
        "standard_erm_layer4_ft_wga": _mean("standard_erm", "layer4_ft"),
        "erm_star_dfr_l2_wga": _mean("erm_star", "dfr_l2"),
        "erm_star_dar_wga": _mean("erm_star", "dar"),
        "gap_ermstar_minus_std_dar_pp": (
            round(_mean("erm_star", "dfr_l2") - _mean("standard_erm", "dar"), 4)
            if _mean("erm_star", "dfr_l2") is not None and _mean("standard_erm", "dar") is not None
            else None
        ),
        "gap_ermstar_minus_std_oracle_mask_pp": (
            round(_mean("erm_star", "dfr_l2") - _mean("standard_erm", "oracle_mask_pool"), 4)
            if _mean("erm_star", "dfr_l2") is not None and _mean("standard_erm", "oracle_mask_pool") is not None
            else None
        ),
        "controls_survive_dar_gate": (
            (_mean("erm_star", "dfr_l2") - _mean("standard_erm", "dar")) >= 1.5
            if _mean("erm_star", "dfr_l2") is not None and _mean("standard_erm", "dar") is not None
            else False
        ),
    }

    write_csv(out_dir / "phase_b_controls_summary.csv", summary_rows)
    write_csv(out_dir / "phase_b_controls_aggregate.csv", agg_rows)
    write_csv(out_dir / "dar_attention_alignment.csv", dar_align_rows)
    np.savez_compressed(out_dir / "phase_b_controls_oof.npz", **oof_store)
    plot_controls(agg_rows, out_dir / "phase_b_controls_ladder.png")
    (out_dir / "gate2_controls_summary.json").write_text(json.dumps(gate2_controls, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Direction 2 · Phase B · Slot 1 Control Ladder Results (Validation Only)",
        "",
        f"**Controls Survive DAR Gate ($\ge 1.5\\text{{ pp}}$ ERM* $-$ std-DAR gap):** "
        f"**{'YES' if gate2_controls['controls_survive_dar_gate'] else 'NO'}** "
        f"(gap = `{gate2_controls['gap_ermstar_minus_std_dar_pp']}` pp; "
        f"vs. Oracle-Mask-Pool gap = `{gate2_controls['gap_ermstar_minus_std_oracle_mask_pp']}` pp).",
        "",
        "| Cohort | Method | OOF WGA (%) | Overall (%) | G01 (%) | G10 (%) | Persistent Failure Rescue (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        wga_str = f"{r['wga_mean']:.2f} ± {r['wga_sd']:.2f}" if r["n_seeds"] > 1 else f"{r['wga_mean']:.2f}"
        lines.append(
            f"| {r['cohort']} | `{r['method']}` | {wga_str} | {r['overall_mean']:.2f} | "
            f"{r['g01_mean']:.2f} | {r['g10_mean']:.2f} | {r['persistent_rescue_pct_mean']:.1f}% |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_b_slot1_controls",
        "split": "val_only_split1",
        "test_files_accessed": 0,
        "device": device,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
        "backbones": backbones,
        "gate2_controls": gate2_controls,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Phase B Slot 1 complete:", json.dumps(gate2_controls, indent=2), flush=True)


if __name__ == "__main__":
    main()
