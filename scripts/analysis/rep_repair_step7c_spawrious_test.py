"""Direction 2 · Phase C · Step 7c-2: Spawrious (o2o_hard) Sealed OOD Shifted Test Confirmation.

Adapts all controls + frozen pre_l4_repair_w005 strictly on the 8-group validation split (3,800 rows)
from saved ResNet-50 backbones (noerula/rep-repair-step7b-spawrious-kaggle), then evaluates once
on the official OOD shifted-background test split (~25,344 rows across 4 shifted breed x bg combos).
Supports dual-GPU (2x T4) parallel execution across seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.rep_repair_phase_c_step6 import write_csv
from src.datasets.spawrious_o2o import (
    CLASSES,
    TEST_SHIFTED_BG,
    SpawriousImageDataset,
    ensure_spawrious_o2o_hard,
    load_o2o_hard_ood_test_rows,
    load_o2o_hard_train_val_rows,
    spawrious_balanced_weights,
    spawrious_group_metrics,
    stratified_8group_folds,
)
from src.methods.rep_repair.controls import DARHead
from src.methods.rep_repair.phase_a_data import file_sha256
from src.methods.rep_repair.phase_b_assets import recover_kaggle_kernel_files
from src.methods.rep_repair.repair import (
    PreLayer4RepairModel,
    compute_group_balanced_layer3_mean,
    context_invariance_loss,
)
from src.models.resnet import load_resnet_checkpoint

ARM_ORDER = ("erm_raw", "dfr_l2", "dar", "affine_dfr_bn_l4", "affine_dfr_bn_all", "pre_l4_repair_w005")


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return "cuda:0"
    if device_arg.startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {device_arg} requested, but CUDA is unavailable")
    return device_arg


@torch.inference_mode()
def _predict_loader(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    use_amp = device.startswith("cuda")
    probs = []
    for _, xb, _, _ in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(xb.to(device))
        probs.append(logits.float().softmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.inference_mode()
def _extract_resnet_cache(
    model: nn.Module, loader: DataLoader, device: str, *, include_l4: bool = True
) -> dict[str, np.ndarray]:
    model.eval()
    use_amp = device.startswith("cuda")
    probs, pooled, l4_maps = [], [], []
    for _, xb, _, _ in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            st = model.extract_stage_maps(xb.to(device))
            logits = model[2](st["pooled"])
        probs.append(logits.float().softmax(dim=1).cpu().numpy().astype(np.float32))
        pooled.append(st["pooled"].float().cpu().numpy().astype(np.float32))
        if include_l4:
            l4_maps.append(st["layer4_map"].float().cpu().numpy().astype(np.float32))
    out = {
        "probs": np.concatenate(probs, axis=0),
        "pooled": np.concatenate(pooled, axis=0),
    }
    if include_l4:
        out["layer4_map"] = np.concatenate(l4_maps, axis=0)
    return out


def _fit_dfr_val_and_ood(
    X_val: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    X_ood: np.ndarray,
    outer_folds: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    c_grid = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    val_oof = np.zeros((len(y_val), 4), dtype=np.float64)
    ood_sum = np.zeros((len(X_ood), 4), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X_val[tr], y_val[tr], g_val[tr]
        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_8group_folds(tr_rows, n_splits=4, seed=seed + 100 + fold)
        best_c, best_key = c_grid[0], (-1.0, -1.0)
        for c_val in c_grid:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(4):
                itr, iva = in_folds != infold, in_folds == infold
                sc = StandardScaler().fit(X_tr[itr])
                clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
                clf.fit(sc.transform(X_tr[itr]), y_tr[itr], sample_weight=spawrious_balanced_weights(g_tr[itr]))
                in_preds[iva] = clf.predict(sc.transform(X_tr[iva]))
            m = spawrious_group_metrics(y_tr, in_preds, g_tr)
            k = (m["wga"], m["minority_acc"])
            if k > best_key:
                best_key, best_c = k, c_val
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X_tr), y_tr, sample_weight=spawrious_balanced_weights(g_tr))
        val_oof[te] = clf.predict_proba(sc.transform(X_val[te]))
        ood_sum += clf.predict_proba(sc.transform(X_ood))
    return val_oof, ood_sum / float(n_outer)


def _fit_dar_val_and_ood(
    base_dev: nn.Module,
    F_val: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    ood_loader: DataLoader,
    outer_folds: np.ndarray,
    *,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train 5-fold DARHeads on val_layer4_map, then stream ood_loader in 1 pass (zero 10 GB RAM cache)."""
    val_oof = np.zeros((len(y_val), 4), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    ce = nn.CrossEntropyLoss(reduction="none")
    n_params = sum(p.numel() for p in DARHead(2048, attn_dim=128, dropout=0.2, num_classes=4).parameters())
    batch_size = 256
    trained_heads: list[tuple[nn.Module, torch.Tensor, torch.Tensor]] = []

    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        F_tr, y_tr, g_tr = F_val[tr], y_val[tr], g_val[tr]
        mean = F_tr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(F_tr.std(axis=(0, 2, 3), keepdims=True), 1e-6).astype(np.float32)
        ft = torch.from_numpy((F_tr - mean) / std)
        yt = torch.from_numpy(y_tr.astype(np.int64))
        wt = torch.from_numpy(spawrious_balanced_weights(g_tr).astype(np.float32))
        mean_t = torch.from_numpy(mean).to(device)
        std_t = torch.from_numpy(std).to(device)

        seeds = (seed + fold, seed + 100 + fold, seed + 200 + fold)
        val_acc_p = np.zeros((int(te.sum()), 4), dtype=np.float64)
        for s in seeds:
            torch.manual_seed(s)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(s)
            head = DARHead(2048, attn_dim=128, dropout=0.2, num_classes=4).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
            gen = torch.Generator().manual_seed(s)
            for _ in range(100):
                head.train()
                perm = torch.randperm(len(F_tr), generator=gen)
                for st in range(0, len(F_tr), batch_size):
                    idx = perm[st : st + batch_size]
                    opt.zero_grad(set_to_none=True)
                    logits, _ = head(ft[idx].to(device))
                    loss = (ce(logits, yt[idx].to(device)) * wt[idx].to(device)).mean()
                    loss.backward()
                    opt.step()
            head.eval()
            with torch.inference_mode():
                chunks = []
                for st in range(0, int(te.sum()), batch_size):
                    xb = torch.from_numpy((F_val[te][st : st + batch_size] - mean) / std).to(device)
                    chunks.append(head(xb)[0].softmax(dim=1).cpu().numpy())
                val_acc_p += np.concatenate(chunks, axis=0)
            trained_heads.append((head, mean_t, std_t))
        val_oof[te] = val_acc_p / float(len(seeds))

    # Single streaming pass over ood_loader applying all 5*3 DARHeads in-flight (<200 MB RAM).
    use_amp = device.startswith("cuda")
    ood_chunks = []
    base_dev.eval()
    with torch.inference_mode():
        for _, xb, _, _ in ood_loader:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                l4 = base_dev.extract_stage_maps(xb.to(device))["layer4_map"].float()
            p_sum = torch.zeros((l4.shape[0], 4), device=device, dtype=torch.float32)
            for head, mean_t, std_t in trained_heads:
                p_sum += head((l4 - mean_t) / std_t)[0].softmax(dim=1)
            ood_chunks.append((p_sum / float(len(trained_heads))).cpu().numpy())
    return val_oof, np.concatenate(ood_chunks, axis=0), n_params


def _train_5fold_repair_val_and_ood(
    base_cpu: nn.Module,
    val_rows: list[dict],
    ood_loader: DataLoader,
    outer_folds: np.ndarray,
    *,
    use_gate: bool,
    bn_scope: str,
    use_inv: bool,
    epochs: int,
    seed: int,
    device: str,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train 5-fold validation adaptation/repair strictly on val_rows and evaluate on OOD shifted test."""
    val_oof = np.zeros((len(val_rows), 4), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    ood_sum = np.zeros((len(ood_loader.dataset), 4), dtype=np.float64)
    ce = nn.CrossEntropyLoss()
    n_trainable = 0

    bg_names = sorted({str(r["bg"]) for r in val_rows})
    bg_to_int = {b: i for i, b in enumerate(bg_names)}

    for fold in range(n_outer):
        te_idx = np.flatnonzero(outer_folds == fold)
        tr_idx = np.flatnonzero(outer_folds != fold)
        tr_rows = [val_rows[i] for i in tr_idx]
        te_rows = [val_rows[i] for i in te_idx]

        inner_folds = stratified_8group_folds(tr_rows, n_splits=4, seed=seed + 300 + fold)
        fit_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds != 0)]
        sel_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds == 0)]
        sel_y = np.array([int(r["y"]) for r in sel_sub], dtype=int)
        sel_g = np.array([str(r["group"]) for r in sel_sub], dtype=str)

        torch.manual_seed(seed + fold)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed + fold)

        model = PreLayer4RepairModel(
            base_cpu,
            use_gate=use_gate,
            gate_mode=("peak_neutral" if use_gate else "raw"),
            head_type="linear",
            bn_scope=bn_scope,
        ).to(device)
        if use_gate:
            fit_eval_loader = DataLoader(SpawriousImageDataset(fit_sub), batch_size=128, shuffle=False, num_workers=workers)
            model.set_neutral_c3(compute_group_balanced_layer3_mean(base_cpu, fit_eval_loader, device))

        params = model.trainable_parameters()
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-2)
        use_amp = device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        w_fit = spawrious_balanced_weights([str(r["group"]) for r in fit_sub])
        gen = torch.Generator().manual_seed(seed + 50 + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_fit).double(), num_samples=len(fit_sub), replacement=True, generator=gen)
        fit_loader = DataLoader(SpawriousImageDataset(fit_sub), batch_size=64, sampler=sampler, num_workers=workers, pin_memory=use_amp)
        sel_loader = DataLoader(SpawriousImageDataset(sel_sub), batch_size=128, shuffle=False, num_workers=workers, pin_memory=use_amp)
        te_loader = DataLoader(SpawriousImageDataset(te_rows), batch_size=128, shuffle=False, num_workers=workers, pin_memory=use_amp)

        best_key = (-1.0, -1.0)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for _ in range(epochs):
            model.set_repair_train_mode()
            for idx_b, xb, yb, _ in fit_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    if use_inv:
                        ab = torch.tensor([bg_to_int[str(fit_sub[int(i)]["bg"])] for i in idx_b], dtype=torch.long, device=device)
                        out = model.forward_with_context_swap(xb, ab)
                        loss_ce = ce(out["logits"].float(), yb)
                        inv = context_invariance_loss(
                            out["logits"].float(), out["logits_swap"].float(),
                            out["layer4_map"].float(), out["layer4_map_swap"].float(), out["gate_map"].float(),
                            drift_weight=0.05, cons_weight=0.10, gate_sparsity=0.005,
                        )
                        loss = loss_ce + inv["total"]
                    else:
                        loss = ce(model(xb).float(), yb)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

            sel_probs = _predict_loader(model, sel_loader, device)
            sel_m = spawrious_group_metrics(sel_y, sel_probs.argmax(axis=1), sel_g)
            key = (round(sel_m["wga"], 3), round(sel_m["minority_acc"], 3))
            if key >= best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        val_oof[te_idx] = _predict_loader(model, te_loader, device)
        ood_sum += _predict_loader(model, ood_loader, device)
    return val_oof, ood_sum / float(n_outer), n_trainable


def eval_val_and_ood(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    ood_probs: np.ndarray,
    y_ood: np.ndarray,
) -> dict:
    vm = spawrious_group_metrics(y_val, val_probs.argmax(axis=1), g_val)
    ood_preds = ood_probs.argmax(axis=1)
    per_cls = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y_ood == cls_idx
        per_cls[cls_name] = float((ood_preds[mask] == y_ood[mask]).mean()) * 100.0
    ood_auc = float(roc_auc_score(y_ood, ood_probs, multi_class="ovr", average="macro")) * 100.0 if len(np.unique(y_ood)) == 4 else float("nan")
    return {
        "val_wga": round(vm["wga"], 4),
        "val_beach_acc": round(vm["minority_acc"], 4),
        "ood_mean_acc": round(float((ood_preds == y_ood).mean()) * 100.0, 4),
        "ood_worst_class_acc": round(float(min(per_cls.values())), 4),
        "ood_macro_auc": round(ood_auc, 4),
        "ood_bulldog_mountain": round(per_cls["bulldog"], 4),
        "ood_corgi_jungle": round(per_cls["corgi"], 4),
        "ood_dachshund_snow": round(per_cls["dachshund"], 4),
        "ood_labrador_desert": round(per_cls["labrador"], 4),
    }


def run_single_seed(
    data_root: Path,
    ckpt_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    adapt_epochs: int,
    device: str,
    workers: int,
) -> dict:
    seed_json = out_dir / f"seed_{seed}_ood_results.json"
    seed_npz = out_dir / f"seed_{seed}_ood_probs.npz"
    if seed_json.is_file() and seed_npz.is_file():
        print(f"    [spawrious s{seed}] Reusing completed seed output", flush=True)
        return json.loads(seed_json.read_text(encoding="utf-8"))

    _, val_rows = load_o2o_hard_train_val_rows(data_root, val_fraction=0.15, split_seed=0)
    ood_rows = load_o2o_hard_ood_test_rows(data_root)
    y_val = np.array([int(r["y"]) for r in val_rows], dtype=int)
    g_val = np.array([str(r["group"]) for r in val_rows], dtype=str)
    y_ood = np.array([int(r["y"]) for r in ood_rows], dtype=int)
    outer_folds = stratified_8group_folds(val_rows, n_splits=5, seed=0)

    member = f"spawrious_o2o_hard_backbones/resnet50_o2o_hard_s{seed}_ckpt_last.pt"
    ckpt_path = recover_kaggle_kernel_files("rep-repair-step7b-spawrious-kaggle", [member], ckpt_dir, workers=1)[member]
    ckpt_sha = file_sha256(ckpt_path)
    base_cpu = load_resnet_checkpoint(ckpt_path, device="cpu", num_classes=4)
    base_dev = copy.deepcopy(base_cpu).to(device)

    pin = device.startswith("cuda")
    val_loader = DataLoader(SpawriousImageDataset(val_rows), batch_size=128, shuffle=False, num_workers=workers, pin_memory=pin)
    ood_loader = DataLoader(SpawriousImageDataset(ood_rows), batch_size=256, shuffle=False, num_workers=workers, pin_memory=pin)
    v_cache = _extract_resnet_cache(base_dev, val_loader, device, include_l4=True)
    o_cache = _extract_resnet_cache(base_dev, ood_loader, device, include_l4=False)

    val_dfr, ood_dfr = _fit_dfr_val_and_ood(v_cache["pooled"], y_val, g_val, o_cache["pooled"], outer_folds, seed=seed)
    val_dar, ood_dar, n_dar = _fit_dar_val_and_ood(
        base_dev, v_cache["layer4_map"], y_val, g_val, ood_loader, outer_folds, seed=seed, device=device
    )
    erm_val_p, erm_ood_p = v_cache["probs"], o_cache["probs"]
    del v_cache, o_cache, base_dev
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    arm_outputs = {
        "erm_raw": (erm_val_p, erm_ood_p, sum(p.numel() for p in base_cpu.parameters())),
        "dfr_l2": (val_dfr, ood_dfr, 2048 * 4 + 4),
        "dar": (val_dar, ood_dar, n_dar),
    }
    for arm_name, use_gate, bn_scope, use_inv in (
        ("affine_dfr_bn_l4", False, "bn_l4", False),
        ("affine_dfr_bn_all", False, "bn_all", False),
        ("pre_l4_repair_w005", True, "bn_all", True),
    ):
        print(f"    [spawrious s{seed} | {device}] starting {arm_name}...", flush=True)
        v_p, o_p, n_p = _train_5fold_repair_val_and_ood(
            base_cpu, val_rows, ood_loader, outer_folds,
            use_gate=use_gate, bn_scope=bn_scope, use_inv=use_inv,
            epochs=adapt_epochs, seed=seed, device=device, workers=workers,
        )
        arm_outputs[arm_name] = (v_p, o_p, n_p)

    rows = []
    prob_store = {}
    for arm in ARM_ORDER:
        vp, op, n_p = arm_outputs[arm]
        m = eval_val_and_ood(vp, y_val, g_val, op, y_ood)
        rows.append({"seed": seed, "arm": arm, "trainable_params": n_p, **m})
        prob_store[f"spawrious_s{seed}__{arm}__val"] = vp.astype(np.float32)
        prob_store[f"spawrious_s{seed}__{arm}__ood"] = op.astype(np.float32)
        print(
            f"    [spawrious s{seed} | {device}] {arm:19s} | Val WGA={m['val_wga']:.2f}% | "
            f"OOD Mean={m['ood_mean_acc']:.2f}% | OOD Worst-Cls={m['ood_worst_class_acc']:.2f}%",
            flush=True,
        )

    payload = {
        "seed": seed, "device": device,
        "n_val": len(val_rows), "n_ood_test": len(ood_rows),
        "checkpoint": str(ckpt_path), "sha256": ckpt_sha,
        "rows": rows,
    }
    (out_dir / f"seed_{seed}_ood_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out_dir / f"seed_{seed}_ood_probs.npz", **prob_store)
    return payload


def plot_step7c_spawrious(agg_rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), layout="constrained")
    arms = [r["arm"] for r in agg_rows]
    pretty = {
        "erm_raw": "ERM\n(Raw)", "dfr_l2": "DFR-L2\n(GAP)", "dar": "DAR\n(7x7 Attn)",
        "affine_dfr_bn_l4": "Affine-DFR\n(BN L4)", "affine_dfr_bn_all": "Affine-DFR\n(BN All)",
        "pre_l4_repair_w005": "Pre-L4 Repair\n(w=0.05)",
    }
    x = np.arange(len(arms))
    w = 0.36

    ax = axes[0]
    om = [r["ood_mean_acc_mean"] for r in agg_rows]
    omsd = [r["ood_mean_acc_sd"] for r in agg_rows]
    ow = [r["ood_worst_class_acc_mean"] for r in agg_rows]
    ax.bar(x - w / 2, om, w, yerr=omsd, capsize=3, color="#245da6", label="OOD Mean Acc (%)")
    ax.bar(x + w / 2, ow, w, color="#bc5100", label="OOD Worst-Class Acc (%)")
    for i, (mv, wv) in enumerate(zip(om, ow)):
        ax.text(i - w / 2, mv + 0.5, f"{mv:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w / 2, wv + 0.5, f"{wv:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Shifted OOD Test Accuracy (%)")
    ax.set_title("A. Spawrious (o2o_hard) Shifted OOD Test: Mean & Worst Class")
    ax.set_ylim(0, 102)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    cols = ["#245da6", "#2e7d32", "#bc5100", "#7b3294"]
    keys = [
        ("ood_bulldog_mountain_mean", "bulldog @ mountain"),
        ("ood_corgi_jungle_mean", "corgi @ jungle"),
        ("ood_dachshund_snow_mean", "dachshund @ snow"),
        ("ood_labrador_desert_mean", "labrador @ desert"),
    ]
    for (k, lbl), col in zip(keys, cols):
        ax.plot(x, [r[k] for r in agg_rows], marker="o", lw=1.9, color=col, label=lbl)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Per-Breed Shifted OOD Accuracy (%)")
    ax.set_title("B. Per-Breed Robustness Under Opposite Spurious Backgrounds")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Direction 2 · Phase C · Step 7c-2 Spawrious (o2o_hard) Sealed OOD Test Confirmation", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--worker-seed", type=int, default=None)
    args = parser.parse_args(argv)

    work_dir = args.work_dir.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "spawrious_backbones"

    data_root = ensure_spawrious_o2o_hard(work_dir)
    if args.worker_seed is not None:
        dev = _resolve_device(args.device)
        run_single_seed(
            data_root, ckpt_dir, out_dir,
            seed=args.worker_seed, adapt_epochs=args.adapt_epochs, device=dev, workers=args.workers,
        )
        return

    recover_kaggle_kernel_files(
        "rep-repair-step7b-spawrious-kaggle",
        [f"spawrious_o2o_hard_backbones/resnet50_o2o_hard_s{s}_ckpt_last.pt" for s in args.seeds],
        ckpt_dir, workers=3,
    )

    base_dev = _resolve_device(args.device)
    n_gpus = min(args.max_gpus, torch.cuda.device_count()) if base_dev.startswith("cuda") else 0
    if n_gpus >= 2 and len(args.seeds) >= 2:
        print(f"=== Dispatching {len(args.seeds)} Spawrious OOD test seeds across {n_gpus} GPUs ===", flush=True)
        gpu_q: queue.Queue[str] = queue.Queue()
        for gid in range(n_gpus):
            gpu_q.put(f"cuda:{gid}")
        abort_event = threading.Event()
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}

        def _launch(s: int):
            if abort_event.is_set():
                return
            dev_id = gpu_q.get()
            try:
                if abort_event.is_set():
                    return
                cmd = [
                    sys.executable, "-B", "-m", "scripts.analysis.rep_repair_step7c_spawrious_test",
                    "--work-dir", str(work_dir), "--output", str(out_dir),
                    "--adapt-epochs", str(args.adapt_epochs),
                    "--device", dev_id, "--workers", "0", "--worker-seed", str(s),
                ]
                print(f"Launching Spawrious OOD seed {s} on {dev_id}...", flush=True)
                proc = subprocess.run(cmd, env=child_env)
                if proc.returncode != 0:
                    abort_event.set()
                    raise RuntimeError(f"Spawrious OOD seed {s} on {dev_id} failed with exit code {proc.returncode}")
                print(f"[DONE] Spawrious OOD seed {s} on {dev_id}", flush=True)
            finally:
                gpu_q.put(dev_id)

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futs = [pool.submit(_launch, s) for s in args.seeds]
            for fut in as_completed(futs):
                fut.result()
    else:
        for s in args.seeds:
            run_single_seed(
                data_root, ckpt_dir, out_dir,
                seed=s, adapt_epochs=args.adapt_epochs, device=base_dev, workers=args.workers,
            )

    per_seed_rows, ckpt_records = [], []
    prob_store: dict[str, np.ndarray] = {}
    n_val = n_ood = 0
    for s in args.seeds:
        payload = json.loads((out_dir / f"seed_{s}_ood_results.json").read_text(encoding="utf-8"))
        n_val, n_ood = payload["n_val"], payload["n_ood_test"]
        ckpt_records.append({"seed": s, "path": payload["checkpoint"], "sha256": payload["sha256"]})
        per_seed_rows.extend(payload["rows"])
        with np.load(out_dir / f"seed_{s}_ood_probs.npz", allow_pickle=False) as zf:
            for k in zf.files:
                prob_store[k] = zf[k]

    dfr_by_seed = {r["seed"]: r for r in per_seed_rows if r["arm"] == "dfr_l2"}
    aff_by_seed = {r["seed"]: r for r in per_seed_rows if r["arm"] == "affine_dfr_bn_all"}

    agg_rows = []
    for arm in ARM_ORDER:
        sub = sorted([r for r in per_seed_rows if r["arm"] == arm], key=lambda r: r["seed"])
        dm_dfr = [r["ood_mean_acc"] - dfr_by_seed[r["seed"]]["ood_mean_acc"] for r in sub]
        dm_aff = [r["ood_mean_acc"] - aff_by_seed[r["seed"]]["ood_mean_acc"] for r in sub]
        dw_aff = [r["ood_worst_class_acc"] - aff_by_seed[r["seed"]]["ood_worst_class_acc"] for r in sub]
        agg_rows.append({
            "arm": arm,
            "n_seeds": len(sub),
            "trainable_params": sub[0]["trainable_params"],
            "val_wga_mean": round(float(np.mean([r["val_wga"] for r in sub])), 2),
            "val_beach_acc_mean": round(float(np.mean([r["val_beach_acc"] for r in sub])), 2),
            "ood_mean_acc_mean": round(float(np.mean([r["ood_mean_acc"] for r in sub])), 2),
            "ood_mean_acc_sd": round(float(np.std([r["ood_mean_acc"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "ood_worst_class_acc_mean": round(float(np.mean([r["ood_worst_class_acc"] for r in sub])), 2),
            "ood_worst_class_acc_sd": round(float(np.std([r["ood_worst_class_acc"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "ood_macro_auc_mean": round(float(np.mean([r["ood_macro_auc"] for r in sub])), 2),
            "delta_ood_mean_vs_dfr": round(float(np.mean(dm_dfr)), 2),
            "delta_ood_mean_vs_affine": round(float(np.mean(dm_aff)), 2),
            "mean_wins_vs_affine": f"{sum(v > 0 for v in dm_aff)}/{len(sub)}",
            "delta_ood_worst_vs_affine": round(float(np.mean(dw_aff)), 2),
            "ood_bulldog_mountain_mean": round(float(np.mean([r["ood_bulldog_mountain"] for r in sub])), 2),
            "ood_corgi_jungle_mean": round(float(np.mean([r["ood_corgi_jungle"] for r in sub])), 2),
            "ood_dachshund_snow_mean": round(float(np.mean([r["ood_dachshund_snow"] for r in sub])), 2),
            "ood_labrador_desert_mean": round(float(np.mean([r["ood_labrador_desert"] for r in sub])), 2),
        })

    write_csv(out_dir / "step7c_spawrious_ood_per_seed.csv", per_seed_rows)
    write_csv(out_dir / "step7c_spawrious_ood_aggregate.csv", agg_rows)
    np.savez_compressed(out_dir / "step7c_spawrious_ood_probs.npz", **prob_store)
    plot_step7c_spawrious(agg_rows, out_dir / "step7c_spawrious_ood.png")

    lines = [
        "# Direction 2 · Phase C · Step 7c-2 Spawrious (`o2o_hard`) Sealed OOD Test Confirmation",
        "",
        f"Validation adaptation set: **{n_val:,}** images | Shifted OOD test set (evaluated once): **{n_ood:,}** images.",
        "",
        "| Arm | Params | Val WGA | OOD Mean Acc (%) | OOD Worst-Cls (%) | Δ Mean vs Affine (Wins) | OOD Macro-AUC (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        lines.append(
            f"| `{r['arm']}` | {r['trainable_params']:,} | {r['val_wga_mean']:.2f} | "
            f"**{r['ood_mean_acc_mean']:.2f} ± {r['ood_mean_acc_sd']:.2f}** | "
            f"**{r['ood_worst_class_acc_mean']:.2f} ± {r['ood_worst_class_acc_sd']:.2f}** | "
            f"{r['delta_ood_mean_vs_affine']:+.2f} ({r['mean_wins_vs_affine']}) | "
            f"**{r['ood_macro_auc_mean']:.2f}** |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_c_step7c_spawrious_ood_one_shot_test",
        "one_shot_test_unsealing": True,
        "frozen_repair_hparams": {"gate_mode": "peak_neutral", "bn_scope": "bn_all", "drift_weight": 0.05, "cons_weight": 0.10, "gate_sparsity": 0.005},
        "gpus_used": n_gpus,
        "checkpoints": ckpt_records,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Step 7c-2 Spawrious OOD test complete. Summary:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
