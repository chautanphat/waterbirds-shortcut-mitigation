"""Direction 2 · Phase C · Step 7b: Dual-GPU (2x T4) Spawrious (o2o_hard) Training & Val Controls.

Runs strictly without OOD test access:
  - Excludes all 4 shifted o2o_hard test combinations via src/datasets/spawrious_o2o.py.
  - Splits official o2o_hard training domains 0 & 1 (stratified across all 8 groups) into
    85% ERM training / 15% held-out validation.
  - Supports multi-GPU parallel execution across seeds (e.g., cuda:0 and cuda:1 on 2x T4).
  - Evaluates 5-fold validation control ladder: erm_raw, dfr_l2, dar, affine_dfr_{bn_l3,bn_l4,bn_all}.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
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
    GROUPS_8,
    SpawriousImageDataset,
    ensure_spawrious_o2o_hard,
    load_o2o_hard_train_val_rows,
    spawrious_balanced_weights,
    spawrious_group_metrics,
    stratified_8group_folds,
)
from src.methods.rep_repair.controls import DARHead
from src.methods.rep_repair.patching import configure_bn_stage_params
from src.methods.rep_repair.phase_a_data import file_sha256
from src.models.resnet import ResNetClassifier, load_resnet_checkpoint
from src.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[2]
ARM_ORDER = ("erm_raw", "dfr_l2", "dar", "affine_dfr_bn_l3", "affine_dfr_bn_l4", "affine_dfr_bn_all")


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


def _eval_spawrious_loader(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    use_amp = device.startswith("cuda")
    probs = []
    with torch.inference_mode():
        for _, xb, _, _ in loader:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xb.to(device))
            probs.append(logits.float().softmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def eval_arm_metrics(probs: np.ndarray, y: np.ndarray, g: np.ndarray, is_beach: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    m = spawrious_group_metrics(y, preds, g)
    all_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro")) * 100.0 if len(np.unique(y)) == 4 else float("nan")
    beach_mask = is_beach == 1
    min_auc = (
        float(roc_auc_score(y[beach_mask], probs[beach_mask], multi_class="ovr", average="macro")) * 100.0
        if beach_mask.any() and len(np.unique(y[beach_mask])) == 4
        else float("nan")
    )
    row = {
        "wga": round(m["wga"], 4),
        "minority_acc": round(m["minority_acc"], 4),
        "majority_acc": round(m["majority_acc"], 4),
        "overall_acc": round(m["overall"], 4),
        "min_macro_auc": round(min_auc, 4),
        "all_macro_auc": round(all_auc, 4),
    }
    for grp in GROUPS_8:
        row[f"acc_{grp}"] = round(m["groups"].get(grp, float("nan")), 4)
    return row


def train_spawrious_erm_backbone(
    train_rows: list[dict],
    val_rows: list[dict],
    ckpt_path: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    workers: int,
) -> list[dict]:
    """Train standard-ERM ResNet-50 (4 classes) on Spawrious train split only (no group labels used)."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.is_file():
        print(f"=== [s{seed} | {device}] Reusing checkpoint {ckpt_path.name} ===", flush=True)
        return []

    seed_everything(seed)
    use_amp = device.startswith("cuda")
    train_loader = DataLoader(
        SpawriousImageDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_amp,
        drop_last=len(train_rows) >= batch_size,
    )
    val_loader = DataLoader(
        SpawriousImageDataset(val_rows), batch_size=256, shuffle=False, num_workers=workers, pin_memory=use_amp
    )
    val_y = np.array([int(r["y"]) for r in val_rows], dtype=int)
    val_g = np.array([str(r["group"]) for r in val_rows], dtype=str)

    model = ResNetClassifier(weights="IMAGENET1K_V2", num_classes=4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    eval_epochs = {1, 5, 10, epochs}
    history: list[dict] = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for _, xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(yb)
            count += len(yb)

        row = {"epoch": ep, "train_loss": round(loss_sum / max(1, count), 4)}
        if ep in eval_epochs:
            val_probs = _eval_spawrious_loader(model, val_loader, device)
            m = spawrious_group_metrics(val_y, val_probs.argmax(axis=1), val_g)
            row["val_wga"] = round(m["wga"], 2)
            row["val_min_acc"] = round(m["minority_acc"], 2)
            row["val_overall"] = round(m["overall"], 2)
            print(
                f"    [s{seed} | {device}] ep {ep:02d}/{epochs} | loss={row['train_loss']:.4f} | "
                f"WGA={row['val_wga']:.2f}% | Beach={row['val_min_acc']:.2f}% ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
        history.append(row)

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(cpu_state, ckpt_path)
    return history


@torch.inference_mode()
def extract_val_cache(model: ResNetClassifier, val_rows: list[dict], device: str, workers: int) -> dict[str, np.ndarray]:
    model.eval()
    use_amp = device.startswith("cuda")
    loader = DataLoader(SpawriousImageDataset(val_rows), batch_size=128, shuffle=False, num_workers=workers, pin_memory=use_amp)
    probs, pooled, l4_maps = [], [], []
    for _, xb, _, _ in loader:
        xb = xb.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            st = model.extract_stage_maps(xb)
            logits = model[2](st["pooled"])
        probs.append(logits.float().softmax(dim=1).cpu().numpy().astype(np.float32))
        pooled.append(st["pooled"].float().cpu().numpy().astype(np.float32))
        l4_maps.append(st["layer4_map"].float().cpu().numpy().astype(np.float32))
    return {
        "erm_probs": np.concatenate(probs, axis=0),
        "pooled": np.concatenate(pooled, axis=0),
        "layer4_map": np.concatenate(l4_maps, axis=0),
    }


def nested_spawrious_dfr_l2_oof(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    outer_folds: np.ndarray,
    *,
    inner_splits: int = 4,
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    seed: int = 0,
) -> np.ndarray:
    probs = np.zeros((len(y), 4), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X[tr], y[tr], g[tr]
        tr_rows = [{"id": str(i), "group": str(grp)} for i, grp in enumerate(g_tr)]
        in_folds = stratified_8group_folds(tr_rows, n_splits=inner_splits, seed=seed + 100 + fold)
        best_c, best_key = c_grid[0], (-1.0, -1.0)
        for c_val in c_grid:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(inner_splits):
                itr = in_folds != infold
                iva = in_folds == infold
                sc = StandardScaler().fit(X_tr[itr])
                clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
                clf.fit(sc.transform(X_tr[itr]), y_tr[itr], sample_weight=spawrious_balanced_weights(g_tr[itr]))
                in_preds[iva] = clf.predict(sc.transform(X_tr[iva]))
            m = spawrious_group_metrics(y_tr, in_preds, g_tr)
            key = (m["wga"], m["minority_acc"])
            if key > best_key:
                best_key, best_c = key, c_val
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X_tr), y_tr, sample_weight=spawrious_balanced_weights(g_tr))
        probs[te] = clf.predict_proba(sc.transform(X[te]))
    return probs


def _fit_dar_4class_single(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    F_te: np.ndarray,
    *,
    attn_dim: int = 128,
    weight_decay: float = 1e-2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 100,
    seeds: tuple[int, ...],
    device: str,
) -> np.ndarray:
    mean = F_tr.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(F_tr.std(axis=(0, 2, 3), keepdims=True), 1e-6)
    ft = torch.from_numpy(((F_tr - mean) / std).astype(np.float32))
    fe = torch.from_numpy(((F_te - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.int64))
    wt = torch.from_numpy(spawrious_balanced_weights(g_tr).astype(np.float32))
    ce = nn.CrossEntropyLoss(reduction="none")

    prob_sum = np.zeros((len(F_te), 4), dtype=np.float64)
    n_tr = len(F_tr)
    batch_size = 256
    for s in seeds:
        torch.manual_seed(s)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(s)
        head = DARHead(F_tr.shape[1], attn_dim=attn_dim, dropout=dropout, num_classes=4).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(s)
        for _ in range(epochs):
            head.train()
            perm = torch.randperm(n_tr, generator=gen)
            for start in range(0, n_tr, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad(set_to_none=True)
                logits, _ = head(ft[idx].to(device))
                loss = (ce(logits, yt[idx].to(device)) * wt[idx].to(device)).mean()
                loss.backward()
                opt.step()
        head.eval()
        fold_probs = []
        with torch.inference_mode():
            for start in range(0, len(F_te), batch_size):
                logits_e, _ = head(fe[start : start + batch_size].to(device))
                fold_probs.append(logits_e.softmax(dim=1).cpu().numpy())
        prob_sum += np.concatenate(fold_probs, axis=0)
    return prob_sum / len(seeds)


def nested_spawrious_dar_oof(
    layer4_map: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    outer_folds: np.ndarray,
    *,
    seed: int = 0,
    device: str = "cuda:0",
) -> np.ndarray:
    probs = np.zeros((len(y), 4), dtype=np.float64)
    for fold in range(int(outer_folds.max()) + 1):
        te = outer_folds == fold
        tr = ~te
        probs[te] = _fit_dar_4class_single(
            layer4_map[tr], y[tr], g[tr], layer4_map[te],
            seeds=(seed + fold, seed + 100 + fold, seed + 200 + fold),
            device=device,
        )
    return probs


def spawrious_affine_dfr_oof(
    base_cpu: ResNetClassifier,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    *,
    bn_scope: str,
    seed: int,
    epochs: int,
    device: str,
    workers: int,
) -> tuple[np.ndarray, int]:
    """5-fold BN-affine + 4-class linear-head adaptation with frozen BN stats & inner selection."""
    probs = np.zeros((len(val_rows), 4), dtype=np.float64)
    ce = nn.CrossEntropyLoss()
    use_amp = device.startswith("cuda")
    n_trainable = 0

    for fold in range(int(outer_folds.max()) + 1):
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
        if use_amp:
            torch.cuda.manual_seed_all(seed + fold)

        model = copy.deepcopy(base_cpu).to(device)
        params = configure_bn_stage_params(model, bn_scope)
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-2)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        w_fit = spawrious_balanced_weights([str(r["group"]) for r in fit_sub])
        gen = torch.Generator().manual_seed(seed + 50 + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_fit).double(), num_samples=len(fit_sub), replacement=True, generator=gen)
        fit_loader = DataLoader(SpawriousImageDataset(fit_sub), batch_size=64, sampler=sampler, num_workers=workers)
        sel_loader = DataLoader(SpawriousImageDataset(sel_sub), batch_size=128, shuffle=False, num_workers=workers)
        te_loader = DataLoader(SpawriousImageDataset(te_rows), batch_size=128, shuffle=False, num_workers=workers)

        best_key = (-1.0, -1.0)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for _ in range(epochs):
            model.eval()
            for _, xb, yb, _ in fit_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = ce(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            sel_probs = _eval_spawrious_loader(model, sel_loader, device)
            sel_m = spawrious_group_metrics(sel_y, sel_probs.argmax(axis=1), sel_g)
            key = (round(sel_m["wga"], 3), round(sel_m["minority_acc"], 3))
            if key >= best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs[te_idx] = _eval_spawrious_loader(model, te_loader, device)
    return probs, n_trainable


def run_single_seed(
    data_root: Path,
    ckpt_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    adapt_epochs: int,
    device: str,
    workers: int,
) -> dict:
    train_rows, val_rows = load_o2o_hard_train_val_rows(data_root, val_fraction=0.15, split_seed=0)
    y = np.array([int(r["y"]) for r in val_rows], dtype=int)
    g = np.array([str(r["group"]) for r in val_rows], dtype=str)
    is_beach = np.array([int(r["is_beach"]) for r in val_rows], dtype=int)
    outer_folds = stratified_8group_folds(val_rows, n_splits=5, seed=0)

    ckpt_path = ckpt_dir / f"resnet50_o2o_hard_s{seed}_ckpt_last.pt"
    history = train_spawrious_erm_backbone(
        train_rows, val_rows, ckpt_path,
        seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, device=device, workers=workers,
    )
    if history:
        (out_dir / f"train_history_s{seed}.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    base_cpu = load_resnet_checkpoint(ckpt_path, device="cpu", num_classes=4)
    cache = extract_val_cache(copy.deepcopy(base_cpu).to(device), val_rows, device=device, workers=workers)

    dfr_probs = nested_spawrious_dfr_l2_oof(cache["pooled"], y, g, outer_folds, seed=seed)
    dar_probs = nested_spawrious_dar_oof(cache["layer4_map"], y, g, outer_folds, seed=seed, device=device)

    arm_outputs: dict[str, tuple[np.ndarray, int]] = {
        "erm_raw": (cache["erm_probs"], sum(p.numel() for p in base_cpu.parameters())),
        "dfr_l2": (dfr_probs, 2048 * 4 + 4),
        "dar": (dar_probs, sum(p.numel() for p in DARHead(num_classes=4).parameters())),
    }
    for scope, arm_name in (("bn_l3", "affine_dfr_bn_l3"), ("bn_l4", "affine_dfr_bn_l4"), ("bn_all", "affine_dfr_bn_all")):
        probs, n_params = spawrious_affine_dfr_oof(
            base_cpu, val_rows, outer_folds,
            bn_scope=scope, seed=seed, epochs=adapt_epochs, device=device, workers=workers,
        )
        arm_outputs[arm_name] = (probs, n_params)

    rows = []
    oof_store = {}
    for arm in ARM_ORDER:
        probs, n_params = arm_outputs[arm]
        oof_store[f"spawrious_s{seed}__{arm}"] = probs.astype(np.float32)
        m = eval_arm_metrics(probs, y, g, is_beach)
        rows.append({"seed": seed, "arm": arm, "trainable_params": n_params, **m})
        print(
            f"    [s{seed} | {device}] {arm:18s}: WGA={m['wga']:.2f}% | Beach={m['minority_acc']:.2f}% | "
            f"MinAUC={m['min_macro_auc']:.2f}% | Acc={m['overall_acc']:.2f}%",
            flush=True,
        )

    seed_payload = {
        "seed": seed,
        "device": device,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "checkpoint": {"seed": seed, "path": str(ckpt_path), "sha256": file_sha256(ckpt_path)},
        "rows": rows,
    }
    (out_dir / f"seed_{seed}_results.json").write_text(json.dumps(seed_payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out_dir / f"seed_{seed}_oof.npz", **oof_store)
    return seed_payload


def plot_step7b(agg_rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), layout="constrained")
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in agg_rows)]
    pretty = {
        "erm_raw": "ERM\n(Raw)",
        "dfr_l2": "DFR-L2\n(GAP)",
        "dar": "DAR\n(7x7 Attn)",
        "affine_dfr_bn_l3": "Affine-DFR\n(BN L3)",
        "affine_dfr_bn_l4": "Affine-DFR\n(BN L4)",
        "affine_dfr_bn_all": "Affine-DFR\n(BN All)",
    }
    x = np.arange(len(arms))
    w = 0.36

    # Panel A: 8-Group WGA & Mean Minority (Beach) Accuracy
    ax = axes[0]
    wgas = [next(r["wga_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    wsds = [next(r["wga_sd"] for r in agg_rows if r["arm"] == a) for a in arms]
    mins = [next(r["minority_acc_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
    ax.bar(x - w / 2, wgas, w, yerr=wsds, capsize=3, color="#245da6", label="8-Group WGA (%)")
    ax.bar(x + w / 2, mins, w, color="#bc5100", label="Mean Minority (`beach`) Acc (%)")
    for i, (wv, mv) in enumerate(zip(wgas, mins)):
        ax.text(i - w / 2, wv + 0.6, f"{wv:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w / 2, mv + 0.6, f"{mv:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A. Spawrious (o2o_hard) Val 8-Group WGA & Minority Acc")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    # Panel B: Per-Class Minority (`beach`) Group Accuracy
    ax = axes[1]
    palette = ["#245da6", "#2e7d32", "#bc5100", "#7b3294"]
    for y_idx, (cls_name, col) in enumerate(zip(CLASSES, palette)):
        vals = [next(r[f"acc_{y_idx}_beach_mean"] for r in agg_rows if r["arm"] == a) for a in arms]
        ax.plot(x, vals, marker="o", lw=1.9, color=col, label=f"{cls_name} + beach")
    ax.set_xticks(x, [pretty[a] for a in arms], fontsize=8)
    ax.set_ylabel("Minority Group Accuracy (%)")
    ax.set_title("B. Per-Breed Minority (`beach`) Breakdown Across Controls")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Direction 2 · Phase C · Step 7b Spawrious (o2o_hard) Validation Control Ladder", fontsize=12.5)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapt-epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-seed", type=int, default=None)
    parser.add_argument("--max-gpus", type=int, default=2)
    args = parser.parse_args(argv)

    work_dir = args.work_dir.resolve()
    ckpt_dir = args.ckpt_dir.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_root = ensure_spawrious_o2o_hard(work_dir)
    if args.worker_seed is not None:
        dev = _resolve_device(args.device)
        run_single_seed(
            data_root, ckpt_dir, out_dir,
            seed=args.worker_seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            adapt_epochs=args.adapt_epochs, device=dev, workers=args.workers,
        )
        return

    base_dev = _resolve_device(args.device)
    n_gpus = min(args.max_gpus, torch.cuda.device_count()) if base_dev.startswith("cuda") else 0

    if n_gpus >= 2 and len(args.seeds) >= 2:
        print(f"=== Dispatching {len(args.seeds)} seeds in parallel across {n_gpus} GPUs ===", flush=True)
        gpu_q: queue.Queue[str] = queue.Queue()
        for gid in range(n_gpus):
            gpu_q.put(f"cuda:{gid}")

        def _launch(s: int):
            dev_id = gpu_q.get()
            try:
                cmd = [
                    sys.executable, "-B", "-m", "scripts.analysis.rep_repair_step7b_spawrious",
                    "--work-dir", str(work_dir),
                    "--ckpt-dir", str(ckpt_dir),
                    "--output", str(out_dir),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--adapt-epochs", str(args.adapt_epochs),
                    "--device", dev_id,
                    "--workers", str(args.workers),
                    "--worker-seed", str(s),
                ]
                print(f"Launching seed {s} on {dev_id}...", flush=True)
                subprocess.run(cmd, check=True)
            finally:
                gpu_q.put(dev_id)

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            list(pool.map(_launch, args.seeds))
    else:
        for s in args.seeds:
            run_single_seed(
                data_root, ckpt_dir, out_dir,
                seed=s, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                adapt_epochs=args.adapt_epochs, device=base_dev, workers=args.workers,
            )

    # Aggregate results across seeds.
    per_seed_rows, ckpt_records = [], []
    oof_store: dict[str, np.ndarray] = {}
    n_train = n_val = 0
    for s in args.seeds:
        payload = json.loads((out_dir / f"seed_{s}_results.json").read_text(encoding="utf-8"))
        n_train, n_val = payload["n_train"], payload["n_val"]
        ckpt_records.append(payload["checkpoint"])
        per_seed_rows.extend(payload["rows"])
        with np.load(out_dir / f"seed_{s}_oof.npz", allow_pickle=False) as zf:
            for k in zf.files:
                oof_store[k] = zf[k]

    agg_rows = []
    for arm in ARM_ORDER:
        sub = [r for r in per_seed_rows if r["arm"] == arm]
        row = {
            "arm": arm,
            "n_seeds": len(sub),
            "trainable_params": sub[0]["trainable_params"],
            "wga_mean": round(float(np.mean([r["wga"] for r in sub])), 2),
            "wga_sd": round(float(np.std([r["wga"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "minority_acc_mean": round(float(np.mean([r["minority_acc"] for r in sub])), 2),
            "majority_acc_mean": round(float(np.mean([r["majority_acc"] for r in sub])), 2),
            "overall_acc_mean": round(float(np.mean([r["overall_acc"] for r in sub])), 2),
            "min_macro_auc_mean": round(float(np.mean([r["min_macro_auc"] for r in sub])), 2),
            "all_macro_auc_mean": round(float(np.mean([r["all_macro_auc"] for r in sub])), 2),
        }
        for y_idx in range(4):
            row[f"acc_{y_idx}_beach_mean"] = round(float(np.mean([r[f"acc_{y_idx}_beach"] for r in sub])), 2)
        agg_rows.append(row)

    write_csv(out_dir / "step7b_spawrious_per_seed.csv", per_seed_rows)
    write_csv(out_dir / "step7b_spawrious_aggregate.csv", agg_rows)
    np.savez_compressed(out_dir / "step7b_oof_predictions.npz", **oof_store)
    plot_step7b(agg_rows, out_dir / "step7b_spawrious_controls.png")

    lines = [
        "# Direction 2 · Phase C · Step 7b Spawrious (`o2o_hard`) ResNet-50 Val Control Ladder",
        "",
        f"Training split (`85%`): **{n_train:,}** images | Held-out 8-group validation (`15%`): **{n_val:,}** images | Shifted OOD test accessed: **0**.",
        "",
        "| Arm | Params | 8-Grp WGA (%) | Minority (`beach`) (%) | Majority (`spur`) (%) | Overall Acc (%) | Minority Macro-AUC (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        lines.append(
            f"| `{r['arm']}` | {r['trainable_params']:,} | **{r['wga_mean']:.2f} ± {r['wga_sd']:.2f}** | "
            f"{r['minority_acc_mean']:.2f} | {r['majority_acc_mean']:.2f} | {r['overall_acc_mean']:.2f} | "
            f"**{r['min_macro_auc_mean']:.2f}** |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_c_step7b_spawrious_o2o_hard_val_controls",
        "split": "train_domains_01_stratified_85_15",
        "test_files_accessed": 0,
        "gpus_used": n_gpus,
        "seeds": list(args.seeds),
        "checkpoints": ckpt_records,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Step 7b Spawrious controls complete. Summary saved to:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
