"""Direction 2 · Phase C · Step 7c-1: One-Shot Waterbirds Held-Out Test Confirmation (ResNet-50 + ViT-B/16).

Unseals Waterbirds split == 2 ONCE with ZERO test-split tuning:
  - Every model checkpoint, neutral baseline buffer, and scalar threshold t*_val is fitted
    strictly on validation (split == 1, cleaned "relabel" labels).
  - Evaluates ResNet-50 (5 standard_erm + 2 erm_star reference seeds) and ViT-B/16 (3 seeds
    from noerula/rep-repair-step7a-vit-kaggle) across all frozen controls + pre_l4_repair_w005
    (and its cross-architecture ViT transfer pre_b6_repair_w005).
  - Supports dual-GPU (2x T4) parallel worker dispatch across seeds.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.rep_repair_phase_c_step6 import write_csv
from src.methods.rep_repair.controls import DARHead, ValImageDataset
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
    ensure_waterbirds_images,
    eval_image_transform,
    recover_backbones,
    recover_kaggle_kernel_files,
)
from src.methods.rep_repair.repair import (
    PreLayer4RepairModel,
    compute_group_balanced_layer3_mean,
    context_invariance_loss,
)
from src.models.resnet import load_resnet_checkpoint
from src.models.vit import (
    PreBlockViTRepairModel,
    compute_group_balanced_vit_block_mean,
    load_vit_checkpoint,
)

RESNET_STD_ARMS = ("erm_raw", "dfr_l2", "dar", "pre_l4_gate_only", "affine_dfr_linear", "pre_l4_repair_w005")
RESNET_STAR_ARMS = ("erm_star_raw", "erm_star_dfr_l2")
VIT_ARMS = ("vit_erm_raw", "vit_dfr_l2", "vit_ln_dfr_late", "vit_pre_b6_gate_only", "vit_pre_b6_repair_w005")


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


def load_waterbirds_test_metadata(metadata_path: Path) -> list[dict]:
    """Load official Waterbirds held-out test rows (split == 2) for Step 7c one-shot confirmation."""
    rows: list[dict] = []
    with Path(metadata_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["split"]) != 2:
                continue
            y = int(row["y"])
            a = int(row["place"])
            rows.append({
                "img_id": int(row["img_id"]),
                "id": row["img_filename"],
                "y": y,
                "a": a,
                "group": f"{y}{a}",
                "split": 2,
            })
    if not rows:
        raise ValueError(f"No split == 2 rows found in {metadata_path}")
    return rows


class WaterbirdsTestDataset(Dataset):
    """Held-out test split (split == 2) dataset strictly for Step 7c inference."""

    def __init__(self, image_dir: Path, rows: list[dict]):
        if any(int(r.get("split", -1)) != 2 for r in rows):
            raise ValueError("WaterbirdsTestDataset accepts split == 2 rows only")
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.transform = eval_image_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        with Image.open(self.image_dir / r["id"]) as im:
            x = self.transform(im.convert("RGB"))
        return idx, x, int(r["y"]), str(r["group"])


def calibrate_val_threshold(val_p1: np.ndarray, val_y: np.ndarray, val_g: np.ndarray) -> float:
    """Select single scalar decision threshold strictly on validation OOF probabilities."""
    best_t, best_k = 0.5, (-1.0, -1.0)
    for t in np.linspace(0.05, 0.95, 181):
        m = group_metrics(val_y, (val_p1 >= t).astype(int), val_g)
        k = (m["wga"], m["overall"])
        if k > best_k:
            best_k, best_t = k, float(t)
    return best_t


def eval_val_oof_summary(val_p1: np.ndarray, val_y: np.ndarray, val_g: np.ndarray, folds: np.ndarray) -> dict:
    preds = np.zeros(len(val_y), dtype=int)
    for f in range(int(folds.max()) + 1):
        tr, te = folds != f, folds == f
        t_f = calibrate_val_threshold(val_p1[tr], val_y[tr], val_g[tr])
        preds[te] = (val_p1[te] >= t_f).astype(int)
    m = group_metrics(val_y, preds, val_g)
    min_mask = np.isin(val_g, ("01", "10"))
    return {
        "val_oof_wga": round(m["wga"], 4),
        "val_oof_acc": round(m["overall"], 4),
        "val_min_auc": round(float(roc_auc_score(val_y[min_mask], val_p1[min_mask])) * 100.0, 4),
    }


def eval_test_probs(
    test_p1: np.ndarray,
    y_off: np.ndarray,
    g_off: np.ndarray,
    y_rel: np.ndarray,
    g_rel: np.ndarray,
    val_thresh: float,
) -> dict:
    """Evaluate official and tern-cleaned Waterbirds held-out test metrics using val-calibrated threshold."""
    m_off_raw = group_metrics(y_off, (test_p1 >= 0.5).astype(int), g_off)
    m_off_thr = group_metrics(y_off, (test_p1 >= val_thresh).astype(int), g_off)
    min_off = np.isin(g_off, ("01", "10"))

    m_rel_raw = group_metrics(y_rel, (test_p1 >= 0.5).astype(int), g_rel)
    m_rel_thr = group_metrics(y_rel, (test_p1 >= val_thresh).astype(int), g_rel)
    min_rel = np.isin(g_rel, ("01", "10"))

    return {
        "val_thresh": round(val_thresh, 4),
        "off_raw_wga": round(m_off_raw["wga"], 4),
        "off_val_thr_wga": round(m_off_thr["wga"], 4),
        "off_val_thr_acc": round(m_off_thr["overall"], 4),
        "off_min_auc": round(float(roc_auc_score(y_off[min_off], test_p1[min_off])) * 100.0, 4),
        "off_all_auc": round(float(roc_auc_score(y_off, test_p1)) * 100.0, 4),
        "rel_raw_wga": round(m_rel_raw["wga"], 4),
        "rel_val_thr_wga": round(m_rel_thr["wga"], 4),
        "rel_val_thr_acc": round(m_rel_thr["overall"], 4),
        "rel_g00": round(m_rel_thr["groups"]["00"], 4),
        "rel_g01": round(m_rel_thr["groups"]["01"], 4),
        "rel_g10": round(m_rel_thr["groups"]["10"], 4),
        "rel_g11": round(m_rel_thr["groups"]["11"], 4),
        "rel_min_auc": round(float(roc_auc_score(y_rel[min_rel], test_p1[min_rel])) * 100.0, 4),
        "rel_all_auc": round(float(roc_auc_score(y_rel, test_p1)) * 100.0, 4),
    }


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
def _extract_resnet_cache(model: nn.Module, loader: DataLoader, device: str, *, include_l4: bool) -> dict[str, np.ndarray]:
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
    out = {"probs": np.concatenate(probs, axis=0), "pooled": np.concatenate(pooled, axis=0)}
    if include_l4:
        out["layer4_map"] = np.concatenate(l4_maps, axis=0)
    return out


@torch.inference_mode()
def _extract_vit_cache(model: nn.Module, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    use_amp = device.startswith("cuda")
    probs, pooled = [], []
    for _, xb, _, _ in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            toks = model.extract_block_tokens(xb.to(device))
        probs.append(toks["logits"].float().softmax(dim=1).cpu().numpy().astype(np.float32))
        pooled.append(toks["pooled"].float().cpu().numpy().astype(np.float32))
    return {"probs": np.concatenate(probs, axis=0), "pooled": np.concatenate(pooled, axis=0)}


def _fit_dfr_val_and_test(
    X_val: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    X_test: np.ndarray,
    outer_folds: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    c_grid = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    val_oof = np.zeros((len(y_val), 2), dtype=np.float64)
    test_sum = np.zeros((len(X_test), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        X_tr, y_tr, g_tr = X_val[tr], y_val[tr], g_val[tr]
        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=4, seed=seed + 100 + fold)
        best_c, best_key = c_grid[0], (-1.0, -1.0)
        for c_val in c_grid:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(4):
                itr, iva = in_folds != infold, in_folds == infold
                sc = StandardScaler().fit(X_tr[itr])
                clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
                clf.fit(sc.transform(X_tr[itr]), y_tr[itr], sample_weight=balanced_group_weights(g_tr[itr]))
                in_preds[iva] = clf.predict(sc.transform(X_tr[iva]))
            m = group_metrics(y_tr, in_preds, g_tr)
            k = (m["wga"], m["overall"])
            if k > best_key:
                best_key, best_c = k, c_val
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X_tr), y_tr, sample_weight=balanced_group_weights(g_tr))
        val_oof[te] = clf.predict_proba(sc.transform(X_val[te]))
        test_sum += clf.predict_proba(sc.transform(X_test))
    return val_oof, test_sum / float(n_outer)


def _fit_dar_val_and_test(
    base_dev: nn.Module,
    F_val: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    test_loader: DataLoader,
    outer_folds: np.ndarray,
    *,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train 5-fold DARHeads on val_layer4_map, then stream test_loader in 1 pass (zero test l4 RAM cache)."""
    cands = [
        {"attn_dim": d, "weight_decay": wd, "dropout": 0.2, "lr": 1e-3, "epochs": 120}
        for d in (64, 128)
        for wd in (1e-2, 1e-1)
    ]
    val_oof = np.zeros((len(y_val), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    ce = nn.CrossEntropyLoss(reduction="none")
    n_params = sum(p.numel() for p in DARHead(F_val.shape[1], attn_dim=128, dropout=0.2, num_classes=2).parameters())
    trained_heads: list[tuple[nn.Module, torch.Tensor, torch.Tensor]] = []

    def _run_one(F_tr, y_tr, g_tr, F_va, cfg, s_list, keep_heads: bool = False):
        mean = F_tr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(F_tr.std(axis=(0, 2, 3), keepdims=True), 1e-6).astype(np.float32)
        ft = torch.from_numpy((F_tr - mean) / std).to(device)
        yt = torch.from_numpy(y_tr.astype(np.int64)).to(device)
        wt = torch.from_numpy(balanced_group_weights(g_tr).astype(np.float32)).to(device)
        mean_t = torch.from_numpy(mean).to(device)
        std_t = torch.from_numpy(std).to(device)
        out_va = np.zeros((len(F_va), 2), dtype=np.float64)
        for s in s_list:
            torch.manual_seed(s)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(s)
            head = DARHead(F_tr.shape[1], attn_dim=cfg["attn_dim"], dropout=cfg["dropout"], num_classes=2).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            for _ in range(cfg["epochs"]):
                head.train()
                opt.zero_grad(set_to_none=True)
                logits, _ = head(ft)
                loss = (ce(logits, yt) * wt).mean()
                loss.backward()
                opt.step()
            head.eval()
            with torch.inference_mode():
                chunks = []
                for st in range(0, len(F_va), 256):
                    x_b = torch.from_numpy((F_va[st : st + 256] - mean) / std).to(device)
                    chunks.append(head(x_b)[0].softmax(dim=1).cpu().numpy())
                out_va += np.concatenate(chunks, axis=0)
            if keep_heads:
                trained_heads.append((head, mean_t, std_t))
        return out_va / float(len(s_list))

    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        F_tr, y_tr, g_tr = F_val[tr], y_val[tr], g_val[tr]
        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=4, seed=seed + 200 + fold)
        best_cfg, best_key = cands[0], (-1.0, -1.0)
        for cfg in cands:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(4):
                itr, iva = in_folds != infold, in_folds == infold
                p_va = _run_one(F_tr[itr], y_tr[itr], g_tr[itr], F_tr[iva], cfg, (seed + fold,), keep_heads=False)
                in_preds[iva] = p_va.argmax(axis=-1)
            m = group_metrics(y_tr, in_preds, g_tr)
            k = (m["wga"], m["overall"])
            if k > best_key:
                best_key, best_cfg = k, cfg
        val_oof[te] = _run_one(
            F_tr, y_tr, g_tr, F_val[te], best_cfg,
            (seed + fold, seed + 100 + fold, seed + 200 + fold),
            keep_heads=True,
        )

    use_amp = device.startswith("cuda")
    test_chunks = []
    base_dev.eval()
    with torch.inference_mode():
        for _, xb, _, _ in test_loader:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                l4 = base_dev.extract_stage_maps(xb.to(device))["layer4_map"].float()
            p_sum = torch.zeros((l4.shape[0], 2), device=device, dtype=torch.float32)
            for head, mean_t, std_t in trained_heads:
                p_sum += head((l4 - mean_t) / std_t)[0].softmax(dim=1)
            test_chunks.append((p_sum / float(len(trained_heads))).cpu().numpy())
    return val_oof, np.concatenate(test_chunks, axis=0), n_params


def _train_5fold_repair_val_and_test(
    make_model_fn,
    set_neutral_fn,
    image_dir: Path,
    val_rows: list[dict],
    test_loader: DataLoader,
    outer_folds: np.ndarray,
    *,
    use_inv: bool,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    workers: int,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train 5-fold validation repair strictly on val folds and return (val_oof, 5-fold mean test probs, n_params)."""
    val_oof = np.zeros((len(val_rows), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    test_sum = np.zeros((len(test_loader.dataset), 2), dtype=np.float64)
    ce = nn.CrossEntropyLoss()
    n_trainable = 0

    for fold in range(n_outer):
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
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed + fold)

        model = make_model_fn().to(device)
        if set_neutral_fn is not None:
            fit_eval_loader = DataLoader(ValImageDataset(image_dir, fit_sub), batch_size=128, shuffle=False, num_workers=workers)
            set_neutral_fn(model, fit_eval_loader)

        params = model.trainable_parameters()
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.startswith("cuda")))

        w_fit = balanced_group_weights([str(r["group"]) for r in fit_sub])
        gen = torch.Generator().manual_seed(seed + 50 + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_fit).double(), num_samples=len(fit_sub), replacement=True, generator=gen)
        fit_loader = DataLoader(ValImageDataset(image_dir, fit_sub), batch_size=64, sampler=sampler, num_workers=workers)
        sel_loader = DataLoader(ValImageDataset(image_dir, sel_sub), batch_size=128, shuffle=False, num_workers=workers)
        te_loader = DataLoader(ValImageDataset(image_dir, te_rows), batch_size=128, shuffle=False, num_workers=workers)

        best_key = (-1.0, -1.0)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for _ in range(epochs):
            model.set_repair_train_mode()
            for _, xb, yb, gb in fit_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_amp and device.startswith("cuda"))):
                    if use_inv:
                        ab = torch.tensor([int(str(s)[1]) for s in gb], dtype=torch.long, device=device)
                        out = model.forward_with_context_swap(xb, ab)
                        spatial_map = out["layer4_map"] if "layer4_map" in out else out["patch_map"]
                        spatial_swap = out["layer4_map_swap"] if "layer4_map_swap" in out else out["patch_map_swap"]
                        loss_ce = ce(out["logits"].float(), yb)
                        inv = context_invariance_loss(
                            out["logits"].float(), out["logits_swap"].float(),
                            spatial_map.float(), spatial_swap.float(), out["gate_map"].float(),
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

            sel_p1 = _predict_loader(model, sel_loader, device)[:, 1]
            sel_min = np.isin(sel_g, ("01", "10"))
            sel_auc = float(roc_auc_score(sel_y[sel_min], sel_p1[sel_min])) * 100.0
            sel_m = group_metrics(sel_y, (sel_p1 >= 0.5).astype(int), sel_g)
            key = (round(sel_auc, 3), round(sel_m["wga"], 3))
            if key >= best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        val_oof[te_idx] = _predict_loader(model, te_loader, device)
        test_sum += _predict_loader(model, test_loader, device)
    return val_oof, test_sum / float(n_outer), n_trainable


def run_worker_job(
    work_dir: Path,
    out_dir: Path,
    *,
    kind: str,
    seed: int,
    device: str,
    workers: int,
) -> dict:
    job_json = out_dir / f"job_{kind}_s{seed}.json"
    job_npz = out_dir / f"probs_{kind}_s{seed}.npz"
    if job_json.is_file() and job_npz.is_file():
        print(f"    [{kind} s{seed}] Reusing completed job output", flush=True)
        return json.loads(job_json.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    image_dir = ensure_waterbirds_images(work_dir / "waterbirds")
    meta_path = image_dir / "metadata.csv"

    val_raw = load_val_metadata(meta_path)
    _, val_rel = apply_label_condition(val_raw, "relabel", species=TERN_SPECIES)
    y_val = np.array([int(r["y"]) for r in val_rel], dtype=int)
    g_val = np.array([str(r["group"]) for r in val_rel], dtype=str)
    outer_folds = stratified_group_folds(val_rel, n_splits=5, seed=0)

    test_off = load_waterbirds_test_metadata(meta_path)
    _, test_rel = apply_label_condition(test_off, "relabel", species=TERN_SPECIES)
    y_off = np.array([int(r["y"]) for r in test_off], dtype=int)
    g_off = np.array([str(r["group"]) for r in test_off], dtype=str)
    y_rel = np.array([int(r["y"]) for r in test_rel], dtype=int)
    g_rel = np.array([str(r["group"]) for r in test_rel], dtype=str)

    pin = device.startswith("cuda")
    val_loader = DataLoader(ValImageDataset(image_dir, val_rel), batch_size=128, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader = DataLoader(WaterbirdsTestDataset(image_dir, test_off), batch_size=256, shuffle=False, num_workers=workers, pin_memory=pin)

    arm_probs: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    ckpt_sha = ""
    cohort_label = kind

    if kind in ("resnet_std", "resnet_star"):
        cohort = "standard_erm" if kind == "resnet_std" else "erm_star"
        cohort_label = cohort
        ckpt_path = work_dir / "backbones" / f"{cohort}_s{seed}_ckpt_last.pt"
        if not ckpt_path.is_file():
            recs = recover_backbones(work_dir / "backbones", cohorts=(cohort,), seeds={cohort: [seed]}, workers=1)
            ckpt_path = Path(recs[0]["path"])
        ckpt_sha = file_sha256(ckpt_path)
        base_cpu = load_resnet_checkpoint(ckpt_path, device="cpu")
        base_dev = copy.deepcopy(base_cpu).to(device)

        v_cache = _extract_resnet_cache(base_dev, val_loader, device, include_l4=(kind == "resnet_std"))
        t_cache = _extract_resnet_cache(base_dev, test_loader, device, include_l4=False)
        n_full = sum(p.numel() for p in base_cpu.parameters())

        val_dfr, test_dfr = _fit_dfr_val_and_test(
            v_cache["pooled"], y_val, g_val, t_cache["pooled"], outer_folds, seed=seed
        )
        erm_val_p, erm_test_p = v_cache["probs"], t_cache["probs"]
        if kind == "resnet_star":
            del v_cache, t_cache, base_dev
            arm_probs["erm_star_raw"] = (erm_val_p, erm_test_p, n_full)
            arm_probs["erm_star_dfr_l2"] = (val_dfr, test_dfr, 2048 * 2 + 2)
        else:
            arm_probs["erm_raw"] = (erm_val_p, erm_test_p, n_full)
            arm_probs["dfr_l2"] = (val_dfr, test_dfr, 2048 * 2 + 2)

            val_dar, test_dar, n_dar = _fit_dar_val_and_test(
                base_dev, v_cache["layer4_map"], y_val, g_val, test_loader, outer_folds, seed=seed, device=device
            )
            arm_probs["dar"] = (val_dar, test_dar, n_dar)
            del v_cache, t_cache, base_dev
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            def _set_resnet_mu(mod: PreLayer4RepairModel, fit_l: DataLoader):
                mod.set_neutral_c3(compute_group_balanced_layer3_mean(base_cpu, fit_l, device))

            for arm_name, use_gate, bn_scope, use_inv, set_mu in (
                ("pre_l4_gate_only", True, "none", False, _set_resnet_mu),
                ("affine_dfr_linear", False, "bn_all", False, None),
                ("pre_l4_repair_w005", True, "bn_all", True, _set_resnet_mu),
            ):
                v_p, t_p, n_p = _train_5fold_repair_val_and_test(
                    lambda ug=use_gate, bs=bn_scope: PreLayer4RepairModel(
                        base_cpu, use_gate=ug, gate_mode="peak_neutral", head_type="linear", bn_scope=bs
                    ),
                    set_mu, image_dir, val_rel, test_loader, outer_folds,
                    use_inv=use_inv, epochs=20, lr=3e-3, seed=seed, device=device, workers=workers, use_amp=True,
                )
                arm_probs[arm_name] = (v_p, t_p, n_p)

    elif kind == "vit":
        member = f"vit_b16_backbones/vit_b16_s{seed}_ckpt_last.pt"
        resolved = recover_kaggle_kernel_files(
            "rep-repair-step7a-vit-kaggle", [member], work_dir / "vit_backbones", workers=1
        )
        ckpt_path = resolved[member]
        ckpt_sha = file_sha256(ckpt_path)
        base_cpu = load_vit_checkpoint(ckpt_path, device="cpu")
        base_dev = copy.deepcopy(base_cpu).to(device)

        v_cache = _extract_vit_cache(base_dev, val_loader, device)
        t_cache = _extract_vit_cache(base_dev, test_loader, device)
        n_full = sum(p.numel() for p in base_cpu.parameters())
        val_dfr, test_dfr = _fit_dfr_val_and_test(
            v_cache["pooled"], y_val, g_val, t_cache["pooled"], outer_folds, seed=seed
        )
        arm_probs["vit_erm_raw"] = (v_cache["probs"], t_cache["probs"], n_full)
        arm_probs["vit_dfr_l2"] = (val_dfr, test_dfr, 768 * 2 + 2)
        del v_cache, t_cache, base_dev
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        def _set_vit_mu(mod: PreBlockViTRepairModel, fit_l: DataLoader):
            mod.set_neutral_tokens(compute_group_balanced_vit_block_mean(base_cpu, fit_l, device, gate_before_block=6))

        for arm_name, use_gate, ln_scope, use_inv, set_mu in (
            ("vit_ln_dfr_late", False, "ln_late", False, None),
            ("vit_pre_b6_gate_only", True, "none", False, _set_vit_mu),
            ("vit_pre_b6_repair_w005", True, "ln_late", True, _set_vit_mu),
        ):
            v_p, t_p, n_p = _train_5fold_repair_val_and_test(
                lambda ug=use_gate, ls=ln_scope: PreBlockViTRepairModel(
                    base_cpu, gate_before_block=6, use_gate=ug, ln_scope=ls
                ),
                set_mu, image_dir, val_rel, test_loader, outer_folds,
                use_inv=use_inv, epochs=20, lr=3e-3, seed=seed, device=device, workers=workers, use_amp=True,
            )
            arm_probs[arm_name] = (v_p, t_p, n_p)
    else:
        raise ValueError(f"Unknown worker kind: {kind}")

    rows = []
    prob_store = {}
    for arm, (val_p, test_p, n_params) in arm_probs.items():
        val_m = eval_val_oof_summary(val_p[:, 1], y_val, g_val, outer_folds)
        val_t = calibrate_val_threshold(val_p[:, 1], y_val, g_val)
        test_m = eval_test_probs(test_p[:, 1], y_off, g_off, y_rel, g_rel, val_t)
        rows.append({"cohort": cohort_label, "seed": seed, "arm": arm, "trainable_params": n_params, **val_m, **test_m})
        prob_store[f"{kind}_s{seed}__{arm}__val"] = val_p.astype(np.float32)
        prob_store[f"{kind}_s{seed}__{arm}__test"] = test_p.astype(np.float32)
        print(
            f"    [{kind} s{seed} | {device}] {arm:23s} | Val WGA={val_m['val_oof_wga']:.2f}% -> "
            f"Test WGA(cln)={test_m['rel_val_thr_wga']:.2f}% (Off={test_m['off_val_thr_wga']:.2f}%) | "
            f"Min-AUC={test_m['rel_min_auc']:.2f}%",
            flush=True,
        )

    payload = {
        "kind": kind, "cohort": cohort_label, "seed": seed, "device": device,
        "checkpoint": str(ckpt_path), "sha256": ckpt_sha,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    (out_dir / f"job_{kind}_s{seed}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out_dir / f"probs_{kind}_s{seed}.npz", **prob_store)
    return payload


def _aggregate_group(rows: list[dict], arm_order: tuple[str, ...], ref_arm: str | None) -> list[dict]:
    ref_by_seed = {r["seed"]: r for r in rows if r["arm"] == ref_arm} if ref_arm else {}
    agg = []
    for arm in arm_order:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["seed"])
        if not sub:
            continue
        item = {
            "cohort": sub[0]["cohort"],
            "arm": arm,
            "n_seeds": len(sub),
            "trainable_params": sub[0]["trainable_params"],
            "val_oof_wga_mean": round(float(np.mean([r["val_oof_wga"] for r in sub])), 2),
            "off_val_thr_wga_mean": round(float(np.mean([r["off_val_thr_wga"] for r in sub])), 2),
            "off_val_thr_wga_sd": round(float(np.std([r["off_val_thr_wga"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "off_val_thr_acc_mean": round(float(np.mean([r["off_val_thr_acc"] for r in sub])), 2),
            "rel_raw_wga_mean": round(float(np.mean([r["rel_raw_wga"] for r in sub])), 2),
            "rel_val_thr_wga_mean": round(float(np.mean([r["rel_val_thr_wga"] for r in sub])), 2),
            "rel_val_thr_wga_sd": round(float(np.std([r["rel_val_thr_wga"] for r in sub], ddof=1)), 2) if len(sub) > 1 else 0.0,
            "rel_val_thr_acc_mean": round(float(np.mean([r["rel_val_thr_acc"] for r in sub])), 2),
            "rel_g01_mean": round(float(np.mean([r["rel_g01"] for r in sub])), 2),
            "rel_g10_mean": round(float(np.mean([r["rel_g10"] for r in sub])), 2),
            "rel_min_auc_mean": round(float(np.mean([r["rel_min_auc"] for r in sub])), 2),
            "rel_all_auc_mean": round(float(np.mean([r["rel_all_auc"] for r in sub])), 2),
        }
        if ref_by_seed and sub[0]["cohort"] != "erm_star":
            dw = [r["rel_val_thr_wga"] - ref_by_seed[r["seed"]]["rel_val_thr_wga"] for r in sub]
            da = [r["rel_min_auc"] - ref_by_seed[r["seed"]]["rel_min_auc"] for r in sub]
            item["delta_rel_wga"] = round(float(np.mean(dw)), 2)
            item["wga_seed_wins"] = f"{sum(v > 0 for v in dw)}/{len(sub)}"
            item["delta_rel_auc"] = round(float(np.mean(da)), 2)
            item["auc_seed_wins"] = f"{sum(v > 0 for v in da)}/{len(sub)}"
        else:
            item["delta_rel_wga"] = ""
            item["wga_seed_wins"] = ""
            item["delta_rel_auc"] = ""
            item["auc_seed_wins"] = ""
        agg.append(item)
    return agg


def plot_step7c_waterbirds(res_agg: list[dict], vit_agg: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), layout="constrained")

    # Panel A: ResNet-50 Held-Out Test WGA (Official & Cleaned) + Min-AUC
    ax = axes[0]
    r_arms = [r["arm"] for r in res_agg]
    x = np.arange(len(r_arms))
    w = 0.27
    off_w = [r["off_val_thr_wga_mean"] for r in res_agg]
    rel_w = [r["rel_val_thr_wga_mean"] for r in res_agg]
    min_a = [r["rel_min_auc_mean"] for r in res_agg]
    ax.bar(x - w, off_w, w, color="#6c757d", label="Official Test WGA (%)")
    ax.bar(x, rel_w, w, color="#245da6", label="Cleaned Test WGA (%)")
    ax.bar(x + w, min_a, w, color="#4a7c59", label="Cleaned Min-Grp AUROC (%)")
    for i, rv in enumerate(rel_w):
        ax.text(i, rv + 0.35, f"{rv:.2f}", ha="center", va="bottom", fontsize=7.2)
    pretty_r = {
        "erm_raw": "ERM", "dfr_l2": "DFR-L2", "dar": "DAR",
        "pre_l4_gate_only": "Pre-L4\nGate", "affine_dfr_linear": "Affine\nDFR",
        "pre_l4_repair_w005": "Pre-L4\nRepair", "erm_star_raw": "ERM*\nRaw", "erm_star_dfr_l2": "ERM*\nDFR-L2",
    }
    ax.set_xticks(x, [pretty_r.get(a, a) for a in r_arms], fontsize=7.6)
    ax.set_ylabel("Score (%)")
    ax.set_title("A. Waterbirds Held-Out Test: ResNet-50 (5 Std ERM + 2 ERM* Seeds)")
    ax.set_ylim(min(60.0, min(off_w) - 5.0), 101.0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    # Panel B: ViT-B/16 Cross-Architecture Transfer (Val OOF vs Cleaned Test WGA & Min-AUC)
    ax = axes[1]
    v_arms = [r["arm"] for r in vit_agg]
    vx = np.arange(len(v_arms))
    val_w = [r["val_oof_wga_mean"] for r in vit_agg]
    t_w = [r["rel_val_thr_wga_mean"] for r in vit_agg]
    t_a = [r["rel_min_auc_mean"] for r in vit_agg]
    ax.bar(vx - w, val_w, w, color="#5b9bd5", label="Val OOF WGA (%)")
    ax.bar(vx, t_w, w, color="#bc5100", label="Cleaned Test WGA (%)")
    ax.bar(vx + w, t_a, w, color="#2e7d32", label="Cleaned Test Min-AUC (%)")
    for i, tv in enumerate(t_w):
        ax.text(i, tv + 0.4, f"{tv:.2f}", ha="center", va="bottom", fontsize=7.5)
    pretty_v = {
        "vit_erm_raw": "ViT\nERM", "vit_dfr_l2": "ViT\nDFR-L2",
        "vit_ln_dfr_late": "ViT\nLN-Late", "vit_pre_b6_gate_only": "Pre-B6\nGate",
        "vit_pre_b6_repair_w005": "Pre-B6\nRepair",
    }
    ax.set_xticks(vx, [pretty_v.get(a, a) for a in v_arms], fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("B. Cross-Architecture Transfer: ViT-B/16 (3 Seeds, Pre-Block6 Repair)")
    ax.set_ylim(min(55.0, min(t_w) - 5.0), 101.0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Direction 2 · Phase C · Step 7c-1 Waterbirds One-Shot Held-Out Test Confirmation", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--std-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--erm-star-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--vit-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-kind", choices=("resnet_std", "resnet_star", "vit"), default=None)
    parser.add_argument("--worker-seed", type=int, default=None)
    args = parser.parse_args(argv)

    work_dir = args.work_dir.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_kind is not None:
        if args.worker_seed is None:
            raise ValueError("--worker-seed required with --worker-kind")
        dev = _resolve_device(args.device)
        run_worker_job(work_dir, out_dir, kind=args.worker_kind, seed=args.worker_seed, device=dev, workers=args.workers)
        return

    # Parent orchestrator prepares all assets once so parallel GPU workers never race.
    ensure_waterbirds_images(work_dir / "waterbirds")
    recover_backbones(work_dir / "backbones", cohorts=("standard_erm",), seeds={"standard_erm": list(args.std_seeds)}, workers=4)
    recover_backbones(work_dir / "backbones", cohorts=("erm_star",), seeds={"erm_star": list(args.erm_star_seeds)}, workers=2)
    recover_kaggle_kernel_files(
        "rep-repair-step7a-vit-kaggle",
        [f"vit_b16_backbones/vit_b16_s{s}_ckpt_last.pt" for s in args.vit_seeds],
        work_dir / "vit_backbones",
        workers=3,
    )

    jobs = (
        [("resnet_std", s) for s in args.std_seeds]
        + [("resnet_star", s) for s in args.erm_star_seeds]
        + [("vit", s) for s in args.vit_seeds]
    )
    base_dev = _resolve_device(args.device)
    n_gpus = min(args.max_gpus, torch.cuda.device_count()) if base_dev.startswith("cuda") else 0

    if n_gpus >= 2 and len(jobs) >= 2:
        print(f"=== Dispatching {len(jobs)} Step 7c-1 jobs across {n_gpus} GPUs ===", flush=True)
        gpu_q: queue.Queue[str] = queue.Queue()
        for gid in range(n_gpus):
            gpu_q.put(f"cuda:{gid}")
        abort_event = threading.Event()
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        def _launch(job: tuple[str, int]):
            k_name, s_val = job
            if abort_event.is_set():
                return
            dev_id = gpu_q.get()
            try:
                if abort_event.is_set():
                    return
                cmd = [
                    sys.executable, "-B", "-m", "scripts.analysis.rep_repair_step7c_waterbirds_test",
                    "--work-dir", str(work_dir), "--output", str(out_dir),
                    "--device", dev_id, "--workers", str(args.workers),
                    "--worker-kind", k_name, "--worker-seed", str(s_val),
                ]
                print(f"Launching ({k_name}, seed={s_val}) on {dev_id}...", flush=True)
                proc = subprocess.run(cmd, env=child_env)
                if proc.returncode != 0:
                    abort_event.set()
                    raise RuntimeError(f"Worker ({k_name}, s{s_val}) on {dev_id} failed with exit code {proc.returncode}")
                print(f"[DONE] ({k_name}, seed={s_val}) on {dev_id}", flush=True)
            finally:
                gpu_q.put(dev_id)

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futs = [pool.submit(_launch, j) for j in jobs]
            for fut in as_completed(futs):
                fut.result()
    else:
        for k_name, s_val in jobs:
            run_worker_job(work_dir, out_dir, kind=k_name, seed=s_val, device=base_dev, workers=args.workers)

    resnet_rows, vit_rows, ckpt_records = [], [], []
    prob_store: dict[str, np.ndarray] = {}
    for k_name, s_val in jobs:
        payload = json.loads((out_dir / f"job_{k_name}_s{s_val}.json").read_text(encoding="utf-8"))
        ckpt_records.append({"kind": k_name, "seed": s_val, "path": payload["checkpoint"], "sha256": payload["sha256"]})
        if k_name == "vit":
            vit_rows.extend(payload["rows"])
        else:
            resnet_rows.extend(payload["rows"])
        with np.load(out_dir / f"probs_{k_name}_s{s_val}.npz", allow_pickle=False) as zf:
            for k in zf.files:
                prob_store[k] = zf[k]

    res_agg = _aggregate_group(resnet_rows, RESNET_STD_ARMS + RESNET_STAR_ARMS, ref_arm="affine_dfr_linear")
    vit_agg = _aggregate_group(vit_rows, VIT_ARMS, ref_arm="vit_ln_dfr_late")

    write_csv(out_dir / "step7c_resnet_test_per_seed.csv", resnet_rows)
    write_csv(out_dir / "step7c_resnet_test_aggregate.csv", res_agg)
    write_csv(out_dir / "step7c_vit_test_per_seed.csv", vit_rows)
    write_csv(out_dir / "step7c_vit_test_aggregate.csv", vit_agg)
    np.savez_compressed(out_dir / "step7c_waterbirds_test_probs.npz", **prob_store)
    plot_step7c_waterbirds(res_agg, vit_agg, out_dir / "step7c_waterbirds_test.png")

    lines = [
        "# Direction 2 · Phase C · Step 7c-1 Waterbirds One-Shot Held-Out Test Confirmation",
        "",
        "All checkpoints, neutral baselines, and decision thresholds calibrated **strictly on validation** (`split == 1`, `1,199` rows); `split == 2` (`5,794` rows) unsealed once.",
        "",
        "## 1. ResNet-50 Held-Out Test (5 Standard-ERM Seeds + 2 ERM* Reference Seeds)",
        "",
        "| Cohort | Arm | Params | Val OOF WGA | Official Test WGA | Cleaned Test WGA | Δ vs Affine (Wins) | Cleaned Acc | Min-Grp AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res_agg:
        dstr = f"{r['delta_rel_wga']:+.2f} ({r['wga_seed_wins']})" if r["delta_rel_wga"] != "" else "—"
        lines.append(
            f"| `{r['cohort']}` | `{r['arm']}` | {r['trainable_params']:,} | {r['val_oof_wga_mean']:.2f} | "
            f"{r['off_val_thr_wga_mean']:.2f} ± {r['off_val_thr_wga_sd']:.2f} | "
            f"**{r['rel_val_thr_wga_mean']:.2f} ± {r['rel_val_thr_wga_sd']:.2f}** | {dstr} | "
            f"{r['rel_val_thr_acc_mean']:.2f} | **{r['rel_min_auc_mean']:.2f}** |"
        )

    lines.extend([
        "",
        "## 2. ViT-B/16 Cross-Architecture Transfer Held-Out Test (3 Standard-ERM Seeds, Pre-`block_6` Repair)",
        "",
        "| Arm | Params | Val OOF WGA | Official Test WGA | Cleaned Test WGA | Δ vs LN-Late (Wins) | Cleaned Acc | Min-Grp AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in vit_agg:
        dstr = f"{r['delta_rel_wga']:+.2f} ({r['wga_seed_wins']})" if r["delta_rel_wga"] != "" else "—"
        lines.append(
            f"| `{r['arm']}` | {r['trainable_params']:,} | {r['val_oof_wga_mean']:.2f} | "
            f"{r['off_val_thr_wga_mean']:.2f} ± {r['off_val_thr_wga_sd']:.2f} | "
            f"**{r['rel_val_thr_wga_mean']:.2f} ± {r['rel_val_thr_wga_sd']:.2f}** | {dstr} | "
            f"{r['rel_val_thr_acc_mean']:.2f} | **{r['rel_min_auc_mean']:.2f}** |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "step": "phase_c_step7c_waterbirds_one_shot_test_confirmation",
        "one_shot_test_unsealing": True,
        "frozen_repair_hparams": {"gate_mode": "peak_neutral", "bn_scope": "bn_all", "ln_scope": "ln_late", "drift_weight": 0.05, "cons_weight": 0.10, "gate_sparsity": 0.005},
        "gpus_used": n_gpus,
        "checkpoints": ckpt_records,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__)},
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Step 7c-1 Waterbirds held-out test complete. Summary:", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
