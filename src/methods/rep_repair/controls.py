"""Direction 2 · Phase B Elimination Controls (Validation-Only 5-Fold CV).

Implements faithful, matched-budget controls on saved ResNet-50 backbones:
  - Rung 0:  dfr_l2           (frozen pooled 2048-D L2 logistic)
  - Rung 3a: dar              (Deep Attention Reweighting over frozen 7x7x2048 layer4_map)
  - Rung 3b: oracle_mask_pool (ground-truth CUB foreground-only 7x7 pooling + DFR-L2)
  - Rung 4a: h2t_dfr          (all-layer frozen pooled layer1..layer4 linear readout)
  - Rung 4b: bn_stat_only     (BN running-stats recalibration on group-balanced val + DFR-L2)
  - Rung 4c: affine_dfr       (only BatchNorm2d gamma/beta + linear head fine-tuned)
  - Rung 5:  layer4_ft        (selective layer4 + linear head fine-tuning)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.models.resnet import load_resnet_checkpoint
from src.methods.rep_repair.phase_a_data import (
    balanced_group_weights,
    group_metrics,
    stratified_group_folds,
)
from src.methods.rep_repair.phase_b_assets import eval_image_transform


class ValImageDataset(Dataset):
    """Validation-only Waterbirds image dataset (explicitly rejects non-val splits)."""

    def __init__(self, image_dir: Path, rows: list[dict], transform=None):
        if any(int(r.get("split", 1)) != 1 for r in rows):
            raise ValueError("ValImageDataset accepts split == 1 validation rows only")
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.transform = transform if transform is not None else eval_image_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        with Image.open(self.image_dir / r["id"]) as im:
            x = self.transform(im.convert("RGB"))
        return idx, x, int(r["y"]), str(r["group"])


@torch.inference_mode()
def extract_val_stage_cache(
    checkpoint_path: Path,
    image_dir: Path,
    val_rows: list[dict],
    device: str = "cuda",
    batch_size: int = 128,
    workers: int = 2,
) -> dict[str, np.ndarray]:
    """Extract frozen stage pooled embeddings (layer1..4) and pre-GAP layer4_map [N, 2048, 7, 7]."""
    model = load_resnet_checkpoint(checkpoint_path, device=device)
    model.eval()
    loader = DataLoader(
        ValImageDataset(image_dir, val_rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
    )
    l1, l2, l3, l4_map, pooled = [], [], [], [], []
    for _, x, _, _ in loader:
        out = model.extract_stage_maps(x.to(device))
        l1.append(out["layer1"].mean(dim=(2, 3)).cpu().numpy().astype(np.float32))
        l2.append(out["layer2"].mean(dim=(2, 3)).cpu().numpy().astype(np.float32))
        l3.append(out["layer3"].mean(dim=(2, 3)).cpu().numpy().astype(np.float32))
        l4_map.append(out["layer4_map"].cpu().numpy().astype(np.float32))
        pooled.append(out["pooled"].cpu().numpy().astype(np.float32))
    return {
        "layer1_pool": np.concatenate(l1, axis=0),
        "layer2_pool": np.concatenate(l2, axis=0),
        "layer3_pool": np.concatenate(l3, axis=0),
        "layer4_map": np.concatenate(l4_map, axis=0),
        "pooled": np.concatenate(pooled, axis=0),
    }


def pool_with_spatial_masks(layer4_map: np.ndarray, masks_7x7: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Oracle foreground-only pooling over [N, C, 7, 7] using [N, 7, 7] CUB foreground weights."""
    w = np.clip(masks_7x7.astype(np.float32), 0.0, 1.0)
    w_sum = w.sum(axis=(1, 2), keepdims=True)
    # If a bird mask is empty inside the 224 crop, fall back cleanly to uniform GAP.
    empty = (w_sum <= eps)
    w_norm = np.where(empty, 1.0 / (w.shape[1] * w.shape[2]), w / np.maximum(w_sum, eps))
    return np.einsum("nchw,nhw->nc", layer4_map, w_norm).astype(np.float32)


def nested_dfr_l2_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    inner_splits: int = 4,
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    seed: int = 0,
) -> np.ndarray:
    """Zero-leakage nested-CV group-weighted L2 logistic regression OOF probabilities [N, 2]."""
    probs = np.zeros((len(y), 2), dtype=np.float64)
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
    return probs


class DARHead(nn.Module):
    """Deep Attention Reweighting (arXiv:2605.20732): post-hoc spatial attention over frozen 7x7 maps."""

    def __init__(self, channels: int = 2048, attn_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.Sequential(
            nn.Linear(channels, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Linear(channels, 2)

    def forward(self, fmap: torch.Tensor):
        # fmap: [B, C, H, W] -> tokens [B, H*W, C]
        b, c, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        tokens_n = self.norm(tokens)
        attn_logits = self.attn(tokens_n).squeeze(-1)  # [B, H*W]
        weights = torch.softmax(attn_logits, dim=-1)   # [B, H*W]
        pooled = torch.einsum("bs,bsc->bc", weights, tokens_n)
        logits = self.classifier(pooled)
        return logits, weights.view(b, h, w)


def _fit_dar_single(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    F_te: np.ndarray,
    *,
    attn_dim: int,
    weight_decay: float,
    dropout: float,
    lr: float,
    epochs: int,
    seeds: tuple[int, ...],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Train DAR attention + classifier on frozen [N, C, 7, 7] feature maps; return (probs, attn_maps)."""
    # Channel standardization fitted strictly on training fold spatial cells.
    mean = F_tr.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(F_tr.std(axis=(0, 2, 3), keepdims=True), 1e-6)
    ft = torch.tensor((F_tr - mean) / std, dtype=torch.float32, device=device)
    fe = torch.tensor((F_te - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    wt = torch.tensor(balanced_group_weights(g_tr), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(reduction="none")

    prob_sum = np.zeros((len(F_te), 2), dtype=np.float64)
    attn_sum = np.zeros((len(F_te), F_te.shape[2], F_te.shape[3]), dtype=np.float64)
    for s in seeds:
        torch.manual_seed(s)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(s)
        head = DARHead(F_tr.shape[1], attn_dim=attn_dim, dropout=dropout).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            head.train()
            opt.zero_grad()
            logits, _ = head(ft)
            loss = (ce(logits, yt) * wt).mean()
            loss.backward()
            opt.step()
        head.eval()
        with torch.inference_mode():
            logits_e, w_e = head(fe)
            prob_sum += logits_e.softmax(dim=1).cpu().numpy()
            attn_sum += w_e.cpu().numpy()
    return prob_sum / len(seeds), (attn_sum / len(seeds)).astype(np.float32)


def nested_dar_oof(
    layer4_map: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    inner_splits: int = 4,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """5-fold nested-CV DAR over frozen pre-GAP 7x7x2048 maps; returns (oof_probs, oof_attn_7x7)."""
    cands = [
        {"attn_dim": d, "weight_decay": wd, "dropout": 0.2, "lr": 1e-3, "epochs": 120}
        for d in (64, 128)
        for wd in (1e-2, 1e-1)
    ]
    probs = np.zeros((len(y), 2), dtype=np.float64)
    attns = np.zeros((len(y), layer4_map.shape[2], layer4_map.shape[3]), dtype=np.float32)
    n_outer = int(outer_folds.max()) + 1
    for fold in range(n_outer):
        te = outer_folds == fold
        tr = ~te
        F_tr, y_tr, g_tr = layer4_map[tr], y[tr], groups[tr]
        tr_rows = [{"id": str(i), "group": str(g)} for i, g in enumerate(g_tr)]
        in_folds = stratified_group_folds(tr_rows, n_splits=inner_splits, seed=seed + 200 + fold)
        best_cfg, best_key = cands[0], (-1.0, -1.0)
        for cfg in cands:
            in_preds = np.zeros(len(y_tr), dtype=int)
            for infold in range(inner_splits):
                itr = in_folds != infold
                iva = in_folds == infold
                p_va, _ = _fit_dar_single(
                    F_tr[itr], y_tr[itr], g_tr[itr], F_tr[iva],
                    seeds=(seed + fold,), device=device, **cfg,
                )
                in_preds[iva] = p_va.argmax(axis=-1)
            m = group_metrics(y_tr, in_preds, g_tr)
            key = (m["wga"], m["overall"])
            if key > best_key:
                best_key, best_cfg = key, cfg
        p_te, a_te = _fit_dar_single(
            F_tr, y_tr, g_tr, layer4_map[te],
            seeds=(seed + fold, seed + 100 + fold, seed + 200 + fold),
            device=device, **best_cfg,
        )
        probs[te] = p_te
        attns[te] = a_te
    return probs, attns


def configure_adaptation_params(model: nn.Module, mode: str) -> list[nn.Parameter]:
    """Freeze everything except the exact target subset for Affine-DFR or Layer4-FT."""
    for p in model.parameters():
        p.requires_grad = False
    if mode == "affine_dfr":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True
        for p in model[2].parameters():
            p.requires_grad = True
    elif mode == "layer4_ft":
        for p in model[0][7].parameters():  # layer4
            p.requires_grad = True
        for p in model[2].parameters():     # fc head
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    return [p for p in model.parameters() if p.requires_grad]


def set_adaptation_train_mode(model: nn.Module, mode: str):
    """Keep frozen BN modules in eval() so running stats only update inside adapted modules."""
    model.eval()
    if mode == "affine_dfr":
        # Affine-DFR adapts BN affine (gamma, beta) & head while preserving stable BN stats or eval mode.
        model[2].train()
    elif mode == "layer4_ft":
        model[0][7].train()
        model[2].train()


def bn_stat_only_oof(
    checkpoint_path: Path,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    seed: int = 0,
    device: str = "cuda",
    batch_size: int = 64,
    workers: int = 2,
) -> np.ndarray:
    """Recalibrate BN running mean/var on group-balanced validation fold (no weight grads), then DFR-L2."""
    base_model = load_resnet_checkpoint(checkpoint_path, device="cpu")
    y_all = np.array([int(r["y"]) for r in val_rows], dtype=int)
    g_all = np.array([str(r["group"]) for r in val_rows], dtype=str)
    probs = np.zeros((len(val_rows), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1

    for fold in range(n_outer):
        te_idx = np.flatnonzero(outer_folds == fold)
        tr_idx = np.flatnonzero(outer_folds != fold)
        model = copy.deepcopy(base_model).to(device)
        for p in model.parameters():
            p.requires_grad = False

        # Reset BN running stats and run 3 group-balanced forward passes to estimate balanced stats.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        tr_rows = [val_rows[i] for i in tr_idx]
        w_tr = balanced_group_weights(g_all[tr_idx])
        gen = torch.Generator().manual_seed(seed + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_tr).double(), num_samples=len(tr_rows) * 3, replacement=True, generator=gen)
        tr_loader = DataLoader(ValImageDataset(image_dir, tr_rows), batch_size=batch_size, sampler=sampler, num_workers=workers)
        model.train()
        with torch.inference_mode():
            for _, x, _, _ in tr_loader:
                _ = model.extract_features(x.to(device))

        # Re-extract features in eval mode and fit L2 DFR strictly on tr_idx.
        model.eval()
        eval_tr = DataLoader(ValImageDataset(image_dir, tr_rows), batch_size=128, shuffle=False, num_workers=workers)
        eval_te = DataLoader(ValImageDataset(image_dir, [val_rows[i] for i in te_idx]), batch_size=128, shuffle=False, num_workers=workers)
        with torch.inference_mode():
            X_tr = np.concatenate([model.extract_features(x.to(device)).cpu().numpy() for _, x, _, _ in eval_tr], axis=0)
            X_te = np.concatenate([model.extract_features(x.to(device)).cpu().numpy() for _, x, _, _ in eval_te], axis=0)
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=0.3, max_iter=1000, solver="lbfgs")
        clf.fit(sc.transform(X_tr), y_all[tr_idx], sample_weight=w_tr)
        probs[te_idx] = clf.predict_proba(sc.transform(X_te))
    return probs


def selective_adaptation_oof(
    checkpoint_path: Path,
    image_dir: Path,
    val_rows: list[dict],
    outer_folds: np.ndarray,
    *,
    mode: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int = 0,
    device: str = "cuda",
    batch_size: int = 64,
    workers: int = 2,
) -> np.ndarray:
    """5-fold OOF selective adaptation (affine_dfr or layer4_ft) on cleaned validation split."""
    base_model = load_resnet_checkpoint(checkpoint_path, device="cpu")
    y_all = np.array([int(r["y"]) for r in val_rows], dtype=int)
    g_all = np.array([str(r["group"]) for r in val_rows], dtype=str)
    probs = np.zeros((len(val_rows), 2), dtype=np.float64)
    n_outer = int(outer_folds.max()) + 1
    ce = nn.CrossEntropyLoss()

    for fold in range(n_outer):
        te_idx = np.flatnonzero(outer_folds == fold)
        tr_idx = np.flatnonzero(outer_folds != fold)
        tr_rows = [val_rows[i] for i in tr_idx]
        te_rows = [val_rows[i] for i in te_idx]

        # Hold out 1/4 of training fold for epoch checkpoint selection (strict zero outer leakage).
        inner_folds = stratified_group_folds(tr_rows, n_splits=4, seed=seed + 300 + fold)
        fit_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds != 0)]
        sel_sub = [tr_rows[i] for i in np.flatnonzero(inner_folds == 0)]
        sel_y = np.array([int(r["y"]) for r in sel_sub], dtype=int)
        sel_g = np.array([str(r["group"]) for r in sel_sub], dtype=str)

        torch.manual_seed(seed + fold)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed + fold)
        model = copy.deepcopy(base_model).to(device)
        params = configure_adaptation_params(model, mode)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        fit_g = np.array([str(r["group"]) for r in fit_sub], dtype=str)
        w_fit = balanced_group_weights(fit_g)
        gen = torch.Generator().manual_seed(seed + 50 + fold)
        sampler = WeightedRandomSampler(torch.from_numpy(w_fit).double(), num_samples=len(fit_sub), replacement=True, generator=gen)
        fit_loader = DataLoader(ValImageDataset(image_dir, fit_sub), batch_size=batch_size, sampler=sampler, num_workers=workers)
        sel_loader = DataLoader(ValImageDataset(image_dir, sel_sub), batch_size=128, shuffle=False, num_workers=workers)
        te_loader = DataLoader(ValImageDataset(image_dir, te_rows), batch_size=128, shuffle=False, num_workers=workers)

        best_key = (-1.0, -1.0)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for _ in range(epochs):
            set_adaptation_train_mode(model, mode)
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
    return probs
