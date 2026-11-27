"""Direction 2 · Phase B: Controlled Counterfactual Background-Swap Causality Probe.

Operates strictly on validation images (split == 1) using pixel-exact CUB_200_2011
foreground masks to build 5 counterfactual views per bird:
  1. orig         : original Waterbirds validation image
  2. matched_bg   : same bird pasted on clean real Places background matching bird class
  3. flipped_bg   : same bird pasted on clean real Places background opposite bird class
  4. neutral_gray : same bird pasted on ImageNet mean-gray background (pure foreground)
  5. ood_bg       : same bird pasted on synthetic out-of-distribution non-land/non-water bg

Measures Counterfactual Flip Rate (CFR), true-class probability recovery, and
layer-wise foreground-region activation drift across layer1..layer4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.methods.rep_repair.phase_b_assets import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    downsample_mask_to_grid,
    load_val_mask_224,
    pil_eval_crop,
)


def inpaint_small_mask(rgb_224: np.ndarray, mask_224: np.ndarray, iterations: int = 12) -> np.ndarray:
    """Clean, dependency-free normalized-convolution inpainting to create bird-free real backgrounds."""
    img = rgb_224.astype(np.float32).copy()
    hole = (mask_224 > 0.15)
    if not hole.any():
        return img
    valid = (~hole).astype(np.float32)
    fill_rgb = img[~hole].mean(axis=0) if (~hole).any() else np.array(IMAGENET_MEAN, dtype=np.float32)
    for c in range(3):
        img[:, :, c] = np.where(hole, fill_rgb[c], img[:, :, c])
    # Iterative Gaussian diffusion restricted to hole pixels preserves real border textures cleanly.
    for sigma in (12.0, 8.0, 5.0, 3.0):
        for _ in range(max(1, iterations // 4)):
            for c in range(3):
                blurred = gaussian_filter(img[:, :, c], sigma=sigma, mode="nearest")
                img[:, :, c] = np.where(hole, blurred, img[:, :, c])
    return np.clip(img, 0.0, 1.0)


def composite_foreground(fg_rgb: np.ndarray, fg_mask: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Alpha-composite foreground bird onto bg_rgb in [0, 1] float32 space."""
    alpha = np.clip(fg_mask, 0.0, 1.0)[:, :, None].astype(np.float32)
    return np.clip(alpha * fg_rgb + (1.0 - alpha) * bg_rgb, 0.0, 1.0)


def make_neutral_gray_bg() -> np.ndarray:
    bg = np.zeros((224, 224, 3), dtype=np.float32)
    for c, v in enumerate(IMAGENET_MEAN):
        bg[:, :, c] = v
    return bg


def make_ood_backgrounds() -> list[np.ndarray]:
    """Three non-land/non-water out-of-distribution synthetic backgrounds (purple, warm-sand, cyan-grid)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, 224, dtype=np.float32), np.linspace(0, 1, 224, dtype=np.float32), indexing="ij")
    bg1 = np.stack([0.48 + 0.12 * yy, 0.30 + 0.06 * xx, 0.56 - 0.10 * yy], axis=-1)  # muted violet gradient
    bg2 = np.stack([0.62 - 0.08 * xx, 0.44 + 0.06 * yy, 0.36 + 0.05 * xx], axis=-1)  # warm terracotta/indoor
    checker = 0.06 * (((np.floor(yy * 14) + np.floor(xx * 14)) % 2) * 2 - 1)
    bg3 = np.stack([0.52 + checker, 0.50 - checker, 0.54 + checker], axis=-1)        # neutral studio pattern
    return [np.clip(b, 0.0, 1.0).astype(np.float32) for b in (bg1, bg2, bg3)]


def build_clean_bg_donors(
    image_dir: Path,
    seg_root: Path,
    val_rows: list[dict],
    per_place: int = 15,
) -> dict[int, list[np.ndarray]]:
    """Select smallest-bird validation images for place=0 (land) & place=1 (water) and inpaint out the bird."""
    by_place: dict[int, list[tuple[float, dict]]] = {0: [], 1: []}
    for r in val_rows:
        mask = load_val_mask_224(seg_root, r["id"])
        area = float((mask > 0.5).mean())
        by_place[int(r["a"])].append((area, r))

    donors: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for place in (0, 1):
        # Smallest birds leave >90% genuine Places background intact.
        ranked = sorted(by_place[place], key=lambda item: item[0])[:per_place]
        for _, r in ranked:
            with Image.open(image_dir / r["id"]) as im:
                rgb = np.asarray(pil_eval_crop(im.convert("RGB")), dtype=np.float32) / 255.0
            mask = load_val_mask_224(seg_root, r["id"])
            donors[place].append(inpaint_small_mask(rgb, mask))
    return donors


def rgb_to_model_tensor(rgb_224: np.ndarray) -> torch.Tensor:
    """Normalize [224, 224, 3] float32 image in [0, 1] into ImageNet-normalized [3, 224, 224] tensor."""
    t = torch.from_numpy(rgb_224).permute(2, 0, 1).float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (t - mean) / std


def foreground_feature_drift(
    stage_maps_a: dict[str, torch.Tensor],
    stage_maps_b: dict[str, torch.Tensor],
    mask_224: np.ndarray,
) -> dict[str, float]:
    """Relative L2 drift inside the TRUE FOREGROUND spatial cells across layer1..layer4 when BG changes."""
    drifts = {}
    for stage in ("layer1", "layer2", "layer3", "layer4_map"):
        fa = stage_maps_a[stage][0].detach().cpu().numpy()  # [C, H, W]
        fb = stage_maps_b[stage][0].detach().cpu().numpy()
        grid = fa.shape[1]
        m = downsample_mask_to_grid(mask_224, grid=grid)
        w = m / max(float(m.sum()), 1e-6) if m.sum() > 1e-4 else np.full((grid, grid), 1.0 / (grid * grid), dtype=np.float32)
        va = np.einsum("chw,hw->c", fa, w)
        vb = np.einsum("chw,hw->c", fb, w)
        denom = max(float(np.linalg.norm(va) + np.linalg.norm(vb)) * 0.5, 1e-6)
        drifts[stage] = float(np.linalg.norm(va - vb) / denom)
    return drifts
