"""Direction 2 · Phase C · Step 5: Causal Block-Level Activation Patching & Drift Attribution.

Operates on ResNetClassifier (0.* backbone Sequential, 1 Flatten, 2 Linear) without
modifying state_dict layout. Provides:
  1. Named block access across layer1, layer2, layer3.{0,2,5}, layer4.{0,1,2};
  2. Spatially masked activation patching (patch_bg_only, patch_fg_only, patch_full);
  3. Bottleneck residual-branch vs. skip-branch foreground drift & ReLU gate-flip decomposition;
  4. Stage-restricted BN-affine adaptation configuration (bn_l1_l2, bn_l3, bn_l4, bn_l3_l4, bn_all).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.methods.rep_repair.phase_b_assets import downsample_mask_to_grid

PATCH_BLOCKS: tuple[tuple[str, int, int | None], ...] = (
    ("layer1", 4, None),
    ("layer2", 5, None),
    ("layer3.0", 6, 0),
    ("layer3.2", 6, 2),
    ("layer3.5", 6, 5),
    ("layer4.0", 7, 0),
    ("layer4.1", 7, 1),
    ("layer4.2", 7, 2),
)


def get_block_module(model: nn.Module, block_name: str) -> nn.Module:
    """Resolve a ResNetClassifier block module by canonical name in PATCH_BLOCKS."""
    lookup = {name: (stage_idx, sub_idx) for name, stage_idx, sub_idx in PATCH_BLOCKS}
    if block_name not in lookup:
        raise ValueError(f"Unknown patch block: {block_name}")
    stage_idx, sub_idx = lookup[block_name]
    stage = model[0][stage_idx]
    return stage if sub_idx is None else stage[sub_idx]


@torch.inference_mode()
def capture_block_outputs(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run one clean forward pass and capture outputs at every PATCH_BLOCKS site plus pooled."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_mod, _inp, out):
            captured[name] = out.detach().clone()
        return _hook

    for name, _, _ in PATCH_BLOCKS:
        handles.append(get_block_module(model, name).register_forward_hook(_make_hook(name)))
    try:
        stages = model.extract_stage_maps(x)
        captured["layer4_map"] = stages["layer4_map"].detach().clone()
        captured["pooled"] = stages["pooled"].detach().clone()
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.inference_mode()
def forward_with_spatial_patch(
    model: nn.Module,
    x_corrupt: torch.Tensor,
    clean_activation: torch.Tensor,
    block_name: str,
    fg_mask_224: np.ndarray,
    patch_mode: str,
) -> dict[str, torch.Tensor]:
    """Forward x_corrupt while spatially patching block_name with clean_activation."""
    if patch_mode not in ("full", "fg_only", "bg_only"):
        raise ValueError(f"Unsupported patch_mode: {patch_mode}")

    module = get_block_module(model, block_name)

    def _patch_hook(_mod, _inp, out: torch.Tensor):
        if clean_activation.shape != out.shape:
            raise ValueError(f"Patch shape mismatch at {block_name}: {clean_activation.shape} vs {out.shape}")
        if patch_mode == "full":
            return clean_activation
        grid_h, grid_w = out.shape[2], out.shape[3]
        if grid_h != grid_w:
            raise ValueError(f"Expected square spatial grid, got {(grid_h, grid_w)}")
        m = downsample_mask_to_grid(fg_mask_224, grid=grid_h)
        fg_t = torch.from_numpy(m).to(device=out.device, dtype=out.dtype)[None, None, :, :]
        alpha = fg_t if patch_mode == "fg_only" else (1.0 - fg_t)
        return alpha * clean_activation + (1.0 - alpha) * out

    handle = module.register_forward_hook(_patch_hook)
    try:
        stages = model.extract_stage_maps(x_corrupt)
    finally:
        handle.remove()
    return stages


@torch.inference_mode()
def bottleneck_branch_decomposition(
    model: nn.Module,
    x_orig: torch.Tensor,
    x_clean: torch.Tensor,
    fg_mask_224: np.ndarray,
    block_names: tuple[str, ...] = ("layer3.0", "layer3.5", "layer4.0", "layer4.1", "layer4.2"),
) -> list[dict]:
    """Decompose foreground-cell drift inside Bottleneck blocks into skip vs. conv branch & ReLU flips."""
    results = []

    def _capture_inputs(x: torch.Tensor) -> dict[str, torch.Tensor]:
        inps = {}
        handles = []
        for name in block_names:
            mod = get_block_module(model, name)
            handles.append(mod.register_forward_hook(lambda _m, inp, _o, n=name: inps.__setitem__(n, inp[0].detach().clone())))
        try:
            _ = model.extract_features(x)
        finally:
            for h in handles:
                h.remove()
        return inps

    in_orig = _capture_inputs(x_orig)
    in_clean = _capture_inputs(x_clean)

    def _branch_tensors(block: nn.Module, x_in: torch.Tensor):
        identity = block.downsample(x_in) if block.downsample is not None else x_in
        f = block.relu(block.bn1(block.conv1(x_in)))
        f = block.relu(block.bn2(block.conv2(f)))
        residual = block.bn3(block.conv3(f))
        pre_relu = identity + residual
        post_relu = block.relu(pre_relu)
        return identity, residual, pre_relu, post_relu

    def _fg_pool(t: torch.Tensor, mask_224: np.ndarray) -> np.ndarray:
        arr = t[0].detach().cpu().numpy()
        m = downsample_mask_to_grid(mask_224, grid=arr.shape[1])
        w = m / max(float(m.sum()), 1e-6) if m.sum() > 1e-4 else np.full_like(m, 1.0 / m.size)
        return np.einsum("chw,hw->c", arr, w)

    def _rel_drift(a: np.ndarray, b: np.ndarray) -> float:
        denom = max(0.5 * float(np.linalg.norm(a) + np.linalg.norm(b)), 1e-6)
        return float(np.linalg.norm(a - b) / denom)

    for name in block_names:
        block = get_block_module(model, name)
        id_o, res_o, pre_o, post_o = _branch_tensors(block, in_orig[name])
        id_c, res_c, pre_c, post_c = _branch_tensors(block, in_clean[name])

        grid = pre_o.shape[2]
        m_bin = downsample_mask_to_grid(fg_mask_224, grid=grid) > 0.25
        if not m_bin.any():
            m_bin = np.ones((grid, grid), dtype=bool)
        po = (pre_o[0].detach().cpu().numpy() > 0.0)[:, m_bin]
        pc = (pre_c[0].detach().cpu().numpy() > 0.0)[:, m_bin]
        relu_flip_rate = float(np.mean(po != pc))

        id_fg_o, id_fg_c = _fg_pool(id_o, fg_mask_224), _fg_pool(id_c, fg_mask_224)
        res_fg_o, res_fg_c = _fg_pool(res_o, fg_mask_224), _fg_pool(res_c, fg_mask_224)
        out_fg_o, out_fg_c = _fg_pool(post_o, fg_mask_224), _fg_pool(post_c, fg_mask_224)

        d_id_norm = float(np.linalg.norm(id_fg_o - id_fg_c))
        d_res_norm = float(np.linalg.norm(res_fg_o - res_fg_c))
        res_share = d_res_norm / max(d_id_norm + d_res_norm, 1e-6)

        results.append({
            "block": name,
            "skip_fg_drift": _rel_drift(id_fg_o, id_fg_c),
            "residual_conv_fg_drift": _rel_drift(res_fg_o, res_fg_c),
            "post_relu_fg_drift": _rel_drift(out_fg_o, out_fg_c),
            "residual_delta_share": res_share,
            "fg_relu_flip_rate": relu_flip_rate,
        })
    return results


def configure_bn_stage_params(model: nn.Module, bn_scope: str) -> list[nn.Parameter]:
    """Unfreeze ONLY final linear head + BatchNorm2d (gamma, beta) in the requested ResNet stage scope."""
    valid_scopes = {
        "bn_l1_l2": (1, 4, 5),
        "bn_l3": (6,),
        "bn_l4": (7,),
        "bn_l3_l4": (6, 7),
        "bn_all": (1, 4, 5, 6, 7),
    }
    if bn_scope not in valid_scopes:
        raise ValueError(f"Unsupported bn_scope: {bn_scope}")

    for p in model.parameters():
        p.requires_grad = False
    for idx in valid_scopes[bn_scope]:
        sub = model[0][idx]
        for m in sub.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True
    for p in model[2].parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def summarize_bn_affine_shifts(base_model: nn.Module, adapted_model: nn.Module) -> list[dict]:
    """Measure per-stage L1 shift in BN (gamma, beta) learned by Affine-DFR."""
    stage_map = [("stem", 1), ("layer1", 4), ("layer2", 5), ("layer3", 6), ("layer4", 7)]
    rows = []
    for stage_name, idx in stage_map:
        base_sub = base_model[0][idx]
        adap_sub = adapted_model[0][idx]
        dw_list, db_list = [], []
        for bm, am in zip(base_sub.modules(), adap_sub.modules()):
            if isinstance(bm, nn.BatchNorm2d):
                dw_list.append((am.weight.detach().cpu() - bm.weight.detach().cpu()).abs().numpy())
                db_list.append((am.bias.detach().cpu() - bm.bias.detach().cpu()).abs().numpy())
        dw = np.concatenate(dw_list) if dw_list else np.array([0.0])
        db = np.concatenate(db_list) if db_list else np.array([0.0])
        rows.append({
            "stage": stage_name,
            "n_channels": int(len(dw)),
            "mean_abs_dgamma": round(float(dw.mean()), 5),
            "max_abs_dgamma": round(float(dw.max()), 5),
            "mean_abs_dbeta": round(float(db.mean()), 5),
            "max_abs_dbeta": round(float(db.max()), 5),
        })
    return rows
