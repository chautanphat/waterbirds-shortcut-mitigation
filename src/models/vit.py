"""ViT-B/16 classifier with block-wise token extraction, spatial patching, and LayerNorm-affine scoping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from src.methods.rep_repair.phase_b_assets import downsample_mask_to_grid

VIT_BLOCK_SITES = ("embed",) + tuple(f"block_{i}" for i in range(12))
VIT_PATCH_SITES = ("embed", "block_0", "block_2", "block_4", "block_6", "block_8", "block_9", "block_10", "block_11")


class ViTClassifier(nn.Module):
    """Standard torchvision ViT-B/16 (224x224 -> 14x14 patches, 12 blocks, 768-D) for binary classification."""

    def __init__(self, weights: str | None = "IMAGENET1K_V1", num_classes: int = 2):
        super().__init__()
        if weights not in (None, "IMAGENET1K_V1", "IMAGENET1K_SWAG_E2E_V1", "IMAGENET1K_SWAG_LINEAR_V1"):
            raise ValueError(f"Unsupported ViT-B/16 weights: {weights}")
        w = None if weights is None else ViT_B_16_Weights[weights]
        self.vit = vit_b_16(weights=w)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        self.hidden_dim = int(in_features)
        self.grid_size = 14

    def _embed_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(inputs)  # [B, 196, 768]
        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)       # [B, 197, 768]
        return self.vit.encoder.dropout(x + self.vit.encoder.pos_embedding)

    def _tokens_to_outputs(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        normed = self.vit.encoder.ln(tokens)
        pooled = normed[:, 0]  # [B, 768]
        patch_map = normed[:, 1:].transpose(1, 2).reshape(
            normed.shape[0], self.hidden_dim, self.grid_size, self.grid_size
        )
        logits = self.vit.heads(pooled)
        return {"final_tokens": normed, "pooled": pooled, "patch_map": patch_map, "logits": logits}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.vit(inputs)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self._embed_tokens(inputs)
        for layer in self.vit.encoder.layers:
            tokens = layer(tokens)
        return self.vit.encoder.ln(tokens)[:, 0]

    def extract_block_tokens(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return token sequences after embedding and each of the 12 Transformer blocks plus final readout."""
        tokens = self._embed_tokens(inputs)
        out: dict[str, torch.Tensor] = {"embed": tokens}
        for idx, layer in enumerate(self.vit.encoder.layers):
            tokens = layer(tokens)
            out[f"block_{idx}"] = tokens
        out.update(self._tokens_to_outputs(tokens))
        return out

    def forward_from_site(self, tokens: torch.Tensor, site: str) -> dict[str, torch.Tensor]:
        """Continue forward pass from a patched token tensor produced immediately after `site`."""
        if site == "embed":
            start_idx = 0
        elif site.startswith("block_"):
            idx = int(site.split("_", 1)[1])
            if not 0 <= idx < len(self.vit.encoder.layers):
                raise ValueError(f"Invalid ViT block site: {site}")
            start_idx = idx + 1
        else:
            raise ValueError(f"Unsupported ViT site: {site}")
        x = tokens
        for layer in self.vit.encoder.layers[start_idx:]:
            x = layer(x)
        return self._tokens_to_outputs(x)


def load_vit_checkpoint(path: Path | str, *, device: str = "cpu") -> ViTClassifier:
    model = ViTClassifier(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device)


def configure_vit_ln_params(model: ViTClassifier, scope: str) -> list[nn.Parameter]:
    """Freeze all attention/MLP/embedding weights; unfreeze ONLY declared LayerNorm affine params + head."""
    if scope not in ("none", "ln_late", "ln_all"):
        raise ValueError(f"Unsupported ViT LN scope: {scope}")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.vit.heads.parameters():
        p.requires_grad = True
    if scope != "none":
        start_block = 8 if scope == "ln_late" else 0
        for layer in list(model.vit.encoder.layers)[start_block:]:
            for mod in layer.modules():
                if isinstance(mod, nn.LayerNorm):
                    for p in mod.parameters():
                        p.requires_grad = True
        for p in model.vit.encoder.ln.parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def pool_vit_fg_tokens(patch_map: np.ndarray, mask_14: np.ndarray) -> np.ndarray:
    """Weighted average of [N, 768, 14, 14] patch tokens strictly inside [N, 14, 14] foreground masks."""
    m = np.clip(mask_14, 0.0, 1.0).astype(np.float32)
    denom = m.sum(axis=(1, 2), keepdims=True)
    fallback = np.full_like(m, 1.0 / (m.shape[1] * m.shape[2]))
    w = np.where(denom > 1e-4, m / np.maximum(denom, 1e-4), fallback)
    return np.einsum("nchw,nhw->nc", patch_map.astype(np.float32), w)


def vit_foreground_token_drift(
    tokens_a: dict[str, torch.Tensor],
    tokens_b: dict[str, torch.Tensor],
    mask_224: np.ndarray,
) -> dict[str, float]:
    """Relative L2 drift inside TRUE foreground 14x14 patch tokens across embed + blocks 0..11 + final CLS."""
    m14 = downsample_mask_to_grid(mask_224, grid=14).reshape(-1)  # [196]
    w = m14 / max(float(m14.sum()), 1e-6) if m14.sum() > 1e-4 else np.full(196, 1.0 / 196.0, dtype=np.float32)
    drifts: dict[str, float] = {}
    for site in VIT_BLOCK_SITES:
        ta = tokens_a[site][0, 1:].detach().cpu().numpy()  # [196, 768]
        tb = tokens_b[site][0, 1:].detach().cpu().numpy()
        va = (ta * w[:, None]).sum(axis=0)
        vb = (tb * w[:, None]).sum(axis=0)
        denom = max(float(np.linalg.norm(va) + np.linalg.norm(vb)) * 0.5, 1e-6)
        drifts[site] = float(np.linalg.norm(va - vb) / denom)
    ca = tokens_a["pooled"][0].detach().cpu().numpy()
    cb = tokens_b["pooled"][0].detach().cpu().numpy()
    cden = max(float(np.linalg.norm(ca) + np.linalg.norm(cb)) * 0.5, 1e-6)
    drifts["final_cls"] = float(np.linalg.norm(ca - cb) / cden)
    return drifts
