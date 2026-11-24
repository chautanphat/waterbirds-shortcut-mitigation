"""ViT-B/16 classifier with block-wise token extraction, spatial patching, and LayerNorm-affine scoping."""

from __future__ import annotations

import copy
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


class PreBlockSpatialGate(nn.Module):
    """Identity-initialized spatial token gate placed before the ViT spatial crossover block (block_6)."""

    def __init__(self, dim: int = 768, hidden: int = 64, dropout: float = 0.2, init_bias: float = 2.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, 196, D] -> gate: [B, 196, 1] in (0, 1)
        logits = self.fc2(self.drop(self.act(self.fc1(self.norm(patch_tokens)))))
        return torch.sigmoid(logits)


class PreBlockViTRepairModel(nn.Module):
    """Cross-architecture transfer of Peak-Neutral Pre-Crossover Gate + LN-affine repair to ViT-B/16."""

    def __init__(
        self,
        base_vit: ViTClassifier,
        *,
        gate_before_block: int = 6,
        use_gate: bool = True,
        ln_scope: str = "ln_late",
        gate_hidden: int = 64,
        gate_dropout: float = 0.2,
    ):
        super().__init__()
        if not 0 < gate_before_block < len(base_vit.vit.encoder.layers):
            raise ValueError(f"Invalid gate_before_block: {gate_before_block}")
        self.backbone = copy.deepcopy(base_vit)
        self.gate_before_block = int(gate_before_block)
        self.use_gate = bool(use_gate)
        self.ln_scope = ln_scope
        configure_vit_ln_params(self.backbone, ln_scope)
        self.gate = PreBlockSpatialGate(self.backbone.hidden_dim, hidden=gate_hidden, dropout=gate_dropout) if use_gate else None
        self.register_buffer("neutral_tokens", torch.zeros(1, 1, self.backbone.hidden_dim))
        self.register_buffer("has_neutral_tokens", torch.tensor(False, dtype=torch.bool))

    def set_neutral_tokens(self, mu: torch.Tensor) -> None:
        vec = mu.detach().reshape(1, 1, self.backbone.hidden_dim).to(
            device=self.neutral_tokens.device, dtype=self.neutral_tokens.dtype
        )
        self.neutral_tokens.copy_(vec)
        self.has_neutral_tokens.fill_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def set_repair_train_mode(self):
        self.backbone.eval()
        if self.gate is not None:
            self.gate.train()

    def _effective_gate(self, g: torch.Tensor) -> torch.Tensor:
        return g / g.amax(dim=1, keepdim=True).clamp_min(1e-4)

    def _neutral_baseline(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if bool(self.has_neutral_tokens.item()):
            return self.neutral_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        return patch_tokens.mean(dim=(0, 1), keepdim=True).detach()

    def _apply_gate(self, patch_tokens: torch.Tensor, g_eff: torch.Tensor) -> torch.Tensor:
        mu = self._neutral_baseline(patch_tokens)
        return g_eff * patch_tokens + (1.0 - g_eff) * mu

    def _forward_to_split(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone._embed_tokens(inputs)
        for layer in list(self.backbone.vit.encoder.layers)[: self.gate_before_block]:
            tokens = layer(tokens)
        return tokens

    def _forward_from_split(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        x = tokens
        for layer in list(self.backbone.vit.encoder.layers)[self.gate_before_block :]:
            x = layer(x)
        return self.backbone._tokens_to_outputs(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self._forward_to_split(inputs)
        if self.gate is not None:
            cls_tok, patch_tok = tokens[:, :1, :], tokens[:, 1:, :]
            g_eff = self._effective_gate(self.gate(patch_tok))
            tokens = torch.cat([cls_tok, self._apply_gate(patch_tok, g_eff)], dim=1)
        return self._forward_from_split(tokens)["logits"]

    def forward_with_context_swap(
        self,
        inputs: torch.Tensor,
        attributes: torch.Tensor,
        zero_context_prob: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        if self.gate is None:
            raise RuntimeError("forward_with_context_swap requires use_gate=True")
        tokens = self._forward_to_split(inputs)
        cls_tok, patch_tok = tokens[:, :1, :], tokens[:, 1:, :]
        g_eff = self._effective_gate(self.gate(patch_tok))  # [B, 196, 1]
        out_orig = self._forward_from_split(torch.cat([cls_tok, self._apply_gate(patch_tok, g_eff)], dim=1))

        m = g_eff.detach()
        bg_w = 1.0 - m
        bg_proto = (bg_w * patch_tok.detach()).sum(dim=1, keepdim=True) / bg_w.sum(dim=1, keepdim=True).clamp_min(1e-4)

        b = patch_tok.shape[0]
        donor_idx = torch.arange(b, device=patch_tok.device)
        attrs = attributes.to(patch_tok.device)
        for i in range(b):
            opp = torch.nonzero(attrs != attrs[i], as_tuple=False).flatten()
            if len(opp) > 0:
                donor_idx[i] = opp[torch.randint(0, len(opp), (1,), device=patch_tok.device).item()]
            elif b > 1:
                donor_idx[i] = (i + 1) % b

        donor_bg = bg_proto[donor_idx]  # [B, 1, D]
        use_neutral = (torch.rand((b, 1, 1), device=patch_tok.device) < zero_context_prob).to(patch_tok.dtype)
        ctx = (1.0 - use_neutral) * donor_bg + use_neutral * self._neutral_baseline(patch_tok)
        patch_swap = m * patch_tok + (1.0 - m) * ctx
        out_swap = self._forward_from_split(torch.cat([cls_tok, self._apply_gate(patch_swap, g_eff)], dim=1))

        gs = self.backbone.grid_size
        gate_map = g_eff.transpose(1, 2).reshape(b, 1, gs, gs)
        return {
            "logits": out_orig["logits"],
            "logits_swap": out_swap["logits"],
            "patch_map": out_orig["patch_map"],
            "patch_map_swap": out_swap["patch_map"],
            "gate_map": gate_map,
        }


@torch.inference_mode()
def compute_group_balanced_vit_block_mean(
    base_vit: ViTClassifier,
    loader,
    device: str,
    *,
    gate_before_block: int = 6,
) -> torch.Tensor:
    """Compute group-balanced mean patch-token activation [1, 1, 768] before gate_before_block strictly on fit split."""
    model = copy.deepcopy(base_vit).to(device)
    model.eval()
    group_sums: dict[str, torch.Tensor] = {}
    group_counts: dict[str, int] = {}
    for _, xb, _, gb in loader:
        tokens = model._embed_tokens(xb.to(device))
        for layer in list(model.vit.encoder.layers)[:gate_before_block]:
            tokens = layer(tokens)
        spatial_mean = tokens[:, 1:, :].mean(dim=1).cpu().double()  # [B, 768]
        for i, g_str in enumerate(gb):
            key = str(g_str)
            if key not in group_sums:
                group_sums[key] = spatial_mean[i].clone()
                group_counts[key] = 1
            else:
                group_sums[key] += spatial_mean[i]
                group_counts[key] += 1
    per_group_means = [group_sums[k] / float(group_counts[k]) for k in sorted(group_sums)]
    return torch.stack(per_group_means, dim=0).mean(dim=0).float().view(1, 1, model.hidden_dim)
