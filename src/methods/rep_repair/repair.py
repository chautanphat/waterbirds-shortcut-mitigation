"""Direction 2 · Phase C · Step 6: Mechanism-Derived Pre-Layer4 Representation Repair.

Derived directly from Step 5 causal localization:
  - Panel A: spatial separation holds through layer3.5 (14x14) and breaks inside layer4.{0,1,2}
    -> insert an identity-initialized Pre-Layer4 Spatial Bleed Gate at layer3.5 (mask-free).
  - Panel B: layer4 3x3 residual branches inject 58%-76% of foreground-cell drift
    -> regularize layer4 foreground output against mask-free layer3.5 low-gate context swaps.
  - Panel C: BN affine parameters (bn_l3_l4 / bn_all) are the highest-leverage minimal parameter set
    -> keep 100% of conv weights frozen; adapt only Pre-Layer4 gate + BN affine + head.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from src.methods.rep_repair.patching import configure_bn_stage_params


class PreLayer4SpatialGate(nn.Module):
    """Lightweight identity-initialized 14x14 spatial gate placed between layer3.5 and layer4.0."""

    def __init__(self, channels: int = 1024, hidden: int = 64, dropout: float = 0.2, init_bias: float = 2.5):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        # Near-identity initialization: sigmoid(+2.5) ~= 0.924 everywhere with healthy gradients.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: [B, C, H, W] -> gate: [B, 1, H, W] in (0, 1)
        b, c, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)  # [B, H*W, C]
        logits = self.fc2(self.drop(self.act(self.fc1(self.norm(tokens)))))  # [B, H*W, 1]
        return torch.sigmoid(logits).transpose(1, 2).view(b, 1, h, w)


class CosineClassifier(nn.Module):
    """Row-standardized L2-unit cosine readout (spherical geometry without bias-shift vulnerability)."""

    def __init__(self, in_dim: int = 2048, num_classes: int = 2, init_scale: float = 12.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x_norm = F.normalize((x - mean) / std, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.scale.abs() * (x_norm @ w_norm.t())


class PreLayer4RepairModel(nn.Module):
    """Wraps a ResNetClassifier with optional Pre-Layer4 Spatial Gate, BN-affine scope, and head."""

    def __init__(
        self,
        base_resnet: nn.Module,
        *,
        use_gate: bool = True,
        head_type: str = "cosine",
        bn_scope: str = "bn_all",
        gate_hidden: int = 64,
        gate_dropout: float = 0.2,
    ):
        super().__init__()
        if head_type not in ("linear", "cosine"):
            raise ValueError(f"Unsupported head_type: {head_type}")
        if bn_scope not in ("none", "bn_l1_l2", "bn_l3", "bn_l4", "bn_l3_l4", "bn_all"):
            raise ValueError(f"Unsupported bn_scope: {bn_scope}")

        self.backbone = copy.deepcopy(base_resnet)
        self.use_gate = bool(use_gate)
        self.head_type = head_type
        self.bn_scope = bn_scope

        # Freeze entire backbone first, then unfreeze only the declared BN affine scope.
        for p in self.backbone.parameters():
            p.requires_grad = False
        if bn_scope != "none":
            configure_bn_stage_params(self.backbone, bn_scope)
            # We manage the classification head via self.head below; keep backbone[2] frozen.
            for p in self.backbone[2].parameters():
                p.requires_grad = False

        self.gate = PreLayer4SpatialGate(1024, hidden=gate_hidden, dropout=gate_dropout) if use_gate else None
        if head_type == "linear":
            self.head = copy.deepcopy(self.backbone[2])
            for p in self.head.parameters():
                p.requires_grad = True
        else:
            self.head = CosineClassifier(2048, 2)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def set_repair_train_mode(self):
        """Keep all frozen BN running stats fixed in eval(); enable dropout only in gate/head."""
        self.backbone.eval()
        if self.gate is not None:
            self.gate.train()
        self.head.train()

    def _forward_to_layer3(self, inputs: torch.Tensor):
        stem = self.backbone[0]
        x = stem[3](stem[2](stem[1](stem[0](inputs))))
        c1 = stem[4](x)
        c2 = stem[5](c1)
        c3 = stem[6](c2)  # [B, 1024, 14, 14]
        return c1, c2, c3

    def _forward_from_layer3(self, c3_in: torch.Tensor):
        stem = self.backbone[0]
        c4 = stem[7](c3_in)               # [B, 2048, 7, 7]
        pooled = self.backbone[1](stem[8](c4))  # [B, 2048]
        logits = self.head(pooled)
        return logits, c4, pooled

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, c3 = self._forward_to_layer3(inputs)
        if self.gate is not None:
            g = self.gate(c3)
            c3 = g * c3
        logits, _, _ = self._forward_from_layer3(c3)
        return logits

    @torch.inference_mode()
    def extract_repaired_stage_maps(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        c1, c2, c3 = self._forward_to_layer3(inputs)
        gate_map = self.gate(c3) if self.gate is not None else torch.ones((c3.shape[0], 1, c3.shape[2], c3.shape[3]), device=c3.device, dtype=c3.dtype)
        c3_gated = gate_map * c3 if self.gate is not None else c3
        logits, c4, pooled = self._forward_from_layer3(c3_gated)
        return {
            "layer1": c1,
            "layer2": c2,
            "layer3": c3,
            "gate_map": gate_map,
            "layer4_map": c4,
            "pooled": pooled,
            "logits": logits,
        }

    def forward_with_context_swap(
        self,
        inputs: torch.Tensor,
        attributes: torch.Tensor,
        zero_context_prob: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        """Forward original batch + mask-free layer3.5 low-gate opposite-background swapped batch."""
        if self.gate is None:
            raise RuntimeError("forward_with_context_swap requires use_gate=True")

        _, _, c3 = self._forward_to_layer3(inputs)
        g = self.gate(c3)  # [B, 1, 14, 14]
        logits, c4, _ = self._forward_from_layer3(g * c3)

        # Build mask-free opposite-background donor context at layer3.5 using detached soft mask m.
        m = g.detach()  # [B, 1, 14, 14]
        bg_w = (1.0 - m)
        bg_proto = (bg_w * c3.detach()).sum(dim=(2, 3), keepdim=True) / bg_w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)

        b = c3.shape[0]
        donor_idx = torch.arange(b, device=c3.device)
        attrs = attributes.to(c3.device)
        for i in range(b):
            opp = torch.nonzero(attrs != attrs[i], as_tuple=False).flatten()
            if len(opp) > 0:
                pick = opp[torch.randint(0, len(opp), (1,), device=c3.device).item()]
                donor_idx[i] = pick
            elif b > 1:
                donor_idx[i] = (i + 1) % b

        donor_bg = bg_proto[donor_idx]  # [B, 1024, 1, 1]
        use_zero = (torch.rand((b, 1, 1, 1), device=c3.device) < zero_context_prob).to(c3.dtype)
        ctx = (1.0 - use_zero) * donor_bg
        c3_swap = m * c3 + (1.0 - m) * ctx

        # Apply the same differentiable gate g to c3_swap so minimizing layer4 drift drives g -> 0 on BG cells.
        logits_swap, c4_swap, _ = self._forward_from_layer3(g * c3_swap)
        return {
            "logits": logits,
            "logits_swap": logits_swap,
            "layer4_map": c4,
            "layer4_map_swap": c4_swap,
            "gate_map": g,
        }


def context_invariance_loss(
    logits: torch.Tensor,
    logits_swap: torch.Tensor,
    l4_map: torch.Tensor,
    l4_map_swap: torch.Tensor,
    gate_map: torch.Tensor,
    *,
    drift_weight: float = 1.0,
    cons_weight: float = 0.5,
    gate_sparsity: float = 0.05,
    min_gate_mean: float = 0.20,
) -> dict[str, torch.Tensor]:
    """Penalize layer4 foreground-cell feature drift + logit shift under layer3.5 BG swaps."""
    # Downsample detached 14x14 gate -> 7x7 foreground weighting for layer4.
    m7 = F.avg_pool2d(gate_map.detach(), kernel_size=2, stride=2)  # [B, 1, 7, 7]
    w7 = m7 / m7.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    fg_orig = (l4_map * w7).sum(dim=(2, 3))       # [B, 2048]
    fg_swap = (l4_map_swap * w7).sum(dim=(2, 3))  # [B, 2048]

    num = (fg_orig - fg_swap).pow(2).sum(dim=1)
    den = fg_orig.detach().pow(2).sum(dim=1).clamp_min(1e-4)
    drift_loss = (num / den).mean()

    p_orig = F.softmax(logits.detach(), dim=1)
    log_p_swap = F.log_softmax(logits_swap, dim=1)
    cons_loss = F.kl_div(log_p_swap, p_orig, reduction="batchmean")

    # Gentle sparsity pressure to attenuate non-discriminative BG cells while forbidding collapse below min_gate_mean.
    g_mean = gate_map.mean(dim=(1, 2, 3))
    anti_collapse = F.relu(min_gate_mean - g_mean).pow(2).mean() * 10.0
    gate_reg = gate_sparsity * g_mean.mean() + anti_collapse

    total = drift_weight * drift_loss + cons_weight * cons_loss + gate_reg
    return {
        "total": total,
        "drift": drift_loss,
        "cons": cons_loss,
        "gate_reg": gate_reg,
        "gate_mean": g_mean.mean(),
    }
