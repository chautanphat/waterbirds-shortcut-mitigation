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


class LoRAConv2d(nn.Module):
    """Identity-initialized low-rank spatial residual wrapper around a frozen nn.Conv2d."""

    def __init__(self, base_conv: nn.Conv2d, rank: int = 8, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.scale = float(alpha if alpha is not None else rank) / float(rank)
        self.lora_a = nn.Conv2d(
            base_conv.in_channels,
            self.rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=base_conv.groups,
            bias=False,
        )
        self.lora_b = nn.Conv2d(self.rank, base_conv.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5.0**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


class PreLayer4RepairModel(nn.Module):
    """Wraps a ResNetClassifier with optional Pre-Layer4 Spatial Gate, layer4.conv2 LoRA, BN-affine scope, and head."""

    def __init__(
        self,
        base_resnet: nn.Module,
        *,
        use_gate: bool = True,
        gate_mode: str = "raw",
        head_type: str = "cosine",
        bn_scope: str = "bn_all",
        gate_hidden: int = 64,
        gate_dropout: float = 0.2,
        use_l4_conv2_lora: bool = False,
        lora_rank: int = 8,
    ):
        super().__init__()
        if head_type not in ("linear", "cosine"):
            raise ValueError(f"Unsupported head_type: {head_type}")
        if gate_mode not in ("raw", "peak_neutral"):
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        if bn_scope not in ("none", "bn_l1_l2", "bn_l3", "bn_l4", "bn_l3_l4", "bn_all"):
            raise ValueError(f"Unsupported bn_scope: {bn_scope}")

        self.backbone = copy.deepcopy(base_resnet)
        self.use_gate = bool(use_gate)
        self.gate_mode = gate_mode
        self.head_type = head_type
        self.bn_scope = bn_scope
        self.use_l4_conv2_lora = bool(use_l4_conv2_lora)
        self.lora_rank = int(lora_rank)

        # Freeze entire backbone first, then unfreeze only the declared BN affine scope.
        for p in self.backbone.parameters():
            p.requires_grad = False
        if bn_scope != "none":
            configure_bn_stage_params(self.backbone, bn_scope)
            # We manage the classification head via self.head below; keep backbone[2] frozen.
            for p in self.backbone[2].parameters():
                p.requires_grad = False

        if self.use_l4_conv2_lora:
            # Step 5 Panel B: target ONLY layer4.{0,1,2}.conv2 (the 3x3 spatial mixing convs).
            for block in self.backbone[0][7]:
                block.conv2 = LoRAConv2d(block.conv2, rank=self.lora_rank)

        self.gate = PreLayer4SpatialGate(1024, hidden=gate_hidden, dropout=gate_dropout) if use_gate else None
        self.register_buffer("neutral_c3", torch.zeros(1, 1024, 1, 1))
        self.register_buffer("has_neutral_c3", torch.tensor(False, dtype=torch.bool))
        if head_type == "linear":
            self.head = copy.deepcopy(self.backbone[2])
            for p in self.head.parameters():
                p.requires_grad = True
        else:
            self.head = CosineClassifier(2048, 2)

    def set_neutral_c3(self, mu_c3: torch.Tensor) -> None:
        """Register fold-training-only group-balanced neutral layer3 channel mean [1, 1024, 1, 1]."""
        mu = mu_c3.detach().reshape(1, 1024, 1, 1).to(device=self.neutral_c3.device, dtype=self.neutral_c3.dtype)
        self.neutral_c3.copy_(mu)
        self.has_neutral_c3.fill_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_groups(self, lr: float, lora_lr: float | None = None) -> list[dict]:
        """Split trainable params into standard (BN/gate/head) and optional lower-LR LoRA group."""
        if lora_lr is None or not self.use_l4_conv2_lora:
            return [{"params": self.trainable_parameters(), "lr": float(lr)}]
        lora_ids = {
            id(p)
            for m in self.backbone[0][7].modules()
            if isinstance(m, LoRAConv2d)
            for p in (m.lora_a.weight, m.lora_b.weight)
            if p.requires_grad
        }
        base_params = [p for p in self.trainable_parameters() if id(p) not in lora_ids]
        lora_params = [p for p in self.trainable_parameters() if id(p) in lora_ids]
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": float(lr)})
        if lora_params:
            groups.append({"params": lora_params, "lr": float(lora_lr)})
        return groups

    def set_repair_train_mode(self):
        """Keep all frozen BN running stats fixed in eval(); enable dropout only in gate/head."""
        self.backbone.eval()
        if self.gate is not None:
            self.gate.train()
        self.head.train()

    def _effective_gate(self, g: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "peak_neutral":
            return g / g.amax(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        return g

    def _neutral_baseline(self, c3: torch.Tensor) -> torch.Tensor:
        if bool(self.has_neutral_c3.item()):
            return self.neutral_c3.to(device=c3.device, dtype=c3.dtype)
        return c3.mean(dim=(0, 2, 3), keepdim=True).detach()

    def _apply_gate(self, c3: torch.Tensor, g_eff: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "peak_neutral":
            mu = self._neutral_baseline(c3)
            return g_eff * c3 + (1.0 - g_eff) * mu
        return g_eff * c3

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
            g_eff = self._effective_gate(self.gate(c3))
            c3 = self._apply_gate(c3, g_eff)
        logits, _, _ = self._forward_from_layer3(c3)
        return logits

    @torch.inference_mode()
    def extract_repaired_stage_maps(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        c1, c2, c3 = self._forward_to_layer3(inputs)
        if self.gate is not None:
            gate_map = self._effective_gate(self.gate(c3))
            c3_gated = self._apply_gate(c3, gate_map)
        else:
            gate_map = torch.ones((c3.shape[0], 1, c3.shape[2], c3.shape[3]), device=c3.device, dtype=c3.dtype)
            c3_gated = c3
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
        g_eff = self._effective_gate(self.gate(c3))  # [B, 1, 14, 14]
        logits, c4, _ = self._forward_from_layer3(self._apply_gate(c3, g_eff))

        # Build mask-free opposite-background donor context at layer3.5 using detached soft mask m.
        m = g_eff.detach()  # [B, 1, 14, 14]
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
        use_neutral = (torch.rand((b, 1, 1, 1), device=c3.device) < zero_context_prob).to(c3.dtype)
        fallback_ctx = self._neutral_baseline(c3) if self.gate_mode == "peak_neutral" else torch.zeros_like(donor_bg)
        ctx = (1.0 - use_neutral) * donor_bg + use_neutral * fallback_ctx
        c3_swap = m * c3 + (1.0 - m) * ctx

        # Apply the same differentiable gate to c3_swap so minimizing layer4 drift drives g_eff -> 0 on BG cells.
        logits_swap, c4_swap, _ = self._forward_from_layer3(self._apply_gate(c3_swap, g_eff))
        return {
            "logits": logits,
            "logits_swap": logits_swap,
            "layer4_map": c4,
            "layer4_map_swap": c4_swap,
            "gate_map": g_eff,
        }


@torch.inference_mode()
def compute_group_balanced_layer3_mean(
    base_resnet: nn.Module,
    loader,
    device: str,
) -> torch.Tensor:
    """Compute group-balanced mean layer3 activation [1, 1024, 1, 1] strictly from a training loader."""
    stem = copy.deepcopy(base_resnet[0]).to(device)
    stem.eval()
    group_sums: dict[str, torch.Tensor] = {}
    group_counts: dict[str, int] = {}
    for _, xb, _, gb in loader:
        xb = xb.to(device)
        x = stem[3](stem[2](stem[1](stem[0](xb))))
        c3 = stem[6](stem[5](stem[4](x)))  # [B, 1024, 14, 14]
        spatial_mean = c3.mean(dim=(2, 3)).cpu().double()  # [B, 1024]
        for i, g_str in enumerate(gb):
            key = str(g_str)
            if key not in group_sums:
                group_sums[key] = spatial_mean[i].clone()
                group_counts[key] = 1
            else:
                group_sums[key] += spatial_mean[i]
                group_counts[key] += 1
    per_group_means = [group_sums[k] / float(group_counts[k]) for k in sorted(group_sums)]
    balanced_mean = torch.stack(per_group_means, dim=0).mean(dim=0)
    return balanced_mean.float().view(1, 1024, 1, 1)


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
