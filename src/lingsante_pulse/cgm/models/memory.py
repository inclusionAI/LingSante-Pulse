"""Frequency-resolved physical-clock transport memory for CGM."""

from __future__ import annotations
import numpy as np, torch
from torch import nn
from lingsante_pulse.cgm.models.distribution_phase import DistributionBarycenter
from lingsante_pulse.cgm.models.hierarchical_continuous_time_day_memory import (
    AGE_SCALES,
    WIDTH,
    pack_hctdm,
)

ARCHITECTURE = "fourier_clock_transport_memory_v1"
PREREGISTERED_STEPS = (600, 6960)
HARMONICS = (1, 2, 3, 4)


def controlled_base_rows(base, mask, control):
    if not control:
        return base
    out = base.clone()
    for e in range(len(base)):
        for d in torch.nonzero(mask[e].any(-1), as_tuple=False).flatten():
            p = torch.nonzero(mask[e, d], as_tuple=False).flatten()
            if p.numel() > 2:
                order = torch.cat(
                    [
                        torch.arange(0, p.numel(), 2, device=p.device),
                        torch.arange(1, p.numel(), 2, device=p.device),
                    ]
                )
                out[e, d, p] = base[e, d, p[order]]
    return out


def spectral_rows(base, clock, mask, control=False):
    base = controlled_base_rows(base, mask, control)
    theta = torch.atan2(clock[..., 0], clock[..., 1])
    parts = []
    mass = mask.to(base.dtype).sum(-1, keepdim=True).clamp_min(1.0)
    for harmonic in HARMONICS:
        cosine = torch.cos(theta * harmonic) * mask
        sine = torch.sin(theta * harmonic) * mask
        real = (base * cosine[..., None]).sum(-2) / mass
        imag = (base * sine[..., None]).sum(-2) / mass
        parts.append(torch.cat([real, imag], -1))
    return torch.stack(parts, -2)


class FourierClockTransportMemory(DistributionBarycenter):
    def __init__(self, clock_ablation_control=False, memory_width=WIDTH):
        super().__init__(control=False)
        if memory_width < 1:
            raise ValueError("memory_width must be positive")
        self.memory_width = int(memory_width)
        self.clock_ablation_control = bool(clock_ablation_control)
        self.fctm_lift = nn.Linear(80, self.memory_width, bias=False)
        self.fctm_qkv = nn.Linear(self.memory_width, 3 * self.memory_width)
        self.fctm_output = nn.Linear(self.memory_width, self.memory_width)
        self.fctm_norm = nn.LayerNorm(self.memory_width)
        self.fctm_ff = nn.Sequential(
            nn.Linear(self.memory_width, 2 * self.memory_width),
            nn.GELU(),
            nn.Linear(2 * self.memory_width, self.memory_width),
        )
        self.fctm_writeback = nn.Linear(self.memory_width, 36, bias=False)
        nn.init.zeros_(self.fctm_writeback.weight)

    def memory_statistics(self, base, clock, point_mask, day_age, day_mask):
        spectral = spectral_rows(base, clock, point_mask, self.clock_ablation_control)
        age = torch.exp(
            -day_age[..., None] / torch.as_tensor(AGE_SCALES, device=base.device)
        )
        harmonic = (
            torch.as_tensor(HARMONICS, device=base.device, dtype=base.dtype)[
                None, None, :, None
            ]
            / 4.0
        )
        x = torch.cat(
            [spectral, spectral * harmonic]
            + [spectral * age[..., i, None, None] for i in range(3)],
            -1,
        )
        b, d, h, _ = x.shape
        x = self.fctm_lift(x.reshape(b, d * h, -1))
        valid = day_mask[..., None].expand(b, d, h).reshape(b, d * h)
        rank = day_age[..., None].expand(b, d, h).reshape(b, d * h)
        q, k, v = self.fctm_qkv(x).chunk(3, -1)
        logits = torch.einsum("bik,bjk->bij", q, k) / (self.memory_width**0.5)
        allowed = (
            valid[:, :, None]
            & valid[:, None, :]
            & (rank[:, None, :] >= rank[:, :, None])
        )
        masked = torch.where(allowed, logits, torch.full_like(logits, -1e9))
        w = torch.softmax(masked, -1) * allowed.to(logits.dtype)
        w /= w.sum(-1, keepdim=True).clamp_min(1e-8)
        x = self.fctm_norm(x + self.fctm_output(w @ v))
        x = (x + self.fctm_ff(x)) * valid[..., None].to(x.dtype)
        weight = valid.to(x.dtype) * torch.exp(-rank / 3.0)
        return (x * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(
            1e-8
        )

    def forward(
        self,
        values,
        features,
        support,
        elapsed_days,
        mask,
        path,
        nodes,
        node_index,
        hctdm_base,
        hctdm_clock,
        hctdm_point_mask,
        hctdm_day_age,
        hctdm_day_mask,
        hctdm_active,
    ):
        embedding, statistics = super().forward(
            values, features, support, elapsed_days, mask, path, nodes, node_index
        )
        memory = self.memory_statistics(
            hctdm_base, hctdm_clock, hctdm_point_mask, hctdm_day_age, hctdm_day_mask
        )
        residual = self.fctm_writeback(memory) * hctdm_active[:, None]
        return embedding + self.readout(residual), statistics + residual


def freeze_observed_state_backbone(model):
    names = []
    for name, p in model.named_parameters():
        enabled = name.startswith("fctm_")
        p.requires_grad_(enabled)
        names.append(name) if enabled else None
    return names


def parameter_budget(memory_width=WIDTH):
    m = FourierClockTransportMemory(memory_width=memory_width)
    freeze_observed_state_backbone(m)
    return {
        "total": sum(p.numel() for p in m.parameters()),
        "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
        "final_dimension": 146,
    }
