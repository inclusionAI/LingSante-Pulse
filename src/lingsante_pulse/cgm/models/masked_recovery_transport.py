#!/usr/bin/env python3
"""Non-JEPA masked encoder for CGM recovery-transport coordinates."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SymmetricResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values.transpose(1, 2)).transpose(1, 2)
        hidden = self.conv(F.gelu(normalized))
        if hidden.shape != values.shape:
            raise RuntimeError("residual block changed temporal support")
        return values + self.projection(F.gelu(hidden))


def masked_statistics(values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    weight = visible.to(values.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weight.unsqueeze(-1)).sum(dim=1) / count
    centered_values = values - mean.unsqueeze(1)
    standard_deviation = torch.sqrt(
        (centered_values.square() * weight.unsqueeze(-1)).sum(dim=1) / count + 1e-6
    )
    time = torch.linspace(-1.0, 1.0, values.shape[1], device=values.device)
    centered_time = (
        time.unsqueeze(0)
        - (time.unsqueeze(0) * weight).sum(dim=1, keepdim=True) / count
    )
    signed = centered_time * weight
    denominator = signed.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
    first_moment = (values * signed.unsqueeze(-1)).sum(dim=1) / denominator
    return torch.cat([mean, standard_deviation, first_moment], dim=1)


class MaskedRecoveryTransportEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dimensions: int = 64,
        local_dimensions: int = 32,
        output_dimensions: int = 24,
        target_dimensions: int = 17,
        dilations: tuple[int, ...] = (1, 3, 9, 27),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(2, hidden_dimensions, 1)
        self.blocks = nn.ModuleList(
            SymmetricResidualBlock(hidden_dimensions, dilation, kernel_size)
            for dilation in dilations
        )
        self.local_head = nn.Linear(hidden_dimensions, local_dimensions)
        self.global_head = nn.Sequential(
            nn.Linear(3 * local_dimensions, 2 * output_dimensions),
            nn.GELU(),
            nn.Linear(2 * output_dimensions, output_dimensions),
        )
        self.target_head = nn.Linear(output_dimensions, target_dimensions)

    def forward(
        self, values: torch.Tensor, visible: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if values.ndim != 2 or visible.shape != values.shape:
            raise ValueError("encoder expects matching [batch,time] tensors")
        visible = visible.bool()
        inputs = torch.stack(
            [torch.where(visible, values, torch.zeros_like(values)), visible.float()],
            dim=1,
        )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        local = self.local_head(hidden.transpose(1, 2))
        code = self.global_head(masked_statistics(local, visible))
        return {"code": code, "prediction": self.target_head(code)}


def vicreg_terms(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg expects paired code matrices")
    invariance = F.mse_loss(first, second)
    centered = [value - value.mean(dim=0, keepdim=True) for value in (first, second)]
    standard_deviation = 0.5 * sum(
        torch.sqrt(value.var(dim=0, unbiased=True) + 1e-4) for value in centered
    )
    variance = F.relu(1.0 - standard_deviation).mean()
    covariance = torch.zeros((), device=first.device)
    for value in centered:
        gram = value.T @ value / max(len(value) - 1, 1)
        off_diagonal = gram - torch.diag(torch.diag(gram))
        covariance = covariance + 0.5 * off_diagonal.square().sum() / first.shape[1]
    return invariance, variance, covariance


def effective_rank(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    if variance.sum() <= 0:
        return 0.0
    probability = variance[variance > 0] / variance.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left, axis=0)
    right = np.asarray(right, dtype=np.float64) - np.mean(right, axis=0)
    cross = np.einsum("ni,nj->ij", left, right, optimize=False)
    left_gram = np.einsum("ni,nj->ij", left, left, optimize=False)
    right_gram = np.einsum("ni,nj->ij", right, right, optimize=False)
    denominator = np.linalg.norm(left_gram) * np.linalg.norm(right_gram)
    return float(np.square(np.linalg.norm(cross)) / max(denominator, 1e-12))
