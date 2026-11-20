#!/usr/bin/env python3
"""Non-JEPA hierarchical temporal contrast sidecar for CARE-CGM."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(values + self.network(values))


def masked_mean_moment(
    values: torch.Tensor, support: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or support.shape != values.shape[:2]:
        raise ValueError(
            "masked pooling expects [batch,time,channels] and [batch,time]"
        )
    weight = support.to(values.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weight.unsqueeze(-1)).sum(dim=1) / count
    time = torch.linspace(-1.0, 1.0, values.shape[1], device=values.device)
    signed = weight * time.unsqueeze(0)
    signed = signed - signed.sum(dim=1, keepdim=True) * weight / count
    denominator = signed.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
    moment = (values * signed.unsqueeze(-1)).sum(dim=1) / denominator
    return mean, moment


class HierarchicalTemporalContrastEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dimensions: int = 64,
        local_dimensions: int = 32,
        output_dimensions: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, hidden_dimensions, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, hidden_dimensions),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualDilatedBlock(hidden_dimensions, dilation, kernel_size)
            for dilation in dilations
        )
        self.local_head = nn.Linear(hidden_dimensions, local_dimensions)
        self.global_head = nn.Sequential(
            nn.Linear(2 * local_dimensions, 2 * output_dimensions),
            nn.GELU(),
            nn.Linear(2 * output_dimensions, output_dimensions),
        )
        self.decoder = nn.Conv1d(hidden_dimensions, 1, 1)

    def forward(
        self, values: torch.Tensor, visible: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if values.ndim != 2 or visible.shape != values.shape:
            raise ValueError("temporal encoder expects matching [batch,time] tensors")
        visible = visible.bool()
        inputs = torch.stack(
            [torch.where(visible, values, torch.zeros_like(values)), visible.float()],
            dim=1,
        )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        local = self.local_head(hidden.transpose(1, 2))
        mean, moment = masked_mean_moment(local, visible)
        code = self.global_head(torch.cat([mean, moment], dim=1))
        reconstruction = self.decoder(hidden).squeeze(1)
        return {
            "local": local,
            "code": code,
            "reconstruction": reconstruction,
        }


def _paired_cross_entropy(
    similarity: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    if similarity.ndim != 3 or similarity.shape[1] != similarity.shape[2]:
        raise ValueError("paired contrast expects [...,items,items]")
    items = similarity.shape[1]
    key_mask = key_valid.unsqueeze(1)
    masked = similarity.masked_fill(~key_mask, -1e4)
    diagonal = torch.diagonal(F.log_softmax(masked, dim=-1), dim1=1, dim2=2)
    valid = query_valid & key_valid
    if not valid.any():
        return similarity.sum() * 0.0
    return -diagonal[valid].mean()


def instance_temporal_contrast(
    first: torch.Tensor,
    second: torch.Tensor,
    support: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if (
        first.shape != second.shape
        or first.ndim != 3
        or support.shape != first.shape[:2]
    ):
        raise ValueError("hierarchical contrast expects paired [batch,time,channels]")
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    instance = torch.einsum("btc,ktc->tbk", first, second) / temperature
    valid_instance = support.transpose(0, 1)
    instance_loss = 0.5 * (
        _paired_cross_entropy(instance, valid_instance, valid_instance)
        + _paired_cross_entropy(
            instance.transpose(1, 2), valid_instance, valid_instance
        )
    )
    temporal = torch.einsum("btc,bsc->bts", first, second) / temperature
    temporal_loss = 0.5 * (
        _paired_cross_entropy(temporal, support, support)
        + _paired_cross_entropy(temporal.transpose(1, 2), support, support)
    )
    return 0.5 * (instance_loss + temporal_loss)


def _pool_time(
    values: torch.Tensor, support: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = support.to(values.dtype).unsqueeze(-1)
    pooled_weight = F.avg_pool1d(weight.transpose(1, 2), 2, stride=2).transpose(1, 2)
    pooled_value = F.avg_pool1d(
        (values * weight).transpose(1, 2), 2, stride=2
    ).transpose(1, 2)
    pooled_value = pooled_value / pooled_weight.clamp_min(1e-6)
    pooled_support = pooled_weight.squeeze(-1) > 0
    return pooled_value, pooled_support


def hierarchical_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    support: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    losses = []
    while first.shape[1] >= 2:
        losses.append(
            instance_temporal_contrast(first, second, support, temperature=temperature)
        )
        first, next_support = _pool_time(first, support)
        second, second_support = _pool_time(second, support)
        support = next_support & second_support
    if not losses:
        raise ValueError("hierarchical contrast needs at least two time points")
    return torch.stack(losses).mean()


def global_contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2:
        raise ValueError("global contrast expects paired nontrivial matrices")
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    similarity = first @ second.T / temperature
    labels = torch.arange(len(first), device=first.device)
    return 0.5 * (
        F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
    )


def effective_rank(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    if variance.sum() <= 0:
        return 0.0
    probability = variance / variance.sum()
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left, axis=0)
    right = np.asarray(right, dtype=np.float64) - np.mean(right, axis=0)
    with np.errstate(all="ignore"):
        cross = left.T @ right
        left_gram = left.T @ left
        right_gram = right.T @ right
    if not all(np.isfinite(value).all() for value in (cross, left_gram, right_gram)):
        raise ValueError("non-finite CKA geometry")
    denominator = np.linalg.norm(left_gram) * np.linalg.norm(right_gram)
    return float(np.square(np.linalg.norm(cross)) / max(denominator, 1e-12))
