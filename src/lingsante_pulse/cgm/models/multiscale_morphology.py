#!/usr/bin/env python3
"""Cadence-observable multiscale morphology encoder for CARE-CGM."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lingsante_pulse.cgm.models.dynamics import AxisEncoder


LENGTH = 288
CANONICAL_STRIDE = 3
MORPHOLOGY_LAGS = (1, 2, 4, 8, 16)
MORPHOLOGY_STATISTICS = (
    "upward_energy",
    "downward_energy",
    "total_variation",
    "reversal_energy",
    "persistence_energy",
)
MORPHOLOGY_TARGET_NAMES = tuple(
    f"{statistic}_lag_{lag}"
    for lag in MORPHOLOGY_LAGS
    for statistic in MORPHOLOGY_STATISTICS
)


def multiscale_morphology_targets(
    values: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Affine-invariant signed excursion and recovery morphology."""
    output, _ = multiscale_morphology_targets_with_validity(values, observed)
    return output


def multiscale_morphology_targets_with_validity(
    values: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Morphology targets plus an explicit mask for unsupported statistics."""
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or values.shape[1] != LENGTH or observed.shape != values.shape:
        raise ValueError("expected aligned [N,288] canonical traces")
    rows = []
    validity_rows = []
    for row, mask in zip(values, observed):
        physical = row[mask]
        if len(physical) < 8:
            raise ValueError("morphology targets require at least eight observations")
        center = float(physical.mean())
        scale = float(physical.std())
        if scale <= 1e-6:
            scale = 1.0
        normalized = (row - center) / scale
        features = []
        validity = []
        for lag in MORPHOLOGY_LAGS:
            step = CANONICAL_STRIDE * lag
            pair = mask[step:] & mask[:-step]
            delta = normalized[step:][pair] - normalized[:-step][pair]
            pair_valid = len(delta) >= 2
            upward = np.maximum(delta, 0.0) if pair_valid else np.zeros(1)
            downward = np.maximum(-delta, 0.0) if pair_valid else np.zeros(1)
            triplet = mask[2 * step :] & mask[step:-step] & mask[: -2 * step]
            first = normalized[step:-step][triplet] - normalized[: -2 * step][triplet]
            second = normalized[2 * step :][triplet] - normalized[step:-step][triplet]
            triplet_valid = len(first) >= 2
            product = first * second if triplet_valid else np.zeros(1)
            features.extend(
                [
                    float(np.square(upward).mean()),
                    float(np.square(downward).mean()),
                    float(np.abs(delta).mean()),
                    float(np.maximum(-product, 0.0).mean()),
                    float(np.maximum(product, 0.0).mean()),
                ]
            )
            validity.extend(
                [pair_valid, pair_valid, pair_valid, triplet_valid, triplet_valid]
            )
        rows.append(features)
        validity_rows.append(validity)
    output = np.asarray(rows, dtype=np.float32)
    valid = np.asarray(validity_rows, dtype=bool)
    if output.shape != (len(values), len(MORPHOLOGY_TARGET_NAMES)):
        raise RuntimeError("invalid morphology target shape")
    if valid.shape != output.shape:
        raise RuntimeError("invalid morphology target validity shape")
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite morphology targets")
    return output, valid


class CAREMultiscaleMorphologySidecar(nn.Module):
    """A non-JEPA TCN trained on cadence-observable CGM morphology."""

    def __init__(self, dimensions: int = 24) -> None:
        super().__init__()
        self.encoder = AxisEncoder(dimensions)
        self.projector = nn.Sequential(
            nn.Linear(dimensions, dimensions),
            nn.GELU(),
            nn.Linear(dimensions, dimensions),
        )
        self.target_head = nn.Linear(dimensions, len(MORPHOLOGY_TARGET_NAMES))

    def forward(
        self,
        normalized: torch.Tensor,
        visible: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        code = self.encoder(normalized, visible, observed)
        projected = self.projector(code)
        return {
            "code": code,
            "projected": projected,
            "prediction": self.target_head(code),
        }


def vicreg_terms(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invariance, per-view variance and covariance terms."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg views must be equally shaped matrices")
    invariance = F.mse_loss(first, second)
    variance_terms = []
    covariance_terms = []
    for values in (first, second):
        centered = values - values.mean(dim=0, keepdim=True)
        standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=True) + 1e-4)
        variance_terms.append(F.relu(1.0 - standard_deviation).mean())
        covariance = centered.T @ centered / max(len(values) - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        covariance_terms.append(off_diagonal.square().sum() / values.shape[1])
    return (
        invariance,
        0.5 * (variance_terms[0] + variance_terms[1]),
        0.5 * (covariance_terms[0] + covariance_terms[1]),
    )
