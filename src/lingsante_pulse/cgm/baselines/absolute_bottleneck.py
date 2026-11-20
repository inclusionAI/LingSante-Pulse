#!/usr/bin/env python3
"""Independent absolute-physiology bottleneck for a frozen GlucoFM-like encoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


PATCH = 12
PATCHES = 24
FEATURE_DIMENSIONS = 2 * PATCHES


def absolute_patch_features(glucose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if (
        glucose.shape != mask.shape
        or glucose.ndim != 2
        or glucose.shape[1] != PATCH * PATCHES
    ):
        raise ValueError("expected matching [N,288] glucose and mask arrays")
    values = glucose.astype(np.float64).reshape(-1, PATCHES, PATCH)
    observed = mask.astype(bool).reshape(-1, PATCHES, PATCH) & np.isfinite(values)
    weight = observed.astype(np.float64)
    count = weight.sum(axis=2)
    mean = (np.nan_to_num(values) * weight).sum(axis=2) / np.maximum(count, 1.0)
    variance = ((np.nan_to_num(values) - mean[:, :, None]) ** 2 * weight).sum(
        axis=2
    ) / np.maximum(count, 1.0)
    std = np.sqrt(variance + 1e-6)
    valid = count > 0
    features = np.concatenate([mean * valid, std * valid], axis=1).astype(np.float32)
    if features.shape[1] != FEATURE_DIMENSIONS or not np.isfinite(features).all():
        raise RuntimeError("invalid absolute patch features")
    return features


class AbsoluteBottleneck(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        if dimensions not in (1, 2):
            raise ValueError("absolute bottleneck dimensions must be 1 or 2")
        self.dimensions = dimensions
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIMENSIONS, 32),
            nn.GELU(),
            nn.Linear(32, dimensions),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimensions, 32),
            nn.GELU(),
            nn.Linear(32, FEATURE_DIMENSIONS),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(features)
        return bottleneck, self.decoder(bottleneck)
