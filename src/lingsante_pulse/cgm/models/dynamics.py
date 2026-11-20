#!/usr/bin/env python3
"""Independent time-even state and time-odd dynamics encoders for CARE-CGM."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LENGTH = 288
CANONICAL_STRIDE = 3
STATE_TARGET_NAMES = (
    "mean",
    "standard_deviation",
    "q05",
    "q25",
    "q75",
    "q95",
    "time_below_70",
    "time_above_180",
    "autocorrelation_lag_1",
    "autocorrelation_lag_4",
    "difference_energy_lag_1",
    "difference_energy_lag_4",
)
DYNAMICS_LAGS = (1, 2, 4, 8, 16)
DYNAMICS_TARGET_NAMES = tuple(
    name
    for lag in DYNAMICS_LAGS
    for name in (f"mean_signed_increment_lag_{lag}", f"signed_energy_lag_{lag}")
)
PAIR_MODES = ("exact", "interval_adaptive", "exact_short_adaptive_long")


def _adaptive_pair_bank(
    row: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every supported ordered observation pair once.

    Pair gaps stay in the canonical five-minute grid.  Enumerating unordered
    pairs makes the target exactly time-reversal equivariant while allowing
    sparse observations to contribute at their actual elapsed interval.
    """
    index = np.flatnonzero(mask)
    left_index, right_index = np.triu_indices(len(index), k=1)
    left_grid = index[left_index]
    right_grid = index[right_index]
    return row[left_grid], row[right_grid], right_grid - left_grid


def _adaptive_scale_mask(
    gap: np.ndarray, target_step: int
) -> tuple[np.ndarray, np.ndarray]:
    target_steps = np.asarray(DYNAMICS_LAGS, dtype=np.float64) * CANONICAL_STRIDE
    position = int(np.flatnonzero(target_steps == target_step)[0])
    lower = (
        np.sqrt(target_steps[position - 1] * target_step)
        if position > 0
        else 0.5 * target_step
    )
    upper = (
        np.sqrt(target_step * target_steps[position + 1])
        if position + 1 < len(target_steps)
        else 1.5 * target_step
    )
    selected = (gap >= np.ceil(lower)) & (gap <= np.floor(upper))
    scale = target_step / gap[selected].astype(np.float64)
    return selected, scale


def responsible_dynamics_coordinates(
    dynamics: torch.Tensor, target_weight: torch.Tensor
) -> torch.Tensor:
    """Target-aligned readout plus the exact target-null innovation coordinates."""
    if dynamics.ndim != 2 or target_weight.ndim != 2:
        raise ValueError("responsible dynamics expects two matrices")
    if dynamics.shape[1] != target_weight.shape[1]:
        raise ValueError("dynamics and target-head width must match")
    _u, singular, vh = torch.linalg.svd(target_weight, full_matrices=True)
    tolerance = torch.finfo(singular.dtype).eps * max(target_weight.shape) * singular[0]
    rank = int((singular > tolerance).sum().item())
    if rank != target_weight.shape[0]:
        raise ValueError("dynamics target head must have full row rank")
    target_aligned = F.linear(dynamics, target_weight, bias=None)
    target_null = dynamics @ vh[rank:].T
    output = torch.cat([target_aligned, target_null], dim=1)
    if output.shape != dynamics.shape:
        raise RuntimeError("responsible dynamics must preserve latent width")
    return output


def _observed_rows(values: np.ndarray, observed: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or values.shape[1] != LENGTH or observed.shape != values.shape:
        raise ValueError("expected aligned [N,288] physical traces")
    for row, mask in zip(values, observed):
        physical = row[mask]
        if len(physical) < 8:
            raise ValueError("multi-axis targets require at least eight observations")
        yield row, mask, physical


def physical_state_targets(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    pair_mode: str = "exact",
    return_support: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Time-reversal-even physical state, range and variability targets."""
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"unsupported pair mode: {pair_mode}")
    rows = []
    support_rows = []
    for row, mask, physical in _observed_rows(values, observed):
        mean = float(np.mean(physical))
        std = float(np.std(physical))
        q05, q25, q75, q95 = np.quantile(physical, [0.05, 0.25, 0.75, 0.95])
        extra = []
        extra_support = []
        adaptive = _adaptive_pair_bank(row, mask) if pair_mode != "exact" else None
        for lag in (1, 4):
            step = CANONICAL_STRIDE * lag
            use_adaptive = pair_mode == "interval_adaptive" or (
                pair_mode == "exact_short_adaptive_long" and lag >= 4
            )
            if not use_adaptive:
                pair = mask[step:] & mask[:-step]
                left = row[:-step][pair]
                right = row[step:][pair]
                scale = np.ones(len(left), dtype=np.float64)
            else:
                all_left, all_right, gap = adaptive
                selected, scale = _adaptive_scale_mask(gap, step)
                left = all_left[selected]
                right = all_right[selected]
            supported = len(left) >= 2
            if supported:
                left_centered = left - left.mean()
                right_centered = right - right.mean()
                denominator = float(
                    np.sqrt(
                        np.square(left_centered).sum() * np.square(right_centered).sum()
                    )
                )
                autocorrelation = (
                    float(np.dot(left_centered, right_centered) / denominator)
                    if denominator > 1e-12
                    else 0.0
                )
                difference_energy = float(np.mean(np.square((right - left) * scale)))
            else:
                autocorrelation = 0.0
                difference_energy = 0.0
            extra.extend([autocorrelation, difference_energy])
            extra_support.extend([supported, supported])
        rows.append(
            [
                mean,
                std,
                q05,
                q25,
                q75,
                q95,
                float(np.mean(physical < 70.0)),
                float(np.mean(physical > 180.0)),
                extra[0],
                extra[2],
                extra[1],
                extra[3],
            ]
        )
        support_rows.append(
            [True] * 8
            + [extra_support[0], extra_support[2], extra_support[1], extra_support[3]]
        )
    output = np.asarray(rows, dtype=np.float32)
    if (
        output.shape != (len(values), len(STATE_TARGET_NAMES))
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid physical-state targets")
    support = np.asarray(support_rows, dtype=bool)
    if support.shape != output.shape:
        raise RuntimeError("invalid physical-state target support")
    return (output, support) if return_support else output


def signed_dynamics_targets(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    glucose_scale: float,
    pair_mode: str = "exact",
    return_support: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Time-reversal-odd multiscale signed transition targets."""
    if glucose_scale <= 0:
        raise ValueError("glucose scale must be positive")
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"unsupported pair mode: {pair_mode}")
    rows = []
    support_rows = []
    for row, mask, _physical in _observed_rows(values, observed):
        features = []
        feature_support = []
        adaptive = _adaptive_pair_bank(row, mask) if pair_mode != "exact" else None
        for lag in DYNAMICS_LAGS:
            step = CANONICAL_STRIDE * lag
            use_adaptive = pair_mode == "interval_adaptive" or (
                pair_mode == "exact_short_adaptive_long" and lag >= 4
            )
            if not use_adaptive:
                pair = mask[step:] & mask[:-step]
                delta = (row[step:][pair] - row[:-step][pair]) / float(glucose_scale)
            else:
                all_left, all_right, gap = adaptive
                selected, scale = _adaptive_scale_mask(gap, step)
                delta = (
                    (all_right[selected] - all_left[selected])
                    * scale
                    / float(glucose_scale)
                )
            supported = len(delta) >= 2
            if supported:
                positive = np.square(np.maximum(delta, 0.0))
                negative = np.square(np.maximum(-delta, 0.0))
                features.extend(
                    [float(delta.mean()), float(positive.mean() - negative.mean())]
                )
            else:
                features.extend([0.0, 0.0])
            feature_support.extend([supported, supported])
        rows.append(features)
        support_rows.append(feature_support)
    output = np.asarray(rows, dtype=np.float32)
    if (
        output.shape
        != (
            len(values),
            len(DYNAMICS_TARGET_NAMES),
        )
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid signed-dynamics targets")
    support = np.asarray(support_rows, dtype=bool)
    if support.shape != output.shape:
        raise RuntimeError("invalid signed-dynamics target support")
    return (output, support) if return_support else output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=5, dilation=dilation, padding=2 * dilation
        )
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(F.gelu(self.norm(values)))
        return values + self.projection(F.gelu(hidden))


class AxisEncoder(nn.Module):
    def __init__(self, output_dimensions: int = 16, hidden_channels: int = 32) -> None:
        super().__init__()
        if hidden_channels < 4 or hidden_channels % 4:
            raise ValueError("hidden_channels must be a positive multiple of four")
        self.input_projection = nn.Conv1d(3, hidden_channels, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.token_norm = nn.LayerNorm(hidden_channels)
        self.output = nn.Linear(2 * hidden_channels, output_dimensions)

    def forward(
        self,
        normalized: torch.Tensor,
        visible: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        weight = visible.to(normalized.dtype)
        inputs = torch.stack(
            [normalized * weight, weight, observed.to(normalized.dtype)], dim=1
        )
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        tokens = self.token_norm(hidden.transpose(1, 2))
        denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (tokens * weight[:, :, None]).sum(dim=1) / denominator
        time = torch.linspace(-1.0, 1.0, tokens.shape[1], device=tokens.device)[None, :]
        center = (time * weight).sum(dim=1, keepdim=True) / denominator
        centered = time - center
        moment_denominator = (
            (centered.square() * weight).sum(dim=1, keepdim=True).clamp_min(1e-6)
        )
        moment = (tokens * (centered * weight)[:, :, None]).sum(
            dim=1
        ) / moment_denominator
        return self.output(torch.cat([mean, moment], dim=1))


class CAREMultiAxisSidecar(nn.Module):
    def __init__(
        self,
        *,
        state_dimensions: int = 16,
        dynamics_dimensions: int = 16,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.state_encoder = AxisEncoder(state_dimensions, hidden_channels)
        self.dynamics_encoder = AxisEncoder(dynamics_dimensions, hidden_channels)
        self.state_reconstruction = nn.Linear(state_dimensions, LENGTH)
        self.state_head = nn.Linear(state_dimensions, len(STATE_TARGET_NAMES))
        self.dynamics_head = nn.Linear(dynamics_dimensions, len(DYNAMICS_TARGET_NAMES))

    @staticmethod
    def _reverse(values: torch.Tensor) -> torch.Tensor:
        return torch.flip(values, dims=(1,))

    def forward(
        self,
        physical_normalized: torch.Tensor,
        relative_normalized: torch.Tensor,
        visible: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reverse_visible = self._reverse(visible)
        reverse_observed = self._reverse(observed)
        state_forward = self.state_encoder(physical_normalized, visible, observed)
        state_reverse = self.state_encoder(
            self._reverse(physical_normalized), reverse_visible, reverse_observed
        )
        dynamics_forward = self.dynamics_encoder(relative_normalized, visible, observed)
        dynamics_reverse = self.dynamics_encoder(
            self._reverse(relative_normalized), reverse_visible, reverse_observed
        )
        state = 0.5 * (state_forward + state_reverse)
        dynamics = 0.5 * (dynamics_forward - dynamics_reverse)
        return {
            "state": state,
            "dynamics": dynamics,
            "dynamics_readout": F.linear(
                dynamics, self.dynamics_head.weight, bias=None
            ),
            "state_prediction": self.state_head(state),
            "dynamics_prediction": self.dynamics_head(dynamics),
            "forward_reconstruction": self.state_reconstruction(state_forward),
            "reverse_reconstruction": self.state_reconstruction(state_reverse),
        }


def effective_rank(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    if variance.sum() <= 0:
        return 0.0
    probability = variance / variance.sum()
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))
