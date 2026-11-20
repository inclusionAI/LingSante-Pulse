#!/usr/bin/env python3
"""Non-JEPA tail-process autoencoder components for the TPA-CGM canary."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LENGTH = 288
SIDES = (1, -1)
QUANTILES = (0.9, 0.95, 0.975)
VARIANTS = (
    "generic_masked_autoencoder",
    "autoencoder_with_rare_event_oversampling",
    "autoencoder_with_scalar_evt_tail_stat_head",
    "plain_unsigned_q95_tail_process_autoencoder",
    "tpa_cgm_signed_nested_cadence_full",
    "range_null_generic_transfer",
    "range_null_cgm_signed_tail",
)
LEGACY_VARIANTS = VARIANTS[:5]


def masked_window_standardize(
    raw_glucose: torch.Tensor, reference_mask: torch.Tensor
) -> torch.Tensor:
    """Standardize each window using only currently visible physical readings."""
    values = torch.nan_to_num(raw_glucose)
    weight = reference_mask.to(values.dtype) * torch.isfinite(raw_glucose).to(
        values.dtype
    )
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weight).sum(dim=1, keepdim=True) / count
    variance = ((values - mean).square() * weight).sum(dim=1, keepdim=True) / count
    return (values - mean) / torch.sqrt(variance + 1e-6)


def masked_local_lowpass(
    values: torch.Tensor, observed: torch.Tensor, kernel_size: int = 3
) -> torch.Tensor:
    """Fixed observation-aware low pass used by the factor-3 cadence decomposition."""
    if kernel_size != 3:
        raise ValueError("the frozen RangeNull-CGM low-pass kernel is factor 3")
    weight = observed.to(values.dtype)
    numerator = F.avg_pool1d(
        (values * weight).unsqueeze(1),
        kernel_size,
        stride=1,
        padding=1,
        count_include_pad=False,
    ).squeeze(1)
    denominator = F.avg_pool1d(
        weight.unsqueeze(1),
        kernel_size,
        stride=1,
        padding=1,
        count_include_pad=False,
    ).squeeze(1)
    return numerator / denominator.clamp_min(1e-6)


@dataclass(frozen=True)
class TailContract:
    center: float
    scale: float
    upper_thresholds: tuple[float, float, float]
    lower_thresholds: tuple[float, float, float]
    lag: int = 24
    minimum_local_support: float = 0.9

    def threshold(self, side_index: int, quantile_index: int) -> float:
        values = self.upper_thresholds if side_index == 0 else self.lower_thresholds
        return float(values[quantile_index])


def _interpolate(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    positions = np.arange(values.size)
    valid = observed.astype(bool) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError("each eligible CGM window needs at least two observations")
    return np.interp(positions, positions[valid], values[valid]).astype(np.float32)


def _runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, cuts) if part.size]


def fit_tail_contract(
    glucose: np.ndarray,
    observed: np.ndarray,
    train_rows: np.ndarray,
    *,
    lag: int = 24,
    minimum_local_support: float = 0.9,
) -> TailContract:
    if glucose.shape != observed.shape or glucose.ndim != 2 or glucose.shape[1] < 2:
        raise ValueError(
            "expected matching non-empty [N,T] glucose and observed arrays"
        )
    if train_rows.size == 0:
        raise ValueError("training rows are empty")
    values = glucose[train_rows]
    mask = observed[train_rows].astype(bool) & np.isfinite(values)
    train_values = values[mask].astype(np.float64)
    center = float(np.median(train_values))
    scale = float(np.std(train_values))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("training glucose scale is degenerate")

    thresholds: list[tuple[float, float, float]] = []
    for side in SIDES:
        deviation = side * (train_values - center)
        positive = deviation[deviation > 0]
        if positive.size == 0:
            raise ValueError("one signed tail has no training support")
        thresholds.append(tuple(float(np.quantile(positive, q)) for q in QUANTILES))
    return TailContract(
        center=center,
        scale=scale,
        upper_thresholds=thresholds[0],
        lower_thresholds=thresholds[1],
        lag=lag,
        minimum_local_support=minimum_local_support,
    )


def _build_tail_targets_serial(
    glucose: np.ndarray, observed: np.ndarray, contract: TailContract
) -> dict[str, np.ndarray]:
    """Build window-level signed cluster profiles and scalar tail summaries."""
    width = 2 * contract.lag + 1
    profiles = np.zeros((len(glucose), 2, 3, width), dtype=np.float32)
    severity = np.zeros((len(glucose), 2, 3), dtype=np.float32)
    valid = np.zeros((len(glucose), 2, 3), dtype=bool)
    scalar = np.zeros((len(glucose), 6), dtype=np.float32)

    sequence_length = glucose.shape[1]
    for row in range(len(glucose)):
        mask = observed[row].astype(bool) & np.isfinite(glucose[row])
        filled = _interpolate(glucose[row], mask)
        q95_summary: list[tuple[float, float, float]] = []
        for side_index, side in enumerate(SIDES):
            signed = side * (filled - contract.center)
            for quantile_index in range(3):
                threshold = contract.threshold(side_index, quantile_index)
                shapes: list[np.ndarray] = []
                excesses: list[float] = []
                lengths: list[int] = []
                exceed = np.flatnonzero(signed > threshold)
                for run in _runs(exceed):
                    anchor = int(run[np.argmax(signed[run])])
                    left = anchor - contract.lag
                    right = anchor + contract.lag + 1
                    if left < 0 or right > sequence_length:
                        continue
                    if float(mask[left:right].mean()) < contract.minimum_local_support:
                        continue
                    peak = float(signed[anchor])
                    if peak <= 0:
                        continue
                    shapes.append((signed[left:right] / peak).astype(np.float32))
                    excesses.append(float(np.log1p(max(peak - threshold, 0.0))))
                    lengths.append(int(run.size))
                if shapes:
                    profiles[row, side_index, quantile_index] = np.mean(shapes, axis=0)
                    severity[row, side_index, quantile_index] = float(np.mean(excesses))
                    valid[row, side_index, quantile_index] = True
                if quantile_index == 1:
                    q95_summary.append(
                        (
                            float(len(shapes)),
                            float(max(excesses, default=0.0)),
                            float(np.mean(lengths) if lengths else 0.0),
                        )
                    )
        if len(q95_summary) != 2:
            raise RuntimeError("signed q95 summaries were not constructed")
        scalar[row] = np.asarray(
            [
                q95_summary[0][0],
                q95_summary[1][0],
                q95_summary[0][1],
                q95_summary[1][1],
                q95_summary[0][2],
                q95_summary[1][2],
            ],
            dtype=np.float32,
        )
    return {
        "profiles": profiles,
        "severity": severity,
        "valid": valid,
        "scalar": scalar,
    }


def build_tail_targets(
    glucose: np.ndarray, observed: np.ndarray, contract: TailContract
) -> dict[str, np.ndarray]:
    """Build tail targets, optionally splitting independent rows across workers."""
    workers = int(os.environ.get("TPA_TARGET_WORKERS", "1"))
    if workers <= 1 or len(glucose) < 2 * workers:
        return _build_tail_targets_serial(glucose, observed, contract)
    indices = [
        part for part in np.array_split(np.arange(len(glucose)), workers) if len(part)
    ]
    tasks = [(glucose[part], observed[part], contract) for part in indices]
    with mp.get_context("fork").Pool(len(tasks)) as pool:
        parts = pool.starmap(_build_tail_targets_serial, tasks)
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("profiles", "severity", "valid", "scalar")
    }


def _deterministic_cadence_view_serial(
    glucose: np.ndarray,
    observed: np.ndarray,
    factor: int = 3,
    native_cadence_min: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty_like(glucose, dtype=np.float32)
    sparse_mask = np.zeros_like(observed, dtype=bool)
    for row in range(len(glucose)):
        source = observed[row].astype(bool) & np.isfinite(glucose[row])
        sparse = np.zeros(glucose.shape[1], dtype=bool)
        if native_cadence_min is not None and int(native_cadence_min[row]) > 5:
            physical = np.flatnonzero(source)
            sparse[physical[::factor]] = True
        else:
            sparse[::factor] = source[::factor]
        values[row] = _interpolate(glucose[row], sparse)
        sparse_mask[row] = sparse
    return values, sparse_mask


def deterministic_cadence_view(
    glucose: np.ndarray,
    observed: np.ndarray,
    factor: int = 3,
    native_cadence_min: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic cadence views, optionally in row-parallel chunks."""
    workers = int(os.environ.get("TPA_TARGET_WORKERS", "1"))
    if workers <= 1 or len(glucose) < 2 * workers:
        return _deterministic_cadence_view_serial(
            glucose, observed, factor, native_cadence_min
        )
    indices = [
        part for part in np.array_split(np.arange(len(glucose)), workers) if len(part)
    ]
    tasks = [
        (
            glucose[part],
            observed[part],
            factor,
            None if native_cadence_min is None else native_cadence_min[part],
        )
        for part in indices
    ]
    with mp.get_context("fork").Pool(len(tasks)) as pool:
        parts = pool.starmap(_deterministic_cadence_view_serial, tasks)
    return (
        np.concatenate([part[0] for part in parts], axis=0),
        np.concatenate([part[1] for part in parts], axis=0),
    )


def deterministic_absolute_cadence_view(
    glucose: np.ndarray,
    observed: np.ndarray,
    *,
    target_cadence_min: int = 15,
    native_cadence_min: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project aligned 5-minute-grid windows into a 15-minute observation space."""
    if target_cadence_min != 15:
        raise ValueError("the frozen RangeNull-CGM canonical cadence is 15 minutes")
    values = np.empty_like(glucose, dtype=np.float32)
    canonical_mask = np.zeros_like(observed, dtype=bool)
    for row in range(len(glucose)):
        source = observed[row].astype(bool) & np.isfinite(glucose[row])
        native = int(native_cadence_min[row])
        if native >= target_cadence_min:
            canonical = source.copy()
        else:
            if native > 5:
                raise ValueError(
                    "sub-15-minute cadence must fit the canonical 5-minute grid"
                )
            physical = np.flatnonzero(source)
            canonical = np.zeros(glucose.shape[1], dtype=bool)
            canonical[physical[:: target_cadence_min // 5]] = True
        values[row] = _interpolate(glucose[row], canonical)
        canonical_mask[row] = canonical
    return values, canonical_mask


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
        )
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(F.gelu(self.norm(values)))
        return values + self.projection(F.gelu(hidden))


class PartitionedResidualConvBlock(nn.Module):
    """Keep bulk and signed-tail channels in disjoint parameter paths."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.bulk = ResidualConvBlock(96, dilation)
        self.tail = ResidualConvBlock(32, dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.bulk(values[:, :96]), self.tail(values[:, 96:])], dim=1)


class TPACGMAutoencoder(nn.Module):
    """Shared-capacity model; variants differ only in pooling and training targets."""

    def __init__(
        self,
        *,
        profile_width: int = 49,
        channels: int = 128,
        profile_input_dim: int = 128,
        responsibility_partitioned: bool = False,
    ) -> None:
        super().__init__()
        if channels != 128:
            raise ValueError("the frozen canary export dimension is 128")
        if profile_input_dim not in (32, 128):
            raise ValueError(
                "profile decoder input must be the 32D tail or full 128D export"
            )
        if responsibility_partitioned and profile_input_dim != 32:
            raise ValueError(
                "responsibility partition requires the 32D tail profile interface"
            )
        self.profile_width = profile_width
        self.profile_input_dim = profile_input_dim
        self.responsibility_partitioned = responsibility_partitioned
        if responsibility_partitioned:
            self.bulk_input_projection = nn.Conv1d(3, 96, kernel_size=5, padding=2)
            self.tail_input_projection = nn.Conv1d(3, 32, kernel_size=5, padding=2)
            self.blocks = nn.ModuleList(
                [PartitionedResidualConvBlock(dilation) for dilation in (1, 2, 4, 8)]
            )
            self.reconstruction = nn.Conv1d(96, 1, kernel_size=1)
            self.bulk_token_norm = nn.LayerNorm(96)
            self.tail_token_norm = nn.LayerNorm(32)
        else:
            self.input_projection = nn.Conv1d(3, channels, kernel_size=5, padding=2)
            self.blocks = nn.ModuleList(
                [ResidualConvBlock(channels, dilation) for dilation in (1, 2, 4, 8)]
            )
            self.reconstruction = nn.Conv1d(channels, 1, kernel_size=1)
            self.token_norm = nn.LayerNorm(channels)
        self.side_embedding = nn.Embedding(2, 4)
        self.quantile_embedding = nn.Embedding(3, 4)
        self.profile_decoder = nn.Sequential(
            nn.Linear(profile_input_dim + 8, 128),
            nn.GELU(),
            nn.Linear(128, profile_width + 1),
        )
        self.scalar_decoder = nn.Sequential(
            nn.Linear(channels, 128), nn.GELU(), nn.Linear(128, 6)
        )

    @staticmethod
    def _pool(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weight = weights.to(tokens.dtype).unsqueeze(2)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)

    def encode(
        self,
        raw_glucose: torch.Tensor,
        normalized_glucose: torch.Tensor,
        observed: torch.Tensor,
        visible: torch.Tensor,
        *,
        variant: str,
        center: float,
        upper_q90: float,
        lower_q90: float,
        upper_q95: float,
        lower_q95: float,
        input_decomposition_mode: str = "none",
        common_raw_glucose: torch.Tensor | None = None,
        common_normalized_glucose: torch.Tensor | None = None,
        private_common_normalized_glucose: torch.Tensor | None = None,
        common_observed: torch.Tensor | None = None,
        common_visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown TPA-CGM variant: {variant}")
        visible_float = visible.to(normalized_glucose.dtype)
        model_input = torch.stack(
            [
                normalized_glucose * visible_float,
                visible_float,
                observed.to(torch.float32),
            ],
            dim=1,
        )
        if self.responsibility_partitioned:
            if input_decomposition_mode == "cadence_range_private":
                lowpass = masked_local_lowpass(normalized_glucose, visible)
                residual = (normalized_glucose - lowpass) * visible_float
                bulk_input = torch.stack(
                    [
                        lowpass * visible_float,
                        visible_float,
                        observed.to(torch.float32),
                    ],
                    dim=1,
                )
                tail_input = torch.stack(
                    [residual, visible_float, observed.to(torch.float32)], dim=1
                )
            elif input_decomposition_mode in (
                "exact_cadence_range_private",
                "physical_common_relative_private",
            ):
                if any(
                    value is None
                    for value in (
                        common_raw_glucose,
                        common_normalized_glucose,
                        common_observed,
                        common_visible,
                    )
                ):
                    raise ValueError(
                        "exact cadence decomposition needs a canonical common view"
                    )
                common_visible_float = common_visible.to(normalized_glucose.dtype)
                bulk_input = torch.stack(
                    [
                        common_normalized_glucose * common_visible_float,
                        common_visible_float,
                        common_observed.to(torch.float32),
                    ],
                    dim=1,
                )
                private_common = (
                    private_common_normalized_glucose
                    if input_decomposition_mode == "physical_common_relative_private"
                    else common_normalized_glucose
                )
                if private_common is None:
                    raise ValueError(
                        "physical common decomposition needs a relative private view"
                    )
                residual = (normalized_glucose - private_common) * visible_float
                tail_input = torch.stack(
                    [residual, visible_float, observed.to(torch.float32)], dim=1
                )
            elif input_decomposition_mode == "none":
                bulk_input = model_input
                tail_input = model_input
            else:
                raise ValueError(
                    f"unknown input decomposition mode: {input_decomposition_mode}"
                )
            hidden = torch.cat(
                [
                    self.bulk_input_projection(bulk_input),
                    self.tail_input_projection(tail_input),
                ],
                dim=1,
            )
        else:
            hidden = self.input_projection(model_input)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2)
        if self.responsibility_partitioned:
            tokens = torch.cat(
                [
                    self.bulk_token_norm(hidden[:, :, :96]),
                    self.tail_token_norm(hidden[:, :, 96:]),
                ],
                dim=2,
            )
        else:
            tokens = self.token_norm(hidden)
        uniform = observed.to(torch.float32)
        bulk_uniform = (
            common_observed.to(torch.float32)
            if input_decomposition_mode
            in (
                "exact_cadence_range_private",
                "physical_common_relative_private",
            )
            else uniform
        )

        if variant in (
            "tpa_cgm_signed_nested_cadence_full",
            "range_null_cgm_signed_tail",
        ):
            bulk = self._pool(tokens[:, :, :96], bulk_uniform)
            upper_weight = (
                F.relu(raw_glucose - float(center) - float(upper_q90)) * uniform
            )
            lower_weight = (
                F.relu(float(center) - raw_glucose - float(lower_q90)) * uniform
            )
            upper_weight = torch.where(
                upper_weight.sum(dim=1, keepdim=True) > 0, upper_weight, uniform
            )
            lower_weight = torch.where(
                lower_weight.sum(dim=1, keepdim=True) > 0, lower_weight, uniform
            )
            upper = self._pool(tokens[:, :, 96:112], upper_weight)
            lower = self._pool(tokens[:, :, 112:128], lower_weight)
            representation = torch.cat([bulk, upper, lower], dim=1)
        elif variant in (
            "plain_unsigned_q95_tail_process_autoencoder",
            "range_null_generic_transfer",
        ):
            bulk = self._pool(tokens[:, :, :96], bulk_uniform)
            deviation = torch.abs(raw_glucose - float(center))
            signed_threshold = torch.where(
                raw_glucose >= float(center),
                torch.full_like(raw_glucose, float(upper_q95)),
                torch.full_like(raw_glucose, float(lower_q95)),
            )
            tail_weight = F.relu(deviation - signed_threshold) * uniform
            tail_weight = torch.where(
                tail_weight.sum(dim=1, keepdim=True) > 0, tail_weight, uniform
            )
            tail = self._pool(tokens[:, :, 96:128], tail_weight)
            representation = torch.cat([bulk, tail], dim=1)
        else:
            representation = self._pool(tokens, uniform)
        reconstruction_tokens = (
            tokens[:, :, :96] if self.responsibility_partitioned else tokens
        )
        reconstructed = self.reconstruction(
            reconstruction_tokens.transpose(1, 2)
        ).squeeze(1)
        return representation, reconstructed

    def decode_profiles(
        self, representation: torch.Tensor, *, use_queries: bool
    ) -> torch.Tensor:
        if (
            representation.ndim != 2
            or representation.shape[1] != self.profile_input_dim
        ):
            raise ValueError(
                f"profile decoder expects [N,{self.profile_input_dim}] representation"
            )
        predictions = []
        for side_index in range(2):
            rows = []
            for quantile_index in range(3):
                if use_queries:
                    side = self.side_embedding(
                        torch.full(
                            (len(representation),),
                            side_index,
                            dtype=torch.long,
                            device=representation.device,
                        )
                    )
                    quantile = self.quantile_embedding(
                        torch.full(
                            (len(representation),),
                            quantile_index,
                            dtype=torch.long,
                            device=representation.device,
                        )
                    )
                else:
                    side = representation.new_zeros((len(representation), 4))
                    quantile = representation.new_zeros((len(representation), 4))
                rows.append(
                    self.profile_decoder(torch.cat([representation, side, quantile], 1))
                )
            predictions.append(torch.stack(rows, dim=1))
        return torch.stack(predictions, dim=1)

    def decode_scalar(self, representation: torch.Tensor) -> torch.Tensor:
        return self.scalar_decoder(representation)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
