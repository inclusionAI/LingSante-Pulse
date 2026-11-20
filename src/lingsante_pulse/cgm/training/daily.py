#!/usr/bin/env python3
"""Train and measure one preregistered TPA-CGM seed43 canary variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from lingsante_pulse.cgm.models.episodic_evidence import ridge_predict
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    standardize_evidence,
    typed_evidence_features,
)
from lingsante_pulse.cgm.training.source_mixture import (
    load_source_family,
    load_source_sampling_stratum,
    source_mixture_probabilities,
    source_replay_probabilities,
    SourceDrawAudit,
    WeightedStratifiedSubjectEpisodes,
)

from lingsante_pulse.cgm.models.daily import (
    QUANTILES,
    VARIANTS,
    TPACGMAutoencoder,
    build_tail_targets,
    deterministic_absolute_cadence_view,
    deterministic_cadence_view,
    fit_tail_contract,
    masked_local_lowpass,
    masked_window_standardize,
    trainable_parameter_count,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum() / weight.expand_as(values).sum().clamp_min(1.0)


def _corrupt(
    observed: torch.Tensor, generator: torch.Generator, rate: float
) -> torch.Tensor:
    keep = (
        torch.rand(observed.shape, generator=generator, device=observed.device) >= rate
    )
    visible = observed & keep
    empty = visible.sum(dim=1) < 2
    if empty.any():
        visible[empty] = observed[empty]
    return visible


def _select_corpus_rows(
    cadence: np.ndarray,
    coverage: np.ndarray,
    split: np.ndarray,
    *,
    training_cadences: tuple[int, ...],
    mechanism_cadence: int,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cadence_adjusted_coverage = (
        coverage * np.maximum(cadence.astype(np.float64), 5.0) / 5.0
    )
    eligible = np.isin(cadence, training_cadences) & (
        cadence_adjusted_coverage >= minimum_coverage
    )
    filtered_cadence = cadence[eligible]
    filtered_split = split[eligible]
    optimization_train = np.flatnonzero(filtered_split == 0)
    mechanism_train = np.flatnonzero(
        (filtered_split == 0) & (filtered_cadence == mechanism_cadence)
    )
    mechanism_validation = np.flatnonzero(
        (filtered_split == 1) & (filtered_cadence == mechanism_cadence)
    )
    if (
        not len(optimization_train)
        or not len(mechanism_train)
        or not len(mechanism_validation)
    ):
        raise ValueError("frozen training/mechanism row selection is empty")
    return eligible, optimization_train, mechanism_train, mechanism_validation


def _cadence_sampling_probability(
    cadence: np.ndarray,
    rows: np.ndarray,
    *,
    native_15min_probability: float | None,
) -> np.ndarray:
    probability = np.ones(len(rows), dtype=np.float64)
    if native_15min_probability is None:
        return probability / probability.sum()
    if not 0.0 < native_15min_probability < 1.0:
        raise ValueError("native 15-minute sampling probability must be in (0,1)")
    is_15min = cadence[rows] == 15
    if not is_15min.any() or is_15min.all():
        raise ValueError("15-minute sampling requires both 15-minute and other rows")
    probability[is_15min] = native_15min_probability / is_15min.sum()
    probability[~is_15min] = (1.0 - native_15min_probability) / (~is_15min).sum()
    return probability / probability.sum()


def _effective_rank(embedding: np.ndarray) -> float:
    centered = embedding.astype(np.float64) - embedding.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigen = np.square(singular)
    return float(eigen.sum() ** 2 / max(float(np.square(eigen).sum()), 1e-12))


def _pc1_absolute_correlation(embedding: np.ndarray, target: np.ndarray) -> float:
    centered = embedding.astype(np.float64) - embedding.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    score = centered @ right[0]
    target = target.astype(np.float64) - float(np.mean(target))
    denominator = float(
        np.sqrt(np.mean(np.square(score))) * np.sqrt(np.mean(np.square(target)))
    )
    return abs(float(np.mean(score * target) / max(denominator, 1e-12)))


def _ridge_nmse(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_valid: np.ndarray,
    validation_valid: np.ndarray,
    *,
    ridge: float,
) -> float:
    train_x = train_x[train_valid].astype(np.float64)
    validation_x = validation_x[validation_valid].astype(np.float64)
    train_y = train_y[train_valid].astype(np.float64)
    validation_y = validation_y[validation_valid].astype(np.float64)
    if len(train_x) < 8 or len(validation_x) < 4:
        return math.inf
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_design = np.concatenate(
        [(train_x - mean) / scale, np.ones((len(train_x), 1))], axis=1
    )
    validation_design = np.concatenate(
        [(validation_x - mean) / scale, np.ones((len(validation_x), 1))], axis=1
    )
    penalty = np.eye(train_design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(
        train_design.T @ train_design + penalty, train_design.T @ train_y
    )
    prediction = validation_design @ coefficient
    numerator = float(np.mean(np.square(prediction - validation_y)))
    baseline = train_y.mean(axis=0, keepdims=True)
    denominator = float(np.mean(np.square(validation_y - baseline)))
    return numerator / max(denominator, 1e-12)


def _nested_profile_loss(
    prediction: torch.Tensor,
    profiles: torch.Tensor,
    target_valid: torch.Tensor,
    offcenter: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    adjacent_valid = target_valid[:, :, 1:] & target_valid[:, :, :-1]
    predicted_contrast = (
        prediction[:, :, 1:, : profiles.shape[-1]]
        - prediction[:, :, :-1, : profiles.shape[-1]]
    )
    if mode == "profile_equality":
        error = predicted_contrast.square()
    elif mode == "target_contrast":
        target_contrast = profiles[:, :, 1:] - profiles[:, :, :-1]
        error = (predicted_contrast - target_contrast).square()
    else:
        raise ValueError(f"unknown nested loss mode: {mode}")
    return _masked_mean(error[..., offcenter], adjacent_valid)


def _cadence_consistency_loss(
    native_tail: torch.Tensor,
    cadence_tail: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "global_tail":
        native = F.normalize(native_tail, dim=1)
        cadence = F.normalize(cadence_tail, dim=1)
    elif mode == "signed_sides":
        if native_tail.shape[1] != 32 or cadence_tail.shape[1] != 32:
            raise ValueError("signed-side cadence consistency requires a 32D tail")
        native = F.normalize(native_tail.reshape(-1, 2, 16), dim=2)
        cadence = F.normalize(cadence_tail.reshape(-1, 2, 16), dim=2)
    elif mode == "bulk_range":
        if native_tail.shape[1] != 96 or cadence_tail.shape[1] != 96:
            raise ValueError("bulk-range cadence consistency requires a 96D branch")
        native = F.normalize(native_tail, dim=1)
        cadence = F.normalize(cadence_tail, dim=1)
    else:
        raise ValueError(f"unknown cadence loss mode: {mode}")
    return (native - cadence).square().mean()


def _tail_only_scalar_input(representation: torch.Tensor, *, mode: str) -> torch.Tensor:
    output = torch.zeros_like(representation)
    if mode == "all_tail":
        output[:, 96:] = representation[:, 96:]
    elif mode == "reserved_signed_moments":
        output[:, 96:99] = representation[:, 96:99]
        output[:, 112:115] = representation[:, 112:115]
    else:
        raise ValueError(f"unknown full scalar input mode: {mode}")
    return output


def _multiview_scalar_loss(
    native_prediction: torch.Tensor,
    cadence_prediction: torch.Tensor,
    native_target: torch.Tensor,
    *,
    cadence_target: torch.Tensor | None = None,
    mode: str,
) -> torch.Tensor:
    native = F.mse_loss(native_prediction, native_target)
    if mode == "native_only":
        return native
    if mode == "native_and_cadence":
        return 0.5 * (native + F.mse_loss(cadence_prediction, native_target))
    if mode == "native_and_cadence_observation_transported":
        if cadence_target is None:
            raise ValueError(
                "observation-transported scalar loss needs a cadence target"
            )
        return 0.5 * (native + F.mse_loss(cadence_prediction, cadence_target))
    raise ValueError(f"unknown full scalar view mode: {mode}")


def _full_scalar_head_active(
    enabled: bool, step: int, active_steps: int | None
) -> bool:
    if not enabled:
        return False
    if active_steps is None:
        return True
    if active_steps < 1:
        raise ValueError("full scalar head active steps must be positive")
    return step < active_steps


def _batch_loss(
    model: TPACGMAutoencoder,
    *,
    variant: str,
    raw: torch.Tensor,
    observed: torch.Tensor,
    cadence_raw: torch.Tensor,
    cadence_observed: torch.Tensor,
    profiles: torch.Tensor,
    severity: torch.Tensor,
    target_valid: torch.Tensor,
    scalar_target: torch.Tensor,
    cadence_scalar_target: torch.Tensor | None,
    center: float,
    scale: float,
    thresholds: dict[str, float],
    corruption_rate: float,
    weights: dict[str, float],
    generator: torch.Generator,
    input_normalization: str,
    nested_loss_mode: str,
    cadence_loss_mode: str,
    full_scalar_head: bool,
    full_scalar_input_mode: str,
    full_scalar_view_mode: str,
    input_decomposition_mode: str,
    bulk_reconstruction_target_mode: str,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    visible = _corrupt(observed, generator, corruption_rate)
    cadence_visible = _corrupt(cadence_observed, generator, corruption_rate)
    if input_normalization == "global_train_center_scale":
        normalized = (raw - center) / scale
        cadence_normalized = (cadence_raw - center) / scale
    elif input_normalization == "window_visible_mean_std":
        normalized = masked_window_standardize(raw, visible)
        cadence_normalized = masked_window_standardize(cadence_raw, cadence_visible)
    else:
        raise ValueError(f"unknown input normalization: {input_normalization}")
    arguments = {
        "variant": variant,
        "center": center,
        "upper_q90": thresholds["upper_q90"],
        "lower_q90": thresholds["lower_q90"],
        "upper_q95": thresholds["upper_q95"],
        "lower_q95": thresholds["lower_q95"],
        "input_decomposition_mode": input_decomposition_mode,
    }
    if input_decomposition_mode in (
        "exact_cadence_range_private",
        "physical_common_relative_private",
    ):
        # P=U∘D must use one coordinate system: derive both views' scale from
        # the canonical cadence observations, then route x-Px only to private.
        normalized = masked_window_standardize(raw, cadence_visible)
        cadence_relative = masked_window_standardize(cadence_raw, cadence_visible)
        cadence_normalized = cadence_relative
        common_normalized = (
            (cadence_raw - center) / scale
            if input_decomposition_mode == "physical_common_relative_private"
            else cadence_relative
        )
        arguments.update(
            common_raw_glucose=cadence_raw,
            common_normalized_glucose=common_normalized,
            private_common_normalized_glucose=cadence_relative,
            common_observed=cadence_observed,
            common_visible=cadence_visible,
        )
    representation, reconstruction = model.encode(
        raw, normalized, observed, visible, **arguments
    )
    cadence_representation, cadence_reconstruction = model.encode(
        cadence_raw,
        cadence_normalized,
        cadence_observed,
        cadence_visible,
        **arguments,
    )
    hidden = observed & ~visible
    cadence_hidden = cadence_observed & ~cadence_visible
    if bulk_reconstruction_target_mode == "native_normalized":
        bulk_target = normalized
        cadence_bulk_target = cadence_normalized
    elif bulk_reconstruction_target_mode == "cadence_lowpass":
        bulk_target = masked_local_lowpass(normalized, observed)
        cadence_bulk_target = masked_local_lowpass(cadence_normalized, cadence_observed)
    elif bulk_reconstruction_target_mode == "exact_cadence_range":
        bulk_target = cadence_normalized
        cadence_bulk_target = cadence_normalized
        hidden = cadence_hidden
    elif bulk_reconstruction_target_mode == "physical_cadence_range":
        physical_common = (cadence_raw - center) / scale
        bulk_target = physical_common
        cadence_bulk_target = physical_common
        hidden = cadence_hidden
    else:
        raise ValueError(
            f"unknown bulk reconstruction target: {bulk_reconstruction_target_mode}"
        )
    bulk = _masked_mean((reconstruction - bulk_target).square(), hidden)
    bulk = 0.5 * (
        bulk
        + _masked_mean(
            (cadence_reconstruction - cadence_bulk_target).square(), cadence_hidden
        )
    )
    total = weights["bulk"] * bulk
    terms = {
        "bulk": float(bulk.detach().cpu()),
        "shape": 0.0,
        "severity": 0.0,
        "nested": 0.0,
        "cadence": 0.0,
        "scalar": 0.0,
    }
    offcenter = torch.ones(model.profile_width, dtype=torch.bool, device=raw.device)
    offcenter[model.profile_width // 2] = False
    profile_representation = (
        representation[:, 96:] if model.profile_input_dim == 32 else representation
    )

    if variant in (
        "tpa_cgm_signed_nested_cadence_full",
        "range_null_cgm_signed_tail",
    ):
        prediction = model.decode_profiles(profile_representation, use_queries=True)
        profile_error = (prediction[..., : model.profile_width] - profiles).square()[
            ..., offcenter
        ]
        shape = _masked_mean(profile_error, target_valid)
        severity_loss = _masked_mean(
            (prediction[..., model.profile_width] - severity).square(), target_valid
        )
        nested = _nested_profile_loss(
            prediction,
            profiles,
            target_valid,
            offcenter,
            mode=nested_loss_mode,
        )
        cadence_slice = (
            slice(0, 96) if cadence_loss_mode == "bulk_range" else slice(96, 128)
        )
        cadence = _cadence_consistency_loss(
            representation[:, cadence_slice],
            cadence_representation[:, cadence_slice],
            mode=cadence_loss_mode,
        )
        total = total + (
            weights["shape"] * shape
            + weights["severity"] * severity_loss
            + weights["nested"] * nested
            + weights["cadence"] * cadence
        )
        if full_scalar_head:
            native_scalar = model.decode_scalar(
                _tail_only_scalar_input(representation, mode=full_scalar_input_mode)
            )
            cadence_scalar = model.decode_scalar(
                _tail_only_scalar_input(
                    cadence_representation, mode=full_scalar_input_mode
                )
            )
            scalar = _multiview_scalar_loss(
                native_scalar,
                cadence_scalar,
                scalar_target,
                cadence_target=cadence_scalar_target,
                mode=full_scalar_view_mode,
            )
            total = total + weights["scalar"] * scalar
            terms["scalar"] = float(scalar.detach().cpu())
        terms.update(
            shape=float(shape.detach().cpu()),
            severity=float(severity_loss.detach().cpu()),
            nested=float(nested.detach().cpu()),
            cadence=float(cadence.detach().cpu()),
        )
    elif variant in (
        "plain_unsigned_q95_tail_process_autoencoder",
        "range_null_generic_transfer",
    ):
        prediction = model.decode_profiles(profile_representation, use_queries=False)[
            :, 0, 1
        ]
        q95_valid = target_valid[:, :, 1]
        divisor = q95_valid.sum(dim=1).clamp_min(1).to(profiles.dtype)
        unsigned_profile = (
            profiles[:, :, 1] * q95_valid[:, :, None].to(profiles.dtype)
        ).sum(dim=1) / divisor[:, None]
        unsigned_severity = (severity[:, :, 1] * q95_valid.to(severity.dtype)).sum(
            dim=1
        ) / divisor
        any_valid = q95_valid.any(dim=1)
        shape = _masked_mean(
            (prediction[:, : model.profile_width] - unsigned_profile).square()[
                :, offcenter
            ],
            any_valid,
        )
        severity_loss = _masked_mean(
            (prediction[:, model.profile_width] - unsigned_severity).square(), any_valid
        )
        total = total + weights["shape"] * shape + weights["severity"] * severity_loss
        if variant == "range_null_generic_transfer":
            cadence = _cadence_consistency_loss(
                representation[:, :96],
                cadence_representation[:, :96],
                mode="bulk_range",
            )
            total = total + weights["cadence"] * cadence
            terms["cadence"] = float(cadence.detach().cpu())
        terms.update(
            shape=float(shape.detach().cpu()),
            severity=float(severity_loss.detach().cpu()),
        )
    elif variant == "autoencoder_with_scalar_evt_tail_stat_head":
        scalar = F.mse_loss(model.decode_scalar(representation), scalar_target)
        total = total + weights["scalar"] * scalar
        terms["scalar"] = float(scalar.detach().cpu())
    terms["total"] = float(total.detach().cpu())
    return total, terms, representation


def _daily_evidence_targets(
    glucose: np.ndarray,
    observed: np.ndarray,
    cadence: np.ndarray,
    starts: np.ndarray,
    train_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the exact four-family, 64-coordinate current-day evidence bank."""
    evidence, _ = typed_evidence_features(glucose, observed, cadence, starts)
    normalized, contract = standardize_evidence(evidence, train_rows)
    normalized = normalized.reshape(len(normalized), -1)
    if normalized.shape[1] != 64:
        raise ValueError("Daily evidence bank must contain four 16D families")
    return normalized, contract["center"], contract["scale"]


def _daily_evidence_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[1] != 64:
        raise ValueError("Daily evidence prediction must be [N,64]")
    return (prediction - target).square().mean()


def _clean_teacher_representation(
    model: TPACGMAutoencoder,
    *,
    variant: str,
    raw: torch.Tensor,
    observed: torch.Tensor,
    cadence_raw: torch.Tensor,
    cadence_observed: torch.Tensor,
    center: float,
    scale: float,
    thresholds: dict[str, float],
    input_normalization: str,
    input_decomposition_mode: str,
) -> torch.Tensor:
    """Encode the complete observed view as a fixed, label-free coordinate target."""
    if input_normalization == "global_train_center_scale":
        normalized = (raw - center) / scale
    elif input_normalization == "window_visible_mean_std":
        normalized = masked_window_standardize(raw, observed)
    else:
        raise ValueError(f"unknown input normalization: {input_normalization}")
    arguments = {
        "variant": variant,
        "center": center,
        "upper_q90": thresholds["upper_q90"],
        "lower_q90": thresholds["lower_q90"],
        "upper_q95": thresholds["upper_q95"],
        "lower_q95": thresholds["lower_q95"],
        "input_decomposition_mode": input_decomposition_mode,
    }
    if input_decomposition_mode in (
        "exact_cadence_range_private",
        "physical_common_relative_private",
    ):
        cadence_relative = masked_window_standardize(cadence_raw, cadence_observed)
        normalized = masked_window_standardize(raw, cadence_observed)
        common_normalized = (
            (cadence_raw - center) / scale
            if input_decomposition_mode == "physical_common_relative_private"
            else cadence_relative
        )
        arguments.update(
            common_raw_glucose=cadence_raw,
            common_normalized_glucose=common_normalized,
            private_common_normalized_glucose=cadence_relative,
            common_observed=cadence_observed,
            common_visible=cadence_observed,
        )
    representation, _ = model.encode(raw, normalized, observed, observed, **arguments)
    return representation


def _extract(
    model: TPACGMAutoencoder,
    raw: np.ndarray,
    observed: np.ndarray,
    cadence_raw: np.ndarray,
    cadence_observed: np.ndarray,
    rows: np.ndarray,
    *,
    variant: str,
    center: float,
    scale: float,
    thresholds: dict[str, float],
    batch_size: int,
    device: torch.device,
    input_normalization: str,
    input_decomposition_mode: str,
    bulk_reconstruction_target_mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    native_embeddings: list[np.ndarray] = []
    cadence_embeddings: list[np.ndarray] = []
    squared_error = 0.0
    squared_baseline = 0.0
    model.eval()
    arguments = {
        "variant": variant,
        "center": center,
        "upper_q90": thresholds["upper_q90"],
        "lower_q90": thresholds["lower_q90"],
        "upper_q95": thresholds["upper_q95"],
        "lower_q95": thresholds["lower_q95"],
        "input_decomposition_mode": input_decomposition_mode,
    }
    with torch.no_grad():
        for left in range(0, len(rows), batch_size):
            index = rows[left : left + batch_size]
            batch_raw = torch.from_numpy(raw[index]).to(device)
            batch_observed = torch.from_numpy(observed[index]).to(device)
            batch_cadence_raw = torch.from_numpy(cadence_raw[index]).to(device)
            batch_cadence_observed = torch.from_numpy(cadence_observed[index]).to(
                device
            )
            if input_normalization == "global_train_center_scale":
                normalized = (batch_raw - center) / scale
                cadence_normalized = (batch_cadence_raw - center) / scale
            elif input_normalization == "window_visible_mean_std":
                normalized = masked_window_standardize(batch_raw, batch_observed)
                cadence_normalized = masked_window_standardize(
                    batch_cadence_raw, batch_cadence_observed
                )
            else:
                raise ValueError(f"unknown input normalization: {input_normalization}")
            if input_decomposition_mode in (
                "exact_cadence_range_private",
                "physical_common_relative_private",
            ):
                normalized = masked_window_standardize(
                    batch_raw, batch_cadence_observed
                )
                cadence_relative = masked_window_standardize(
                    batch_cadence_raw, batch_cadence_observed
                )
                cadence_normalized = cadence_relative
                common_normalized = (
                    (batch_cadence_raw - center) / scale
                    if input_decomposition_mode == "physical_common_relative_private"
                    else cadence_relative
                )
            representation, reconstruction = model.encode(
                batch_raw,
                normalized,
                batch_observed,
                batch_observed,
                common_raw_glucose=batch_cadence_raw
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                common_normalized_glucose=common_normalized
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                private_common_normalized_glucose=cadence_relative
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                common_observed=batch_cadence_observed
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                common_visible=batch_cadence_observed
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                **arguments,
            )
            cadence_representation, _ = model.encode(
                batch_cadence_raw,
                cadence_normalized,
                batch_cadence_observed,
                batch_cadence_observed,
                common_raw_glucose=batch_cadence_raw
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                common_normalized_glucose=common_normalized
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                private_common_normalized_glucose=cadence_relative
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                common_observed=batch_cadence_observed
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                common_visible=batch_cadence_observed
                if input_decomposition_mode
                in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                )
                else None,
                **arguments,
            )
            valid = batch_observed.to(normalized.dtype)
            if bulk_reconstruction_target_mode == "native_normalized":
                reconstruction_target = normalized
            elif bulk_reconstruction_target_mode == "cadence_lowpass":
                reconstruction_target = masked_local_lowpass(normalized, batch_observed)
            elif bulk_reconstruction_target_mode == "exact_cadence_range":
                reconstruction_target = cadence_normalized
                valid = batch_cadence_observed.to(normalized.dtype)
            elif bulk_reconstruction_target_mode == "physical_cadence_range":
                reconstruction_target = (batch_cadence_raw - center) / scale
                valid = batch_cadence_observed.to(normalized.dtype)
            else:
                raise ValueError(
                    f"unknown bulk reconstruction target: {bulk_reconstruction_target_mode}"
                )
            squared_error += float(
                ((reconstruction - reconstruction_target).square() * valid).sum().cpu()
            )
            squared_baseline += float(
                (reconstruction_target.square() * valid).sum().cpu()
            )
            native_embeddings.append(representation.cpu().numpy())
            cadence_embeddings.append(cadence_representation.cpu().numpy())
    return (
        np.concatenate(native_embeddings),
        np.concatenate(cadence_embeddings),
        squared_error / max(squared_baseline, 1e-12),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--model-init-seed",
        type=int,
        help="Optional diagnostic override for random parameter initialization.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="Optional diagnostic override for row sampling and input corruption.",
    )
    parser.add_argument(
        "--daily-evidence-reader",
        choices=("none", "persistent", "fresh"),
        default="none",
        help="Optional matched daily evidence objective; default preserves legacy training.",
    )
    parser.add_argument(
        "--daily-coordinate-teacher",
        type=Path,
        help="Optional fixed unlabeled Stage-A checkpoint defining canonical output coordinates.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seed = int(config["seed"])
    steps = int(config["max_optimizer_steps"])
    training = config["training"]
    input_normalization = training.get(
        "input_normalization", "global_train_center_scale"
    )
    nested_loss_mode = training.get("nested_loss_mode", "profile_equality")
    cadence_loss_mode = training.get("cadence_loss_mode", "global_tail")
    full_scalar_head = bool(training.get("full_scalar_head", False))
    full_scalar_input_mode = training.get("full_scalar_input_mode", "all_tail")
    full_scalar_view_mode = training.get("full_scalar_view_mode", "native_only")
    input_decomposition_mode = training.get("input_decomposition_mode", "none")
    bulk_reconstruction_target_mode = training.get(
        "bulk_reconstruction_target_mode", "native_normalized"
    )
    cadence_projector_mode = training.get("cadence_projector_mode", "relative_factor3")
    full_scalar_head_steps = training.get("full_scalar_head_steps")
    if full_scalar_head_steps is not None:
        full_scalar_head_steps = int(full_scalar_head_steps)
    profile_input_dim = int(training.get("profile_decoder_input_dim", 128))
    responsibility_partitioned = bool(training.get("responsibility_partitioned", False))
    model_init_seed = seed if args.model_init_seed is None else args.model_init_seed
    sampling_seed = seed if args.sampling_seed is None else args.sampling_seed
    np.random.seed(model_init_seed)
    torch.manual_seed(model_init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    with np.load(args.npz, allow_pickle=False) as archive:
        glucose = archive["glucose_mg_dl"].astype(np.float32)
        observed = archive["observed_mask"].astype(bool)
        split = archive["split"].astype(np.int64)
        cadence = archive["native_cadence_min"].astype(np.int64)
        coverage = archive["coverage"].astype(np.float32)
        subjects = (
            archive["subject_id"].astype(str)
            if args.daily_evidence_reader != "none"
            else None
        )
        starts = (
            archive["shifted_window_start"].astype("datetime64[m]")
            if args.daily_evidence_reader != "none"
            else None
        )
    if set(np.unique(split)) - {0, 1}:
        raise ValueError("TPA canary requires a public-only split0/1 artifact")
    training_cadences = tuple(
        int(value) for value in training.get("native_cadences", [5])
    )
    mechanism_cadence = int(training.get("mechanism_evaluation_native_cadence", 5))
    eligible, train_rows, mechanism_train_rows, validation_rows = _select_corpus_rows(
        cadence,
        coverage,
        split,
        training_cadences=training_cadences,
        mechanism_cadence=mechanism_cadence,
        minimum_coverage=float(training["minimum_window_coverage"]),
    )
    glucose = glucose[eligible]
    observed = observed[eligible]
    split = split[eligible]
    cadence = cadence[eligible]
    if subjects is not None:
        subjects = subjects[eligible]
        starts = starts[eligible]
    pbq_probability = training.get("pbq_sampling_probability")
    source_families = (
        None
        if pbq_probability is None
        else load_source_family(args.npz, len(eligible))[eligible]
    )
    replay_stratum_key = training.get("source_replay_stratum_array")
    anchor_replay_probability = training.get("anchor_replay_probability")
    if (replay_stratum_key is None) != (anchor_replay_probability is None):
        raise ValueError(
            "Replay strata and anchor probability must be configured together"
        )
    source_sampling_stratum = None
    if replay_stratum_key is not None:
        if source_families is None:
            raise ValueError("Replay weighting requires explicit source mixing")
        source_sampling_stratum = load_source_sampling_stratum(
            args.npz, len(eligible), str(replay_stratum_key)
        )[eligible]
    source_audit = (
        None
        if source_families is None
        else SourceDrawAudit(source_families, source_sampling_stratum)
    )
    contract = fit_tail_contract(
        glucose,
        observed,
        mechanism_train_rows,
        lag=int(training["tail_lag_grid_steps"]),
        minimum_local_support=float(training["minimum_local_support"]),
    )
    targets = build_tail_targets(glucose, observed, contract)
    daily_evidence = None
    daily_evidence_center = None
    daily_evidence_scale = None
    if args.daily_evidence_reader != "none":
        (daily_evidence, daily_evidence_center, daily_evidence_scale) = (
            _daily_evidence_targets(glucose, observed, cadence, starts, train_rows)
        )
    if cadence_projector_mode == "relative_factor3":
        cadence_glucose, cadence_observed = deterministic_cadence_view(
            glucose, observed, native_cadence_min=cadence
        )
    elif cadence_projector_mode == "absolute_15min":
        cadence_glucose, cadence_observed = deterministic_absolute_cadence_view(
            glucose,
            observed,
            target_cadence_min=15,
            native_cadence_min=cadence,
        )
    else:
        raise ValueError(f"unknown cadence projector mode: {cadence_projector_mode}")
    scalar_mean = targets["scalar"][train_rows].mean(axis=0)
    scalar_scale = targets["scalar"][train_rows].std(axis=0)
    scalar_scale[scalar_scale < 1e-6] = 1.0
    normalized_scalar = ((targets["scalar"] - scalar_mean) / scalar_scale).astype(
        np.float32
    )
    normalized_cadence_scalar: np.ndarray | None = None
    if full_scalar_view_mode == "native_and_cadence_observation_transported":
        completed_contract = replace(contract, minimum_local_support=1.0)
        cadence_targets = build_tail_targets(
            cadence_glucose,
            np.ones_like(cadence_observed, dtype=bool),
            completed_contract,
        )
        normalized_cadence_scalar = (
            (cadence_targets["scalar"] - scalar_mean) / scalar_scale
        ).astype(np.float32)
    thresholds = {
        "upper_q90": contract.upper_thresholds[0],
        "upper_q95": contract.upper_thresholds[1],
        "lower_q90": contract.lower_thresholds[0],
        "lower_q95": contract.lower_thresholds[1],
    }

    model = TPACGMAutoencoder(
        profile_width=2 * contract.lag + 1,
        profile_input_dim=profile_input_dim,
        responsibility_partitioned=responsibility_partitioned,
    ).to(device)
    persistent_evidence_reader = None
    if args.daily_evidence_reader == "persistent":
        persistent_evidence_reader = torch.nn.Linear(128, daily_evidence.shape[1]).to(
            device
        )
    coordinate_teacher = None
    coordinate_teacher_sha256 = None
    if args.daily_coordinate_teacher is not None:
        if args.daily_evidence_reader != "none":
            raise ValueError(
                "Coordinate teacher and daily evidence reader are separate arms"
            )
        teacher_checkpoint = torch.load(
            args.daily_coordinate_teacher, map_location="cpu", weights_only=False
        )
        coordinate_teacher = TPACGMAutoencoder(
            profile_width=2 * contract.lag + 1,
            profile_input_dim=profile_input_dim,
            responsibility_partitioned=responsibility_partitioned,
        ).to(device)
        coordinate_teacher.load_state_dict(teacher_checkpoint["model"], strict=True)
        coordinate_teacher.eval()
        coordinate_teacher.requires_grad_(False)
        coordinate_teacher_sha256 = sha256(args.daily_coordinate_teacher)
    init_checkpoint_sha256 = None
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(args.init_checkpoint)
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial["model"], strict=True)
        init_checkpoint_sha256 = sha256(args.init_checkpoint)
    optimizer_parameters = list(model.parameters())
    if persistent_evidence_reader is not None:
        optimizer_parameters.extend(persistent_evidence_reader.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(sampling_seed)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(sampling_seed)
    rare = targets["valid"][:, :, 1].any(axis=1)
    cadence_probability = _cadence_sampling_probability(
        cadence,
        train_rows,
        native_15min_probability=training.get("native_15min_sampling_probability"),
    )
    rare_probability = cadence_probability.copy()
    rare_probability *= (
        1.0
        + (float(training["rare_event_sampling_multiplier"]) - 1.0) * rare[train_rows]
    )
    rare_probability /= rare_probability.sum()
    if source_families is not None:
        if source_sampling_stratum is None:
            cadence_probability = source_mixture_probabilities(
                cadence_probability, source_families[train_rows], pbq_probability
            )
            rare_probability = source_mixture_probabilities(
                rare_probability, source_families[train_rows], pbq_probability
            )
        else:
            cadence_probability = source_replay_probabilities(
                cadence_probability,
                source_families[train_rows],
                pbq_probability,
                source_sampling_stratum[train_rows],
                anchor_replay_probability,
            )
            rare_probability = source_replay_probabilities(
                rare_probability,
                source_families[train_rows],
                pbq_probability,
                source_sampling_stratum[train_rows],
                anchor_replay_probability,
            )
    daily_episodes = None
    if args.daily_evidence_reader != "none":
        episode_strata = (
            source_sampling_stratum
            if source_sampling_stratum is not None
            else source_families
            if source_families is not None
            else np.repeat("all", len(glucose))
        )
        daily_episodes = WeightedStratifiedSubjectEpisodes(
            subjects,
            train_rows,
            episode_strata,
            cadence_probability,
            batch_size=int(training["batch_size"]) // 2,
            seed=sampling_seed,
        )
    batch_weights = dict(training["loss_weights"])
    daily_evidence_weight = float(training.get("daily_evidence_weight", 1.0))
    daily_coordinate_weight = float(training.get("daily_coordinate_weight", 1.0))
    history: list[dict[str, float | int]] = []
    started = time.monotonic()

    for step in range(steps):
        support_size = 0
        if daily_episodes is not None:
            support_rows, query_rows = daily_episodes.next()
            support_size = len(support_rows)
            chosen = np.concatenate([support_rows, query_rows])
        elif args.variant == "autoencoder_with_rare_event_oversampling":
            chosen = rng.choice(
                train_rows,
                size=int(training["batch_size"]),
                replace=True,
                p=rare_probability,
            )
        else:
            chosen = rng.choice(
                train_rows,
                size=int(training["batch_size"]),
                replace=True,
                p=cadence_probability,
            )
        if source_audit is not None:
            source_audit.update(chosen)
        raw = torch.from_numpy(glucose[chosen]).to(device)
        mask = torch.from_numpy(observed[chosen]).to(device)
        cadence_raw = torch.from_numpy(cadence_glucose[chosen]).to(device)
        cadence_mask = torch.from_numpy(cadence_observed[chosen]).to(device)
        profile = torch.from_numpy(targets["profiles"][chosen]).to(device)
        severity = torch.from_numpy(targets["severity"][chosen]).to(device)
        chosen_target_valid = targets["valid"][chosen].copy()
        if cadence_projector_mode == "absolute_15min":
            # At the canonical cadence P=I, so there is no private/null target.
            chosen_target_valid[cadence[chosen] >= 15] = False
        target_valid = torch.from_numpy(chosen_target_valid).to(device)
        scalar_target = torch.from_numpy(normalized_scalar[chosen]).to(device)
        cadence_scalar_target = (
            torch.from_numpy(normalized_cadence_scalar[chosen]).to(device)
            if normalized_cadence_scalar is not None
            else None
        )
        model.train()
        if persistent_evidence_reader is not None:
            persistent_evidence_reader.train()
        loss, terms, representation = _batch_loss(
            model,
            variant=args.variant,
            raw=raw,
            observed=mask,
            cadence_raw=cadence_raw,
            cadence_observed=cadence_mask,
            profiles=profile,
            severity=severity,
            target_valid=target_valid,
            scalar_target=scalar_target,
            cadence_scalar_target=cadence_scalar_target,
            center=contract.center,
            scale=contract.scale,
            thresholds=thresholds,
            corruption_rate=float(training["corruption_rate"]),
            weights=batch_weights,
            generator=torch_generator,
            input_normalization=input_normalization,
            nested_loss_mode=nested_loss_mode,
            cadence_loss_mode=cadence_loss_mode,
            full_scalar_head=_full_scalar_head_active(
                full_scalar_head, step, full_scalar_head_steps
            ),
            full_scalar_input_mode=full_scalar_input_mode,
            full_scalar_view_mode=full_scalar_view_mode,
            input_decomposition_mode=input_decomposition_mode,
            bulk_reconstruction_target_mode=bulk_reconstruction_target_mode,
        )
        if daily_episodes is not None:
            evidence_target = torch.from_numpy(daily_evidence[chosen]).to(device)
            evidence_representation = representation
            if args.daily_evidence_reader == "persistent":
                evidence_prediction = persistent_evidence_reader(
                    evidence_representation
                )
                evidence_loss = _daily_evidence_mse(
                    evidence_prediction, evidence_target
                )
            else:
                evidence_prediction = ridge_predict(
                    evidence_representation[:support_size],
                    evidence_target[:support_size],
                    evidence_representation[support_size:],
                )
                evidence_loss = _daily_evidence_mse(
                    evidence_prediction, evidence_target[support_size:]
                )
            loss = loss + daily_evidence_weight * evidence_loss
            terms["daily_evidence"] = float(evidence_loss.detach().cpu())
            terms["total"] = float(loss.detach().cpu())
        if coordinate_teacher is not None:
            with torch.no_grad():
                coordinate_target = _clean_teacher_representation(
                    coordinate_teacher,
                    variant=args.variant,
                    raw=raw,
                    observed=mask,
                    cadence_raw=cadence_raw,
                    cadence_observed=cadence_mask,
                    center=contract.center,
                    scale=contract.scale,
                    thresholds=thresholds,
                    input_normalization=input_normalization,
                    input_decomposition_mode=input_decomposition_mode,
                )
            coordinate_loss = F.mse_loss(representation, coordinate_target)
            loss = loss + daily_coordinate_weight * coordinate_loss
            terms["daily_coordinate"] = float(coordinate_loss.detach().cpu())
            terms["total"] = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        if step == 0 or (step + 1) % int(training["history_interval_steps"]) == 0:
            history.append({"step": step + 1, **terms})

    train_embedding, _, _ = _extract(
        model,
        glucose,
        observed,
        cadence_glucose,
        cadence_observed,
        mechanism_train_rows,
        variant=args.variant,
        center=contract.center,
        scale=contract.scale,
        thresholds=thresholds,
        batch_size=int(training["evaluation_batch_size"]),
        device=device,
        input_normalization=input_normalization,
        input_decomposition_mode=input_decomposition_mode,
        bulk_reconstruction_target_mode=bulk_reconstruction_target_mode,
    )
    validation_embedding, validation_cadence_embedding, bulk_nmse = _extract(
        model,
        glucose,
        observed,
        cadence_glucose,
        cadence_observed,
        validation_rows,
        variant=args.variant,
        center=contract.center,
        scale=contract.scale,
        thresholds=thresholds,
        batch_size=int(training["evaluation_batch_size"]),
        device=device,
        input_normalization=input_normalization,
        input_decomposition_mode=input_decomposition_mode,
        bulk_reconstruction_target_mode=bulk_reconstruction_target_mode,
    )
    keep = np.ones(2 * contract.lag + 1, dtype=bool)
    keep[contract.lag] = False
    profile_nmse: dict[str, float] = {}
    cadence_nmse: dict[str, float] = {}
    private_profile_nmse: dict[str, float] = {}
    for side_index, side_name in enumerate(("upper", "lower")):
        for quantile_index, quantile in enumerate(QUANTILES):
            key = f"{side_name}_q{int(round(1000 * quantile)):03d}"
            profile_nmse[key] = _ridge_nmse(
                train_embedding,
                validation_embedding,
                targets["profiles"][mechanism_train_rows, side_index, quantile_index][
                    :, keep
                ],
                targets["profiles"][validation_rows, side_index, quantile_index][
                    :, keep
                ],
                targets["valid"][mechanism_train_rows, side_index, quantile_index],
                targets["valid"][validation_rows, side_index, quantile_index],
                ridge=float(training["mechanism_probe_ridge"]),
            )
            cadence_nmse[key] = _ridge_nmse(
                train_embedding,
                validation_cadence_embedding,
                targets["profiles"][mechanism_train_rows, side_index, quantile_index][
                    :, keep
                ],
                targets["profiles"][validation_rows, side_index, quantile_index][
                    :, keep
                ],
                targets["valid"][mechanism_train_rows, side_index, quantile_index],
                targets["valid"][validation_rows, side_index, quantile_index],
                ridge=float(training["mechanism_probe_ridge"]),
            )
            private_profile_nmse[key] = _ridge_nmse(
                train_embedding[:, 96:],
                validation_embedding[:, 96:],
                targets["profiles"][mechanism_train_rows, side_index, quantile_index][
                    :, keep
                ],
                targets["profiles"][validation_rows, side_index, quantile_index][
                    :, keep
                ],
                targets["valid"][mechanism_train_rows, side_index, quantile_index],
                targets["valid"][validation_rows, side_index, quantile_index],
                ridge=float(training["mechanism_probe_ridge"]),
            )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "variant": args.variant,
            "seed": seed,
            "model_init_seed": model_init_seed,
            "sampling_seed": sampling_seed,
            "steps": steps,
            "init_checkpoint_sha256": init_checkpoint_sha256,
            "tail_contract": contract.__dict__,
            "config_sha256": sha256(args.config),
            "dataset_sha256": sha256(args.npz),
            "input_normalization": input_normalization,
            "nested_loss_mode": nested_loss_mode,
            "cadence_loss_mode": cadence_loss_mode,
            "full_scalar_head": full_scalar_head,
            "full_scalar_input_mode": full_scalar_input_mode,
            "full_scalar_view_mode": full_scalar_view_mode,
            "full_scalar_head_steps": full_scalar_head_steps,
            "profile_input_dim": profile_input_dim,
            "responsibility_partitioned": responsibility_partitioned,
            "input_decomposition_mode": input_decomposition_mode,
            "bulk_reconstruction_target_mode": bulk_reconstruction_target_mode,
            "cadence_projector_mode": cadence_projector_mode,
            "daily_evidence_reader": args.daily_evidence_reader,
            "daily_evidence_weight": daily_evidence_weight,
            "daily_evidence_target_dim": (
                None if daily_evidence is None else int(daily_evidence.shape[1])
            ),
            "daily_evidence_center": daily_evidence_center,
            "daily_evidence_scale": daily_evidence_scale,
            "daily_coordinate_teacher_sha256": coordinate_teacher_sha256,
            "daily_coordinate_weight": daily_coordinate_weight,
            "persistent_evidence_reader": (
                None
                if persistent_evidence_reader is None
                else persistent_evidence_reader.state_dict()
            ),
        },
        checkpoint,
    )
    q95_keys = ("upper_q950", "lower_q950")
    validation_daily_mean = np.asarray(
        [
            values[mask].mean()
            for values, mask in zip(glucose[validation_rows], observed[validation_rows])
        ]
    )
    report = {
        "variant": args.variant,
        "seed": seed,
        "model_init_seed": model_init_seed,
        "sampling_seed": sampling_seed,
        "optimizer_steps": steps,
        "encoder_forward_count_during_training": 2 * steps,
        "representation_dim": 128,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "input_normalization": input_normalization,
        "nested_loss_mode": nested_loss_mode,
        "cadence_loss_mode": cadence_loss_mode,
        "full_scalar_head": full_scalar_head,
        "full_scalar_input_mode": full_scalar_input_mode,
        "full_scalar_view_mode": full_scalar_view_mode,
        "full_scalar_head_steps": full_scalar_head_steps,
        "profile_input_dim": profile_input_dim,
        "responsibility_partitioned": responsibility_partitioned,
        "input_decomposition_mode": input_decomposition_mode,
        "bulk_reconstruction_target_mode": bulk_reconstruction_target_mode,
        "cadence_projector_mode": cadence_projector_mode,
        "daily_evidence_reader": args.daily_evidence_reader,
        "daily_evidence_weight": daily_evidence_weight,
        "daily_evidence_target_dim": (
            None if daily_evidence is None else int(daily_evidence.shape[1])
        ),
        "daily_evidence_sampler": (
            None if daily_episodes is None else daily_episodes.summary()
        ),
        "daily_coordinate_teacher_sha256": coordinate_teacher_sha256,
        "daily_coordinate_weight": daily_coordinate_weight,
        "trainable_parameters": trainable_parameter_count(model),
        "temporary_reader_parameters": (
            0
            if persistent_evidence_reader is None
            else sum(
                parameter.numel()
                for parameter in persistent_evidence_reader.parameters()
            )
        ),
        "dataset": str(args.npz),
        "dataset_sha256": sha256(args.npz),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": sha256(checkpoint),
        "train_windows": int(len(train_rows)),
        "mechanism_train_windows": int(len(mechanism_train_rows)),
        "validation_windows": int(len(validation_rows)),
        "training_native_cadences": list(training_cadences),
        "mechanism_evaluation_native_cadence": mechanism_cadence,
        "native_15min_sampling_probability": training.get(
            "native_15min_sampling_probability"
        ),
        "tail_contract": contract.__dict__,
        "anchor_windows": {
            "train": {
                f"{side}_q{int(round(1000 * q)):03d}": int(
                    targets["valid"][
                        mechanism_train_rows, side_index, quantile_index
                    ].sum()
                )
                for side_index, side in enumerate(("upper", "lower"))
                for quantile_index, q in enumerate(QUANTILES)
            },
            "validation": {
                f"{side}_q{int(round(1000 * q)):03d}": int(
                    targets["valid"][validation_rows, side_index, quantile_index].sum()
                )
                for side_index, side in enumerate(("upper", "lower"))
                for quantile_index, q in enumerate(QUANTILES)
            },
        },
        "metrics": {
            "tail_profile_nmse": profile_nmse,
            "mean_q95_upper_lower_tail_profile_nmse": float(
                np.mean([profile_nmse[key] for key in q95_keys])
            ),
            "nested_threshold_profile_error": float(
                np.mean(list(profile_nmse.values()))
            ),
            "cadence_profile_nmse": cadence_nmse,
            "cadence_profile_error_q95": float(
                np.mean([cadence_nmse[key] for key in q95_keys])
            ),
            "private_tail_profile_nmse": private_profile_nmse,
            "private_mean_q95_upper_lower_tail_profile_nmse": float(
                np.mean([private_profile_nmse[key] for key in q95_keys])
            ),
            "common_cadence_cosine_distance": float(
                np.mean(
                    1.0
                    - np.sum(
                        validation_embedding[:, :96]
                        * validation_cadence_embedding[:, :96],
                        axis=1,
                    )
                    / np.maximum(
                        np.linalg.norm(validation_embedding[:, :96], axis=1)
                        * np.linalg.norm(validation_cadence_embedding[:, :96], axis=1),
                        1e-12,
                    )
                )
            ),
            "bulk_reconstruction_nmse": bulk_nmse,
            "embedding_effective_rank": _effective_rank(validation_embedding),
            "bulk_pc1_daily_mean_abs_pearson": _pc1_absolute_correlation(
                validation_embedding[:, :96], validation_daily_mean
            ),
        },
        "isolation": {
            "phenotype_label_reads": 0,
            "downstream_identity_reads": 0,
            "private_test_optimization_reads": 0,
            "loaded_archive_arrays": [
                "glucose_mg_dl",
                "observed_mask",
                "split",
                "native_cadence_min",
                "coverage",
            ],
        },
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = args.output_dir / "training_report.json"
    if source_audit is not None:
        report["source_sampling"] = dict(
            source_audit.summary(),
            pbq_probability=pbq_probability,
            within_source_weights="original cadence/event weights",
            validation_policy="unchanged",
            policy="source_mass_reweight_v1",
        )
        if source_sampling_stratum is not None:
            report["source_sampling"].update(
                within_source_weights="original cadence/event weights within fixed anchor/increment strata",
                replay_stratum_array=str(replay_stratum_key),
                anchor_replay_probability=float(anchor_replay_probability),
            )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checkpoint.chmod(0o600)
    report_path.chmod(0o600)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "history"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
