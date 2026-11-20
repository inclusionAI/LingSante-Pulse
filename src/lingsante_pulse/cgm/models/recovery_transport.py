#!/usr/bin/env python3
"""Label-free causal rise-to-recovery coordinates for CGM trajectories."""

from __future__ import annotations

from collections import deque

import numpy as np


FEATURE_NAMES = (
    "matched_rise_fraction",
    "matched_fall_fraction",
    "unmatched_rise_fraction",
    "unmatched_fall_fraction",
    "delay_mean_fraction",
    "delay_std_fraction",
    "delay_q25_fraction",
    "delay_q50_fraction",
    "delay_q75_fraction",
    "delay_q90_fraction",
    "recovery_within_30m",
    "recovery_within_60m",
    "recovery_within_120m",
    "recovery_within_240m",
    "recovery_within_480m",
    "recovery_within_720m",
    "variation_mass_imbalance",
)


def fill_observed(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    positions = np.arange(values.size)
    support = np.flatnonzero(observed)
    if support.size < 2:
        raise ValueError("at least two observed glucose values are required")
    return np.interp(positions, support, values[support])


def normalized_trace(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    filled = fill_observed(values, observed)
    center = float(np.median(filled))
    scale = float(np.subtract(*np.percentile(filled, [75, 25])))
    if scale <= 1e-6:
        scale = float(np.std(filled))
    return (filled - center) / max(scale, 1e-6)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(values, dtype=np.float64).copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    if values.size == 0 or float(weights.sum()) <= 0:
        return 1.0
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    threshold = probability * cumulative[-1]
    index = np.searchsorted(cumulative, threshold, side="left")
    return float(values[order][index])


def recovery_transport_features(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    smoothing_sigma: float,
    maximum_horizon: int,
    recovery_profile: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Greedily couple positive variation to future negative variation."""
    normalized = normalized_trace(values, observed)
    difference = np.diff(gaussian_smooth(normalized, smoothing_sigma))
    rise = np.maximum(difference, 0.0)
    fall = np.maximum(-difference, 0.0)

    pending: deque[list[float]] = deque()
    delay_values: list[float] = []
    delay_weights: list[float] = []
    unmatched_rise = 0.0
    unmatched_fall = 0.0
    tolerance = 1e-12
    for time_index, (rise_mass, fall_mass) in enumerate(zip(rise, fall), start=1):
        while pending and time_index - int(pending[0][0]) > maximum_horizon:
            unmatched_rise += pending.popleft()[1]
        if rise_mass > tolerance:
            pending.append([float(time_index), float(rise_mass)])
        remaining_fall = float(fall_mass)
        while remaining_fall > tolerance and pending:
            rise_time, available = pending[0]
            matched = min(available, remaining_fall)
            delay_values.append(float(time_index - rise_time))
            delay_weights.append(matched)
            available -= matched
            remaining_fall -= matched
            if available <= tolerance:
                pending.popleft()
            else:
                pending[0][1] = available
        unmatched_fall += remaining_fall
    unmatched_rise += sum(item[1] for item in pending)

    delays = np.asarray(delay_values, dtype=np.float64)
    weights = np.asarray(delay_weights, dtype=np.float64)
    matched = float(weights.sum())
    total_rise = float(rise.sum())
    total_fall = float(fall.sum())
    horizon = float(maximum_horizon)
    if matched > tolerance:
        mean_delay = float(np.average(delays, weights=weights))
        delay_std = float(
            np.sqrt(np.average(np.square(delays - mean_delay), weights=weights))
        )
    else:
        mean_delay, delay_std = horizon, 0.0
    quantiles = [
        weighted_quantile(delays, weights, probability) / horizon
        for probability in (0.25, 0.5, 0.75, 0.9)
    ]
    recovery = [
        float(weights[delays <= step].sum()) / max(total_rise, tolerance)
        for step in recovery_profile
    ]
    denominator = max(total_rise + total_fall, tolerance)
    result = np.asarray(
        [
            matched / max(total_rise, tolerance),
            matched / max(total_fall, tolerance),
            unmatched_rise / max(total_rise, tolerance),
            unmatched_fall / max(total_fall, tolerance),
            mean_delay / horizon,
            delay_std / horizon,
            *quantiles,
            *recovery,
            abs(total_rise - total_fall) / denominator,
        ],
        dtype=np.float64,
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("invalid recovery-transport coordinates")
    return result


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    left = values[:-lag] - values[:-lag].mean()
    right = values[lag:] - values[lag:].mean()
    denominator = float(np.sqrt(np.square(left).sum() * np.square(right).sum()))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def generic_temporal_summary(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Equal-width generic control without rise-to-future-fall pairing."""
    normalized = normalized_trace(values, observed)
    difference = np.diff(normalized)
    positive = float(np.maximum(difference, 0.0).sum())
    negative = float(np.maximum(-difference, 0.0).sum())
    result = np.asarray(
        [
            *np.quantile(normalized, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
            float(np.std(normalized)),
            float(np.ptp(normalized)),
            positive,
            negative,
            abs(positive - negative) / max(positive + negative, 1e-12),
            *[_autocorrelation(normalized, lag) for lag in (1, 3, 6, 12, 24)],
        ],
        dtype=np.float64,
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("invalid generic temporal summary")
    return result
