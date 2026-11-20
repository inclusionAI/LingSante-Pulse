#!/usr/bin/env python3
"""Label-free covariance-preserving transport for longitudinal innovations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    product = np.einsum("ij,jk->ik", left, right, optimize=True)
    if not np.isfinite(product).all():
        raise ValueError("non-finite covariance transport product")
    return product


@dataclass(frozen=True)
class CovarianceTransport:
    daily_center: np.ndarray
    memory_center: np.ndarray
    daily_to_memory: np.ndarray
    memory_whitener: np.ndarray
    daily_color: np.ndarray

    def innovation(self, daily: np.ndarray, memory: np.ndarray) -> np.ndarray:
        daily_centered = np.asarray(daily, dtype=np.float64) - self.daily_center
        memory_centered = np.asarray(memory, dtype=np.float64) - self.memory_center
        unexplained = memory_centered - _matrix_product(
            daily_centered, self.daily_to_memory
        )
        white = _matrix_product(unexplained, self.memory_whitener)
        return _matrix_product(white, self.daily_color).astype(np.float32)

    def embedding(
        self,
        daily: np.ndarray,
        memory: np.ndarray,
        valid_history: np.ndarray,
        *,
        innovation_ratio: float,
    ) -> np.ndarray:
        daily = np.asarray(daily, dtype=np.float32)
        valid = np.asarray(valid_history, dtype=bool)
        output = daily.copy()
        output[valid] += float(innovation_ratio) * self.innovation(
            daily[valid], memory[valid]
        )
        return output


def fit_covariance_transport(
    daily: np.ndarray,
    memory: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[CovarianceTransport, dict[str, float | int]]:
    """Fit a memory innovation whose split-fit covariance matches daily CARE.

    The memory must have at least as many coordinates as daily CARE.  First,
    ordinary least squares removes all linearly daily-predictable memory.  The
    residual is whitened and then colored by the daily covariance square root.
    Consequently, on fit rows the innovation is linearly uncorrelated with the
    daily code and has the same covariance spectrum.
    """
    daily = np.asarray(daily, dtype=np.float64)
    memory = np.asarray(memory, dtype=np.float64)
    rows = np.asarray(fit_rows, dtype=np.int64)
    if daily.ndim != 2 or memory.ndim != 2 or len(daily) != len(memory):
        raise ValueError("daily and memory must be aligned matrices")
    if memory.shape[1] < daily.shape[1] or len(rows) <= daily.shape[1]:
        raise ValueError("memory width and fit support must cover daily dimensions")

    daily_fit = daily[rows]
    memory_fit = memory[rows]
    daily_center = daily_fit.mean(axis=0, keepdims=True)
    memory_center = memory_fit.mean(axis=0, keepdims=True)
    daily_centered = daily_fit - daily_center
    memory_centered = memory_fit - memory_center
    daily_to_memory = np.linalg.lstsq(daily_centered, memory_centered, rcond=None)[0]
    unexplained = memory_centered - _matrix_product(daily_centered, daily_to_memory)

    _, memory_singular, memory_right = np.linalg.svd(unexplained, full_matrices=False)
    dimensions = daily.shape[1]
    retained = memory_singular[:dimensions]
    numerical_floor = max(float(memory_singular[0]) * 1e-7, 1e-10)
    if len(retained) < dimensions or float(retained[-1]) <= numerical_floor:
        raise ValueError("longitudinal memory lacks full innovation support")
    memory_scores = _matrix_product(unexplained, memory_right[:dimensions].T)
    memory_score_scale = memory_scores.std(axis=0, ddof=0)
    memory_whitener = memory_right[:dimensions].T / memory_score_scale[None, :]

    _, daily_singular, daily_right = np.linalg.svd(daily_centered, full_matrices=False)
    daily_scale = daily_singular / np.sqrt(len(rows))
    daily_color = (daily_scale[:, None] * daily_right).astype(np.float64)
    transport = CovarianceTransport(
        daily_center=daily_center,
        memory_center=memory_center,
        daily_to_memory=daily_to_memory,
        memory_whitener=memory_whitener,
        daily_color=daily_color,
    )

    innovation = transport.innovation(daily_fit, memory_fit).astype(np.float64)
    daily_covariance = _matrix_product(daily_centered.T, daily_centered) / len(rows)
    innovation_covariance = _matrix_product(innovation.T, innovation) / len(rows)
    cross_covariance = _matrix_product(daily_centered.T, innovation) / len(rows)
    diagnostics: dict[str, float | int] = {
        "dimensions": dimensions,
        "memory_dimensions": memory.shape[1],
        "retained_memory_singular_ratio": float(retained[-1] / retained[0]),
        "fit_covariance_relative_frobenius_error": float(
            np.linalg.norm(innovation_covariance - daily_covariance, ord="fro")
            / max(np.linalg.norm(daily_covariance, ord="fro"), 1e-12)
        ),
        "fit_daily_innovation_cross_covariance_relative_frobenius": float(
            np.linalg.norm(cross_covariance, ord="fro")
            / max(np.linalg.norm(daily_covariance, ord="fro"), 1e-12)
        ),
    }
    return transport, diagnostics


def calibrate_innovation_gain(
    daily: np.ndarray,
    innovation: np.ndarray,
    target_rms_ratio: float,
) -> float:
    daily_rms = float(np.sqrt(np.mean(np.square(np.asarray(daily, dtype=np.float64)))))
    innovation_rms = float(
        np.sqrt(np.mean(np.square(np.asarray(innovation, dtype=np.float64))))
    )
    if not 0.0 < target_rms_ratio < 1.0 or innovation_rms <= 1e-12:
        raise ValueError("invalid innovation RMS calibration")
    return float(target_rms_ratio * daily_rms / innovation_rms)


def transport_state(transport: CovarianceTransport) -> dict[str, np.ndarray]:
    return {
        "daily_center": transport.daily_center.astype(np.float32),
        "memory_center": transport.memory_center.astype(np.float32),
        "daily_to_memory": transport.daily_to_memory.astype(np.float32),
        "memory_whitener": transport.memory_whitener.astype(np.float32),
        "daily_color": transport.daily_color.astype(np.float32),
    }


def transport_from_state(state: dict[str, np.ndarray]) -> CovarianceTransport:
    required = (
        "daily_center",
        "memory_center",
        "daily_to_memory",
        "memory_whitener",
        "daily_color",
    )
    if any(name not in state for name in required):
        raise ValueError("incomplete covariance transport state")
    return CovarianceTransport(
        **{name: np.asarray(state[name], dtype=np.float64) for name in required}
    )
