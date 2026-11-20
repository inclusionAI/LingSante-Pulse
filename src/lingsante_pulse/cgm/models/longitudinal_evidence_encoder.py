#!/usr/bin/env python3
"""Causal, responsibility-typed long-term evidence encoder for CARE-CGM.

The module deliberately keeps the frozen daily CARE representation as the
only exported coordinate system.  Four small memories summarize different
longitudinal evidence regimes and predict a history-supported CARE code.  A
row without prior context is returned bit-for-bit unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

import numpy as np
import torch
from torch import nn


DAY_MINUTES = 24 * 60
GRID_MINUTES = 5
RESPONSIBILITIES = ("state", "challenge", "extreme", "rhythm")
FEATURES_PER_RESPONSIBILITY = 16


def causal_history_indices(
    subject_id: np.ndarray,
    start_time: np.ndarray,
    cohort: np.ndarray,
    split: np.ndarray,
    cadence: np.ndarray,
    source: np.ndarray,
    *,
    max_days: int = 7,
) -> list[np.ndarray]:
    """Build causal non-overlapping histories inside strict provenance groups.

    Gaps are allowed but represented explicitly by elapsed time.  The oldest
    admitted window must begin no more than ``max_days`` before the endpoint.
    """
    subjects = np.asarray(subject_id).astype(str)
    starts = np.asarray(start_time).astype("datetime64[m]")
    cohorts = np.asarray(cohort).astype(str)
    splits = np.asarray(split)
    cadences = np.asarray(cadence)
    sources = np.asarray(source)
    arrays = (starts, cohorts, splits, cadences, sources)
    if max_days < 1 or not all(len(value) == len(subjects) for value in arrays):
        raise ValueError("invalid history inputs")
    if np.any(np.isnat(starts)):
        raise ValueError("start_time contains NaT")

    groups: dict[tuple[str, str, int, int, int], list[int]] = defaultdict(list)
    for index in range(len(subjects)):
        groups[
            (
                subjects[index],
                cohorts[index],
                int(splits[index]),
                int(cadences[index]),
                int(sources[index]),
            )
        ].append(index)

    histories: list[np.ndarray | None] = [None] * len(subjects)
    maximum_span = max_days * DAY_MINUTES
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: starts[index])
        minutes = np.asarray(
            [int(starts[index].astype(np.int64)) for index in ordered], dtype=np.int64
        )
        if len(np.unique(minutes)) != len(minutes):
            raise ValueError("duplicate start_time within a provenance group")
        for position, current_index in enumerate(ordered):
            current_start = minutes[position]
            cursor = current_start
            selected = [current_index]
            for prior_position in range(position - 1, -1, -1):
                prior_start = minutes[prior_position]
                if current_start - prior_start > maximum_span:
                    break
                if cursor - prior_start < DAY_MINUTES:
                    continue
                selected.append(ordered[prior_position])
                cursor = prior_start
                if len(selected) == max_days:
                    break
            histories[current_index] = np.asarray(selected[::-1], dtype=np.int64)
    if any(value is None for value in histories):
        raise RuntimeError("not every row received a causal history")
    return [value for value in histories if value is not None]


def _interpolate(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    index = np.arange(len(values))
    valid = np.flatnonzero(observed)
    if not len(valid):
        return np.zeros_like(values, dtype=np.float32)
    return np.interp(index, valid, values[valid]).astype(np.float32)


def _run_statistics(condition: np.ndarray) -> tuple[float, float]:
    padded = np.r_[False, np.asarray(condition, dtype=bool), False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    lengths = changes[1::2] - changes[::2]
    return float(len(lengths)), float(lengths.max(initial=0))


def _phase_means(
    values: np.ndarray, observed: np.ndarray, start_minute: int
) -> list[float]:
    phase = (start_minute + np.arange(len(values)) * GRID_MINUTES) % DAY_MINUTES
    result = []
    for left, right in ((0, 360), (360, 600), (600, 1080), (1080, 1440)):
        selected = observed & (phase >= left) & (phase < right)
        result.append(float(values[selected].mean()) if np.any(selected) else 0.0)
    return result


def _challenge_features(
    values: np.ndarray, observed: np.ndarray, cadence_min: float
) -> tuple[np.ndarray, float]:
    filled = _interpolate(values, observed)
    expected = min(1.0, GRID_MINUTES / float(cadence_min))
    horizon = 12
    recovery = 24
    proposals: list[tuple[float, np.ndarray]] = []
    for start in range(len(filled) - horizon - 1):
        end = start + horizon
        if not (observed[start] and observed[end]):
            continue
        if observed[start : end + 1].mean() < 0.5 * expected:
            continue
        rise = float(filled[end] - filled[start])
        if rise < 30.0:
            continue
        stop = min(len(filled), end + recovery + 1)
        local_observed = np.flatnonzero(observed[end:stop]) + end
        if not len(local_observed):
            continue
        peak = int(local_observed[np.argmax(filled[local_observed])])
        peak_amplitude = float(filled[peak] - filled[start])
        after_60 = min(len(filled) - 1, peak + 12)
        after_120 = min(len(filled) - 1, peak + 24)
        recover_60 = float(filled[peak] - filled[after_60])
        recover_120 = float(filled[peak] - filled[after_120])
        morphology = np.asarray(
            [rise, peak_amplitude, (peak - start) / 12.0, recover_60, recover_120],
            dtype=np.float32,
        )
        proposals.append((peak_amplitude + max(recover_120, 0.0), morphology))
    proposals.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in proposals[:3]]
    output = np.zeros(FEATURES_PER_RESPONSIBILITY, dtype=np.float32)
    output[0] = min(len(proposals), 6) / 6.0
    for index, morphology in enumerate(selected):
        output[1 + 5 * index : 6 + 5 * index] = morphology
    support = min(1.0, len(selected) / 3.0) * float(
        observed.mean() / max(expected, 1e-6)
    )
    return output, float(np.clip(support, 0.0, 1.0))


def typed_evidence_features(
    glucose: np.ndarray,
    observed_mask: np.ndarray,
    cadence_min: np.ndarray,
    start_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return four 16D CGM evidence tokens and per-token support."""
    glucose = np.asarray(glucose, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    cadence = np.asarray(cadence_min, dtype=np.float32)
    starts = np.asarray(start_time).astype("datetime64[m]").astype(np.int64)
    if glucose.ndim != 2 or glucose.shape != observed.shape:
        raise ValueError("glucose and observed_mask must be aligned matrices")
    if len(cadence) != len(glucose) or len(starts) != len(glucose):
        raise ValueError("metadata length mismatch")
    if np.any(cadence <= 0) or np.any(observed.sum(axis=1) < 2):
        raise ValueError("every row needs a valid cadence and two observations")

    features = np.zeros(
        (len(glucose), len(RESPONSIBILITIES), FEATURES_PER_RESPONSIBILITY),
        dtype=np.float32,
    )
    support = np.zeros((len(glucose), len(RESPONSIBILITIES)), dtype=np.float32)
    for row in range(len(glucose)):
        mask = observed[row]
        raw = glucose[row]
        values = raw[mask]
        filled = _interpolate(raw, mask)
        expected = min(1.0, GRID_MINUTES / float(cadence[row]))
        coverage = float(np.clip(mask.mean() / max(expected, 1e-6), 0.0, 1.0))

        quantiles = np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        state = np.asarray(
            [
                values.mean(),
                values.std(),
                *quantiles,
                np.mean(values < 70.0),
                np.mean((values >= 70.0) & (values <= 180.0)),
                np.mean(values > 180.0),
                *_phase_means(raw, mask, int(starts[row] % DAY_MINUTES)),
            ],
            dtype=np.float32,
        )
        features[row, 0] = state
        support[row, 0] = coverage

        challenge, challenge_support = _challenge_features(
            raw, mask, float(cadence[row])
        )
        features[row, 1] = challenge
        support[row, 1] = challenge_support

        low = mask & (raw < 70.0)
        very_low = mask & (raw < 54.0)
        high = mask & (raw > 180.0)
        very_high = mask & (raw > 250.0)
        low_runs, low_max = _run_statistics(low)
        high_runs, high_max = _run_statistics(high)
        change = filled[6:] - filled[:-6]
        low_positions = np.flatnonzero(low)
        high_positions = np.flatnonzero(high)
        low_rebound = max(
            (
                float(filled[min(len(filled) - 1, index + 24)] - filled[index])
                for index in low_positions
            ),
            default=0.0,
        )
        high_recovery = max(
            (
                float(filled[index] - filled[min(len(filled) - 1, index + 24)])
                for index in high_positions
            ),
            default=0.0,
        )
        features[row, 2] = np.asarray(
            [
                np.mean(values < 70.0),
                np.mean(values < 54.0),
                np.mean(values > 180.0),
                np.mean(values > 250.0),
                values.min(),
                values.max(),
                np.quantile(values, 0.05),
                np.quantile(values, 0.95),
                low_runs,
                high_runs,
                low_max * GRID_MINUTES,
                high_max * GRID_MINUTES,
                float(max(-change.min(initial=0.0), 0.0)),
                float(max(change.max(initial=0.0), 0.0)),
                low_rebound,
                high_recovery,
            ],
            dtype=np.float32,
        )
        support[row, 2] = coverage

        phase = int(starts[row] % DAY_MINUTES) + np.arange(len(raw)) * GRID_MINUTES
        angle = 2.0 * np.pi * phase / DAY_MINUTES
        normalized = (filled - values.mean()) / max(float(values.std()), 1.0)
        weights = mask.astype(np.float32)
        harmonic = []
        for order in (1, 2):
            harmonic.extend(
                [
                    float(
                        np.sum(normalized * np.sin(order * angle) * weights)
                        / weights.sum()
                    ),
                    float(
                        np.sum(normalized * np.cos(order * angle) * weights)
                        / weights.sum()
                    ),
                ]
            )
        blocks = np.asarray(
            [filled[left : left + 48].mean() for left in range(0, len(filled), 48)],
            dtype=np.float32,
        )
        features[row, 3] = np.asarray(
            [
                *blocks,
                *harmonic,
                float(np.std(blocks)),
                float(blocks[-1] - blocks[0]),
                float(np.mean(np.abs(np.diff(blocks)))),
                float(np.corrcoef(filled[:-12], filled[12:])[0, 1]),
                coverage,
                float(cadence[row] / 15.0),
            ],
            dtype=np.float32,
        )
        support[row, 3] = coverage
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, support


def generic_multiscale_features(
    glucose: np.ndarray, observed_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width generic temporal summaries for the ablation control."""
    glucose = np.asarray(glucose, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    if glucose.ndim != 2 or glucose.shape != observed.shape:
        raise ValueError("glucose and observed_mask must align")
    output = np.zeros(
        (len(glucose), len(RESPONSIBILITIES), FEATURES_PER_RESPONSIBILITY),
        dtype=np.float32,
    )
    support = np.repeat(observed.mean(axis=1, keepdims=True), 4, axis=1).astype(
        np.float32
    )
    for row in range(len(glucose)):
        filled = _interpolate(glucose[row], observed[row])
        normalized = (filled - filled.mean()) / max(float(filled.std()), 1.0)
        for branch, scale in enumerate((1, 3, 6, 12)):
            smoothed = np.convolve(
                normalized, np.ones(scale, dtype=np.float32) / scale, mode="same"
            )
            chunks = np.array_split(smoothed, 4)
            statistics = []
            for chunk in chunks:
                statistics.extend([chunk.mean(), chunk.std(), chunk.min(), chunk.max()])
            output[row, branch] = np.asarray(statistics, dtype=np.float32)
    return output, support


def standardize_evidence(
    features: np.ndarray, fit_rows: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(features, dtype=np.float32)
    fit = np.asarray(fit_rows, dtype=np.int64)
    center = values[fit].mean(axis=0, keepdims=True)
    scale = values[fit].std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-5, 1.0, scale)
    return ((values - center) / scale).astype(np.float32), {
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def pack_histories(
    features: np.ndarray,
    support: np.ndarray,
    starts: np.ndarray,
    histories: list[np.ndarray],
    *,
    max_prior_days: int = 6,
) -> dict[str, np.ndarray]:
    """Pack only prior rows; the endpoint itself is never a memory value."""
    features = np.asarray(features, dtype=np.float32)
    support = np.asarray(support, dtype=np.float32)
    minutes = np.asarray(starts).astype("datetime64[m]").astype(np.int64)
    if len(histories) != len(features) or support.shape != features.shape[:2]:
        raise ValueError("history inputs do not align")
    packed = np.zeros(
        (len(features), max_prior_days, *features.shape[1:]), dtype=np.float32
    )
    packed_support = np.zeros(
        (len(features), max_prior_days, support.shape[1]), dtype=np.float32
    )
    elapsed = np.zeros((len(features), max_prior_days), dtype=np.float32)
    mask = np.zeros((len(features), max_prior_days), dtype=bool)
    for endpoint, history in enumerate(histories):
        if len(history) == 0 or int(history[-1]) != endpoint:
            raise ValueError("each history must end at its endpoint")
        prior = np.asarray(history[:-1], dtype=np.int64)[-max_prior_days:]
        if not len(prior):
            continue
        if np.any(minutes[prior] >= minutes[endpoint]):
            raise ValueError("future or simultaneous window in history")
        offset = max_prior_days - len(prior)
        packed[endpoint, offset:] = features[prior]
        packed_support[endpoint, offset:] = support[prior]
        elapsed[endpoint, offset:] = (minutes[endpoint] - minutes[prior]) / DAY_MINUTES
        mask[endpoint, offset:] = True
    return {
        "features": packed,
        "support": packed_support,
        "elapsed_days": elapsed,
        "history_mask": mask,
    }


class _ResponsibilityMemory(nn.Module):
    def __init__(self, input_dimensions: int, hidden_dimensions: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.GELU(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
        )
        self.key = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.value = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.gate = nn.Linear(2 * hidden_dimensions + 1, 1)
        self.target_head = nn.Linear(hidden_dimensions, input_dimensions)

    def forward(
        self,
        history: torch.Tensor,
        elapsed_code: torch.Tensor,
        support: torch.Tensor,
        history_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = self.token(history) + elapsed_code
        score = torch.sum(self.key(token) * query[:, None, :], dim=-1) / sqrt(
            query.shape[-1]
        )
        score = score + torch.log(torch.clamp(support, min=1e-4))
        score = score.masked_fill(~history_mask, -1e4)
        attention = torch.softmax(score, dim=1) * history_mask
        attention = attention / torch.clamp(
            attention.sum(dim=1, keepdim=True), min=1e-6
        )
        memory = torch.sum(attention[:, :, None] * self.value(token), dim=1)
        mean_support = torch.sum(attention * support, dim=1, keepdim=True)
        gate = torch.sigmoid(self.gate(torch.cat([query, memory, mean_support], dim=1)))
        valid = history_mask.any(dim=1, keepdim=True)
        code = gate * memory * valid
        return code, self.target_head(code), attention


class CausalLongitudinalEvidenceEncoder(nn.Module):
    """Four signal-specific memories that emit one history-supported CARE code."""

    def __init__(
        self,
        *,
        daily_dimensions: int = 146,
        feature_dimensions: int = FEATURES_PER_RESPONSIBILITY,
        hidden_dimensions: int = 24,
    ) -> None:
        super().__init__()
        self.daily_dimensions = daily_dimensions
        self.query = nn.Sequential(
            nn.LayerNorm(daily_dimensions),
            nn.Linear(daily_dimensions, hidden_dimensions),
            nn.Tanh(),
        )
        self.elapsed = nn.Sequential(nn.Linear(3, hidden_dimensions), nn.Tanh())
        self.memories = nn.ModuleList(
            [
                _ResponsibilityMemory(feature_dimensions, hidden_dimensions)
                for _ in RESPONSIBILITIES
            ]
        )
        self.long_prediction = nn.Sequential(
            nn.LayerNorm(len(RESPONSIBILITIES) * hidden_dimensions),
            nn.Linear(len(RESPONSIBILITIES) * hidden_dimensions, daily_dimensions),
        )

    def forward(
        self,
        daily_embedding: torch.Tensor,
        history_features: torch.Tensor,
        history_support: torch.Tensor,
        elapsed_days: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if history_features.ndim != 4 or history_features.shape[2] != len(
            RESPONSIBILITIES
        ):
            raise ValueError("history_features must be [batch,day,4,feature]")
        if history_support.shape != history_features.shape[:3]:
            raise ValueError("history_support does not align")
        if elapsed_days.shape != history_mask.shape != history_features.shape[:2]:
            raise ValueError("elapsed_days/history_mask do not align")
        query = self.query(daily_embedding)
        elapsed_input = torch.stack(
            [
                torch.log1p(elapsed_days),
                torch.sin(2.0 * torch.pi * elapsed_days / 7.0),
                torch.cos(2.0 * torch.pi * elapsed_days / 7.0),
            ],
            dim=-1,
        )
        elapsed_code = self.elapsed(elapsed_input)
        codes = []
        predictions = []
        attentions = []
        for branch, memory in enumerate(self.memories):
            code, prediction, attention = memory(
                history_features[:, :, branch],
                elapsed_code,
                history_support[:, :, branch],
                history_mask,
                query,
            )
            codes.append(code)
            predictions.append(prediction)
            attentions.append(attention)
        long_prediction = self.long_prediction(torch.cat(codes, dim=1))
        history_count = history_mask.sum(dim=1, keepdim=True).to(daily_embedding.dtype)
        blend = history_count / (history_count + 1.0)
        embedding = daily_embedding + blend * (long_prediction - daily_embedding)
        return {
            "embedding": embedding,
            "long_prediction": long_prediction,
            "typed_predictions": torch.stack(predictions, dim=1),
            "memory_codes": torch.stack(codes, dim=1),
            "attention": torch.stack(attentions, dim=1),
            "history_blend": blend,
        }
