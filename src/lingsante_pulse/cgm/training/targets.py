"""Fixed reconstruction targets for TER objective controls."""

from __future__ import annotations

import numpy as np

from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import bounded_histories


MAX_DAYS = 7
HOURS_PER_DAY = 24
POINTS_PER_HOUR = 12
TARGET_DIMENSION = MAX_DAYS * HOURS_PER_DAY * 2


def raw_hourly_history_targets(data, fit_rows, normalization=None):
    """Return 7-day hourly glucose-and-mask targets with fit-only scaling.

    The target is deliberately generic: each hourly value is only the mean of
    actually observed measurements on the canonical five-minute storage grid.
    Empty hours are never interpolated.  Days are right-aligned so the endpoint
    day is always the last of seven slots.
    """
    glucose = np.asarray(data["glucose"], np.float32)
    observed = np.asarray(data["observed"], bool) & np.isfinite(glucose)
    if glucose.ndim != 2 or glucose.shape != observed.shape:
        raise ValueError("Glucose and observation mask must be aligned matrices")
    if glucose.shape[1] != HOURS_PER_DAY * POINTS_PER_HOUR:
        raise ValueError("Raw control requires the canonical 288-point daily grid")

    histories = bounded_histories(
        causal_history_indices(
            data["subject"],
            data["starts"],
            data["cohort"],
            data["split"],
            data["cadence"],
            data["source"],
            max_days=MAX_DAYS,
        ),
        data["starts"],
    )
    values = np.zeros((len(glucose), MAX_DAYS, HOURS_PER_DAY), np.float32)
    masks = np.zeros_like(values)
    for endpoint, history in enumerate(histories):
        rows = np.asarray(history, np.int64)[-MAX_DAYS:]
        if rows[-1] != endpoint:
            raise ValueError("Causal history must end at the target window")
        for key in ("subject", "cohort", "split", "cadence", "source"):
            if not np.all(
                np.asarray(data[key])[rows] == np.asarray(data[key])[endpoint]
            ):
                raise ValueError("Raw target crossed a provenance boundary")
        offset = MAX_DAYS - len(rows)
        for local, row in enumerate(rows):
            hourly_glucose = glucose[row].reshape(HOURS_PER_DAY, POINTS_PER_HOUR)
            hourly_observed = observed[row].reshape(HOURS_PER_DAY, POINTS_PER_HOUR)
            count = hourly_observed.sum(1)
            supported = count > 0
            mean = np.divide(
                np.where(hourly_observed, hourly_glucose, 0.0).sum(1),
                np.maximum(count, 1),
            )
            slot = offset + local
            values[endpoint, slot, supported] = np.tanh(
                (mean[supported] - 100.0) / 40.0
            )
            masks[endpoint, slot, supported] = 1.0

    raw = np.concatenate(
        [values.reshape(len(values), -1), masks.reshape(len(masks), -1)], axis=1
    )
    if raw.shape[1] != TARGET_DIMENSION:
        raise ValueError("Raw reconstruction target dimension drift")
    fit_rows = np.asarray(fit_rows, np.int64)
    if normalization is None:
        center = raw[fit_rows].mean(0)
        scale = np.maximum(raw[fit_rows].std(0), 1e-5)
    else:
        center = np.asarray(normalization["center"], np.float32)
        scale = np.asarray(normalization["scale"], np.float32)
        if center.shape != (TARGET_DIMENSION,) or scale.shape != (TARGET_DIMENSION,):
            raise ValueError("Raw target normalization contract mismatch")
    target = ((raw - center) / scale).astype(np.float32)
    return target, {
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "target_dimension": TARGET_DIMENSION,
        "value_dimension": MAX_DAYS * HOURS_PER_DAY,
        "mask_dimension": MAX_DAYS * HOURS_PER_DAY,
        "history_days": MAX_DAYS,
        "hours_per_day": HOURS_PER_DAY,
        "value_transform": "tanh((observed_hourly_mean_mg_dl-100)/40)",
        "missing_policy": "zero_value_plus_explicit_binary_mask_no_interpolation",
    }
