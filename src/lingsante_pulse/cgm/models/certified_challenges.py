"""Observed challenge trajectories: non-overlapping rises, explicit censoring.

These operational glucose challenges are not identified meals. Grid is the
existing canonical five minutes, independent of native acquisition cadence.
"""

import numpy as np

from lingsante_pulse.cgm.diagnostics.challenge_support import proposal_support

MAX_EVENTS = 4
OFFSETS = np.arange(-12, 25, 3)  # -60 to +120 minutes around an observed peak.


def day_challenge_tokens(glucose, observed, cadence, start_minute=0):
    """Return four 33D event tokens and masks, without interpolation or fitting.

    13 signed relative values + 13 observation indicators + baseline, rise,
    rise duration, coverage, clock sin/cos, and zero age (filled by packer).
    Missing post-peak values remain unobserved, never zero-valued recovery.
    """
    raw = np.asarray(glucose, np.float32)
    mask = np.asarray(observed, bool)
    if (
        raw.ndim != 1
        or raw.shape != mask.shape
        or cadence <= 0
        or not np.isfinite(cadence)
    ):
        raise ValueError("Aligned one-dimensional CGM and positive cadence required")
    if not np.isfinite(raw[mask]).all():
        raise ValueError("Observed glucose must be finite")
    tokens = np.zeros((MAX_EVENTS, 33), np.float32)
    valid = np.zeros(MAX_EVENTS, bool)
    if mask.sum() < 2:
        return tokens, valid
    proposals = [
        p
        for p in proposal_support(raw, mask)
        if p["rising_coverage"] >= 0.5 * min(1.0, 5.0 / cadence)
    ]
    # Do not rank by recovery: this would prefer well-recovered challenges.
    proposals.sort(key=lambda p: (-float(raw[p["peak"]] - raw[p["start"]]), p["start"]))
    chosen = []
    for p in proposals:
        # Intersecting rise intervals cannot establish independent challenges.
        if any(
            max(p["start"], q["start"]) <= min(p["peak"], q["peak"]) for q in chosen
        ):
            continue
        chosen.append(p)
        if len(chosen) == MAX_EVENTS:
            break
    chosen.sort(key=lambda p: p["start"])
    for i, p in enumerate(chosen):
        positions = p["peak"] + OFFSETS
        inside = (positions >= 0) & (positions < len(raw))
        available = np.zeros(13, bool)
        available[inside] = mask[positions[inside]]
        relative = np.zeros(13, np.float32)
        relative[available] = (raw[positions[available]] - raw[p["start"]]) / 100.0
        clock = 2 * np.pi * ((start_minute + p["peak"] * 5) % 1440) / 1440.0
        meta = [
            raw[p["start"]] / 100.0,
            (raw[p["peak"]] - raw[p["start"]]) / 100.0,
            (p["peak"] - p["start"]) / 36.0,
            available.mean(),
            np.sin(clock),
            np.cos(clock),
            0.0,
        ]
        tokens[i] = np.r_[relative, available.astype(np.float32), meta]
        valid[i] = True
    return tokens, valid


def pack_challenge_tokens(data):
    """Re-use strict7d subject/source/split/cadence histories; no labels read."""
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        causal_history_indices,
    )
    from lingsante_pulse.cgm.models.parallel_longitudinal_models import (
        bounded_histories,
    )

    starts = np.asarray(data["starts"]).astype("datetime64[m]")
    minutes = starts.astype(np.int64)
    days = [
        day_challenge_tokens(x, m, float(c), int(t % 1440))
        for x, m, c, t in zip(
            data["glucose"], data["observed"], data["cadence"], minutes
        )
    ]
    histories = bounded_histories(
        causal_history_indices(
            data["subject"],
            starts,
            data["cohort"],
            data["split"],
            data["cadence"],
            data["source"],
            max_days=7,
        ),
        starts,
    )
    tokens = np.zeros((len(starts), 7 * MAX_EVENTS, 33), np.float32)
    mask = np.zeros((len(starts), 7 * MAX_EVENTS), bool)
    for i, history in enumerate(histories):
        rows = history[::-1]
        ages = minutes[i] - minutes[rows]
        if (
            rows[0] != i
            or np.any(ages < 0)
            or np.any(ages > 8640)
            or np.any(np.diff(ages) < 1440)
        ):
            raise ValueError("Noncausal, overlapping or oversized context")
        for key in ("subject", "cohort", "split", "source", "cadence"):
            if not np.all(data[key][rows] == data[key][i]):
                raise ValueError("Cross-provenance challenge")
        for j, row in enumerate(rows):
            dest = slice(j * MAX_EVENTS, (j + 1) * MAX_EVENTS)
            tokens[i, dest], mask[i, dest] = days[row]
            tokens[i, dest, -1] = np.where(mask[i, dest], ages[j] / (1440.0 * 7), 0.0)
    return dict(challenge_tokens=tokens, challenge_mask=mask)
