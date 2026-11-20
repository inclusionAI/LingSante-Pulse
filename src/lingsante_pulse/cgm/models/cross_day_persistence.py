"""Matched, subject-balanced raw-state repeatability; no model or label reads."""

import numpy as np
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import bounded_histories

STATE_GROUPS = {
    "level": [0, 2, 3, 4, 5, 6, 7, 8],
    "variability": [1],
    "occupancy": [9, 10, 11],
    "phase_baseline": [12, 13, 14, 15],
}


def phase_supported(data):
    seen = np.asarray(data["observed"], bool) & np.isfinite(data["glucose"])
    starts = np.asarray(data["starts"]).astype("datetime64[m]").astype(np.int64)
    # Match the existing state16 clock intervals on the canonical5min grid.
    phase = (starts[:, None] + np.arange(seen.shape[1])[None, :] * 5) % 1440
    intervals = ((0, 360), (360, 600), (600, 1080), (1080, 1440))
    return np.stack(
        [(seen & (phase >= a) & (phase < b)).sum(1) >= 2 for a, b in intervals], 1
    ).all(1)


def eligible_pairs(data, valid):
    histories = bounded_histories(
        causal_history_indices(
            data["subject"],
            data["starts"],
            data["cohort"],
            data["split"],
            data["cadence"],
            data["source"],
            max_days=7,
        ),
        data["starts"],
    )
    starts = np.asarray(data["starts"]).astype("datetime64[m]")
    pairs = []
    for i, history in enumerate(histories):
        if not valid[i]:
            continue
        prior = [int(j) for j in history[:-1] if valid[j]]
        if not prior:
            continue
        j = prior[-1]
        assert np.timedelta64(1, "D") <= starts[i] - starts[j] <= np.timedelta64(6, "D")
        for key in ("subject", "cohort", "split", "cadence", "source"):
            assert data[key][i] == data[key][j], key
        pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def paired_repeatability(current, past, subjects):
    x = np.asarray(current, np.float64)
    y = np.asarray(past, np.float64)
    subjects = np.asarray(subjects).astype(str)
    if x.shape != y.shape or x.ndim != 2 or len(x) != len(subjects):
        raise ValueError("Unaligned pair arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite states")
    ids = np.unique(subjects)
    if len(ids) < 2:
        return dict(pairs=len(x), subjects=len(ids), evaluable=False)
    means = np.stack([y[subjects == s].mean(0) for s in ids])
    squares = np.stack([np.square(y[subjects == s]).mean(0) for s in ids])
    same = []
    different = []
    for k, s in enumerate(ids):
        rows = subjects == s
        other_mean = (means.sum(0) - means[k]) / (len(ids) - 1)
        other_square = (squares.sum(0) - squares[k]) / (len(ids) - 1)
        same.append(np.square(x[rows] - y[rows]).mean(0))
        different.append(
            (np.square(x[rows]) - 2 * x[rows] * other_mean + other_square).mean(0)
        )
    a = np.mean(same, 0)
    b = np.maximum(np.mean(different, 0), 0.0)
    valid = b > 1e-10
    advantage = np.divide(b - a, b, out=np.zeros_like(b), where=valid)
    groups = {}
    for name, index in STATE_GROUPS.items():
        use = np.asarray(index)[valid[index]]
        groups[name] = dict(
            valid_coordinates=len(use),
            advantage=float(advantage[use].mean()) if len(use) else None,
        )
    return dict(
        pairs=len(x),
        subjects=len(ids),
        evaluable=True,
        same_person_mse=a.tolist(),
        different_person_mse=b.tolist(),
        advantage=[float(v) if ok else None for v, ok in zip(advantage, valid)],
        groups=groups,
    )


def supports_next_experiment(result):
    if result.get("subjects", 0) < 8 or not result.get("evaluable", False):
        return False
    groups = result["groups"]
    values = {k: r["advantage"] for k, r in groups.items()}
    return (
        sum(v is not None for v in values.values()) >= 3
        and values["level"] is not None
        and values["level"] >= 0.10
        and any(v is not None and v >= 0.10 for k, v in values.items() if k != "level")
    )
