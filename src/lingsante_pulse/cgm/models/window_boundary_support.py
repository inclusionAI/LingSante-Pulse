"""Exact common-anchor continuation within already selected causal histories."""

from collections import defaultdict
import numpy as np
from lingsante_pulse.cgm.models.local_trajectory_occupation import (
    daily_local_trajectory,
)
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import bounded_histories


def boundary_paths(data, day_valid):
    daily, _ = daily_local_trajectory(data)
    g = np.asarray(data["glucose"], np.float32)
    seen = np.asarray(data["observed"], bool) & np.isfinite(g)
    phase = np.stack([seen[:, k::3].sum(1) for k in range(3)], 1).argmax(1)
    starts = np.asarray(data["starts"]).astype("datetime64[m]").astype(np.int64)
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
    for i, history in enumerate(histories):
        rows = np.asarray(history[::-1])
        old = daily[rows]
        new = old.copy()
        past = {}
        if rows[0] != i or starts[i] - starts[rows].min() > 6 * 1440:
            raise ValueError("Wrong causal span")
        for key in ("subject", "cohort", "split", "cadence", "source"):
            if not np.all(data[key][rows] == data[key][i]):
                raise ValueError("Cross provenance")
        for slot in range(len(rows) - 1, -1, -1):
            j = rows[slot]
            if not day_valid[j]:
                continue
            for pos in np.flatnonzero((old[slot, :, 14] == 0) & (old[slot, :, -1] > 0)):
                end = phase[j] + ((pos - phase[j]) // 3) * 3
                index = end - np.arange(8, -1, -1) * 3
                if index[0] >= 0:
                    continue
                values = []
                for k in index:
                    if k >= 0:
                        if not seen[j, k]:
                            break
                        values.append(float(g[j, k] / 100.0))
                    else:
                        stamp = int(starts[j] + 5 * k)
                        if stamp not in past:
                            break
                        values.append(past[stamp])
                if len(values) == 9:
                    new[slot, pos, 5:14] = values
                    new[slot, pos, 14] = 1.0
            for k in range(int(phase[j]), 288, 3):
                if seen[j, k]:
                    past[int(starts[j] + k * 5)] = float(g[j, k] / 100.0)
        # Other common trajectories, all physical/signed/clock/mass coordinates stay exact.
        assert np.array_equal(new[..., :5], old[..., :5]) and np.array_equal(
            new[..., -1], old[..., -1]
        )
        assert np.array_equal(new[old[..., 14] > 0], old[old[..., 14] > 0])
        assert np.array_equal(new[-1], old[-1])
        yield i, rows, old, new


def summarize_boundary_support(data, day_valid):
    groups = defaultdict(
        lambda: dict(
            endpoints=0,
            changed_endpoints=0,
            old_trajectory_points=0,
            new_trajectory_points=0,
            recovered_points=0,
            recovered_current_points=0,
        )
    )
    for i, rows, old, new in boundary_paths(data, day_valid):
        active = np.asarray(day_valid)[rows, None] & (old[..., -1] > 0)
        changed = active & (old[..., 14] == 0) & (new[..., 14] > 0)
        key = f"{data['cohort'][i]}/split{int(data['split'][i])}"
        r = groups[key]
        r["endpoints"] += 1
        r["changed_endpoints"] += int(changed.any())
        r["old_trajectory_points"] += int((active & (old[..., 14] > 0)).sum())
        r["new_trajectory_points"] += int((active & (new[..., 14] > 0)).sum())
        r["recovered_points"] += int(changed.sum())
        r["recovered_current_points"] += int(changed[0].sum())
    for r in groups.values():
        r["changed_endpoint_fraction"] = r["changed_endpoints"] / max(r["endpoints"], 1)
    supported = []
    for source in np.unique(data["cohort"]):
        if source == "private_source_private":
            continue
        if all(
            groups.get(f"{source}/split{k}", {}).get("recovered_points", 0) > 0
            for k in (0, 1)
        ):
            supported.append(str(source))
    splits_ok = all(
        sum(
            r["recovered_points"] for k, r in groups.items() if k.endswith(f"/split{s}")
        )
        > 0
        for s in (0, 1)
    )
    return dict(
        groups=dict(groups),
        supported_public_sources=supported,
        candidate_support_passed=bool(splits_ok and len(supported) >= 2),
        original_features_unchanged=True,
        original_supported_trajectories_unchanged=True,
        left_boundary_unchanged=True,
        interpolation=False,
        unselected_window_reads=False,
    )
