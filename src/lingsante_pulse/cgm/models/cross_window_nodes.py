"""Observed 15-minute cells for pre-pooling cross-window correspondence."""

from collections import defaultdict
import numpy as np
from lingsante_pulse.cgm.models.local_trajectory_occupation import (
    daily_local_trajectory,
)
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import bounded_histories


def recurrence_nodes(data):
    path, original = daily_local_trajectory(data)
    g = np.asarray(data["glucose"], np.float32)
    seen = np.asarray(data["observed"], bool) & np.isfinite(g)
    phase = np.stack([seen[:, k::3].sum(1) for k in range(3)], 1).argmax(1)
    nodes = np.zeros((len(g), 96, 16), np.float32)
    for i, first in enumerate(phase):
        for cell, anchor in enumerate(range(int(first), 288, 3)):
            block = path[i, anchor : min(anchor + 3, 288)]
            mass = block[:, -1].sum()
            if mass <= 0:
                continue
            nodes[i, cell] = path[i, anchor]
            nodes[i, cell, 1:3] = (block[:, 1:3] * block[:, -1:]).sum(0) / mass
            nodes[i, cell, 5:14] = (
                (path[i, anchor, 5:14] - path[i, anchor, 0])
                if path[i, anchor, 14]
                else 0
            )
            nodes[i, cell, -1] = mass
    np.testing.assert_allclose(
        nodes[..., -1].sum(1), path[..., -1].sum(1), rtol=1e-6, atol=1e-7
    )
    for k in (1, 2):
        np.testing.assert_allclose(
            (nodes[..., k] * nodes[..., -1]).sum(1),
            (path[..., k] * path[..., -1]).sum(1),
            rtol=1e-6,
            atol=1e-7,
        )
    return nodes, original


def node_histories(data, day_valid):
    nodes, _ = recurrence_nodes(data)
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
    for i, history in enumerate(histories):
        rows = np.asarray(history[::-1])
        assert rows[0] == i and (starts[i] - starts[rows].min()) <= np.timedelta64(
            6, "D"
        )
        for key in ("subject", "cohort", "split", "cadence", "source"):
            assert np.all(data[key][rows] == data[key][i]), key
        yield i, rows, nodes[rows], np.asarray(day_valid, bool)[rows]


def add_cross_window_nodes(data, inputs):
    from lingsante_pulse.cgm.models.local_trajectory_occupation import (
        add_local_trajectory,
    )

    report = add_local_trajectory(data, inputs)
    g = np.asarray(data["glucose"])
    seen = np.asarray(data["observed"], bool) & np.isfinite(g)
    phase = np.stack([seen[:, k::3].sum(1) for k in range(3)], 1).argmax(1)
    inputs["nodes"] = np.zeros((len(g), 7, 96, 16), np.float32)
    inputs["node_index"] = np.zeros((len(g), 7, 288), np.int64)
    for i, rows, nodes, _ in node_histories(data, inputs["support"][:, 0, 0] > 0):
        if len(rows) != int(inputs["mask"][i].sum()):
            raise ValueError("Node history mismatch")
        inputs["nodes"][i, : len(rows)] = nodes
        inputs["node_index"][i, : len(rows)] = np.maximum(
            (np.arange(288)[None, :] - phase[rows, None]) // 3, 0
        )
    report["cross_window_nodes"] = True
    return report


def summarize_node_support(data, day_valid):
    groups = defaultdict(
        lambda: dict(
            endpoints=0,
            historical_endpoints=0,
            paired_endpoints=0,
            directed_cross_window_pairs=0,
            complete_nodes=0,
            total_nodes=0,
            positive_private_nodes=0,
            negative_private_nodes=0,
        )
    )
    for i, rows, nodes, valid in node_histories(data, day_valid):
        if not valid[0]:
            continue
        key = f"{data['cohort'][i]}/split{int(data['split'][i])}"
        r = groups[key]
        r["endpoints"] += 1
        active = (nodes[..., -1] > 0) & valid[:, None]
        complete = active & (nodes[..., 14] > 0)
        counts = complete.sum(1).astype(np.int64)
        # Current-to-earlier and earlier-to-still-earlier; no same-window edges.
        pairs = sum(int(counts[j] * counts[j + 1 :].sum()) for j in range(len(rows)))
        r["historical_endpoints"] += int(valid.sum() > 1)
        r["paired_endpoints"] += int(pairs > 0)
        r["directed_cross_window_pairs"] += pairs
        r["complete_nodes"] += int(complete.sum())
        r["total_nodes"] += int(active.sum())
        r["positive_private_nodes"] += int((complete & (nodes[..., 1] > 0)).sum())
        r["negative_private_nodes"] += int((complete & (nodes[..., 2] > 0)).sum())
    for r in groups.values():
        r["historical_pair_fraction"] = r["paired_endpoints"] / max(
            r["historical_endpoints"], 1
        )
    public = [
        str(c)
        for c in np.unique(data["cohort"])
        if c != "private_source_private"
        and all(
            groups.get(f"{c}/split{s}", {}).get("paired_endpoints", 0) > 0
            for s in (0, 1)
        )
    ]
    split_fractions = {}
    for s in (0, 1):
        part = [r for k, r in groups.items() if k.endswith(f"/split{s}")]
        split_fractions[str(s)] = sum(r["paired_endpoints"] for r in part) / max(
            sum(r["historical_endpoints"] for r in part), 1
        )
    return dict(
        groups=dict(groups),
        supported_public_sources=public,
        split_pair_fractions=split_fractions,
        candidate_support_passed=len(public) >= 2
        and min(split_fractions.values()) >= 0.5,
        interpolation=False,
        cross_boundary_stitching=False,
        extra_observations=False,
        common_key_columns=[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        private_key_columns=[],
        note="Pairs are input support opportunities, not independently verified repeated meals or subjects.",
    )
