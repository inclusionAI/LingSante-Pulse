"""Timestamp-deduplicated history observations with unchanged occupation weights."""

from collections import defaultdict
import numpy as np
import torch
from lingsante_pulse.cgm.models.occupation_path import (
    OccupationPath,
    daily_path,
    occupation_statistics,
)


def merge_history_points(rows, starts, glucose, observed, paths, left, current):
    all_t = []
    all_g = []
    ft = []
    fv = []
    for row in rows:
        pos = np.flatnonzero(observed[row] & np.isfinite(glucose[row]))
        t = starts[row] + pos * 5
        keep = (t >= left) & (t < current + 1440)
        pos, t = pos[keep], t[keep]
        all_t.append(t)
        all_g.append(glucose[row, pos])
        supported = (t < current) & (paths[row, pos, -1] > 0)
        ft.append(t[supported])
        fv.append(paths[row, pos[supported]])
    if not rows:
        return np.zeros((0, 6), np.float32), np.zeros(0, np.int64)
    t = np.concatenate(all_t)
    g = np.concatenate(all_g)
    order = np.argsort(t)
    t, g = t[order], g[order]
    if len(t):
        _, first = np.unique(t, return_index=True)
        if np.any(np.maximum.reduceat(g, first) - np.minimum.reduceat(g, first) > 1e-3):
            raise ValueError("Conflicting glucose at duplicated timestamp")
    t = np.concatenate(ft)
    v = np.concatenate(fv)
    # np.unique returns sorted timestamps and the first source in priority order.
    unique, first = np.unique(t, return_index=True)
    return v[first], unique


def add_observed_union(data, inputs, selected_only=False):
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        causal_history_indices,
    )
    from lingsante_pulse.cgm.models.parallel_longitudinal_models import (
        bounded_histories,
    )

    paths, report = daily_path(data)
    starts = np.asarray(data["starts"]).astype("datetime64[m]").astype(np.int64)
    groups = defaultdict(list)

    def key(i):
        return tuple(
            str(data[k][i]) for k in ("subject", "cohort", "split", "cadence", "source")
        )

    for i in range(len(paths)):
        groups[key(i)].append(i)
    for group in groups.values():
        group.sort(key=lambda i: (starts[i], i))
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
    past = []
    has_history = []
    totals = defaultdict(
        lambda: dict(selected_points=0, union_points=0, extra_points=0)
    )
    valid = inputs["mask"] & (inputs["support"][:, :, 0] > 0)
    current_paths = paths.copy()
    for i, h in enumerate(histories):
        selected = list(h[::-1])
        assert selected[0] == i and len(selected) == int(inputs["mask"][i].sum())
        assert (
            all(key(j) == key(i) for j in selected)
            and starts[i] - starts[selected].min() <= 6 * 1440
        )
        active = [int(j) for slot, j in enumerate(selected) if valid[i, slot]]
        if not valid[i, 0]:
            current_paths[i] = 0
        supported_days = sum(bool(paths[j, :, -1].sum() > 0) for j in active)
        has_history.append(supported_days > 1)
        left = int(starts[selected].min())
        old, old_t = merge_history_points(
            active, starts, data["glucose"], data["observed"], paths, left, starts[i]
        )
        rows = active
        if not selected_only and supported_days > 1:
            inactive = set(selected) - set(active)
            rows = active + [
                j
                for j in groups[key(i)]
                if j not in selected
                and j not in inactive
                and starts[j] <= starts[i]
                and starts[j] + 1440 > left
            ]
        merged, t = (
            (old, old_t)
            if selected_only
            else merge_history_points(
                rows, starts, data["glucose"], data["observed"], paths, left, starts[i]
            )
        )
        old_indices = np.searchsorted(t, old_t)
        if len(old_t):
            assert np.array_equal(t[old_indices], old_t) and np.array_equal(
                merged[old_indices], old
            )
        past.append(merged)
        name = f"split{int(data['split'][i])}/{data['cohort'][i]}"
        r = totals[name]
        current = int(np.count_nonzero(current_paths[i, :, -1]))
        r["selected_points"] += current + len(old)
        r["union_points"] += current + len(merged)
        r["extra_points"] += len(merged) - len(old)
    maximum = 288 + max(map(len, past), default=0)
    packed = np.zeros((len(paths), maximum, 6), np.float32)
    packed[:, :288] = current_paths
    for i, p in enumerate(past):
        packed[i, 288 : 288 + len(p)] = p
    inputs["union_path"] = packed
    inputs["union_has_history"] = np.asarray(has_history, bool)
    report.update(
        selected_only=bool(selected_only),
        points_by_source_split=dict(totals),
        maximum_packed_points=maximum,
        extra_points=sum(r["extra_points"] for r in totals.values()),
        original_point_features_unchanged=True,
        duplicate_conflicts=0,
        maximum_span_days=max(
            (starts[i] - starts[h].min()) / 1440 + 1 for i, h in enumerate(histories)
        ),
    )
    return report


def validate_observed_union_export(report, plan):
    if (
        plan.get("policy") != "observed_union_post_canary_exploration_v1"
        or plan.get("exploratory_followup") is not True
        or plan.get("original_canary_acceptance_passed") is not False
    ):
        raise ValueError(
            "Explicit union exploration required; original failure is preserved"
        )
    if (
        plan.get("canary_checkpoint_sha256")
        != "cc5014b0d2898e203d2bc92530229b1b18a388cb27122497b44e97417f88266c"
    ):
        raise ValueError("Wrong canary")
    if (
        report["kind"],
        report["steps"],
        report["seed"],
        report["parameters"],
        report["final_dimension"],
    ) != ("candidate", 6960, 43, 5488, 146):
        raise ValueError("Wrong full model")
    if (
        report.get("observed_union") is not True
        or report.get("checkpoint_selection") != "last_step"
    ):
        raise ValueError("Wrong union model or checkpoint selection")
    hard = (
        "finite",
        "initial_stable",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
        "shared_lift_gradient",
        "two_day_update",
        "slot_permutation",
    )
    if not all(report["gates"].get(k) is True for k in hard):
        raise ValueError("Structural or isolation failure")
    if (
        report.get("phenotype_or_private_test_reads") != 0
        or report.get("glucofm_components") != 0
    ):
        raise ValueError("Forbidden training inputs")
    p = report["occupation_path"]
    if (
        p["selected_only"]
        or p["extra_points"] <= 0
        or p["duplicate_conflicts"] != 0
        or p["original_point_features_unchanged"] is not True
        or p["maximum_span_days"] > 7
    ):
        raise ValueError("Invalid observation union")
    if p["retained_observed_fraction"] < 0.8 or p["signed_anchor_residual_max"] > 1e-6:
        raise ValueError("Observation contract failed")


class ObservedUnionOccupation(OccupationPath):
    def __init__(self):
        super().__init__(control=True)

    def forward(
        self,
        values,
        features,
        support,
        elapsed_days,
        mask,
        union_path,
        union_has_history,
    ):
        valid = mask & (support[:, :, 0] > 0)
        w = torch.where(valid, support[:, :, 0], 0.0)
        stable = (
            torch.where(valid[:, :, None], values, 0.0)
            * (w / w.sum(1, keepdim=True).clamp_min(1e-8))[:, :, None]
        ).sum(1)
        weight = union_path[:, :, -1]
        h = self.lifts(torch.where(weight[:, :, None] > 0, union_path[:, :, :5], 0.0))
        stats = occupation_statistics(h, weight, True)
        return stable + self.readout(stats) * union_has_history[:, None], stats


def union_decision(candidate, reference):
    hard = (
        "finite",
        "initial_stable",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
        "shared_lift_gradient",
        "two_day_update",
        "slot_permutation",
    )
    useful = (
        any(
            candidate["learned"][k] > reference["learned"][k]
            for k in ("current_mean", "longitudinal_mean")
        )
        or candidate["event_ev"] > reference["event_ev"]
    )
    gates = dict(
        structural_isolation=all(candidate["gates"].get(k) is True for k in hard),
        forbidden_inputs_absent=candidate["phenotype_or_private_test_reads"] == 0
        and candidate["glucofm_components"] == 0,
        same_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 5488, 146)
            for r in (candidate, reference)
        ),
        useful_progress=candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"]
        and useful,
        actual_observation_gain=candidate["occupation_path"]["extra_points"] > 0
        and candidate["occupation_path"]["original_point_features_unchanged"] is True,
        support=candidate["occupation_path"]["retained_observed_fraction"] >= 0.8
        and candidate["occupation_path"]["duplicate_conflicts"] == 0,
        frozen_reference=reference["checkpoint_sha256"]
        == "1573ae71f70b33d21da8bad4d3d6dbfb6691393cae07af7e6079767fbd29ecbc",
        identical_episodes=candidate["episode_sequence_sha256"]
        == reference["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == reference["target_sha256"],
    )
    return dict(
        policy="observed_union_occupation_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
