"""Boundary-complete inputs to the unchanged anchored global encoder."""

import numpy as np
from lingsante_pulse.cgm.models.local_trajectory_occupation import (
    daily_local_trajectory,
    trajectory_support_valid,
)
from lingsante_pulse.cgm.models.window_boundary_support import boundary_paths


def add_boundary_trajectory(data, inputs):
    _, report = daily_local_trajectory(data)
    out = np.zeros((len(data["glucose"]), 7, 288, 16), np.float32)
    day_valid = inputs["support"][:, 0, 0] > 0
    recovered = 0
    for i, rows, old, new in boundary_paths(data, day_valid):
        if len(rows) != int(inputs["mask"][i].sum()):
            raise ValueError("History packing mismatch")
        out[i, : len(rows)] = new
        active = day_valid[rows, None] & (old[..., -1] > 0)
        recovered += int((active & (old[..., 14] == 0) & (new[..., 14] > 0)).sum())
    inputs["path"] = out
    report.update(
        cross_day_stitching=True,
        boundary_recovered_points=recovered,
        original_features_unchanged=True,
        original_supported_trajectories_unchanged=True,
        earliest_slot_unchanged=True,
        exact_common_timestamps_only=True,
    )
    return report


def boundary_decision(candidate, reference, support):
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
        "temporal_gradient",
        "temporal_order_response",
        "temporal_gap_response",
    )
    p = candidate["occupation_path"]
    challenge = any(
        candidate["learned"][k][1]
        > max(reference["learned"][k][1], candidate["baseline"][k][1])
        for k in ("current", "longitudinal")
    )
    gates = dict(
        structural_isolation=all(candidate["gates"].get(k) is True for k in hard),
        original_support=trajectory_support_valid(p)
        and support["candidate_support_passed"] is True,
        exact_boundary_support=p["boundary_recovered_points"]
        == sum(x["recovered_points"] for x in support["groups"].values())
        > 0
        and all(
            p.get(k) is True
            for k in (
                "cross_day_stitching",
                "original_features_unchanged",
                "original_supported_trajectories_unchanged",
                "earliest_slot_unchanged",
                "exact_common_timestamps_only",
            )
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 8928, 146)
            for r in (candidate, reference)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, reference)
        ),
        candidate_definition=candidate.get("boundary_trajectory") is True
        and candidate.get("anchored_trajectory") is True
        and candidate["gap_recurrent"]["regular_control"] is False,
        frozen_control=reference["checkpoint_sha256"]
        == "6ebdc4be4a03c4b30c771af148c27e1b068cf4c944ed1298233de0745c215513",
        identical_episodes=candidate["episode_sequence_sha256"]
        == reference["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == reference["target_sha256"],
        challenge_progress=challenge
        and candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"],
    )
    return dict(
        policy="boundary_trajectory_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
        challenge_ev_delta={
            k: candidate["learned"][k][1] - reference["learned"][k][1]
            for k in ("current", "longitudinal")
        },
    )
