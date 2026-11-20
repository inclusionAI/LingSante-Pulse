"""Preserve physical level while exposing local common-view excursions."""

import torch
from lingsante_pulse.cgm.models.gap_recurrent_occupation import GapRecurrentOccupation


def anchored_path(path):
    supported = path[..., 14:15] > 0
    relative = torch.where(supported, path[..., 5:14] - path[..., 0:1], 0.0)
    return torch.cat([path[..., :5], relative, path[..., 14:]], -1)


class AnchoredTrajectory(GapRecurrentOccupation):
    def forward(self, values, features, support, elapsed_days, mask, path):
        return super().forward(
            values, features, support, elapsed_days, mask, anchored_path(path)
        )


def anchored_decision(candidate, reference):
    from lingsante_pulse.cgm.models.local_trajectory_occupation import (
        trajectory_support_valid,
    )

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
    challenge = any(
        candidate["learned"][k][1]
        > max(reference["learned"][k][1], candidate["baseline"][k][1])
        for k in ("current", "longitudinal")
    )
    gates = dict(
        structural_isolation=all(candidate["gates"].get(k) is True for k in hard),
        support=trajectory_support_valid(candidate["occupation_path"])
        and candidate["occupation_path"]["sorted_control"] is False,
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 8928, 146)
            for r in (candidate, reference)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, reference)
        ),
        candidate_definition=candidate.get("anchored_trajectory") is True
        and candidate["gap_recurrent"]["regular_control"] is False,
        frozen_control=reference["checkpoint_sha256"]
        == "a8d30ff4896a715c0a2013bf9279263da932f8160c18fe3d175ba074a5a4563c",
        identical_episodes=candidate["episode_sequence_sha256"]
        == reference["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == reference["target_sha256"],
        challenge_progress=challenge
        and candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"],
    )
    return dict(
        policy="anchored_trajectory_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
        challenge_ev_delta={
            k: candidate["learned"][k][1] - reference["learned"][k][1]
            for k in ("current", "longitudinal")
        },
    )
