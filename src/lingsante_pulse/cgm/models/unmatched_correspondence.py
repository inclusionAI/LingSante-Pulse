"""Common-content acceptance versus equal-budget global message scaling."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.cross_window_correspondence import (
    CrossWindowCorrespondence,
)


def match_acceptance(scores, mass, admitted, bias, constant=False):
    mass = torch.where(admitted, mass, 0.0)
    prior = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
    log_evidence = torch.logsumexp(
        (scores + prior.clamp_min(1e-8).log()[:, None]).masked_fill(
            ~admitted[:, None], -1e4
        ),
        -1,
    )
    gate = torch.sigmoid(
        torch.zeros_like(log_evidence) - bias if constant else log_evidence - bias
    )
    return gate * admitted.any(-1)[:, None]


class UnmatchedCorrespondence(CrossWindowCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.global_gate_control = control
        self.unmatched_logit = nn.Parameter(torch.zeros(()))

    def node_messages(self, nodes, valid, elapsed):
        messages = super().node_messages(nodes, valid, elapsed)
        eligible = valid[..., None] & (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
        common = torch.where(
            eligible[..., None],
            nodes[..., [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
            0.0,
        )
        keys = self.key(common.flatten(1, 2))
        mass = nodes[..., -1].flatten(1, 2)
        gates = []
        for j in range(nodes.shape[1]):
            scores = self.query(common[:, j]) @ keys.transpose(1, 2) / (8**0.5)
            admitted = (
                eligible & (elapsed > elapsed[:, j : j + 1])[..., None]
            ).flatten(1, 2)
            g = match_acceptance(
                scores, mass, admitted, self.unmatched_logit, self.global_gate_control
            )
            gates.append(g * eligible[:, j])
        return messages * torch.stack(gates, 1)[..., None]


def unmatched_decision(candidate, control, support):
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
        "correspondence_gradient",
        "query_specific_attention",
        "unmatched_gradient",
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 9249, 146)
            for r in (candidate, control)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=support.get("candidate_support_passed") is True,
        definitions=candidate["unmatched_correspondence"]["global_gate_control"]
        is False
        and control["unmatched_correspondence"]["global_gate_control"] is True,
        identical_episodes=candidate["episode_sequence_sha256"]
        == control["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == control["target_sha256"],
    )
    c, g = candidate["learned"], control["learned"]
    gates["useful_conditional_content"] = (
        any(c[k][i] > g[k][i] for k in ("current", "longitudinal") for i in (1, 2))
        or candidate["event_ev"] > control["event_ev"]
    )
    gates["useful_progress"] = (
        c["current_mean"] > 0.48521103244274855
        or c["longitudinal_mean"] > 0.3768560569583168
        or candidate["event_ev"] > 0.34062106534838676
    )
    gates["useful_loss"] = candidate["episode_validation_loss"] < 4.149182319641113
    return dict(
        policy="unmatched_correspondence_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
