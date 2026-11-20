"""Common-only channel filters retain edge/value association before pooling."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.unmatched_correspondence import (
    UnmatchedCorrespondence,
    match_acceptance,
)

COMMON = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def filtered_values(attention, filters, source, current, control=False):
    mean_filter = (attention[..., None] * filters).sum(-2)
    if control:
        return mean_filter * (attention @ source - current)
    return (
        torch.einsum("bqs,bqsc,bsc->bqc", attention, filters, source)
        - mean_filter * current
    )


class EdgeConditionedValue(UnmatchedCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.mean_field_control = control
        self.edge_filter = nn.Linear(12, 8, bias=False)
        self.edge_writeback = nn.Linear(8, 8, bias=False)
        nn.init.zeros_(self.edge_writeback.weight)

    def edge_messages(self, nodes, valid, elapsed, control=None):
        control = self.mean_field_control if control is None else control
        nodes = torch.where(valid[..., None, None], nodes, 0.0)
        eligible = valid[..., None] & (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
        common = torch.where(eligible[..., None], nodes[..., COMMON], 0.0)
        value = self.lifts(
            torch.where((nodes[..., -1] > 0)[..., None], nodes[..., :15], 0.0)
        )
        source = value.flatten(1, 2)
        keys = self.key(common.flatten(1, 2))
        projected = self.edge_filter(common)
        source_filter = projected.flatten(1, 2)
        mass = nodes[..., -1].flatten(1, 2)
        messages = []
        for j in range(nodes.shape[1]):
            attention = self.attention_for_day(nodes, valid, elapsed, j)
            scores = self.query(common[:, j]) @ keys.transpose(1, 2) / (8**0.5)
            admitted = (
                eligible & (elapsed > elapsed[:, j : j + 1])[..., None]
            ).flatten(1, 2)
            gate = (
                match_acceptance(scores, mass, admitted, self.unmatched_logit, False)
                * eligible[:, j]
            )
            # Project before broadcasting: U(cs-cq)=Ucs-Ucq, without pairwise12D storage.
            filters = torch.tanh(source_filter[:, None] - projected[:, j, :, None])
            content = filtered_values(attention, filters, source, value[:, j], control)
            messages.append(self.edge_writeback(content) * gate[..., None])
        return torch.stack(messages, 1)

    def node_messages(self, nodes, valid, elapsed):
        return super().node_messages(nodes, valid, elapsed) + self.edge_messages(
            nodes, valid, elapsed
        )


def edge_diagnostics(model):
    from lingsante_pulse.cgm.models.cross_window_correspondence import (
        synthetic_node_inputs,
    )

    device = next(model.parameters()).device
    p = synthetic_node_inputs([[1.0, 3.0, 2.0, 4.0]], [4.5, 3.0, 1.0, 0.0], device)
    x = p["nodes"].clone()
    x[:, :, 1, 0] += 0.7
    x[:, :, 2, 1] += 0.2
    v, t = p["mask"], p["elapsed_days"]
    with torch.no_grad():
        a = model.edge_messages(x, v, t, False)
        b = model.edge_messages(x, v, t, True)
        changed = x.clone()
        changed[:, 3] = 99.0
        future_error = (
            (
                model.edge_messages(changed, v, t)[:, :3]
                - model.edge_messages(x, v, t)[:, :3]
            )
            .abs()
            .max()
        )
        altered = x.clone()
        altered[..., 1:3] = float("nan")
        filter_error = (
            (
                model.edge_filter(x[..., COMMON])
                - model.edge_filter(altered[..., COMMON])
            )
            .abs()
            .max()
        )
    return dict(
        association_writeback_difference=float((a - b).abs().max()),
        future_message_max_error=float(future_error),
        private_filter_max_error=float(filter_error),
    )


HARD_GATES = (
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
    "edge_gradient",
    "edge_association_response",
    "edge_future_isolation",
    "edge_private_filter_isolation",
)


def edge_decision(candidate, control, support):
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in HARD_GATES
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 9409, 146)
            for r in (candidate, control)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=support.get("supported_for_canary") is True,
        definitions=candidate["edge_conditioned_value"]["mean_field_control"] is False
        and control["edge_conditioned_value"]["mean_field_control"] is True
        and all(
            r["unmatched_correspondence"]["global_gate_control"] is False
            for r in (candidate, control)
        ),
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
        c["current_mean"] > 0.48457906767725945
        or c["longitudinal_mean"] > 0.37659279330663625
        or candidate["event_ev"] > 0.3407187722623348
    )
    gates["useful_loss"] = candidate["episode_validation_loss"] < 4.149182319641113
    return dict(
        policy="edge_conditioned_value_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
