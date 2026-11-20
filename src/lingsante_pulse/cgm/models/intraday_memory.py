"""Gap-reset within-window memory before existing across-window aggregation."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.unmatched_correspondence import UnmatchedCorrespondence


def segment_order(anchor, active):
    # Invalid nodes remain between runs; ties retain original order.
    segment = (~active).long().cumsum(1) * 2 + active.long()
    by_value = torch.where(active, anchor, 0.0).argsort(dim=1, stable=True)
    by_segment = segment.gather(1, by_value).argsort(dim=1, stable=True)
    return by_value.gather(1, by_segment)


class IntradayMemory(UnmatchedCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.sorted_intraday_control = control
        self.intraday = nn.GRUCell(8, 8)
        self.intraday_writeback = nn.Linear(8, 8, bias=False)
        nn.init.zeros_(self.intraday_writeback.weight)

    def intraday_states(self, values, active, anchor):
        order = segment_order(anchor, active) if self.sorted_intraday_control else None
        values = torch.where(active[..., None], values, 0.0)
        if order is not None:
            values = values.gather(1, order[..., None].expand_as(values))
            active = active.gather(1, order)
        state = torch.zeros_like(values[:, 0])
        states = []
        for j in range(values.shape[1]):
            state = torch.where(
                active[:, j, None], self.intraday(values[:, j], state), 0.0
            )
            states.append(state)
        states = torch.stack(states, 1)
        if order is not None:
            states = states.gather(1, order.argsort(1)[..., None].expand_as(states))
        return states

    def node_messages(self, nodes, valid, elapsed):
        messages = super().node_messages(nodes, valid, elapsed)
        active = valid[..., None] & (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
        values = (
            self.lifts(torch.where(active[..., None], nodes[..., :15], 0.0)) + messages
        )
        b, d, t, _ = values.shape
        states = self.intraday_states(
            values.reshape(b * d, t, 8),
            active.reshape(b * d, t),
            nodes[..., 0].reshape(b * d, t),
        )
        return messages + self.intraday_writeback(states).reshape_as(messages)


def intraday_diagnostics(model):
    device = next(model.parameters()).device
    x = torch.arange(32, device=device, dtype=torch.float32).reshape(1, 4, 8) / 20
    active = torch.ones(1, 4, device=device, dtype=torch.bool)
    anchor = torch.tensor([[1.0, 3.0, 2.0, 5.0]], device=device)
    perm = torch.tensor([1, 0, 2, 3], device=device)
    with torch.no_grad():
        before = model.intraday_writeback(model.intraday_states(x, active, anchor))
        after = model.intraday_writeback(
            model.intraday_states(x[:, perm], active, anchor[:, perm])
        )[:, perm.argsort()]
        changed = x.clone()
        changed[:, :2] += 5
        active[:, 2] = False
        a = model.intraday_writeback(model.intraday_states(x, active, anchor))
        b = model.intraday_writeback(model.intraday_states(changed, active, anchor))
    return dict(
        order_writeback_difference=float((before - after).abs().max()),
        post_gap_max_error=float((a[:, 3:] - b[:, 3:]).abs().max()),
    )


def intraday_decision(candidate, control, support):
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
        "intraday_gradient",
        "intraday_order_response",
        "intraday_gap_reset",
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 9745, 146)
            for r in (candidate, control)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=support.get("supported_for_canary") is True,
        definitions=candidate["intraday_memory"]["sorted_control"] is False
        and control["intraday_memory"]["sorted_control"] is True
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
        policy="intraday_memory_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
