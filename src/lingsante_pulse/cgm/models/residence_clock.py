"""Common sampled-state age modulates the existing shared value path."""

import torch
from torch import nn
from torch.nn import functional as F
from lingsante_pulse.cgm.models.unmatched_correspondence import UnmatchedCorrespondence


def clock_codes(nodes, accumulated=False):
    common = nodes[..., [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
    active = (
        (nodes[..., 14] > 0) & (nodes[..., -1] > 0) & torch.isfinite(common).all(-1)
    )
    zone = torch.where(nodes[..., 0] < 0.7, 0, torch.where(nodes[..., 0] > 1.8, 2, 1))
    previous = F.pad(active[..., :-1], (1, 0), value=False)
    same = zone == F.pad(zone[..., :-1], (1, 0), value=-1)
    continuing = active & previous & same
    index = torch.arange(nodes.shape[-2], device=nodes.device).expand_as(zone)
    start = torch.where(continuing, 0, index).cummax(-1).values
    onehot = F.one_hot(zone, 3).to(nodes.dtype)
    if accumulated:
        total = (continuing[..., None] * onehot).cumsum(-2) * 0.25
        reset = torch.where(active[..., None], 0.0, total).cummax(-2).values
        span = total - reset
    else:
        span = ((index - start) * 0.25)[..., None]
    onset = (active & previous & ~same).gather(-1, start)
    return torch.cat(
        [onehot * span * active[..., None], (onset & active)[..., None]], -1
    )


class ResidenceClock(UnmatchedCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.accumulated_clock_control = control
        self.clock_lift = nn.Linear(4, 16)
        self.clock_writeback = nn.Linear(16, 8, bias=False)
        nn.init.zeros_(self.clock_writeback.weight)

    def clock_modulation(self, nodes, accumulated=None):
        codes = clock_codes(
            nodes,
            self.accumulated_clock_control if accumulated is None else accumulated,
        )
        codes = torch.cat([torch.log1p(codes[..., :3]), codes[..., 3:]], -1)
        active = (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
        return (
            torch.tanh(self.clock_writeback(F.silu(self.clock_lift(codes))))
            * active[..., None]
        )

    def node_messages(self, nodes, valid, elapsed):
        messages = super().node_messages(nodes, valid, elapsed)
        nodes = torch.where(valid[..., None, None], nodes, 0.0)
        active = (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
        values = (
            self.lifts(torch.where(active[..., None], nodes[..., :15], 0.0)) + messages
        )
        return messages + self.clock_modulation(nodes) * values


def clock_diagnostics(model):
    device = next(model.parameters()).device
    x = torch.zeros(1, 6, 16, device=device)
    x[..., 14:] = 1.0
    x[..., 0] = torch.tensor([2.0, 2.1, 1.0, 1.1, 2.0, 2.05], device=device)
    with torch.no_grad():
        c = model.clock_modulation(x, False)
        g = model.clock_modulation(x, True)
        prefix = x.clone()
        prefix[:, 4:, 0] = 0.5
        prefix_error = (
            (model.clock_modulation(prefix)[:, :4] - model.clock_modulation(x)[:, :4])
            .abs()
            .max()
        )
        x[:, 2, 14] = 0.0
        changed = x.clone()
        changed[:, :2, 0] = 0.5
        gap_error = (
            (model.clock_modulation(x)[:, 3:] - model.clock_modulation(changed)[:, 3:])
            .abs()
            .max()
        )
        changed = x.clone()
        changed[..., 1:3] = float("nan")
        private_error = (
            (model.clock_modulation(x) - model.clock_modulation(changed)).abs().max()
        )
    return dict(
        clock_writeback_difference=float((c - g).abs().max()),
        prefix_max_error=float(prefix_error),
        post_gap_max_error=float(gap_error),
        private_max_error=float(private_error),
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
    "clock_gradient",
    "clock_response",
    "clock_gap_reset",
    "clock_prefix_causal",
    "clock_private_isolation",
)


def clock_decision(candidate, control, support):
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in HARD_GATES
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 9457, 146)
            for r in (candidate, control)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=support.get("supported_for_canary") is True,
        definitions=candidate["residence_clock"]["accumulated_control"] is False
        and control["residence_clock"]["accumulated_control"] is True
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
        policy="residence_clock_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
