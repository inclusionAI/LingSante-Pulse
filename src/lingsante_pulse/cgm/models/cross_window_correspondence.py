"""Common-key correspondence updates dense point lifts before daily pooling."""

import torch
from torch import nn
from torch.nn import functional as F
from lingsante_pulse.cgm.models.anchored_trajectory import anchored_path
from lingsante_pulse.cgm.models.gap_recurrent_occupation import (
    GapRecurrentOccupation,
    synthetic_inputs,
)
from lingsante_pulse.cgm.models.occupation_path import occupation_statistics


class CrossWindowCorrespondence(GapRecurrentOccupation):
    def __init__(self, control=False):
        super().__init__(control=False)  # Both arms retain actual elapsed-time memory.
        self.pooled_query_control = control
        self.query = nn.Linear(12, 8, bias=False)
        self.key = nn.Linear(12, 8, bias=False)
        self.message = nn.Linear(16, 8, bias=False)
        nn.init.zeros_(self.message.weight)

    def attention_for_day(self, nodes, valid, elapsed, j):
        common = nodes[..., [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
        eligible = valid[..., None] & (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
        common = torch.where(eligible[..., None], common, 0.0)
        raw_query = common[:, j]
        if self.pooled_query_control:
            mass = torch.where(eligible[:, j], nodes[:, j, :, -1], 0.0)
            raw_query = (
                (raw_query * mass[..., None]).sum(1)
                / mass.sum(1, keepdim=True).clamp_min(1e-8)
            )[:, None].expand_as(raw_query)
        q = self.query(raw_query)
        k = self.key(common.flatten(1, 2))
        older = elapsed > elapsed[:, j : j + 1]
        admitted = (eligible & older[..., None]).flatten(1, 2)
        source_mass = nodes[..., -1].flatten(1, 2)
        scores = (
            q @ k.transpose(1, 2) / (8**0.5)
            + source_mass.clamp_min(1e-8).log()[:, None]
        )
        weights = (
            scores.masked_fill(~admitted[:, None], -1e4).softmax(-1) * admitted[:, None]
        )
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return weights * eligible[:, j, :, None]

    def node_messages(self, nodes, valid, elapsed):
        nodes = torch.where(valid[..., None, None], nodes, 0.0)
        value = self.lifts(
            torch.where((nodes[..., -1] > 0)[..., None], nodes[..., :15], 0.0)
        )
        source = value.flatten(1, 2)
        messages = []
        for j in range(nodes.shape[1]):
            weights = self.attention_for_day(nodes, valid, elapsed, j)
            matched = weights @ source
            msg = self.message(torch.cat([value[:, j], matched - value[:, j]], -1))
            messages.append(msg * (weights.sum(-1) > 0)[..., None])
        return torch.stack(messages, 1)

    def statistics_residual(self, h, path, valid):
        return 0.0

    def path_lift(self, path):
        return self.lifts(torch.where(path[..., -1, None] > 0, path[..., :15], 0.0))

    def forward(
        self, values, features, support, elapsed_days, mask, path, nodes, node_index
    ):
        valid = mask & (support[:, :, 0] > 0)
        order = torch.where(valid, elapsed_days, -torch.inf).argsort(1, descending=True)
        values, support, elapsed_days, mask, path, nodes, node_index = tuple(
            v.gather(1, order.reshape(*order.shape, *([1] * (v.ndim - 2))).expand_as(v))
            for v in (values, support, elapsed_days, mask, path, nodes, node_index)
        )
        valid = mask & (support[:, :, 0] > 0)
        w = torch.where(valid, support[:, :, 0], 0.0)
        stable = (
            torch.where(valid[:, :, None], values, 0.0)
            * (w / w.sum(1, keepdim=True).clamp_min(1e-8))[:, :, None]
        ).sum(1)
        path = anchored_path(torch.where(valid[:, :, None, None], path, 0.0))
        nodes = torch.where(valid[:, :, None, None], nodes, 0.0)
        mass = path[..., -1].sum(2)
        valid = valid & (mass > 0)
        b, d, t, _ = path.shape
        h = self.path_lift(path)
        extra = self.statistics_residual(h, path, valid)
        msg = self.node_messages(nodes, valid, elapsed_days)
        msg = msg.gather(2, node_index[..., None].expand(b, d, t, 8))
        h = h + msg * (path[..., -1] > 0)[..., None]
        global_stats = occupation_statistics(
            h.flatten(1, 2), path[..., -1].flatten(1, 2), True
        )
        daily = occupation_statistics(
            h.reshape(b * d, t, 8), path[..., -1].reshape(b * d, t), True
        ).reshape(b, d, 36)
        state = torch.zeros(b, 16, device=values.device, dtype=values.dtype)
        pooled = torch.zeros_like(state)
        previous = torch.zeros_like(elapsed_days[:, 0])
        seen = torch.zeros_like(valid[:, 0])
        for j in range(d):
            active = valid[:, j]
            delta = torch.where(
                seen & active, (previous - elapsed_days[:, j]).clamp_min(0.0), 0.0
            )
            relaxed = state * torch.exp(
                -F.softplus(self.decay)[None, :] * delta[:, None]
            )
            new = self.memory(
                torch.cat(
                    [daily[:, j], mass[:, j, None], torch.log1p(delta)[:, None]], 1
                ),
                relaxed,
            )
            state = torch.where(active[:, None], new, state)
            pooled = pooled + state * torch.where(active, mass[:, j], 0.0)[:, None]
            previous = torch.where(active, elapsed_days[:, j], previous)
            seen = seen | active
        pooled = pooled / mass.sum(1, keepdim=True).clamp_min(1e-8)
        stats = global_stats + self.bridge(pooled) + extra
        return stable + self.readout(stats) * (valid.sum(1) > 1)[:, None], stats


def synthetic_node_inputs(levels, times, device="cpu"):
    p = synthetic_inputs(levels, times, device)
    p["nodes"] = anchored_path(p["path"][:, :, ::3].clone())
    p["nodes"][..., -1] *= 3
    p["node_index"] = (torch.arange(9, device=device) // 3)[None, None].expand(
        p["path"].shape[0], p["path"].shape[1], 9
    )
    return p


def cross_window_temporal_collision(model, prepare_inputs=None):
    device = next(model.parameters()).device
    with torch.no_grad():
        p = synthetic_node_inputs(
            [[2, 2, 2, 1, 1, 1, 1], [2, 1, 2, 1, 2, 1, 1]],
            [6, 5, 4, 3, 2, 1, 0],
            device,
        )
        if prepare_inputs is not None:
            p = prepare_inputs(p)
        z, s = model(**p)
        a = synthetic_node_inputs([[2, 1, 2, 1]], [3, 2, 1, 0], device)
        b = synthetic_node_inputs([[2, 1, 2, 1]], [4.5, 3, 1, 0], device)
        if prepare_inputs is not None:
            a, b = prepare_inputs(a), prepare_inputs(b)
        za, sa = model(**a)
        zb, sb = model(**b)
    return dict(
        day_order_embedding_difference=float((z[0] - z[1]).abs().max()),
        day_order_statistics_difference=float((s[0] - s[1]).abs().max()),
        gap_embedding_difference=float((za - zb).abs().max()),
        gap_statistics_difference=float((sa - sb).abs().max()),
    )


def query_collision(model):
    device = next(model.parameters()).device
    p = synthetic_node_inputs([[1.0, 2.0]], [1.0, 0.0], device)
    nodes = p["nodes"].clone()
    nodes[:, :, :, 5:14] += torch.arange(3, device=device)[None, None, :, None] / 3
    changed = nodes.clone()
    changed[:, 1] = changed[:, 1, [2, 1, 0]]
    with torch.no_grad():
        a = model.attention_for_day(nodes, p["mask"], p["elapsed_days"], 1)
        b = model.attention_for_day(changed, p["mask"], p["elapsed_days"], 1)
    return float((a - b).abs().max())


def correspondence_decision(candidate, control, support):
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
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 9248, 146)
            for r in (candidate, control)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=support["candidate_support_passed"] is True,
        definitions=candidate["cross_window_correspondence"]["pooled_query_control"]
        is False
        and control["cross_window_correspondence"]["pooled_query_control"] is True,
        identical_episodes=candidate["episode_sequence_sha256"]
        == control["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == control["target_sha256"],
        useful_challenge=any(
            candidate["learned"][k][1]
            > max(control["learned"][k][1], candidate["baseline"][k][1])
            for k in ("current", "longitudinal")
        ),
        useful_loss=candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"],
    )
    return dict(
        policy="cross_window_correspondence_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
