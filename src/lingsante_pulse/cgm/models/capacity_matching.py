"""One-sided capacity projection of an already accepted common correspondence."""

import torch
from lingsante_pulse.cgm.models.unmatched_correspondence import (
    match_acceptance,
    UnmatchedCorrespondence,
)


def capacity_weights(weights, target_mass, source_mass):
    p = target_mass / target_mass.sum(-1, keepdim=True).clamp_min(1e-8)
    b = source_mass / source_mass.sum(-1, keepdim=True).clamp_min(1e-8)
    load = (p[..., None] * weights).sum(-2)
    factor = (b / load.clamp_min(1e-8)).clamp(max=1.0)
    capped = weights * factor[..., None, :]
    original = (p * weights.sum(-1)).sum(-1)
    retained = (p * capped.sum(-1)).sum(-1)
    # No accepted mass means no removal, rather than a 100% removal event.
    rho = torch.where(
        original > 0, retained / original.clamp_min(1e-8), torch.ones_like(original)
    )
    return capped, weights * rho[..., None, None], rho, load, b


def accepted_weights(model, nodes, valid, elapsed, j=0):
    eligible = valid[..., None] & (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
    common = torch.where(
        eligible[..., None], nodes[..., [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], 0.0
    )
    admitted = (eligible & (elapsed > elapsed[:, j : j + 1])[..., None]).flatten(1, 2)
    source_mass = torch.where(admitted, nodes[..., -1].flatten(1, 2), 0.0)
    target_mass = torch.where(eligible[:, j], nodes[:, j, :, -1], 0.0)
    scores = (
        model.query(common[:, j])
        @ model.key(common.flatten(1, 2)).transpose(1, 2)
        / (8**0.5)
    )
    gate = match_acceptance(scores, source_mass, admitted, model.unmatched_logit, False)
    weights = model.attention_for_day(nodes, valid, elapsed, j) * gate[..., None]
    return weights, target_mass, source_mass


class CapacityMatching(UnmatchedCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.uniform_capacity_control = control

    def node_messages(self, nodes, valid, elapsed):
        nodes = torch.where(valid[..., None, None], nodes, 0.0)
        value = self.lifts(
            torch.where((nodes[..., -1] > 0)[..., None], nodes[..., :15], 0.0)
        )
        source = value.flatten(1, 2)
        messages = []
        for j in range(nodes.shape[1]):
            w, p, b = accepted_weights(self, nodes, valid, elapsed, j)
            capped, uniform, _, _, _ = capacity_weights(w, p, b)
            weights = uniform if self.uniform_capacity_control else capped
            current = weights.sum(-1)[..., None] * value[:, j]
            # Linear writeback includes acceptance without re-normalizing rows.
            messages.append(
                self.message(torch.cat([current, weights @ source - current], -1))
            )
        return torch.stack(messages, 1)


def capacity_diagnostics(model, nodes, valid, elapsed):
    out = dict(
        column_excess=0.0,
        row_excess=0.0,
        equal_mass_error=0.0,
        selective_difference=0.0,
    )
    with torch.no_grad():
        for start in range(0, len(nodes), 32):
            x, v, t = (
                nodes[start : start + 32],
                valid[start : start + 32],
                elapsed[start : start + 32],
            )
            for j in range(x.shape[1]):
                w, p, b = accepted_weights(model, x, v, t, j)
                c, g, _, _, cap = capacity_weights(w, p, b)
                p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
                checks = dict(
                    column_excess=float(
                        ((p[..., None] * c).sum(-2) - cap).clamp_min(0).max()
                    ),
                    row_excess=float((c.sum(-1) - w.sum(-1)).clamp_min(0).max()),
                    equal_mass_error=float(
                        ((p[..., None] * (c - g)).sum((-1, -2))).abs().max()
                    ),
                    selective_difference=float((c - g).abs().max()),
                )
                for k in out:
                    out[k] = max(out[k], checks[k])
    return out


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
    "capacity_columns",
    "capacity_rows",
    "capacity_equal_mass",
    "capacity_selective_response",
)


def capacity_decision(candidate, control, support):
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in HARD_GATES
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
        support=support.get("supported_for_canary") is True,
        definitions=candidate["capacity_matching"]["uniform_control"] is False
        and control["capacity_matching"]["uniform_control"] is True
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
        policy="capacity_matching_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
