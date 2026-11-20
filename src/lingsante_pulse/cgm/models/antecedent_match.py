"""Match on early observed conditions; keep later responses in values."""

import math
import numpy as np
import torch
from torch import nn
from lingsante_pulse.cgm.models.distribution_phase import DistributionBarycenter


def matching_coordinates(common, control=False):
    # common = endpoint level, sin/cos(endpoint), nine endpoint-relative values.
    first = 9 if control else 3
    minutes = 30 if control else 120
    angle = 2 * math.pi * minutes / 1440
    start = common[..., 0] + common[..., first]
    rise15 = common[..., first + 1] - common[..., first]
    rise30 = common[..., first + 2] - common[..., first]
    sine = common[..., 1] * math.cos(angle) - common[..., 2] * math.sin(angle)
    cosine = common[..., 2] * math.cos(angle) + common[..., 1] * math.sin(angle)
    parts = (start, rise15, rise30, sine, cosine)
    return (
        torch.stack(parts, -1)
        if isinstance(common, torch.Tensor)
        else np.stack(parts, -1)
    )


class ConditionProjection(nn.Module):
    def __init__(self, control=False):
        super().__init__()
        self.control = control
        self.projection = nn.Linear(5, 8, bias=False)

    def forward(self, common):
        return self.projection(matching_coordinates(common, self.control))


class AntecedentMatch(DistributionBarycenter):
    def __init__(self, control=False):
        super().__init__(
            control=False
        )  # Both arms retain the successful daily barycenter.
        self.match_tail_control = control
        self.query = ConditionProjection(control)
        self.key = ConditionProjection(control)


def matching_diagnostics(model):
    from lingsante_pulse.cgm.models.cross_window_correspondence import (
        synthetic_node_inputs,
    )

    device = next(model.parameters()).device
    p = synthetic_node_inputs([[1.0, 2.0, 3.0]], [2.0, 1.0, 0.0], device)
    common = p["nodes"][..., [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].clone()

    # Change physical phases, then re-anchor to the updated endpoint.
    def changed(first, last):
        x = common.clone()
        g = x[..., 3:] + x[..., :1]
        g[..., first:last] += 0.4
        x[..., 0] = g[..., -1]
        x[..., 3:] = g - g[..., -1:]
        return x

    irrelevant = changed(0, 6) if model.match_tail_control else changed(3, 9)
    relevant = changed(6, 9) if model.match_tail_control else changed(0, 3)
    with torch.no_grad():
        error = max(
            float((module(common) - module(irrelevant)).abs().max())
            for module in (model.query, model.key)
        )
        response = max(
            float((module(common) - module(relevant)).abs().max())
            for module in (model.query, model.key)
        )
        valid = p["mask"] & (p["support"][..., 0] > 0)
        nodes = p["nodes"].clone()
        nodes[..., 1:3] += 100.0
        private = max(
            float(
                (
                    model.attention_for_day(p["nodes"], valid, p["elapsed_days"], j)
                    - model.attention_for_day(nodes, valid, p["elapsed_days"], j)
                )
                .abs()
                .max()
            )
            for j in range(3)
        )
    return dict(
        irrelevant_phase_key_error=error,
        relevant_phase_key_difference=response,
        private_attention_error=private,
    )


def selection_witness(nodes, valid):
    active = valid[:, None] & (nodes[..., 14] > 0) & (nodes[..., -1] > 0)
    if len(nodes) < 2 or not active[0].any() or not active[1:].any():
        return 0.0
    common = nodes[..., [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
    query = common[0, active[0]]
    source = common[1:][active[1:]]
    mass = nodes[1:, :, -1][active[1:]]
    response = -nodes[1:, :, 7][active[1:]]  # g120-g30, in units of100mg/dL.
    estimates = []
    for control in (False, True):
        q = matching_coordinates(query, control)
        k = matching_coordinates(source, control)
        logits = -0.5 * (
            (q * q).sum(-1)[:, None] + (k * k).sum(-1)[None, :] - 2 * q @ k.T
        )
        logits += np.log(mass)[None, :]
        logits -= logits.max(-1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(-1, keepdims=True)
        estimates.append(w @ response)
    query_mass = nodes[0, active[0], -1]
    query_mass = query_mass / query_mass.sum()
    return float(np.sqrt(np.sum(query_mass * (estimates[0] - estimates[1]) ** 2)))


def canary_decision(candidate, control, support):
    from lingsante_pulse.cgm.models.distribution_phase import HARD_GATES

    hard = HARD_GATES + (
        "matching_condition_isolation",
        "matching_condition_response",
        "matching_private_isolation",
    )
    reports = (candidate, control)
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in reports for k in hard
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 10577, 146)
            for r in reports
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in reports
        ),
        support=support.get("supported_for_canary") is True
        and support["route"] == "antecedent_match",
        definitions=all(
            r["distribution_phase"]["route"] == "antecedent_match"
            and r["antecedent_match"]["distribution_control"] is False
            and r["unmatched_correspondence"]["global_gate_control"] is False
            for r in reports
        )
        and candidate["antecedent_match"]["tail_control"] is False
        and control["antecedent_match"]["tail_control"] is True,
        identical_episodes=candidate["episode_sequence_sha256"]
        == control["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == control["target_sha256"],
    )
    c, g = candidate["learned"], control["learned"]
    gates["useful_conditional_content"] = (
        any(c[k][i] > g[k][i] for k in ("current", "longitudinal") for i in (1, 2))
        or candidate["event_ev"] > control["event_ev"]
    )
    ref = (0.4853538740426302, 0.37716955442221894, 0.34321602061390877)
    values = (c["current_mean"], c["longitudinal_mean"], candidate["event_ev"])
    gates["reference_preserved"] = all(v >= r - 0.01 for v, r in zip(values, ref))
    gates["useful_progress"] = any(v > r for v, r in zip(values, ref))
    gates["useful_loss"] = candidate["episode_validation_loss"] < 4.149182319641113
    return dict(
        policy="antecedent_match_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
