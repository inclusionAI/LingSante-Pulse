"""Two independent raw-CGM extensions, not terminal feature concatenations."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.unmatched_correspondence import UnmatchedCorrespondence


def quantiles(h, mass):
    active = mass > 0
    clean = torch.where(active[..., None], h, 0.0)
    ordered, index = clean.sort(dim=-2, stable=True)
    # Compare unnormalised masses in float64: float32 parallel prefix sums can
    # cross exact quantile boundaries differently on CPU and accelerators.
    weight = mass[..., None].expand_as(h).gather(-2, index).to(torch.float64)
    cumulative = weight.cumsum(-2)
    total = weight.sum(-2, keepdim=True)
    q = h.new_tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
    # First observed sample whose cumulative mass reaches each fixed quantile.
    hit = cumulative.unsqueeze(-2) >= total.unsqueeze(-2) * q[:, None]
    positions = hit.to(torch.int64).argmax(-3)
    result = ordered.gather(-2, positions)
    return torch.where(active.any(-1)[..., None, None], result, 0.0)


def daily_quantiles(h, mass, control=False):
    mass = torch.where(mass > 0, mass, 0.0)
    day = mass.sum(-1) > 0
    weight = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
    if control:
        result = quantiles(h.flatten(1, 2), weight.flatten(1, 2))
    else:
        result = (quantiles(h, weight) * day[..., None, None]).sum(1) / day.sum(
            1
        ).clamp_min(1)[:, None, None]
    return result.flatten(1)


class DistributionBarycenter(UnmatchedCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.distribution_control = control
        self.distribution_writeback = nn.Linear(40, 36, bias=False)
        nn.init.zeros_(self.distribution_writeback.weight)

    def statistics_residual(self, h, path, valid):
        mass = torch.where(valid[..., None] & (path[..., 14] > 0), path[..., -1], 0.0)
        return self.distribution_writeback(
            daily_quantiles(h, mass, self.distribution_control)
        )


class PhaseLift(nn.Module):
    def __init__(self, base, control=False):
        super().__init__()
        self.base = base
        self.control = control
        self.early = nn.Linear(2, 8, bias=False)
        self.late = nn.Linear(6, 8, bias=False)
        self.writeback = nn.Linear(8, 8, bias=False)
        nn.init.zeros_(self.writeback.weight)

    def phase_content(self, x, control=None):
        control = self.control if control is None else control
        g = torch.where((x[..., 14] > 0)[..., None], x[..., 5:14], 0.0)
        e = torch.tanh(self.early(g[..., 1:3] - g[..., :1]))
        l = torch.tanh(self.late(g[..., 3:] - g[..., 2:3]))
        return (e + l) / 2 if control else e * l

    def forward(self, x):
        return self.base(x) + self.writeback(self.phase_content(x))


class PhaseCoupling(UnmatchedCorrespondence):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.lifts = PhaseLift(self.lifts, control)


def branch_modules(model):
    return (
        {"distribution_writeback": model.distribution_writeback}
        if isinstance(model, DistributionBarycenter)
        else {
            "early": model.lifts.early,
            "late": model.lifts.late,
            "writeback": model.lifts.writeback,
        }
    )


def route_diagnostics(model):
    device = next(model.parameters()).device
    from lingsante_pulse.cgm.models.cross_window_correspondence import (
        synthetic_node_inputs,
    )

    p = synthetic_node_inputs([[1.0, 3.0, 2.0, 4.0]], [4.5, 3.0, 1.0, 0.0], device)
    x = p["nodes"].clone()
    x[:, :, 1, 0] += 0.7
    x[:, :, 2, 1] += 0.2
    with torch.no_grad():
        changed = x.clone()
        changed[:, 3] = 99.0
        future = float(
            (
                model.node_messages(changed, p["mask"], p["elapsed_days"])[:, :3]
                - model.node_messages(x, p["mask"], p["elapsed_days"])[:, :3]
            )
            .abs()
            .max()
        )
        if isinstance(model, DistributionBarycenter):
            h = (
                torch.arange(8, device=device, dtype=torch.float32)[None, None, None, :]
                .expand(1, 2, 4, 8)
                .clone()
            )
            h[:, 1] += 2
            mass = torch.ones(1, 2, 4, device=device)
            a = model.distribution_writeback(daily_quantiles(h, mass, False))
            b = model.distribution_writeback(daily_quantiles(h, mass, True))
            masked = h.clone()
            masked[:, 1] = float("nan")
            mass[:, 1] = 0
            mask_error = float(
                (daily_quantiles(masked, mass, False) - daily_quantiles(h, mass, False))
                .abs()
                .max()
            )
            return dict(
                branch_response=float((a - b).abs().max()),
                mask_max_error=mask_error,
                future_message_max_error=future,
            )
        x = torch.zeros(4, 15, device=device)
        x[:, 14] = 1
        e = x.new_tensor([0.0, 1.0, 0.0, 1.0])
        l = x.new_tensor([0.0, 0.0, 1.0, 1.0])
        x[:, 6:8] = e[:, None]
        x[:, 8:14] = 2 * e[:, None]
        y = x.clone()
        y[:, 8:14] = e[:, None] + l[:, None]
        a = model.lifts.writeback(model.lifts.phase_content(x, False).mean(0))
        b = model.lifts.writeback(model.lifts.phase_content(y, False).mean(0))
        additive = float(
            (
                model.lifts.phase_content(x, True).mean(0)
                - model.lifts.phase_content(y, True).mean(0)
            )
            .abs()
            .max()
        )
        changed = x.clone()
        changed[:, 1:3] = float("nan")
        private = float(
            (model.lifts.phase_content(x) - model.lifts.phase_content(changed))
            .abs()
            .max()
        )
        changed = x.clone()
        changed[:, 14] = 0
        changed[:, 5:14] = float("nan")
        mask_error = float(model.lifts.phase_content(changed).abs().max())
        return dict(
            branch_response=float((a - b).abs().max()),
            additive_pairing_max_error=additive,
            private_max_error=private,
            mask_max_error=mask_error,
            future_message_max_error=future,
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
    "route_gradient",
    "route_response",
    "route_mask_isolation",
    "route_future_isolation",
)


def route_decision(candidate, control, support, route):
    hard = HARD_GATES + (
        ("route_private_isolation", "route_additive_control")
        if route == "phase"
        else ()
    )
    params = {"distribution": 10689, "phase": 9377}[route]
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, params, 146)
            for r in (candidate, control)
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=support.get("supported_for_canary") is True
        and support["route"] == route,
        definitions=candidate["distribution_phase"]["route"]
        == control["distribution_phase"]["route"]
        == route
        and candidate["distribution_phase"]["control"] is False
        and control["distribution_phase"]["control"] is True
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
        policy=route + "_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
