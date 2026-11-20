"""Joint raw-trajectory occupation and gap-aware cross-window memory."""

import torch
from torch import nn
from torch.nn import functional as F
from lingsante_pulse.cgm.models.occupation_path import (
    OccupationPath,
    occupation_statistics,
)


def canonical_slots(values, features, support, elapsed_days, mask, path):
    """Fix reduction order, without changing the chronological model or weights."""
    valid = mask & (support[:, :, 0] > 0)
    order = torch.where(valid, elapsed_days, -torch.inf).argsort(1, descending=True)
    return tuple(
        v.gather(1, order.reshape(*order.shape, *([1] * (v.ndim - 2))).expand_as(v))
        for v in (values, features, support, elapsed_days, mask, path)
    )


class GapRecurrentOccupation(OccupationPath):
    def __init__(self, control=False):
        super().__init__(control=True, input_dimensions=15)
        self.regular_control = control
        self.memory = nn.GRUCell(38, 16)
        self.decay = nn.Parameter(torch.zeros(16))
        self.bridge = nn.Linear(16, 36, bias=False)

    def forward(self, values, features, support, elapsed_days, mask, path):
        values, features, support, elapsed_days, mask, path = canonical_slots(
            values, features, support, elapsed_days, mask, path
        )
        valid = mask & (support[:, :, 0] > 0)
        w = torch.where(valid, support[:, :, 0], 0.0)
        stable = (
            torch.where(valid[:, :, None], values, 0.0)
            * (w / w.sum(1, keepdim=True).clamp_min(1e-8))[:, :, None]
        ).sum(1)
        path = torch.where(valid[:, :, None, None], path, 0.0)
        mass = path[:, :, :, -1].sum(2)
        valid = valid & (mass > 0)
        b, d, t, _ = path.shape
        h = self.lifts(
            torch.where(path[:, :, :, -1, None] > 0, path[:, :, :, :15], 0.0)
        )
        global_stats = occupation_statistics(
            h.flatten(1, 2), path[:, :, :, -1].flatten(1, 2), True
        )
        daily = occupation_statistics(
            h.reshape(b * d, t, 8), path[:, :, :, -1].reshape(b * d, t), True
        ).reshape(b, d, 36)
        order = torch.where(valid, elapsed_days, -torch.inf).argsort(1, descending=True)
        daily = daily.gather(1, order[:, :, None].expand_as(daily))
        times = elapsed_days.gather(1, order)
        mass = mass.gather(1, order)
        valid = valid.gather(1, order)
        state = torch.zeros(b, 16, device=values.device, dtype=values.dtype)
        pooled = torch.zeros_like(state)
        previous = torch.zeros_like(times[:, 0])
        seen = torch.zeros_like(valid[:, 0])
        for j in range(d):
            active = valid[:, j]
            delta = (previous - times[:, j]).clamp_min(0.0)
            if self.regular_control:
                delta = torch.ones_like(delta)
            delta = torch.where(seen & active, delta, 0.0)
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
            previous = torch.where(active, times[:, j], previous)
            seen = seen | active
        pooled = pooled / mass.sum(1, keepdim=True).clamp_min(1e-8)
        stats = global_stats + self.bridge(pooled)
        return stable + self.readout(stats) * (valid.sum(1) > 1)[:, None], stats


def synthetic_inputs(levels, times, device="cpu"):
    levels = torch.tensor(levels, dtype=torch.float32, device=device)
    b, d = levels.shape
    t = 9
    path = torch.zeros(b, d, t, 16, device=device)
    path[:, :, :, 0] = levels[:, :, None]
    path[:, :, :, 5:14] = levels[:, :, None, None]
    path[:, :, :, 14] = 1.0
    path[:, :, :, -1] = 1.0 / t
    return dict(
        values=levels[:, :, None].expand(b, d, 146),
        features=torch.zeros(b, d, 4, 16, device=device),
        support=torch.ones(b, d, 4, device=device),
        elapsed_days=torch.tensor(times, dtype=torch.float32, device=device).expand(
            b, d
        ),
        mask=torch.ones(b, d, dtype=torch.bool, device=device),
        path=path,
    )


def temporal_collision(model):
    device = next(model.parameters()).device
    with torch.no_grad():
        p = synthetic_inputs(
            [[2, 2, 2, 1, 1, 1, 1], [2, 1, 2, 1, 2, 1, 1]],
            [6, 5, 4, 3, 2, 1, 0],
            device,
        )
        z, s = model(**p)
        a = synthetic_inputs([[2, 1, 2, 1]], [3, 2, 1, 0], device)
        b = synthetic_inputs([[2, 1, 2, 1]], [4.5, 3, 1, 0], device)
        za, sa = model(**a)
        zb, sb = model(**b)
    return dict(
        day_order_embedding_difference=float((z[0] - z[1]).abs().max()),
        day_order_statistics_difference=float((s[0] - s[1]).abs().max()),
        gap_embedding_difference=float((za - zb).abs().max()),
        gap_statistics_difference=float((sa - sb).abs().max()),
    )


def gap_recurrent_decision(candidate, control, reference):
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
    values = lambda r: [
        r["learned"]["current_mean"],
        r["learned"]["longitudinal_mean"],
        r["event_ev"],
    ]
    c, g, ref = map(values, (candidate, control, reference))
    base = [
        candidate["baseline"]["current_mean"],
        candidate["baseline"]["longitudinal_mean"],
        candidate["stable_event_ev"],
    ]
    dominated = (
        all(y >= x for x, y in zip(c, g))
        and control["episode_validation_loss"] <= candidate["episode_validation_loss"]
        and (
            any(y > x for x, y in zip(c, g))
            or control["episode_validation_loss"] < candidate["episode_validation_loss"]
        )
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        support=all(
            trajectory_support_valid(r["occupation_path"])
            and r["occupation_path"]["sorted_control"] is False
            for r in (candidate, control)
        ),
        forbidden_inputs_absent=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        equal_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 8928, 146)
            for r in (candidate, control)
        ),
        regular_control=candidate["gap_recurrent"]["regular_control"] is False
        and control["gap_recurrent"]["regular_control"] is True,
        useful_progress=any(x > max(y, z) for x, y, z in zip(c, ref, base))
        and candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"],
        not_dominated_by_generic=not dominated,
        frozen_reference=reference["checkpoint_sha256"]
        == "50d7bed5bbffccec8662d213f31bbfadf685b6669f1f0c4f02aeb72cbeba1e14",
        identical_episodes=len(
            {r["episode_sequence_sha256"] for r in (candidate, control, reference)}
        )
        == 1,
        identical_targets=len(
            {r["target_sha256"] for r in (candidate, control, reference)}
        )
        == 1,
    )
    return dict(
        policy="gap_recurrent_occupation_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        candidate_minus_generic=[x - y for x, y in zip(c, g)],
        rank_policy="diagnostic_only",
    )


def validate_gap_recurrent_export(report, decision):
    from lingsante_pulse.cgm.models.local_trajectory_occupation import (
        trajectory_support_valid,
    )

    expected = {
        "structural_isolation",
        "support",
        "forbidden_inputs_absent",
        "equal_budget",
        "regular_control",
        "useful_progress",
        "not_dominated_by_generic",
        "frozen_reference",
        "identical_episodes",
        "identical_targets",
    }
    if (
        decision.get("policy") != "gap_recurrent_occupation_v1"
        or decision.get("acceptance_passed") is not True
        or set(decision.get("gates", {})) != expected
        or not all(v is True for v in decision["gates"].values())
    ):
        raise ValueError("Original ten-gate canary decision required")
    if (
        report["kind"],
        report["steps"],
        report["seed"],
        report["parameters"],
        report["final_dimension"],
    ) != ("candidate", 6960, 43, 8928, 146):
        raise ValueError("Wrong full model")
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
    if not all(report["gates"].get(k) is True for k in hard):
        raise ValueError("Structural or isolation failure")
    if (
        report.get("gap_recurrent", {}).get("regular_control") is not False
        or report.get("checkpoint_selection") != "last_step"
    ):
        raise ValueError("Wrong recurrent model or checkpoint")
    if (
        report.get("phenotype_or_private_test_reads") != 0
        or report.get("glucofm_components") != 0
        or "responsibility_gradient" in report
    ):
        raise ValueError("Forbidden training inputs or gradient projection")
    p = report["occupation_path"]
    if (
        p.get("sorted_control") is not False
        or not trajectory_support_valid(p)
        or (p["trajectory_minutes"], p["common_samples"]) != (120, 9)
    ):
        raise ValueError("Wrong raw trajectory support")
