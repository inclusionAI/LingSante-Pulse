"""Joint raw occupation and observed adjacent-day distribution relations."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.occupation_path import occupation_statistics


def day_relations(days, mass, elapsed, control=False):
    n = days.shape[1]
    ij = torch.triu_indices(n, n, 1, device=days.device)
    a, b = days[:, ij[0]], days[:, ij[1]]
    dt = (elapsed[:, ij[0]] - elapsed[:, ij[1]]).abs()
    valid = (dt > 0) if control else ((dt - 1).abs() < 1e-5)
    weight = torch.minimum(mass[:, ij[0]], mass[:, ij[1]]) * valid
    features = torch.cat([0.5 * (a + b), (a - b).square()], -1)
    return features, weight


class DayGraphOccupation(nn.Module):
    def __init__(self, control=False):
        super().__init__()
        self.control = control
        self.lifts = nn.Sequential(
            nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 8), nn.Tanh()
        )
        self.edge_lift = nn.Sequential(
            nn.Linear(72, 16), nn.GELU(), nn.Linear(16, 36), nn.Tanh()
        )
        self.readout = nn.Linear(72, 146, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, values, features, support, elapsed_days, mask, path):
        valid = mask & (support[:, :, 0] > 0)
        w = torch.where(valid, support[:, :, 0], 0.0)
        stable = (
            torch.where(valid[:, :, None], values, 0.0)
            * (w / w.sum(1, keepdim=True).clamp_min(1e-8))[:, :, None]
        ).sum(1)
        path = torch.where(valid[:, :, None, None], path, 0.0)
        point_weight = path[:, :, :, -1]
        mass = point_weight.sum(2)
        h = self.lifts(
            torch.where(point_weight[:, :, :, None] > 0, path[:, :, :, :5], 0.0)
        )
        batch, days, points, _ = h.shape
        global_stats = occupation_statistics(
            h.flatten(1, 2), point_weight.flatten(1, 2), True
        )
        daily = occupation_statistics(
            h.reshape(batch * days, points, 8),
            point_weight.reshape(batch * days, points),
            True,
        ).reshape(batch, days, 36)
        e, ew = day_relations(daily, mass, elapsed_days, self.control)
        messages = self.edge_lift(e)
        edge_stats = (
            messages * (ew / ew.sum(1, keepdim=True).clamp_min(1e-8))[:, :, None]
        ).sum(1)
        stats = torch.cat([global_stats, edge_stats], 1)
        return stable + self.readout(stats) * ((mass > 0).sum(1) > 1)[:, None], stats


def graph_collision(model):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    order = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0]], device=device, dtype=dtype
    )
    t = torch.arange(288, device=device, dtype=dtype)
    path = torch.zeros(2, 7, 288, 6, device=device, dtype=dtype)
    path[:, :, :, 0] = 1.0 + 0.8 * order[:, :, None] + 0.08 * torch.sin(t / 18)
    path[:, :, :, 3] = torch.sin(2 * torch.pi * t / 288)
    path[:, :, :, 4] = torch.cos(2 * torch.pi * t / 288)
    path[:, :, :, 5] = 5 / 1440
    with torch.no_grad():
        z, stats = model(
            values=order[:, :, None].expand(2, 7, 146),
            features=torch.zeros(2, 7, 1, device=device, dtype=dtype),
            support=torch.ones(2, 7, 4, device=device, dtype=dtype),
            elapsed_days=torch.arange(6, -1, -1, device=device, dtype=dtype).expand(
                2, -1
            ),
            mask=torch.ones(2, 7, device=device, dtype=torch.bool),
            path=path,
        )
    return dict(
        global_moment_max_difference=float((stats[0, :36] - stats[1, :36]).abs().max()),
        edge_moment_max_difference=float((stats[0, 36:] - stats[1, 36:]).abs().max()),
        embedding_max_difference=float((z[0] - z[1]).abs().max()),
        synthetic_only=True,
        longest_high_run_days=[3, 1],
        high_bouts=[1, 3],
    )


def day_graph_decision(candidate, control, reference):
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
        "day_edge_gradient",
        "day_topology_response",
    )
    useful = (
        any(
            candidate["learned"][k] > reference["learned"][k]
            for k in ("current_mean", "longitudinal_mean")
        )
        or candidate["event_ev"] > reference["event_ev"]
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        forbidden_inputs_absent=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=all(
            r["occupation_path"]["retained_observed_fraction"] >= 0.8
            and r["day_graph"]["audit_adjacent_edge_fraction"] >= 0.7
            for r in (candidate, control)
        ),
        actual_progress=candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"]
        and useful,
        equal_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 12524, 146)
            for r in (candidate, control)
        ),
        control_definition=candidate["day_graph"]["complete_graph_control"] is False
        and control["day_graph"]["complete_graph_control"] is True,
        frozen_reference=reference["checkpoint_sha256"]
        == "1573ae71f70b33d21da8bad4d3d6dbfb6691393cae07af7e6079767fbd29ecbc",
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
        policy="day_graph_occupation_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
        control_advantage_diagnostic=dict(
            current=candidate["learned"]["current_mean"]
            - control["learned"]["current_mean"],
            longitudinal=candidate["learned"]["longitudinal_mean"]
            - control["learned"]["longitudinal_mean"],
            event=candidate["event_ev"] - control["event_ev"],
        ),
    )
