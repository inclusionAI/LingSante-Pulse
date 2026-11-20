"""Learn daily evidence before shared, support-weighted pair statistics."""

import torch
from torch import nn


def pair_statistics(h, support, elapsed, valid, control=False):
    # i is the newer observation, j the older one; storage order is irrelevant.
    gap = elapsed[:, None, :] - elapsed[:, :, None]
    pairs = (gap > 0) & valid[:, :, None] & valid[:, None, :]
    weight = (
        torch.minimum(support[:, :, None, :], support[:, None, :, :])
        * pairs[:, :, :, None]
    )
    weight = weight / weight.sum((1, 2), keepdim=True).clamp_min(1e-8)
    newer, older = h[:, :, None], h[:, None, :]
    coincidence = (newer.square() + older.square()) * 0.5 if control else newer * older
    rate = (newer - older) / gap.clamp_min(1e-3)[:, :, :, None, None]
    c = (weight[:, :, :, :, None] * coincidence).sum((1, 2)).flatten(1)
    r = (weight[:, :, :, :, None] * rate).sum((1, 2)).flatten(1)
    return torch.cat([c, r], 1)


class SharedPairLift(nn.Module):
    def __init__(self, control=False):
        super().__init__()
        self.control = control
        self.lifts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(17, 16), nn.GELU(), nn.Linear(16, 16), nn.Tanh()
                )
                for _ in range(4)
            ]
        )
        self.readout = nn.Linear(128, 146, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, values, features, support, elapsed_days, mask):
        valid = mask & (support[:, :, 0] > 0)
        clean_support = torch.where(valid[:, :, None], support, 0.0)
        clean_features = torch.where(
            valid[:, :, None, None] & (clean_support[:, :, :, None] > 0), features, 0.0
        )
        h = torch.stack(
            [
                net(
                    torch.cat(
                        [clean_features[:, :, k], clean_support[:, :, k, None]], -1
                    )
                )
                for k, net in enumerate(self.lifts)
            ],
            2,
        )
        relations = pair_statistics(h, clean_support, elapsed_days, valid, self.control)
        weights = clean_support[:, :, 0]
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        stable = (
            torch.where(valid[:, :, None], values, 0.0) * weights[:, :, None]
        ).sum(1)
        return stable + self.readout(relations), relations


def pair_lift_decision(candidate, control):
    c, g = candidate, control
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
    )
    long_ok = c["learned"]["longitudinal_mean"] > max(
        c["baseline"]["longitudinal_mean"], g["learned"]["longitudinal_mean"]
    )
    event_ok = c["event_ev"] > max(c["stable_event_ev"], g["event_ev"])
    gates = dict(
        structural_isolation=all(
            r.get("gates", {}).get(k) is True for r in (c, g) for k in hard
        ),
        useful_relational_progress=c["episode_validation_loss"]
        < c["stable_episode_validation_loss"]
        and (long_ok or event_ok),
        equal_parameters=c["parameters"] == g["parameters"] == 20928,
        fixed_budget=all(r["seed"] == 43 and r["steps"] == 600 for r in (c, g)),
        identical_episodes=c["episode_sequence_sha256"] == g["episode_sequence_sha256"],
        identical_targets=c["target_sha256"] == g["target_sha256"],
    )
    return dict(
        policy="shared_pair_lift_progress_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        original_proxy_acceptance_passed=c["acceptance_passed"],
        full_or_public_launched=False,
    )


def validate_pair_lift_export(report, decision):
    if (
        decision.get("policy") != "shared_pair_lift_progress_v1"
        or decision.get("acceptance_passed") is not True
    ):
        raise ValueError("Prospective pair-lift decision required")
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
    )
    if not all(report.get("gates", {}).get(k) is True for k in hard):
        raise ValueError("Structural/isolation failure")
    if (
        report["steps"],
        report["seed"],
        report["kind"],
        report["final_dimension"],
        report["parameters"],
    ) != (6960, 43, "candidate", 146, 20928):
        raise ValueError("Only fixed full candidate may be exported")
    if report["shared_pair_lift"]["control"] is not False:
        raise ValueError("Wrong relation operator")
