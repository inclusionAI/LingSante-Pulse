"""Learn CGM relations as finite supported measures, not unit-mass distributions."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.shared_pair_lift import SharedPairLift


def exposure_measure(h, support, elapsed, valid, control=False):
    gap = elapsed[:, None, :] - elapsed[:, :, None]
    pairs = (gap > 0) & (gap <= 6) & valid[:, :, None] & valid[:, None, :]
    weight = (
        torch.minimum(support[:, :, None, :], support[:, None, :, :])
        * pairs[:, :, :, None]
    )
    if control:
        weight = weight / weight.sum((1, 2), keepdim=True).clamp_min(1e-8)
    else:
        weight = (
            weight / 21.0
        )  # C(7,2), the fixed horizon; never fit to cohort/history length.
    newer, older = h[:, :, None], h[:, None, :]
    product = newer * older
    rate = (newer - older) / gap.clamp_min(1e-3)[:, :, :, None, None]
    c = (weight[:, :, :, :, None] * product).sum((1, 2))
    r = (weight[:, :, :, :, None] * rate).sum((1, 2))
    mass = weight.sum((1, 2))[:, :, None]
    return torch.cat([c, r, mass], -1)


class ExposurePairMeasure(SharedPairLift):
    def __init__(self, control=False):
        super().__init__(control=control)
        self.integrators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(33, 32), nn.GELU(), nn.Linear(32, 32), nn.Tanh()
                )
                for _ in range(4)
            ]
        )

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
        measure = exposure_measure(h, clean_support, elapsed_days, valid, self.control)
        # Subtract the network's empty-measure response: one day has no relation write.
        relations = torch.stack(
            [
                net(measure[:, k]) - net(torch.zeros_like(measure[:, k]))
                for k, net in enumerate(self.integrators)
            ],
            1,
        ).flatten(1)
        weights = clean_support[:, :, 0]
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        stable = (
            torch.where(valid[:, :, None], values, 0.0) * weights[:, :, None]
        ).sum(1)
        return stable + self.readout(relations), relations


def exposure_collision_check(model, packed, row):
    repeated = {
        k: v[row : row + 1, :1].expand(1, 7, *v.shape[2:]).clone()
        for k, v in packed.items()
    }
    repeated["mask"] = torch.ones_like(repeated["mask"])
    repeated["elapsed_days"] = torch.arange(
        7, device=repeated["values"].device
    ).float()[None, :]
    with torch.no_grad():
        seven, r7 = model(**repeated)
        repeated["mask"][:, 2:] = False
        two, r2 = model(**repeated)
    return dict(
        output_max_difference=float((seven - two).abs().max()),
        relation_max_difference=float((r7 - r2).abs().max()),
    )


def exposure_decision(candidate, control):
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
        "exposure_response",
    )
    long_ok = c["learned"]["longitudinal_mean"] > max(
        c["baseline"]["longitudinal_mean"], g["learned"]["longitudinal_mean"]
    )
    event_ok = c["event_ev"] > max(c["stable_event_ev"], g["event_ev"])
    gates = dict(
        structural_isolation=all(
            r.get("gates", {}).get(k) is True for r in (c, g) for k in hard
        ),
        useful_measure_progress=c["episode_validation_loss"]
        < c["stable_episode_validation_loss"]
        and (long_ok or event_ok),
        equal_parameters=c["parameters"] == g["parameters"] == 29504,
        fixed_budget=all(r["seed"] == 43 and r["steps"] == 600 for r in (c, g)),
        identical_episodes=c["episode_sequence_sha256"] == g["episode_sequence_sha256"],
        identical_targets=c["target_sha256"] == g["target_sha256"],
    )
    return dict(
        policy="exposure_pair_measure_progress_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        original_proxy_acceptance_passed=c["acceptance_passed"],
        full_or_public_launched=False,
    )
