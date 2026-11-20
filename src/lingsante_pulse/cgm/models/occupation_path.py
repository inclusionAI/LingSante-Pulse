"""Learned observed occupation path: mask-aware signed temporal integrals."""

import numpy as np
import torch
from torch import nn


def daily_path(data):
    g = np.asarray(data["glucose"], dtype=np.float32)
    seen = np.asarray(data["observed"], bool) & np.isfinite(g)
    n, t = g.shape
    if t != 288:
        raise ValueError("Frozen five-minute day grid required")
    # D selects one observed 15-minute phase, determined only by observation mask.
    phase = np.stack([seen[:, k::3].sum(1) for k in range(3)], 1).argmax(1)
    pos = np.arange(t)[None, :]
    anchor = phase[:, None] + ((pos - phase[:, None]) // 3) * 3
    safe = np.maximum(anchor, 0)
    base = np.take_along_axis(g, safe, 1)
    valid = seen & (anchor >= 0) & np.take_along_axis(seen, safe, 1)
    clean = np.where(valid, g, 0.0)
    base = np.where(valid, base, 0.0)
    residual = clean - base
    minutes = np.asarray(data["starts"]).astype("datetime64[m]").astype(np.int64) % 1440
    clock = 2 * np.pi * (minutes[:, None] + pos * 5) / 1440
    features = np.stack(
        [
            base / 100,
            np.maximum(residual, 0) / 100,
            np.maximum(-residual, 0) / 100,
            np.sin(clock),
            np.cos(clock),
        ],
        -1,
    ).astype(np.float32)
    features = np.where(valid[:, :, None], features, 0.0)
    weight = valid * np.maximum(5, np.asarray(data["cadence"]))[:, None] / 1440
    return np.concatenate([features, weight[:, :, None].astype(np.float32)], -1), dict(
        observed_points=int(seen.sum()),
        supported_points=int(valid.sum()),
        retained_observed_fraction=float(valid.sum() / max(seen.sum(), 1)),
        signed_anchor_residual_max=float(
            np.max(
                np.abs(
                    residual[np.arange(n)[:, None], safe][
                        np.take_along_axis(valid, safe, 1)
                    ]
                )
            )
        )
        if valid.any()
        else 0.0,
    )


def add_occupation_path(data, inputs, builder=None):
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        causal_history_indices,
    )
    from lingsante_pulse.cgm.models.parallel_longitudinal_models import (
        bounded_histories,
    )

    daily, report = (daily_path if builder is None else builder)(data)
    histories = bounded_histories(
        causal_history_indices(
            data["subject"],
            data["starts"],
            data["cohort"],
            data["split"],
            data["cadence"],
            data["source"],
            max_days=7,
        ),
        data["starts"],
    )
    out = np.zeros((len(daily), 7, 288, daily.shape[-1]), np.float32)
    for i, h in enumerate(histories):
        rows = h[::-1]
        if rows[0] != i or len(rows) != int(inputs["mask"][i].sum()):
            raise ValueError("History mismatch")
        for key in ("subject", "cohort", "split", "cadence", "source"):
            if not np.all(data[key][rows] == data[key][i]):
                raise ValueError("Cross provenance")
        out[i, : len(rows)] = daily[rows]
    inputs["path"] = out
    return report


def occupation_statistics(h, weight, control=False):
    weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
    increment = h * weight[:, :, None]
    mean = increment.sum(1)
    if control:
        matrix = torch.einsum("bti,btj->bij", increment, h)
    else:
        prefix = increment.cumsum(1) - increment
        order = torch.einsum("bti,btj->bij", prefix, increment)
        matrix = order - order.transpose(1, 2)
    ij = torch.triu_indices(h.shape[-1], h.shape[-1], 1, device=h.device)
    return torch.cat([mean, matrix[:, ij[0], ij[1]]], 1)


class OccupationPath(nn.Module):
    def __init__(self, control=False, input_dimensions=5):
        super().__init__()
        self.control = control
        self.input_dimensions = input_dimensions
        self.lifts = nn.Sequential(
            nn.Linear(input_dimensions, 16), nn.GELU(), nn.Linear(16, 8), nn.Tanh()
        )
        self.readout = nn.Linear(36, 146, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, values, features, support, elapsed_days, mask, path):
        valid = mask & (support[:, :, 0] > 0)
        w = torch.where(valid, support[:, :, 0], 0.0)
        stable = (
            torch.where(valid[:, :, None], values, 0.0)
            * (w / w.sum(1, keepdim=True).clamp_min(1e-8))[:, :, None]
        ).sum(1)
        path = torch.where(valid[:, :, None, None], path, 0.0)
        order = elapsed_days.argsort(1, descending=True)
        path = path.gather(1, order[:, :, None, None].expand_as(path))
        has_history = (path[:, :, :, -1].sum(2) > 0).sum(1) > 1
        flat = path.flatten(1, 2)
        point_valid = flat[:, :, -1] > 0
        h = self.lifts(
            torch.where(
                point_valid[:, :, None], flat[:, :, : self.input_dimensions], 0.0
            )
        )
        stats = occupation_statistics(h, flat[:, :, -1], self.control)
        update = self.readout(stats) * has_history[:, None]
        return stable + update, stats


def occupation_decision(candidate, control):
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
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        observed_support=all(
            r["occupation_path"]["retained_observed_fraction"] >= 0.8
            and r["occupation_path"]["signed_anchor_residual_max"] <= 1e-6
            for r in (candidate, control)
        ),
        useful_progress=candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"]
        and (
            candidate["learned"]["longitudinal_mean"]
            > max(
                candidate["baseline"]["longitudinal_mean"],
                control["learned"]["longitudinal_mean"],
            )
            or candidate["event_ev"]
            > max(candidate["stable_event_ev"], control["event_ev"])
        ),
        equal_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 5488, 146)
            for r in (candidate, control)
        ),
        identical_episodes=candidate["episode_sequence_sha256"]
        == control["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == control["target_sha256"],
    )
    return dict(
        policy="occupation_path_order_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )


def validate_occupation_moment_export(report, plan):
    if (
        plan.get("policy") != "occupation_moment_post_canary_exploration_v1"
        or plan.get("exploratory_followup") is not True
        or plan.get("original_ordered_candidate_acceptance_passed") is not False
    ):
        raise ValueError("Explicit control followup required, not ordered acceptance")
    if (
        plan.get("canary_checkpoint_sha256")
        != "1573ae71f70b33d21da8bad4d3d6dbfb6691393cae07af7e6079767fbd29ecbc"
    ):
        raise ValueError("Wrong canary")
    if (
        report["kind"],
        report["steps"],
        report["seed"],
        report["parameters"],
        report["final_dimension"],
    ) != ("generic", 6960, 43, 5488, 146):
        raise ValueError("Wrong full model")
    if report["shared_pair_lift"]["control"] is not True:
        raise ValueError("Ordered arm is closed")
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
    if not all(report["gates"].get(k) is True for k in hard):
        raise ValueError("Structural or isolation failure")
    if (
        report.get("phenotype_or_private_test_reads") != 0
        or report.get("glucofm_components") != 0
    ):
        raise ValueError("Forbidden training inputs")
    if (
        report["occupation_path"]["retained_observed_fraction"] < 0.8
        or report["occupation_path"]["signed_anchor_residual_max"] > 1e-6
    ):
        raise ValueError("Observation contract failed")
