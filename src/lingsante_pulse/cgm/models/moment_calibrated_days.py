"""Observed-day probability calibration, not a final-embedding projection."""

import numpy as np
import torch
from lingsante_pulse.cgm.models.observation_supported_attention import (
    ObservationSupportedAttention,
    pack_observations,
)


def moment_projectors(moments, valid, physical=True):
    """Nullspace of mass and observed glucose first/second moment constraints.

    Fixed observed covariates only; no target fit, identity or phenotype inputs.
    Physical moments are per-day E[g] and E[g^2], not squared daily E[g].
    """
    moments = np.asarray(moments, dtype=np.float64)
    valid = np.asarray(valid, bool)
    if moments.shape != (*valid.shape, 2):
        raise ValueError("Two physical moments per slot required")
    result = np.zeros((*valid.shape, valid.shape[1]), np.float32)
    for i, mask in enumerate(valid):
        rows = np.flatnonzero(mask)
        if not len(rows):
            raise ValueError("No observed day")
        x = moments[i, rows]
        if not np.isfinite(x).all():
            raise ValueError("Invalid observed moments")
        a = np.ones((1, len(rows)))
        if physical:
            centered = (x - x.mean(0)).T
            centered /= np.maximum(
                np.linalg.norm(centered, axis=1, keepdims=True), 1e-12
            )
            a = np.concatenate([a, centered], axis=0)
        # All matrices are at most 3 x 7. This is deterministic geometry, not training.
        p = np.eye(len(rows)) - np.linalg.pinv(a, rcond=1e-12) @ a
        result[i][np.ix_(rows, rows)] = p.astype(np.float32)
    return result


def calibrate_weights(proposed, base, projector):
    """Keep positive mass and fixed observed moments with no tunable strength."""
    delta = torch.bmm(projector, (proposed - base).unsqueeze(-1)).squeeze(-1)
    valid = base > 0
    delta = torch.where(valid, delta, 0.0)
    relative = torch.where(valid, delta.abs() / base.clamp_min(1e-12), 0.0)
    scale = 1 + relative.amax(dim=1, keepdim=True)
    return base + delta / scale


def add_calibration(data, packed, physical=True):
    features = np.zeros((len(data["glucose"]), 4, 16), np.float32)
    for i, (x, m) in enumerate(zip(data["glucose"], data["observed"])):
        observed = np.asarray(x, dtype=np.float64)[np.asarray(m, bool) & np.isfinite(x)]
        if not len(observed):
            raise ValueError("Empty observed day")
        features[i, 0, :2] = (observed.mean(), (observed**2).mean())
    moments = pack_observations(
        data, packed["values"][:, 0], features, packed["support"][:, 0]
    )["features"][:, :, 0, :2]
    valid = packed["mask"] & (packed["support"][:, :, 0] > 0)
    packed.update(
        moments=moments, projector=moment_projectors(moments, valid, physical)
    )


class MomentCalibratedDayAttention(ObservationSupportedAttention):
    def __init__(self):
        super().__init__(coherent=True)

    def forward(self, projector, moments, **observations):
        _, proposed = super().forward(**observations)
        valid = observations["mask"] & (observations["support"][:, :, 0] > 0)
        base = torch.where(valid, observations["support"][:, :, 0], 0.0)
        base = base / base.sum(1, keepdim=True)
        weights = calibrate_weights(proposed[:, :, 0], base, projector)
        clean = torch.where(valid[:, :, None], observations["values"], 0.0)
        final = (weights[:, :, None] * clean).sum(1)
        final = torch.where(
            (~valid[:, 1:].any(1))[:, None], observations["values"][:, 0], final
        )
        return final, weights[:, :, None]


def moment_canary_decision(candidate, control):
    from lingsante_pulse.cgm.models.episodic_evidence import coherent_canary_decision

    decision = coherent_canary_decision(candidate, control)
    decision["policy"] = "moment_calibrated_days_progress_v1"
    decision["gates"]["physical_moments"] = (
        candidate["gates"].get("physical_moments") is True
    )
    decision["gates"]["nonzero_learning"] = all(
        r["gates"].get("nonzero_day_update") is True for r in (candidate, control)
    )
    decision["acceptance_passed"] = all(decision["gates"].values())
    return decision


def validate_moment_export(report, decision):
    if (
        decision.get("policy") != "moment_calibrated_days_progress_v1"
        or decision.get("acceptance_passed") is not True
    ):
        raise ValueError("Prospective moment-calibration decision required")
    hard = (
        "finite",
        "initial_stable",
        "value_bounds",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
        "joint_convex",
        "physical_moments",
        "nonzero_day_update",
    )
    if not all(report.get("gates", {}).get(k) is True for k in hard):
        raise ValueError("Invalid moment calibration or structural/isolation evidence")
    if (
        report["steps"],
        report["seed"],
        report["kind"],
        report["final_dimension"],
        report["parameters"],
    ) != (6960, 43, "candidate", 146, 8641):
        raise ValueError("Only fixed full candidate may be exported")
    info = report["moment_calibration"]
    if (
        info["physical_constraints"] is not True
        or not info["relative_physical_error"] <= 1e-5
    ):
        raise ValueError("Physical calibration missing or invalid")
