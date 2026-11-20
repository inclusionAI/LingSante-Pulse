"""Local observed signed velocity jointly lifted with CGM occupation states."""

import numpy as np
from lingsante_pulse.cgm.models.occupation_path import (
    OccupationPath,
    daily_path,
    add_occupation_path,
)


def daily_state_flux(data, unsigned=False):
    path, report = daily_path(data)
    g = np.asarray(data["glucose"], np.float32)
    observed = np.asarray(data["observed"], bool) & np.isfinite(g)
    flux = np.zeros((*g.shape, 3), np.float32)
    supported = 0
    possible = int(np.count_nonzero(path[:, :, -1] > 0))
    for i in range(len(g)):
        pos = np.flatnonzero(observed[i])
        if len(pos) < 2:
            continue
        previous, current = pos[:-1], pos[1:]
        dt = (current - previous) * 5.0
        valid = (
            (dt <= max(5.0, float(data["cadence"][i])) + 1e-5)
            & (path[i, current, -1] > 0)
            & (path[i, previous, -1] > 0)
        )
        previous, current, dt = previous[valid], current[valid], dt[valid]
        velocity = (
            (g[i, current] - g[i, previous]) / dt * 0.6
        )  # mg/dL/hour divided by100.
        positive = np.maximum(velocity, 0)
        negative = np.maximum(-velocity, 0)
        if unsigned:
            positive = negative = (positive + negative) * 0.5
        flux[i, current, 0] = positive
        flux[i, current, 1] = negative
        flux[i, current, 2] = 1.0
        supported += len(current)
    report.update(
        flow_supported_points=supported,
        flow_support_fraction=supported / max(possible, 1),
        direction_control=bool(unsigned),
        cross_gap_velocity_edges=0,
        cross_day_velocity_edges=0,
    )
    return np.concatenate([path[:, :, :5], flux, path[:, :, 5:]], -1), report


def add_state_flux(data, inputs, unsigned=False):
    return add_occupation_path(
        data, inputs, builder=lambda d: daily_state_flux(d, unsigned)
    )


class StateFluxOccupation(OccupationPath):
    def __init__(self, control=False):
        super().__init__(control=True, input_dimensions=8)
        self.direction_control = control


def state_flux_decision(candidate, control, reference):
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
    long_ok = candidate["learned"]["longitudinal_mean"] > max(
        candidate["baseline"]["longitudinal_mean"],
        control["learned"]["longitudinal_mean"],
        reference["learned"]["longitudinal_mean"],
    )
    event_ok = candidate["event_ev"] > max(
        candidate["stable_event_ev"], control["event_ev"], reference["event_ev"]
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        direction_control=candidate["occupation_path"]["direction_control"] is False
        and control["occupation_path"]["direction_control"] is True,
        forbidden_inputs_absent=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=all(
            r["occupation_path"]["flow_support_fraction"] >= 0.7
            and r["occupation_path"]["retained_observed_fraction"] >= 0.8
            for r in (candidate, control)
        ),
        useful_signed_progress=candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"]
        and (long_ok or event_ok),
        equal_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 5536, 146)
            for r in (candidate, control)
        ),
        frozen_reference=reference["checkpoint_sha256"]
        == "1573ae71f70b33d21da8bad4d3d6dbfb6691393cae07af7e6079767fbd29ecbc"
        and (reference["seed"], reference["steps"], reference["kind"])
        == (43, 600, "generic"),
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
        policy="state_flux_occupation_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
