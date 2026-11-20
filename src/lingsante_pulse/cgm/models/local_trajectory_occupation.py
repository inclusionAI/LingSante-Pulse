"""Observed common-view local trajectories, jointly lifted before occupation pooling."""

import numpy as np
from lingsante_pulse.cgm.models.occupation_path import (
    OccupationPath,
    daily_path,
    add_occupation_path,
)


def daily_local_trajectory(data, sorted_control=False):
    path, report = daily_path(data)
    g = np.asarray(data["glucose"], np.float32)
    seen = np.asarray(data["observed"], bool) & np.isfinite(g)
    n, t = g.shape
    phase = np.stack([seen[:, k::3].sum(1) for k in range(3)], 1).argmax(1)
    pos = np.arange(t)[None, :]
    end = phase[:, None] + ((pos - phase[:, None]) // 3) * 3
    index = end[:, :, None] - np.arange(8, -1, -1)[None, None, :] * 3
    safe = np.maximum(index, 0)
    rows = np.arange(n)[:, None, None]
    supported = (index[:, :, 0] >= 0) & seen[rows, safe].all(2) & (path[:, :, -1] > 0)
    values = np.where(supported[:, :, None], g[rows, safe] / 100.0, 0.0).astype(
        np.float32
    )
    if sorted_control:
        values = np.sort(values, axis=2)
    counts = {}
    for cohort in np.unique(data["cohort"]):
        for split in np.unique(data["split"]):
            selected = (data["cohort"] == cohort) & (data["split"] == split)
            total = int(np.count_nonzero(path[selected, :, -1] > 0))
            if not total:
                continue
            usable = int(supported[selected].sum())
            counts[f"{cohort}/split{int(split)}"] = dict(
                valid_points=total, trajectory_points=usable, fraction=usable / total
            )
    report.update(
        sorted_control=bool(sorted_control),
        trajectory_supported_points=int(supported.sum()),
        trajectory_support_fraction=float(
            supported.sum() / max(np.count_nonzero(path[:, :, -1] > 0), 1)
        ),
        trajectory_source_support=counts,
        trajectory_minutes=120,
        common_samples=9,
        missing_anchor_interpolation=False,
        cross_day_stitching=False,
    )
    return np.concatenate(
        [
            path[:, :, :5],
            values,
            supported[:, :, None].astype(np.float32),
            path[:, :, 5:],
        ],
        2,
    ), report


def add_local_trajectory(data, inputs, sorted_control=False):
    return add_occupation_path(
        data, inputs, builder=lambda d: daily_local_trajectory(d, sorted_control)
    )


class LocalTrajectoryOccupation(OccupationPath):
    def __init__(self, control=False):
        super().__init__(control=True, input_dimensions=15)
        self.sorted_control = control


def trajectory_support_valid(report):
    return (
        report["retained_observed_fraction"] >= 0.8
        and report["signed_anchor_residual_max"] <= 1e-6
        and bool(report["trajectory_source_support"])
        and all(
            x["fraction"] >= 0.5 for x in report["trajectory_source_support"].values()
        )
    )


def validate_masked_sparse_support(report):
    """Validate real support; historical density thresholds are diagnostics only.

    Does not alter input construction: incomplete nine-anchor trajectories stay
    zero with support=0 and cannot enter trajectory attention or quantiles.
    """
    observed = report["observed_points"]
    supported = report["supported_points"]
    groups = report["trajectory_source_support"]
    if not (0 < supported <= observed and groups):
        raise ValueError("No usable observed path support")
    if not (
        np.isfinite(report["signed_anchor_residual_max"])
        and 0 <= report["signed_anchor_residual_max"] <= 1e-6
    ):
        raise ValueError("Signed anchor isolation failure")
    total = complete = 0
    for group in groups.values():
        n, k = group["valid_points"], group["trajectory_points"]
        if not (0 <= k <= n and n > 0 and np.isclose(group["fraction"], k / n)):
            raise ValueError("Invalid trajectory support accounting")
        total += n
        complete += k
    if (
        total != supported
        or complete != report["trajectory_supported_points"]
        or complete <= 0
        or not np.isclose(report["retained_observed_fraction"], supported / observed)
        or not np.isclose(report["trajectory_support_fraction"], complete / supported)
        or report["missing_anchor_interpolation"]
        or report["cross_day_stitching"]
    ):
        raise ValueError("Invalid masked sparse support contract")
    report["support_policy"] = "masked_sparse_v1"
    report["historical_density_gate_passed"] = bool(trajectory_support_valid(report))
    return report


def validate_local_trajectory_export(report, decision):
    expected = {
        "structural_isolation",
        "sorted_control",
        "forbidden_inputs_absent",
        "support",
        "useful_progress",
        "equal_budget",
        "frozen_reference",
        "identical_episodes",
        "identical_targets",
    }
    if (
        decision.get("policy") != "local_trajectory_occupation_v1"
        or decision.get("acceptance_passed") is not True
        or set(decision.get("gates", {})) != expected
        or not all(v is True for v in decision["gates"].values())
    ):
        raise ValueError("Original nine-gate canary decision required")
    if (
        report["kind"],
        report["steps"],
        report["seed"],
        report["parameters"],
        report["final_dimension"],
    ) != ("candidate", 6960, 43, 5648, 146):
        raise ValueError("Wrong full model")
    if (
        report.get("local_trajectory_occupation") is not True
        or report.get("checkpoint_selection") != "last_step"
    ):
        raise ValueError("Wrong local trajectory or checkpoint")
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
    p = report["occupation_path"]
    if (
        p.get("sorted_control") is not False
        or not trajectory_support_valid(p)
        or (p["trajectory_minutes"], p["common_samples"]) != (120, 9)
    ):
        raise ValueError("Wrong trajectory support or control")


def local_trajectory_decision(candidate, control, reference):
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
    higher = any(
        candidate["learned"][k] > max(reference["learned"][k], candidate["baseline"][k])
        for k in ("current_mean", "longitudinal_mean")
    ) or candidate["event_ev"] > max(
        reference["event_ev"], candidate["stable_event_ev"]
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        sorted_control=candidate["occupation_path"]["sorted_control"] is False
        and control["occupation_path"]["sorted_control"] is True,
        forbidden_inputs_absent=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        support=all(
            trajectory_support_valid(r["occupation_path"]) for r in (candidate, control)
        ),
        useful_progress=higher
        and candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"],
        equal_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 5648, 146)
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
    deltas = {
        k: candidate["learned"][k] - control["learned"][k]
        for k in ("current_mean", "longitudinal_mean")
    }
    deltas["event_ev"] = candidate["event_ev"] - control["event_ev"]
    return dict(
        policy="local_trajectory_occupation_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        candidate_minus_sorted_control=deltas,
        rank_policy="diagnostic_only",
        control_is_attribution_not_sole_veto=True,
    )
