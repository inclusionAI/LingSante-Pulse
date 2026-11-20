"""Auxiliary causal physical-state prediction, with unchanged Distribution encoder."""

import hashlib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lingsante_pulse.cgm.models.cross_day_persistence import (
    STATE_GROUPS,
    eligible_pairs,
    phase_supported,
)
from lingsante_pulse.cgm.models.episodic_evidence import ridge_predict
from lingsante_pulse.cgm.models.distribution_phase import HARD_GATES


def prediction_pairs(data, inputs):
    valid = phase_supported(data) & (inputs["support"][:, 0, 0] > 0)
    starts = np.asarray(data["starts"]).astype("datetime64[m]").astype(np.int64)
    provenance = lambda i: tuple(
        data[k][i] for k in ("subject", "cohort", "split", "cadence", "source")
    )
    lookup = {(provenance(i), int(t)): i for i, t in enumerate(starts)}
    result = []
    masks = []
    used = set()
    pairs = eligible_pairs(data, valid)
    for target, origin in sorted(pairs.tolist(), key=lambda p: (starts[p[0]], p[0])):
        if origin in used:
            continue
        delta = int(starts[target] - starts[origin])
        age = np.rint(inputs["elapsed_days"][origin] * 1440).astype(np.int64)
        keep = (
            inputs["mask"][origin]
            & (inputs["support"][origin, :, 0] > 0)
            & (age + delta <= 8640)
        )
        for slot in np.flatnonzero(keep):
            row = lookup[(provenance(origin), int(starts[origin] - age[slot]))]
            keep[slot] = valid[row]
        if "path" in inputs:
            keep = keep & (inputs["path"][origin, ..., -1].sum(1) > 0)
        if keep.sum() < 2:
            continue
        assert keep[0] and delta >= 1440 and np.all(age[keep] >= 0)
        assert int(age[keep].max()) + delta + 1440 <= 10080
        result.append((origin, target))
        masks.append(keep)
        used.add(origin)
    return np.asarray(result, np.int64).reshape(-1, 2), np.asarray(masks, bool).reshape(
        -1, 7
    )


def prediction_inputs(inputs, pairs, masks):
    out = {}
    for k, v in inputs.items():
        if v.shape[:2] != (len(inputs["mask"]), 7):
            raise ValueError("Unexpected packing")
        x = v[pairs[:, 0]].copy()
        out[k] = np.where(
            masks.reshape(*masks.shape, *([1] * (x.ndim - 2))), x, 0
        ).astype(x.dtype)
    out["mask"] = masks.copy()
    return out


def support_summary(data, pairs, masks):
    origins = pairs[:, 0]
    groups = {}
    for c in np.unique(data["cohort"]):
        for s in (0, 1):
            rows = origins[
                (data["cohort"][origins] == c) & (data["split"][origins] == s)
            ]
            groups[f"{c}/split{s}"] = dict(
                endpoints=len(rows), subjects=len(np.unique(data["subject"][rows]))
            )
    totals = {
        str(s): dict(
            endpoints=sum(
                v["endpoints"] for k, v in groups.items() if k.endswith(f"/split{s}")
            ),
            subjects=len(
                np.unique(data["subject"][origins[data["split"][origins] == s]])
            ),
        )
        for s in (0, 1)
    }
    public = [
        str(c)
        for c in np.unique(data["cohort"])
        if c != "private_source_private"
        and all(groups[f"{c}/split{s}"]["endpoints"] > 0 for s in (0, 1))
    ]
    return dict(
        groups=groups,
        totals=totals,
        public_sources=public,
        supported_for_canary=totals["0"]["endpoints"] >= 64
        and totals["1"]["endpoints"] >= 32
        and totals["0"]["subjects"] >= 4
        and totals["1"]["subjects"] >= 2
        and len(public) >= 2,
        minimum_input_days=int(masks.sum(1).min()) if len(masks) else 0,
        pair_sha256=hashlib.sha256(
            pairs.astype("<i8").tobytes() + masks.tobytes()
        ).hexdigest(),
    )


def state_loss(support, support_targets, query, query_targets, stop_through_fit=False):
    error = (
        ridge_predict(
            support,
            support_targets,
            query,
            stop_through_fit=stop_through_fit,
        )
        - query_targets
    ).square()
    return torch.stack([error[:, idx].mean() for idx in STATE_GROUPS.values()]).mean()


def state_readout(z, y, fit, audit):
    pred = (
        make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        .fit(z[fit], y[fit])
        .predict(z[audit])
    )
    truth = y[audit]
    den = truth.var(0)
    ev = 1 - (truth - pred).var(0) / np.maximum(den, 1e-8)
    ev = np.where(den > 1e-8, ev, 0.0)
    groups = {name: float(ev[index].mean()) for name, index in STATE_GROUPS.items()}
    return dict(
        groups=groups,
        mean_ev=float(np.mean(list(groups.values()))),
        coordinate_ev=ev.tolist(),
    )


def prediction_decision(candidate, control):
    both = (candidate, control)
    a = candidate["cross_day_prediction"]
    b = control["cross_day_prediction"]
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in both for k in HARD_GATES
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 10689, 146)
            for r in both
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in both
        ),
        definitions=a["target_kind"] == "unseen_future_day"
        and b["target_kind"] == "observed_last_day"
        and all(r["distribution_control"] is False for r in (a, b)),
        support=all(
            r["support"]["supported_for_canary"]
            and r["support"]["minimum_input_days"] >= 2
            for r in (a, b)
        ),
        identical_main_episodes=candidate["episode_sequence_sha256"]
        == control["episode_sequence_sha256"],
        identical_aux_episodes=a["episode_sha256"] == b["episode_sha256"],
        identical_main_targets=candidate["target_sha256"] == control["target_sha256"],
        identical_aux_inputs=a["support"]["pair_sha256"] == b["support"]["pair_sha256"]
        and a["input_sha256"] == b["input_sha256"],
        identical_target_definitions=all(
            a[k] == b[k] for k in ("future_target_sha256", "observed_target_sha256")
        ),
        auxiliary_gradient=all(r["max_aux_readout_gradient"] > 0 for r in (a, b)),
        unseen_state_gain=a["future_readout"]["mean_ev"]
        > b["future_readout"]["mean_ev"]
        and a["future_readout"]["mean_ev"] > a["stable_future_readout"]["mean_ev"],
        reference_preserved=candidate["learned"]["current_mean"]
        >= 0.4853538740426302 - 0.02
        and candidate["learned"]["longitudinal_mean"] >= 0.37716955442221894 - 0.02
        and candidate["event_ev"] >= 0.34321602061390877 - 0.02,
    )
    return dict(
        policy="cross_day_prediction_exploratory_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
