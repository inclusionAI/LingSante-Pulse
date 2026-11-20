"""Physiology-oriented, uncertainty-scaled assimilation into one CARE code."""

import numpy as np
from sklearn.linear_model import Ridge


def supported_reliability(packed, current_support):
    mask = np.asarray(packed["history_mask"], dtype=bool)
    quality = np.clip(np.asarray(packed["support"])[:, :, 0], 0, 1) * mask
    mass = quality.sum(1)
    return np.sqrt(
        np.clip(mass / 6, 0, 1) * np.clip(current_support[:, 0], 0, 1)
    ).astype(np.float32)


def _normalize(x, rows):
    center = np.asarray(x[rows], dtype=np.float64).mean(0)
    scale = np.maximum(np.asarray(x[rows], dtype=np.float64).std(0), 1e-5)
    return center, scale


def fit_assimilation(stable, memory, target, fit_rows):
    """Fit a linear observation model and posterior mean update on split0.

    CARE-to-physiology D fixes semantic orientation. Joint memory readout
    supplies evidence. Residual prediction variance sets measurement noise;
    no phenotype labels, validation tuning or covariance-axis pairing occurs.
    """
    sc, ss = _normalize(stable, fit_rows)
    mc, ms = _normalize(memory, fit_rows)
    s = (stable[fit_rows] - sc) / ss
    m = (memory[fit_rows] - mc) / ms
    y = target[fit_rows]
    base = Ridge(alpha=1.0).fit(s, y)
    joint = Ridge(alpha=1.0).fit(np.concatenate((s, m), 1), y)
    D = base.coef_.T
    covariance = s.T @ s / len(s)
    noise = np.maximum(
        np.var(y - joint.predict(np.concatenate((s, m), 1)), axis=0), 1e-6
    )
    system = D.T @ covariance @ D + np.diag(noise)
    gain = np.linalg.solve(system, D.T @ covariance).T
    return dict(
        stable_center=sc,
        stable_scale=ss,
        memory_center=mc,
        memory_scale=ms,
        base_weight=D,
        base_bias=base.intercept_,
        joint_weight=joint.coef_.T,
        joint_bias=joint.intercept_,
        gain=gain,
        noise=noise,
    )


def apply_assimilation(stable, memory, reliability, state):
    s = (stable - state["stable_center"]) / state["stable_scale"]
    m = (memory - state["memory_center"]) / state["memory_scale"]
    prior = s @ state["base_weight"] + state["base_bias"]
    evidence = np.concatenate((s, m), 1) @ state["joint_weight"] + state["joint_bias"]
    correction = (evidence - prior) @ state["gain"].T
    z = stable + np.asarray(reliability)[:, None] * correction * state["stable_scale"]
    z = np.asarray(z, dtype=np.float32)
    z[np.asarray(reliability) == 0] = stable[np.asarray(reliability) == 0]
    if not np.isfinite(z).all():
        raise ValueError("non-finite semantic assimilation")
    return z
