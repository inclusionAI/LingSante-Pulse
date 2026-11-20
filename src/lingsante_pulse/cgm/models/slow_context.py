"""Observed six-hour common context for the existing two-hour local response."""

import numpy as np
import torch
from torch import nn
from lingsante_pulse.cgm.models.distribution_phase import DistributionBarycenter


def daily_slow_context(data):
    g = np.asarray(data["glucose"], np.float32)
    seen = np.asarray(data["observed"], bool) & np.isfinite(g)
    n, t = g.shape
    phase = np.stack([seen[:, k::3].sum(1) for k in range(3)], 1).argmax(1)
    pos = np.arange(t)[None, :]
    end = phase[:, None] + ((pos - phase[:, None]) // 3) * 3
    # Require every 15-minute anchor in the six-hour interval, not only nine sparse samples.
    index = end[:, :, None] - np.arange(24, -1, -1)[None, None, :] * 3
    safe = np.maximum(index, 0)
    rows = np.arange(n)[:, None, None]
    valid = (index[:, :, 0] >= 0) & seen[rows, safe].all(2)
    values = np.where(valid[:, :, None], g[rows, safe[:, :, ::3]] / 100.0, 0.0).astype(
        np.float32
    )
    return values, valid


def add_slow_context(data, inputs):
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        causal_history_indices,
    )
    from lingsante_pulse.cgm.models.parallel_longitudinal_models import (
        bounded_histories,
    )

    values, valid = daily_slow_context(data)
    history = bounded_histories(
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
    extra = np.zeros((*inputs["path"].shape[:-1], 10), np.float32)
    for i, h in enumerate(history):
        rows = np.asarray(h[::-1])
        assert rows[0] == i and len(rows) == int(inputs["mask"][i].sum())
        extra[i, : len(rows), :, :9] = values[rows]
        extra[i, : len(rows), :, 9] = valid[rows]
    inputs["path"] = np.concatenate(
        [inputs["path"][..., :15], extra, inputs["path"][..., -1:]], -1
    )


class SlowContext(DistributionBarycenter):
    def __init__(self, control=False):
        super().__init__(control=False)
        self.additive_context_control = control
        self.context_writeback = nn.Linear(8, 8, bias=False)
        nn.init.zeros_(self.context_writeback.weight)

    def context_content(self, path):
        valid = (path[..., 14] > 0) & (path[..., 24] > 0) & (path[..., -1] > 0)
        fast = path[..., :15].clone()
        fast[..., 1:3] = 0.0
        slow = fast.clone()
        slow[..., 5:14] = path[..., 15:24] - path[..., 0:1]
        fast = torch.where(valid[..., None], fast, 0.0)
        slow = torch.where(valid[..., None], slow, 0.0)
        hs = torch.tanh(self.lifts(slow))
        content = (
            hs if self.additive_context_control else hs * torch.tanh(self.lifts(fast))
        )
        return torch.where(valid[..., None], content, 0.0)

    def path_lift(self, path):
        base = super().path_lift(path)
        return base + self.context_writeback(self.context_content(path))


def synthetic_context_inputs(inputs):
    # Synthetic diagnostic only: give each existing trajectory a supported slow path.
    p = inputs["path"]
    return dict(
        inputs,
        path=torch.cat([p[..., :15], p[..., 5:14], p[..., 14:15], p[..., -1:]], -1),
    )


def branch_modules(model):
    return dict(
        distribution_writeback=model.distribution_writeback,
        context_writeback=model.context_writeback,
    )


def context_diagnostics(model):
    device = next(model.parameters()).device
    x = torch.zeros(4, 26, device=device)
    x[:, 0] = 1.0
    x[:, 14] = 1.0
    x[:, 24:] = 1.0
    x[:, 15:24] = 1.0
    x[1::2, 5:13] = 0.4
    x[2:, 15:23] += 0.6
    with torch.no_grad():
        h = model.context_writeback(model.context_content(x))
        changed = x.clone()
        changed[:, 1:3] = float("nan")
        private = float(
            (model.context_content(changed) - model.context_content(x)).abs().max()
        )
        changed = x.clone()
        changed[:, 24] = 0.0
        changed[:, 15:24] = float("nan")
        gap = float(model.context_writeback(model.context_content(changed)).abs().max())
    return dict(
        slow_change_writeback=float((h[2:] - h[:2]).abs().max()),
        pairing_difference=float((h[0] + h[3] - h[1] - h[2]).abs().max()),
        private_error=private,
        gap_error=gap,
    )


def canary_decision(candidate, control, support):
    from lingsante_pulse.cgm.models.distribution_phase import HARD_GATES

    hard = HARD_GATES + (
        "context_response",
        "context_private_isolation",
        "context_gap_isolation",
        "context_conditioning",
    )
    reports = (candidate, control)
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in reports for k in hard
        ),
        fixed_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 10753, 146)
            for r in reports
        ),
        isolated=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in reports
        ),
        support=support.get("supported_for_canary") is True
        and support["route"] == "slow_context",
        definitions=all(
            r["distribution_phase"]["route"] == "slow_context"
            and r["slow_context"]["distribution_control"] is False
            and r["unmatched_correspondence"]["global_gate_control"] is False
            for r in reports
        )
        and candidate["slow_context"]["additive_control"] is False
        and control["slow_context"]["additive_control"] is True,
        identical_episodes=candidate["episode_sequence_sha256"]
        == control["episode_sequence_sha256"],
        identical_targets=candidate["target_sha256"] == control["target_sha256"],
    )
    c, g = candidate["learned"], control["learned"]
    gates["useful_conditional_content"] = (
        any(c[k][i] > g[k][i] for k in ("current", "longitudinal") for i in (1, 2))
        or candidate["event_ev"] > control["event_ev"]
    )
    ref = (0.4853538740426302, 0.37716955442221894, 0.34321602061390877)
    values = (c["current_mean"], c["longitudinal_mean"], candidate["event_ev"])
    gates["reference_preserved"] = all(v >= r - 0.01 for v, r in zip(values, ref))
    gates["useful_progress"] = any(v > r for v, r in zip(values, ref))
    gates["useful_loss"] = candidate["episode_validation_loss"] < 4.149182319641113
    return dict(
        policy="slow_context_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
    )
