"""Hierarchical physical-time interactions over gap-safe CGM observations and days."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lingsante_pulse.cgm.models.distribution_phase import DistributionBarycenter
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import bounded_histories


ARCHITECTURE = "hierarchical_continuous_time_day_memory_v1"
PREREGISTERED_STEPS = (600, 6960)
MAX_DAYS = 7
MAX_POINTS = 96
BASE_DIM = 8
WIDTH = 16
SCALES = np.asarray([10.0, 20.0, 40.0, 80.0], np.float32)
AGE_SCALES = np.asarray([1.0, 3.0, 7.0], np.float32)


def _phase(observed, cadence):
    if cadence in (3, 5):
        return 0
    if cadence == 15:
        return int(np.stack([observed[k::3].sum() for k in range(3)]).argmax())
    raise ValueError("HCTDM requires canonical-grid 3-, 5- or 15-minute cadence")


def pack_hctdm(data, observation_policy="fixed15_v1"):
    if observation_policy not in ("fixed15_v1", "observed_interval_v1"):
        raise ValueError("Unknown memory observation policy")
    histories = bounded_histories(
        causal_history_indices(
            data["subject"],
            data["starts"],
            data["cohort"],
            data["split"],
            data["cadence"],
            data["source"],
            max_days=MAX_DAYS,
        ),
        data["starts"],
    )
    count = len(histories)
    capacity = (
        MAX_POINTS if observation_policy == "fixed15_v1" else data["glucose"].shape[1]
    )
    base = np.zeros((count, MAX_DAYS, capacity, BASE_DIM), np.float32)
    clock = np.zeros((count, MAX_DAYS, capacity, 2), np.float32)
    point_mask = np.zeros((count, MAX_DAYS, capacity), bool)
    day_age = np.zeros((count, MAX_DAYS), np.float32)
    day_mask = np.zeros((count, MAX_DAYS), bool)
    starts = np.asarray(data["starts"]).astype("datetime64[m]").astype(np.int64)
    for endpoint, history in enumerate(histories):
        rows = np.asarray(history, np.int64)[-MAX_DAYS:]
        if rows[-1] != endpoint:
            raise ValueError("Causal history must end at endpoint")
        for key in ("subject", "cohort", "split", "cadence", "source"):
            if not np.all(
                np.asarray(data[key])[rows] == np.asarray(data[key])[endpoint]
            ):
                raise ValueError("Cross-provenance HCTDM history")
        offset = MAX_DAYS - len(rows)
        for local, row in enumerate(rows):
            slot = offset + local
            glucose = np.asarray(data["glucose"][row], np.float32)
            observed = np.asarray(data["observed"][row], bool) & np.isfinite(glucose)
            phase = _phase(observed, int(data["cadence"][row]))
            positions = np.arange(phase, len(glucose), 3, dtype=np.int64)[:MAX_POINTS]
            values = glucose[positions]
            seen = observed[positions]
            valid = seen.copy()
            valid[0] = False
            valid[1:] &= seen[:-1]
            irregular = (
                observation_policy == "observed_interval_v1" and int(valid.sum()) < 24
            )
            if irregular:
                # Preserve every real sample, without inventing intermediate points.
                positions = np.flatnonzero(observed)
                values = glucose[positions]
                seen = np.ones(len(positions), bool)
                valid = seen.copy()
                if len(valid):
                    valid[0] = False
                    valid[1:] = np.diff(positions) * 5 <= 60
            safe_values = np.where(seen, values, 100.0)
            delta = np.zeros_like(safe_values)
            delta[1:] = safe_values[1:] - safe_values[:-1]
            if irregular:
                # A 15-minute-equivalent observed slope, not an interpolated value.
                delta[1:] *= 15.0 / (np.diff(positions) * 5)
            length = len(positions)
            base[endpoint, slot, :length, :4] = np.tanh(
                (safe_values[:, None] - 100.0) / SCALES
            )
            base[endpoint, slot, :length, 4:] = np.tanh(delta[:, None] / SCALES)
            minute = (starts[row] + positions * 5) % 1440
            angle = 2.0 * np.pi * minute / 1440.0
            clock[endpoint, slot, :length, 0] = np.sin(angle)
            clock[endpoint, slot, :length, 1] = np.cos(angle)
            point_mask[endpoint, slot, :length] = valid
            day_age[endpoint, slot] = (
                float(max(starts[endpoint] - starts[row], 0)) / 1440.0
            )
            day_mask[endpoint, slot] = int(valid.sum()) >= 24
    active = day_mask.sum(1) >= 3
    point_mask &= day_mask[..., None] & active[:, None, None]
    day_mask &= active[:, None]
    return {
        "hctdm_base": base,
        "hctdm_clock": clock,
        "hctdm_point_mask": point_mask,
        "hctdm_day_age": day_age,
        "hctdm_day_mask": day_mask,
        "hctdm_active": active,
    }


def controlled_time_rows(clock, point_mask, day_age, day_mask, control):
    if not control:
        return clock, day_age
    shifted_clock = clock.clone()
    shifted_age = day_age.clone()
    for endpoint in range(len(clock)):
        days = torch.nonzero(day_mask[endpoint], as_tuple=False).flatten()
        if days.numel() > 1:
            shifted_age[endpoint, days] = day_age[endpoint, days].flip(0)
        for day in days:
            points = torch.nonzero(point_mask[endpoint, day], as_tuple=False).flatten()
            if points.numel() > 1:
                shifted_clock[endpoint, day, points] = clock[
                    endpoint, day, points
                ].roll(1, 0)
    return shifted_clock, shifted_age


def _attention(hidden, valid, qkv, output, norm, feedforward, causal_rank=None):
    query, key, value = qkv(hidden).chunk(3, -1)
    logits = torch.einsum("bik,bjk->bij", query, key) / WIDTH**0.5
    allowed = valid[:, :, None] & valid[:, None, :]
    if causal_rank is not None:
        allowed &= causal_rank[:, None, :] >= causal_rank[:, :, None]
    masked = torch.where(allowed, logits, torch.full_like(logits, -1e9))
    weight = torch.softmax(masked, -1) * allowed.to(logits.dtype)
    weight /= weight.sum(-1, keepdim=True).clamp_min(1e-8)
    mixed = norm(hidden + output(weight @ value))
    mixed = mixed + feedforward(mixed)
    return mixed * valid[..., None].to(mixed.dtype)


class HierarchicalContinuousTimeDayMemory(DistributionBarycenter):
    def __init__(self, time_ablation_control=False):
        super().__init__(control=False)
        self.time_ablation_control = bool(time_ablation_control)
        self.hctdm_observation_lift = nn.Linear(3 * BASE_DIM, WIDTH, bias=False)
        self.hctdm_intra_qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.hctdm_intra_output = nn.Linear(WIDTH, WIDTH)
        self.hctdm_intra_norm = nn.LayerNorm(WIDTH)
        self.hctdm_intra_ff = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH)
        )
        self.hctdm_day_time_lift = nn.Linear(
            4 * WIDTH + len(AGE_SCALES), WIDTH, bias=False
        )
        self.hctdm_inter_qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.hctdm_inter_output = nn.Linear(WIDTH, WIDTH)
        self.hctdm_inter_norm = nn.LayerNorm(WIDTH)
        self.hctdm_inter_ff = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH)
        )
        self.hctdm_writeback = nn.Linear(WIDTH, 36, bias=False)
        nn.init.zeros_(self.hctdm_writeback.weight)

    def memory_statistics(self, base, clock, point_mask, day_age, day_mask):
        clock, day_age = controlled_time_rows(
            clock, point_mask, day_age, day_mask, self.time_ablation_control
        )
        interaction = torch.cat(
            [base, base * clock[..., :1], base * clock[..., 1:]], -1
        )
        batch, days, points, _ = interaction.shape
        observation = self.hctdm_observation_lift(interaction).reshape(
            batch * days, points, WIDTH
        )
        valid_points = point_mask.reshape(batch * days, points)
        observation = _attention(
            observation,
            valid_points,
            self.hctdm_intra_qkv,
            self.hctdm_intra_output,
            self.hctdm_intra_norm,
            self.hctdm_intra_ff,
        )
        mass = valid_points.to(observation.dtype)
        day = (observation * mass[..., None]).sum(1) / mass.sum(
            1, keepdim=True
        ).clamp_min(1.0)
        day = day.reshape(batch, days, WIDTH)
        age_kernel = torch.exp(
            -day_age[..., None] / torch.as_tensor(AGE_SCALES, device=day.device)
        )
        timed_day = torch.cat(
            [
                day,
                day * age_kernel[..., 0:1],
                day * age_kernel[..., 1:2],
                day * age_kernel[..., 2:3],
                age_kernel,
            ],
            -1,
        )
        day = self.hctdm_day_time_lift(timed_day)
        day = _attention(
            day,
            day_mask,
            self.hctdm_inter_qkv,
            self.hctdm_inter_output,
            self.hctdm_inter_norm,
            self.hctdm_inter_ff,
            day_age,
        )
        weight = day_mask.to(day.dtype) * age_kernel[..., 1]
        return (day * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(
            1e-8
        )

    def forward(
        self,
        values,
        features,
        support,
        elapsed_days,
        mask,
        path,
        nodes,
        node_index,
        hctdm_base,
        hctdm_clock,
        hctdm_point_mask,
        hctdm_day_age,
        hctdm_day_mask,
        hctdm_active,
    ):
        embedding, statistics = super().forward(
            values, features, support, elapsed_days, mask, path, nodes, node_index
        )
        memory = self.memory_statistics(
            hctdm_base, hctdm_clock, hctdm_point_mask, hctdm_day_age, hctdm_day_mask
        )
        residual = self.hctdm_writeback(memory) * hctdm_active[:, None]
        return embedding + self.readout(residual), statistics + residual


def freeze_observed_state_backbone(model):
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("hctdm_")
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    return trainable


def parameter_budget():
    model = HierarchicalContinuousTimeDayMemory()
    freeze_observed_state_backbone(model)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_dimension": 146,
    }
