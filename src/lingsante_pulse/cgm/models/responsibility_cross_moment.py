"""Observed responsibility/value cross moments; stable evidence is never rewritten."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import (
    LongitudinalStatisticsEncoder,
    statistics_tokens,
)


def cross_moment_pool(values, scores, support, mask):
    valid = mask & (support[:, :, 0] > 0)
    if not bool(valid[:, 0].all()):
        raise ValueError("Current observation requires positive support")
    d = torch.where(valid[..., None], values, torch.zeros_like(values))
    w = torch.where(valid, support[:, :, 0], torch.zeros_like(support[:, :, 0]))
    stable = (w[..., None] * d).sum(1) / w.sum(1, keepdim=True)
    rw = torch.where(
        valid[..., None] & (support > 0),
        torch.minimum(support, w[..., None]),
        torch.zeros_like(support),
    )
    rw = rw / rw.sum(1, keepdim=True).clamp_min(1e-8)
    q = torch.where(rw > 0, scores, torch.zeros_like(scores))
    q = q - (rw * q).sum(1, keepdim=True)
    mean_d = torch.einsum("btr,btd->brd", rw, d)
    centered = d[:, :, None, :] - mean_d[:, None, :, :]
    moment = torch.einsum("btr,btrd->brd", rw * q, centered)
    no_history = ~valid[:, 1:].any(1)
    stable = torch.where(no_history[:, None], values[:, 0], stable)
    moment = torch.where(no_history[:, None, None], torch.zeros_like(moment), moment)
    return torch.cat([stable[:, None, :], moment], 1).flatten(1)


class ResponsibilityCrossMomentEncoder(LongitudinalStatisticsEncoder):
    def __init__(self):
        super().__init__()
        del self.output
        self.day_projection = nn.Linear(20, 48)
        self.responsibility_type = nn.Parameter(torch.randn(1, 1, 4, 48) * 0.02)
        self.day_score = nn.Linear(48, 1)

    def forward(self, values, features, support, elapsed_days, mask, events):
        valid = mask & (support[:, :, 0] > 0)
        clean = torch.where(valid[..., None], values, torch.zeros_like(values))
        weight = torch.where(
            valid, support[:, :, 0], torch.zeros_like(support[:, :, 0])
        )
        stable = (weight[..., None] * clean).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp_min(1e-8)
        stats, smask = statistics_tokens(features, support, elapsed_days, mask, events)
        tokens = (
            torch.cat(
                [
                    self.stat_projection(stats),
                    self.value_projection(torch.stack([clean[:, 0], stable], 1)),
                ],
                1,
            )
            + self.token_type
        )
        token_mask = torch.cat([smask, torch.ones_like(valid[:, :2])], 1)
        sequence = torch.cat([self.query.expand(len(values), -1, -1), tokens], 1)
        padding = ~torch.cat([torch.ones_like(valid[:, :1]), token_mask], 1)
        context = self.block(sequence, src_key_padding_mask=padding)[:, 0]
        day = torch.cat(
            [
                features,
                support[..., None],
                (elapsed_days / 6)[:, :, None, None].expand(-1, -1, 4, 1),
                events[:, :, None, :].expand(-1, -1, 4, -1),
            ],
            -1,
        )
        day = torch.where(
            (valid[..., None] & (support > 0))[..., None], day, torch.zeros_like(day)
        )
        hidden = (
            self.day_projection(day)
            + self.responsibility_type
            + context[:, None, None, :]
        )
        scores = torch.tanh(self.day_score(torch.nn.functional.gelu(hidden))).squeeze(
            -1
        )
        return cross_moment_pool(values, scores, support, mask)
