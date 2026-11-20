"""Observed distributions join the CARE global query, not its output vector."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import (
    LongitudinalStatisticsEncoder,
    statistics_tokens,
)

CACHE_SHA = "e76484e6f144f3c7c84b0219c079c1626f67954d3bee1d9d1be18ac5331faad8"


class GlucodensityEncoder(LongitudinalStatisticsEncoder):
    def __init__(self):
        super().__init__()
        self.density_projection = nn.Sequential(
            nn.Linear(44, 48), nn.GELU(), nn.Linear(48, 48)
        )
        self.density_type = nn.Parameter(torch.randn(1, 7, 48) * 0.02)

    def forward(
        self,
        values,
        features,
        support,
        elapsed_days,
        mask,
        events,
        density_tokens,
        density_mask,
    ):
        valid = mask & (support[:, :, 0] > 0)
        if not bool(valid[:, 0].all()):
            raise ValueError("Current observation requires positive support")
        clean = torch.where(valid[..., None], values, torch.zeros_like(values))
        weight = torch.where(
            valid, support[:, :, 0], torch.zeros_like(support[:, :, 0])
        )
        stable = (weight[..., None] * clean).sum(1) / weight.sum(1, keepdim=True)
        stats, stats_mask = statistics_tokens(
            features, support, elapsed_days, mask, events
        )
        stats_mask = stats_mask.clone()
        stats_mask[:, :7] = (
            False  # Replace state summaries; retain other CARE responsibilities.
        )
        original = (
            torch.cat(
                [
                    self.stat_projection(stats),
                    self.value_projection(torch.stack([clean[:, 0], stable], 1)),
                ],
                1,
            )
            + self.token_type
        )
        density = torch.where(
            density_mask[..., None], density_tokens, torch.zeros_like(density_tokens)
        )
        density = self.density_projection(density) + self.density_type.repeat(1, 7, 1)
        tokens = torch.cat([original, density], 1)
        token_mask = torch.cat(
            [stats_mask, torch.ones_like(valid[:, :2]), density_mask], 1
        )
        sequence = torch.cat([self.query.expand(len(values), -1, -1), tokens], 1)
        padding = ~torch.cat([torch.ones_like(valid[:, :1]), token_mask], 1)
        z = self.output(self.block(sequence, src_key_padding_mask=padding)[:, 0])
        return torch.where(~valid[:, 1:].any(1)[:, None], values[:, 0], z)
