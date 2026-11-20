"""Joint evidence learning with an exactly recoverable stable CARE estimate."""

import torch
from torch import nn
from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import (
    LongitudinalStatisticsEncoder,
    statistics_tokens,
)


class ReversibleEvidenceEncoder(LongitudinalStatisticsEncoder):
    def __init__(self):
        super().__init__()
        del self.output
        self.shift = nn.Sequential(
            nn.Linear(48, 64, bias=False), nn.GELU(), nn.Linear(64, 146, bias=False)
        )
        self.log_scale = nn.Sequential(nn.Linear(146, 64), nn.GELU(), nn.Linear(64, 48))
        nn.init.zeros_(self.shift[-1].weight)
        nn.init.zeros_(self.log_scale[-1].weight)
        nn.init.zeros_(self.log_scale[-1].bias)

    def couple(self, stable, evidence):
        u = stable + self.shift(evidence)
        v = evidence * torch.exp(torch.tanh(self.log_scale(u)))
        return torch.cat([u, v], -1)

    def inverse(self, code):
        u, v = code[..., :146], code[..., 146:]
        evidence = v * torch.exp(-torch.tanh(self.log_scale(u)))
        return u - self.shift(evidence), evidence

    def forward(self, values, features, support, elapsed_days, mask, events):
        valid = mask & (support[:, :, 0] > 0)
        if not bool(valid[:, 0].all()):
            raise ValueError("Current observation requires positive support")
        clean = torch.where(valid[..., None], values, torch.zeros_like(values))
        weight = torch.where(
            valid, support[:, :, 0], torch.zeros_like(support[:, :, 0])
        )
        stable = (weight[..., None] * clean).sum(1) / weight.sum(1, keepdim=True)
        no_history = ~valid[:, 1:].any(1)
        stable = torch.where(no_history[:, None], values[:, 0], stable)
        stats, stats_mask = statistics_tokens(
            features, support, elapsed_days, mask, events
        )
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
        token_mask = torch.cat([stats_mask, torch.ones_like(valid[:, :2])], 1)
        sequence = torch.cat([self.query.expand(len(values), -1, -1), tokens], 1)
        padding = ~torch.cat([torch.ones_like(valid[:, :1]), token_mask], 1)
        evidence = self.block(sequence, src_key_padding_mask=padding)[:, 0]
        evidence = torch.where(
            no_history[:, None], torch.zeros_like(evidence), evidence
        )
        return self.couple(stable, evidence)
