"""One shared longitudinal query, with observation-certified challenge sets."""

import torch
from torch import nn

from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import (
    LongitudinalStatisticsEncoder,
    statistics_tokens,
)


class ChallengeSetEncoder(LongitudinalStatisticsEncoder):
    """Replace legacy challenge summaries, not the CARE day encoder or classifier.

    Pooled control has identical parameters and valid token counts, but all
    challenge tokens contain their common mean: event heterogeneity is removed.
    """

    def __init__(self, pooled_control=False):
        super().__init__()
        self.pooled_control = pooled_control
        self.challenge_projection = nn.Sequential(
            nn.Linear(33, 48), nn.GELU(), nn.Linear(48, 48)
        )
        self.challenge_type = nn.Parameter(torch.randn(1, 1, 48) * 0.02)

    def forward(
        self,
        values,
        features,
        support,
        elapsed_days,
        mask,
        events,
        challenge_tokens,
        challenge_mask,
    ):
        valid = mask & (support[:, :, 0] > 0)
        if not bool(valid[:, 0].all()):
            raise ValueError("Current observation requires positive support")
        clean = torch.where(valid[..., None], values, torch.zeros_like(values))
        w = torch.where(valid, support[:, :, 0], torch.zeros_like(support[:, :, 0]))
        stable = (w[..., None] * clean).sum(1) / w.sum(1, keepdim=True)
        stats, smask = statistics_tokens(features, support, elapsed_days, mask, events)
        # Legacy top3 challenge summaries (one of the four responsibilities) are
        # replaced by certified raw trajectories; state/extreme/rhythm unchanged.
        smask = smask.clone()
        smask[:, 7:14] = False
        stats = torch.where(smask[..., None], stats, torch.zeros_like(stats))
        base = (
            torch.cat(
                [
                    self.stat_projection(stats),
                    self.value_projection(torch.stack([clean[:, 0], stable], 1)),
                ],
                1,
            )
            + self.token_type
        )
        ch = torch.where(
            challenge_mask[..., None],
            challenge_tokens,
            torch.zeros_like(challenge_tokens),
        )
        if self.pooled_control:
            avg = ch.sum(1, keepdim=True) / challenge_mask.sum(1)[
                :, None, None
            ].clamp_min(1)
            ch = avg.expand_as(ch)
        event_values = self.challenge_projection(ch) + self.challenge_type
        tokens = torch.cat([base, event_values], 1)
        tm = torch.cat([smask, torch.ones_like(valid[:, :2]), challenge_mask], 1)
        seq = torch.cat([self.query.expand(len(values), -1, -1), tokens], 1)
        padding = ~torch.cat([torch.ones_like(valid[:, :1]), tm], 1)
        result = self.output(self.block(seq, src_key_padding_mask=padding)[:, 0])
        return torch.where((~valid[:, 1:].any(1))[:, None], values[:, 0], result)
