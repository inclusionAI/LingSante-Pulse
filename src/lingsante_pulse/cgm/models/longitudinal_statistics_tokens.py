"""Joint statistics-token encoding; no appended feature block or residual write."""

import torch
from torch import nn


def statistics_tokens(features, support, elapsed_days, mask, events):
    """Seven functionals per responsibility plus separate low/high event tokens.

    Current is slot 0. Events are observed fractions below70/above180 mg/dL.
    Gaps are not interpolated: adjacency is exactly 1440 integer minutes.
    """
    valid = mask[..., None] & (support > 0)
    x = torch.where(valid[..., None], features, torch.zeros_like(features))
    w = torch.where(valid, support, torch.zeros_like(support))
    total = w.sum(1).clamp_min(1e-8)
    p = w / total[:, None]
    mean = (p[..., None] * x).sum(1)
    delta = x - mean[:, None]
    # Epsilon-stabilized root is exactly zero for a constant supported sequence.
    root = lambda v: (v.clamp_min(0) + 1e-8).sqrt() - 1e-4
    std = root((p[..., None] * delta.square()).sum(1))
    upper = root((p[..., None] * delta.relu().square()).sum(1))
    lower = root((p[..., None] * (-delta).relu().square()).sum(1))
    age = torch.where(mask, elapsed_days, torch.zeros_like(elapsed_days))
    time_mean = (p * age[..., None]).sum(1)
    dt = time_mean[:, None] - age[..., None]  # Positive toward current time.
    time_var = (p * dt.square()).sum(1)
    trend = (p[..., None] * dt[..., None] * delta).sum(1) / time_var.clamp_min(1e-6)[
        ..., None
    ]
    minutes = torch.round(age * 1440).to(torch.int64)
    adjacent = mask[:, :-1] & mask[:, 1:] & ((minutes[:, 1:] - minutes[:, :-1]) == 1440)
    pair_w = torch.minimum(w[:, :-1], w[:, 1:]) * adjacent[..., None]
    pair_total = pair_w.sum(1)
    lag = (pair_w[..., None] * delta[:, :-1] * delta[:, 1:]).sum(1)
    lag = lag / pair_total.clamp_min(1e-8)[..., None] / std.clamp_min(1e-4)
    lag = torch.where(pair_total[..., None] > 0, lag, torch.zeros_like(lag))
    moments = torch.stack([x[:, 0], mean, std, upper, lower, trend, lag], dim=2)
    count = valid.sum(1)
    moment_mask = (count[..., None] > 0).expand(-1, -1, 7).clone()
    moment_mask[:, :, 0] = valid[:, 0]
    moment_mask[:, :, 2:6] &= count[..., None] > 1
    moment_mask[:, :, 6] = pair_total > 0
    neff = total.square() / w.square().sum(1).clamp_min(1e-8)
    span = torch.where(valid, age[..., None], torch.zeros_like(w)).amax(1)
    meta = torch.stack([w.sum(1) / 7, neff / 7, span / 6, pair_total / 6], dim=-1)
    moment_input = torch.cat([moments, meta[:, :, None].expand(-1, -1, 7, -1)], dim=-1)

    state_valid = mask & (support[:, :, 0] > 0)
    e = torch.where(state_valid[..., None], events, torch.zeros_like(events))
    occurred = (e > 0).to(e.dtype)
    ew = w[:, :, 0]
    ep = ew / ew.sum(1, keepdim=True).clamp_min(1e-8)
    pair = torch.minimum(ew[:, :-1], ew[:, 1:]) * adjacent
    pair_den = pair.sum(1, keepdim=True).clamp_min(1e-8)
    pair_mean = lambda v: (pair[..., None] * v).sum(1) / pair_den
    prevalence = (ep[..., None] * occurred).sum(1)
    burden = (ep[..., None] * e).sum(1)
    burden_std = root((ep[..., None] * (e - burden[:, None]).square()).sum(1))
    event_stats = torch.stack(
        [
            prevalence,
            occurred.sum(1) / 7,
            pair_mean(occurred[:, :-1] * occurred[:, 1:]),
            pair_mean(
                occurred[:, :-1] * (1 - occurred[:, 1:])
            ),  # Past -> present onset.
            pair_mean((1 - occurred[:, :-1]) * occurred[:, 1:]),
            occurred[:, 0],
            burden,
            burden_std,
        ],
        dim=-1,
    )
    # Eight explicit event descriptors; the other slots are padding, not new data.
    event_input = torch.cat(
        [event_stats, torch.zeros_like(event_stats), meta[:, :1].expand(-1, 2, -1)],
        dim=-1,
    )
    token_input = torch.cat([moment_input.flatten(1, 2), event_input], dim=1)
    token_mask = torch.cat(
        [moment_mask.flatten(1), state_valid.any(1)[:, None].expand(-1, 2)], dim=1
    )
    return torch.where(
        token_mask[..., None], token_input, torch.zeros_like(token_input)
    ), token_mask


class LongitudinalStatisticsEncoder(nn.Module):
    """One shared 146D representation from 30 statistics + two CARE-value tokens."""

    def __init__(self, mean_only=False):
        super().__init__()
        self.mean_only = mean_only
        self.stat_projection = nn.Linear(20, 48)
        self.value_projection = nn.Linear(146, 48)
        self.token_type = nn.Parameter(torch.randn(1, 32, 48) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, 48) * 0.02)
        self.block = nn.TransformerEncoderLayer(
            48, 4, 96, dropout=0.0, batch_first=True, norm_first=True
        )
        self.output = nn.Linear(48, 146)

    def forward(self, values, features, support, elapsed_days, mask, events):
        valid = mask & (support[:, :, 0] > 0)
        if not bool(valid[:, 0].all()):
            raise ValueError("Current observation requires positive support")
        clean_values = torch.where(valid[..., None], values, torch.zeros_like(values))
        weight = torch.where(
            valid, support[:, :, 0], torch.zeros_like(support[:, :, 0])
        )
        stable = (weight[..., None] * clean_values).sum(1) / weight.sum(1, keepdim=True)
        stats, stats_mask = statistics_tokens(
            features, support, elapsed_days, mask, events
        )
        if self.mean_only:
            # Same architecture/parameters: remove higher functionals, keep current/mean.
            stats = stats.clone()
            for group in range(4):
                stats[:, group * 7 + 2 : group * 7 + 7] = stats[
                    :, group * 7 + 1 : group * 7 + 2
                ]
            # Presence/absence is no longer exposed by this ablation.
            stats[:, 28:] = 0
            stats_mask = stats_mask.clone()
            for group in range(4):
                stats_mask[:, group * 7 + 2 : group * 7 + 7] = stats_mask[
                    :, group * 7 + 1 : group * 7 + 2
                ]
            stats_mask[:, 28:] = True
        tokens = torch.cat(
            [
                self.stat_projection(stats),
                self.value_projection(torch.stack([clean_values[:, 0], stable], dim=1)),
            ],
            dim=1,
        )
        tokens = tokens + self.token_type
        token_mask = torch.cat([stats_mask, torch.ones_like(valid[:, :2])], dim=1)
        sequence = torch.cat([self.query.expand(len(values), -1, -1), tokens], dim=1)
        padding = ~torch.cat([torch.ones_like(valid[:, :1]), token_mask], dim=1)
        z = self.output(self.block(sequence, src_key_padding_mask=padding)[:, 0])
        no_history = ~valid[:, 1:].any(1)
        return torch.where(no_history[:, None], values[:, 0], z)
