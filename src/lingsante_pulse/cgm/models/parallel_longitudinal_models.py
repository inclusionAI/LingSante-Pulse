"""Four independently motivated longitudinal encoders, sharing CARE readouts."""

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("event_set", "decay_state", "day_contrast", "masked_scale")


def bounded_histories(histories, starts, maximum_days=7):
    """Bound the whole observed span, including the current 24-hour window."""
    starts = np.asarray(starts).astype("datetime64[ns]")
    return [
        v[(starts[i] - starts[v]) <= np.timedelta64(maximum_days - 1, "D")]
        for i, v in enumerate(histories)
    ]


class LongitudinalIdea(nn.Module):
    def __init__(self, variant, dimensions=146, width=40):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.width = width
        # All branches retain signs, physical coordinates, coverage and elapsed time.
        self.token = nn.Linear(20, width)
        self.branch = nn.Parameter(torch.zeros(4, width))
        self.query = nn.Linear(dimensions, 4 * width)
        if variant == "event_set":
            self.events = nn.Linear(5, width)
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        elif variant == "decay_state":
            self.rate = nn.Parameter(torch.linspace(-2.0, 2.0, width))
            self.update = nn.GRUCell(width, width)
        elif variant == "day_contrast":
            self.convolution = nn.Conv1d(width, width, 3, padding=1)
            self.projection = nn.Linear(4 * width, 64)
        else:
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in (1, 2, 4)]
            )
            self.reconstruct = nn.Linear(width, 16)
        self.norm = nn.LayerNorm(width)
        self.typed_heads = nn.ModuleList([nn.Linear(width, 16) for _ in range(4)])
        self.transition_heads = nn.ModuleList([nn.Linear(width, 16) for _ in range(4)])
        self.care_head = nn.Linear(4 * width, dimensions)

    def tokens(self, features, support, elapsed, mask):
        # Mask content before any mixing, so padded tokens cannot leak through convs.
        features = features * mask[:, :, None, None]
        support = support * mask[:, :, None]
        elapsed = elapsed * mask
        clock = torch.stack(
            (
                torch.log1p(elapsed),
                torch.sin(elapsed * 2 * math.pi / 7),
                torch.cos(elapsed * 2 * math.pi / 7),
            ),
            -1,
        )
        clock = clock[:, :, None, :].expand(-1, -1, 4, -1)
        x = torch.cat((features, support[..., None], clock), -1)
        return torch.tanh(self.token(x) + self.branch) * mask[:, :, None, None]

    def encode(self, daily, features, support, elapsed, mask, *, corrupt=False):
        b, t = mask.shape
        valid = mask[:, None, :].expand(-1, 4, -1).reshape(b * 4, t)
        tokens = self.tokens(features, support, elapsed, mask)
        x = tokens.permute(0, 2, 1, 3).reshape(b * 4, t, self.width)
        aux = daily.new_zeros(())
        if self.variant == "event_set":
            # Three separate natural-challenge proposals per day, not their mean.
            events = features[:, :, 1, 1:].reshape(b, t, 3, 5)
            event_tokens = torch.tanh(self.events(events))
            event_tokens = event_tokens + tokens[:, :, 1, None, :]
            expanded = tokens[:, :, :, None, :].expand(-1, -1, -1, 3, -1).clone()
            expanded[:, :, 1] = event_tokens
            x = expanded.permute(0, 2, 1, 3, 4).reshape(b * 4, t * 3, self.width)
            valid = valid.repeat_interleave(3, 1)
            safe = valid.clone()
            safe[~safe.any(1), 0] = True
            q = self.query(daily).reshape(b * 4, 1, self.width)
            h = self.attention(q, x, x, key_padding_mask=~safe, need_weights=False)[0][
                :, 0
            ]
        elif self.variant == "decay_state":
            h = x.new_zeros(b * 4, self.width)
            age = elapsed[:, None, :].expand(-1, 4, -1).reshape(b * 4, t)
            prev_age = age[:, 0]
            for k in range(t):
                gap = (prev_age - age[:, k]).clamp(min=0)
                decay = torch.exp(-gap[:, None] * F.softplus(self.rate)[None])
                update = self.update(x[:, k], h * decay)
                h = torch.where(valid[:, k, None], update, h)
                prev_age = torch.where(valid[:, k], age[:, k], prev_age)
            h = h * torch.exp(
                -prev_age[:, None].clamp(min=0) * F.softplus(self.rate)[None]
            )
        elif self.variant == "day_contrast":
            z = torch.tanh(self.convolution(x.transpose(1, 2)).transpose(1, 2))
            h = (z * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        else:
            selected = (
                (torch.rand_like(valid.float()) < 0.25) & valid
                if corrupt
                else torch.zeros_like(valid)
            )
            z = x.masked_fill(selected[..., None], 0)
            z = (
                sum(
                    torch.tanh(conv(z.transpose(1, 2)).transpose(1, 2))
                    for conv in self.convolutions
                )
                / 3
            )
            if corrupt:
                truth = features.permute(0, 2, 1, 3).reshape(b * 4, t, 16)
                aux = (
                    (self.reconstruct(z) - truth).square().mean(-1) * selected
                ).sum() / selected.sum().clamp(min=1)
            h = (z * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        h = self.norm(h).reshape(b, 4, self.width)
        h = h * mask.any(1)[:, None, None]
        return h, aux

    def forward(self, daily, features, support, elapsed, mask):
        h, aux = self.encode(
            daily,
            features,
            support,
            elapsed,
            mask,
            corrupt=self.training and self.variant == "masked_scale",
        )
        return {
            "memory_codes": h,
            "typed_predictions": torch.stack(
                [head(h[:, j]) for j, head in enumerate(self.typed_heads)], 1
            ),
            "transition_predictions": torch.stack(
                [head(h[:, j]) for j, head in enumerate(self.transition_heads)], 1
            ),
            "long_prediction": self.care_head(h.flatten(1)),
            "auxiliary": aux,
        }

    def contrast_loss(self, daily, features, support, elapsed, mask, groups):
        """Two non-overlapping day subsets; same-subject rows are not negatives."""
        ranks = mask.long().cumsum(1)
        left = mask & (ranks % 2 == 1)
        right = mask & (ranks % 2 == 0)
        eligible = left.any(1) & right.any(1)
        if int(eligible.sum()) < 2:
            return daily.sum() * 0
        args = (
            daily[eligible],
            features[eligible],
            support[eligible],
            elapsed[eligible],
        )
        h1, _ = self.encode(*args, left[eligible])
        h2, _ = self.encode(*args, right[eligible])
        z1 = F.normalize(self.projection(h1.flatten(1)), dim=1)
        z2 = F.normalize(self.projection(h2.flatten(1)), dim=1)
        logits = z1 @ z2.T / 0.2
        ids = groups[eligible]
        duplicate = ids[:, None] == ids[None, :]
        duplicate.fill_diagonal_(False)
        logits = logits.masked_fill(duplicate, -1e4)
        target = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
