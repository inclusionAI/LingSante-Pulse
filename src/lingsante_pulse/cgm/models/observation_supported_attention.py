"""Learn where to read longitudinal evidence, without generating new values."""

import torch
from torch import nn
import numpy as np


def pack_observations(data, daily, features, support):
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        causal_history_indices,
    )
    from lingsante_pulse.cgm.models.parallel_longitudinal_models import (
        bounded_histories,
    )

    histories = bounded_histories(
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
    n = len(daily)
    out = dict(
        values=np.zeros((n, 7, 146), np.float32),
        features=np.zeros((n, 7, 4, 16), np.float32),
        support=np.zeros((n, 7, 4), np.float32),
        elapsed_days=np.zeros((n, 7), np.float32),
        mask=np.zeros((n, 7), bool),
    )
    starts = np.asarray(data["starts"]).astype("datetime64[m]")
    for i, history in enumerate(histories):
        rows = history[::-1]
        if rows[0] != i:
            raise ValueError("Current day must be first")
        minutes = (starts[i] - starts[rows]).astype("timedelta64[m]").astype(np.int64)
        if (
            np.any(minutes < 0)
            or np.any(minutes > 8640)
            or np.any(np.diff(minutes) < 1440)
        ):
            raise ValueError("Noncausal, overlapping, or oversized history")
        age = (minutes / 1440.0).astype(np.float32)
        for key in ("subject", "cohort", "split", "cadence", "source"):
            if not np.all(data[key][rows] == data[key][i]):
                raise ValueError("Cross-provenance read")
        size = len(rows)
        out["values"][i, :size] = daily[rows]
        out["features"][i, :size] = features[rows]
        out["support"][i, :size] = support[rows]
        out["elapsed_days"][i, :size] = age
        out["mask"][i, :size] = True
    return out


class ObservationSupportedAttention(nn.Module):
    def __init__(self, coherent=False):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(133, 64), nn.GELU(), nn.Linear(64, 1 if coherent else 146)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, values, features, support, elapsed_days, mask):
        # Slot zero is the current observation; remaining slots are legal past days.
        if not bool(mask[:, 0].all()):
            raise ValueError("Current observation must be present")
        valid = mask & (support[:, :, 0] > 0)
        if not bool(valid[:, 0].all()):
            raise ValueError("Current observation must have positive support")
        # Sanitize padding before scoring: invalid data must not leak through NaN*0.
        clean_values = torch.where(valid[:, :, None], values, torch.zeros_like(values))
        clean_features = torch.where(
            valid[:, :, None, None], features, torch.zeros_like(features)
        )
        clean_support = torch.where(
            valid[:, :, None], support, torch.zeros_like(support)
        )
        clean_elapsed = torch.where(valid, elapsed_days, torch.zeros_like(elapsed_days))
        current = (
            clean_features[:, 0].flatten(1)[:, None].expand(-1, values.shape[1], -1)
        )
        inputs = torch.cat(
            [
                clean_features.flatten(2),
                current,
                clean_support,
                clean_elapsed[:, :, None] / 6.0,
            ],
            dim=-1,
        )
        logits = (
            self.score(inputs)
            + clean_support[:, :, 0].clamp(min=1e-8).log()[:, :, None]
        )
        weights = logits.masked_fill(~valid[:, :, None], float("-inf")).softmax(dim=1)
        output = (weights * clean_values).sum(1)
        no_history = ~valid[:, 1:].any(1)
        output = torch.where(no_history[:, None], values[:, 0], output)
        return output, weights
