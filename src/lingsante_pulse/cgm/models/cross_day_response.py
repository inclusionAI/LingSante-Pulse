"""Withhold an entire past day; predict its challenge recovery from other days.

All context is inside the current endpoint's causal seven-day history. This is
masked historical response modelling, not prospective glucose forecasting.
"""

import torch
from torch import nn
import numpy as np


class ConditionalRecoveryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(151, 64), nn.GELU(), nn.Linear(64, 8))

    def forward(self, embedding, condition):
        return self.net(torch.cat([embedding, condition], -1))


def eligible_responses(packed):
    """Two recovery observations; withhold past day while retaining >=2 days."""
    tokens, valid = packed["challenge_tokens"], packed["challenge_mask"]
    days = packed["mask"] & (packed["support"][:, :, 0] > 0)
    supported_days = days.sum(1) >= 3
    result = valid & (tokens[:, :, 18:26].sum(-1) >= 2)
    result &= days.repeat_interleave(4, dim=1)
    result &= supported_days[:, None]
    result[:, :4] = False
    return result


def response_batch(packed, rows, event_indices):
    """Return masked context and the condition/target, never exposing held-out day.

    Conditions: initial glucose, rise amplitude, rise duration, peak clock sin/cos.
    Targets: fraction recovered at +15..+120min; unobserved coordinates ignored.
    """
    sub = {k: v[rows].clone() for k, v in packed.items()}
    indexes = torch.as_tensor(
        event_indices, dtype=torch.long, device=sub["mask"].device
    )
    if len(indexes) != len(sub["mask"]):
        raise ValueError("One event index per row required")
    active = indexes >= 0
    safe = indexes.clamp_min(0)
    if bool((safe >= 28).any()):
        raise ValueError("Event index out of range")
    positions = torch.arange(len(indexes), device=indexes.device)
    eligible = eligible_responses(sub)
    if not bool(eligible[positions[active], safe[active]].all()):
        raise ValueError("Unsupported or non-past response target")
    token = sub["challenge_tokens"][positions, safe].clone()
    condition = torch.stack(
        [token[:, 26], token[:, 27], token[:, 28], token[:, 30], token[:, 31]], -1
    )
    observed = (token[:, 18:26] > 0) & active[:, None]
    target = 1.0 - token[:, 5:13] / token[:, 27:28].clamp_min(1e-6)
    target = torch.where(observed, target, torch.zeros_like(target))
    condition = torch.where(active[:, None], condition, torch.zeros_like(condition))
    day = safe // 4
    for k in ("values", "features", "support", "elapsed_days", "mask", "events"):
        sub[k][positions[active], day[active]] = 0
    for k in ("challenge_tokens", "challenge_mask"):
        for offset in range(4):
            sub[k][positions[active], day[active] * 4 + offset] = 0
    return sub, condition, target, observed


def observed_mse(prediction, target, observed):
    error = torch.where(observed, prediction - target, torch.zeros_like(prediction))
    return error.square().sum() / observed.sum().clamp_min(1)


def recovery_loss(model, decoder, packed, rows, event_indices, no_context=False):
    context, condition, target, observed = response_batch(packed, rows, event_indices)
    z = model(**context)
    if no_context:
        z = z * 0.0
    return observed_mse(decoder(z, condition), target, observed)


def choose_response_events(eligible, rows, rng=None):
    """Uniform training choice; deterministic first eligible event for reporting."""
    chosen = []
    for row in rows:
        available = np.flatnonzero(eligible[row])
        chosen.append(
            -1
            if not len(available)
            else int(available[0] if rng is None else rng.choice(available))
        )
    return np.asarray(chosen, dtype=np.int64)


@torch.no_grad()
def evaluate_recovery(model, decoder, packed, rows, eligible, no_context=False):
    model.eval()
    decoder.eval()
    totals = np.zeros(3, np.float64)
    endpoints = 0
    for batch in np.array_split(
        np.asarray(rows), max(1, int(np.ceil(len(rows) / 128)))
    ):
        chosen = choose_response_events(eligible, batch)
        context, condition, target, observed = response_batch(packed, batch, chosen)
        z = model(**context)
        pred = decoder(z * 0.0 if no_context else z, condition)
        zero = decoder(z * 0.0, condition)
        totals += [
            float(torch.where(observed, pred - target, 0.0).square().sum()),
            float(torch.where(observed, zero - target, 0.0).square().sum()),
            float(observed.sum()),
        ]
        endpoints += int((chosen >= 0).sum())
    if totals[2] == 0:
        raise ValueError("No observed recovery targets in evaluation split")
    return dict(
        mse=float(totals[0] / totals[2]),
        zero_context_mse=float(totals[1] / totals[2]),
        observed_points=int(totals[2]),
        endpoints=endpoints,
    )
