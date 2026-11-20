"""Directly learn the exported CARE146 update, not a post-hoc memory transport."""

import numpy as np
import torch
from torch import nn

from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    typed_evidence_features,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import LongitudinalIdea


def cadence_run_statistics(values, observed, cadence, threshold, low=True):
    positions = np.flatnonzero(observed)
    active = values[positions] < threshold if low else values[positions] > threshold
    step = max(5.0, float(cadence))
    runs, length, longest = 0, 0, 0
    for j, yes in enumerate(active):
        connected = j > 0 and (positions[j] - positions[j - 1]) * 5 <= step + 1e-5
        if yes:
            if not connected or not active[j - 1]:
                runs += 1
                length = 0
            length += 1
            longest = max(longest, length)
        else:
            length = 0
    return runs, longest * step


def evidence_v2(glucose, observed, cadence, starts):
    features, support = typed_evidence_features(glucose, observed, cadence, starts)
    for i in range(len(glucose)):
        lr, ld = cadence_run_statistics(glucose[i], observed[i], cadence[i], 70.0)
        hr, hd = cadence_run_statistics(
            glucose[i], observed[i], cadence[i], 180.0, low=False
        )
        features[i, 2, [8, 9, 10, 11]] = [lr, hr, ld, hd]
    return features, support


def longitudinal_targets(features, support, histories):
    """4 responsibilities x (mean, spread, signed tail, ordered change) x 16."""
    result = np.zeros((len(features), 4, 4, 16), np.float32)
    for i, rows in enumerate(histories):
        for branch in range(4):
            valid = rows[support[rows, branch] > 0]
            if not len(valid):
                continue
            x = features[valid, branch]
            weight = support[valid, branch].astype(np.float64)
            weight /= weight.sum()
            mean = np.sum(x * weight[:, None], axis=0)
            spread = np.sqrt(np.sum((x - mean) ** 2 * weight[:, None], axis=0))
            tail = x.max(axis=0)
            if branch == 2:
                tail[[4, 6]] = x[:, [4, 6]].min(axis=0)
            result[i, branch] = np.stack([mean, spread, tail, x[-1] - x[0]])
    return result


def supported_mean(daily, coverage, histories):
    out = daily.copy()
    for i, rows in enumerate(histories):
        weights = coverage[rows]
        if weights.sum() > 0:
            out[i] = (daily[rows] * weights[:, None]).sum(0) / weights.sum()
    return out


class FinalSpaceLongitudinal(nn.Module):
    def __init__(self, variant, decoder_weight, decoder_bias):
        super().__init__()
        self.encoder = LongitudinalIdea(variant)
        # Remove unused heads rather than report them as trained capacity.
        del self.encoder.typed_heads
        del self.encoder.transition_heads
        del self.encoder.query
        nn.init.zeros_(self.encoder.care_head.weight)
        nn.init.zeros_(self.encoder.care_head.bias)
        self.register_buffer(
            "decoder_weight", torch.as_tensor(decoder_weight, dtype=torch.float32)
        )
        self.register_buffer(
            "decoder_bias", torch.as_tensor(decoder_bias, dtype=torch.float32)
        )

    def forward(self, daily, stable, reliability, features, support, elapsed, mask):
        memory, auxiliary = self.encoder.encode(
            daily,
            features,
            support,
            elapsed,
            mask,
            corrupt=self.training and self.encoder.variant == "masked_scale",
        )
        residual = self.encoder.care_head(memory.flatten(1))
        final = stable + reliability[:, None] * residual
        final = torch.where(mask.any(1)[:, None], final, daily)
        prediction = final @ self.decoder_weight.T + self.decoder_bias
        return final, prediction, auxiliary


def final_space_loss(final, stable, prediction, target):
    current = (prediction[:, :64] - target[:, :64]).square().mean()
    long_term = (prediction[:, 64:] - target[:, 64:]).square().mean()
    residual = (final - stable).square().mean()
    x, y = final - final.mean(0), stable - stable.mean(0)
    cx, cy = x.T @ x / max(len(x) - 1, 1), y.T @ y / max(len(y) - 1, 1)
    covariance = (cx - cy).square().sum() / cy.square().sum().clamp(min=1e-8)
    return current + long_term + 0.1 * residual + 0.1 * covariance
