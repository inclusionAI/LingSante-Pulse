#!/usr/bin/env python3
"""Ordered longitudinal evidence with an orthogonal CARE-space residual."""

from __future__ import annotations

import torch
from torch import nn

try:
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        RESPONSIBILITIES,
    )
except ModuleNotFoundError:  # package import in local unit tests
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        RESPONSIBILITIES,
    )


class _RecurrentResponsibilityMemory(nn.Module):
    def __init__(self, feature_dimensions: int, hidden_dimensions: int) -> None:
        super().__init__()
        recurrent_input = 2 * feature_dimensions + 4
        self.cell = nn.GRUCell(recurrent_input, hidden_dimensions)
        self.gate = nn.Linear(2 * hidden_dimensions + 1, 1)
        self.target_head = nn.Linear(hidden_dimensions, feature_dimensions)
        self.transition_head = nn.Linear(hidden_dimensions, feature_dimensions)

    def forward(
        self,
        history: torch.Tensor,
        support: torch.Tensor,
        elapsed_days: torch.Tensor,
        history_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = history.new_zeros((len(history), self.cell.hidden_size))
        previous = history.new_zeros((len(history), history.shape[-1]))
        previous_valid = torch.zeros(
            len(history), 1, dtype=torch.bool, device=history.device
        )
        for step in range(history.shape[1]):
            valid = history_mask[:, step : step + 1]
            level = history[:, step]
            transition = torch.where(
                previous_valid, level - previous, torch.zeros_like(level)
            )
            elapsed = elapsed_days[:, step : step + 1]
            time_code = torch.cat(
                [
                    torch.log1p(elapsed),
                    torch.sin(2.0 * torch.pi * elapsed / 7.0),
                    torch.cos(2.0 * torch.pi * elapsed / 7.0),
                ],
                dim=1,
            )
            cell_input = torch.cat(
                [level, transition, time_code, support[:, step : step + 1]], dim=1
            )
            update = self.cell(cell_input, hidden)
            hidden = torch.where(valid, update, hidden)
            previous = torch.where(valid, level, previous)
            previous_valid = previous_valid | valid
        valid_history = history_mask.any(dim=1, keepdim=True)
        mean_support = torch.sum(
            support * history_mask, dim=1, keepdim=True
        ) / torch.clamp(history_mask.sum(dim=1, keepdim=True), min=1)
        gate = torch.sigmoid(self.gate(torch.cat([query, hidden, mean_support], dim=1)))
        code = gate * hidden * valid_history
        return code, self.target_head(code), self.transition_head(code)


class RecurrentOrthogonalEvidenceEncoder(nn.Module):
    """Four ordered memories that can only augment the daily CARE complement."""

    def __init__(
        self,
        daily_basis: torch.Tensor,
        *,
        daily_dimensions: int = 146,
        feature_dimensions: int = 16,
        hidden_dimensions: int = 24,
    ) -> None:
        super().__init__()
        basis = torch.as_tensor(daily_basis, dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] != daily_dimensions:
            raise ValueError("daily_basis must be [daily_dimensions, components]")
        if basis.shape[1] >= daily_dimensions:
            raise ValueError("daily_basis must leave a non-empty orthogonal complement")
        self.register_buffer("daily_basis", basis)
        self.register_buffer("residual_gain", torch.ones(1))
        self.query = nn.Sequential(
            nn.LayerNorm(daily_dimensions),
            nn.Linear(daily_dimensions, hidden_dimensions),
            nn.Tanh(),
        )
        self.memories = nn.ModuleList(
            [
                _RecurrentResponsibilityMemory(
                    feature_dimensions=feature_dimensions,
                    hidden_dimensions=hidden_dimensions,
                )
                for _ in RESPONSIBILITIES
            ]
        )
        self.long_prediction = nn.Sequential(
            nn.LayerNorm(len(RESPONSIBILITIES) * hidden_dimensions),
            nn.Linear(len(RESPONSIBILITIES) * hidden_dimensions, daily_dimensions),
        )

    def orthogonal_residual(
        self, daily_embedding: torch.Tensor, long_prediction: torch.Tensor
    ) -> torch.Tensor:
        residual = long_prediction - daily_embedding
        coefficients = residual @ self.daily_basis
        return residual - coefficients @ self.daily_basis.T

    def set_residual_gain(self, gain: float) -> None:
        if not 0.0 < gain < 100.0:
            raise ValueError("residual gain must be positive and bounded")
        self.residual_gain.fill_(float(gain))

    def forward(
        self,
        daily_embedding: torch.Tensor,
        history_features: torch.Tensor,
        history_support: torch.Tensor,
        elapsed_days: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_features.ndim != 4 or history_features.shape[2] != len(
            RESPONSIBILITIES
        ):
            raise ValueError("history_features must be [batch,day,4,feature]")
        if history_support.shape != history_features.shape[:3]:
            raise ValueError("history_support does not align")
        if elapsed_days.shape != history_mask.shape != history_features.shape[:2]:
            raise ValueError("history time arrays do not align")
        query = self.query(daily_embedding)
        codes = []
        predictions = []
        transition_predictions = []
        for branch, memory in enumerate(self.memories):
            code, prediction, transition = memory(
                history_features[:, :, branch],
                history_support[:, :, branch],
                elapsed_days,
                history_mask,
                query,
            )
            codes.append(code)
            predictions.append(prediction)
            transition_predictions.append(transition)
        memory_codes = torch.stack(codes, dim=1)
        long_prediction = self.long_prediction(torch.cat(codes, dim=1))
        residual = self.orthogonal_residual(daily_embedding, long_prediction)
        valid = history_mask.any(dim=1, keepdim=True)
        embedding = daily_embedding + valid * self.residual_gain * residual
        return {
            "embedding": embedding,
            "long_prediction": long_prediction,
            "orthogonal_residual": residual * valid,
            "typed_predictions": torch.stack(predictions, dim=1),
            "transition_predictions": torch.stack(transition_predictions, dim=1),
            "memory_codes": memory_codes,
        }
