"""Hard-preserve fit-only current CGM readouts while learning long evidence."""

import numpy as np
import torch
from torch import nn
from lingsante_pulse.cgm.models.final_space_longitudinal import FinalSpaceLongitudinal


def current_physiology_null_basis(decoder_weight):
    weight = np.asarray(decoder_weight, dtype=np.float64)[:64]
    if weight.shape != (64, 146) or not np.isfinite(weight).all():
        raise ValueError("Expected finite 64x146 current-physiology decoder")
    _, _, vh = np.linalg.svd(weight, full_matrices=True)
    basis = vh[64:].T
    residual = weight @ basis
    if not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1e-8:
        raise ValueError("Unsupported numerical nullspace")
    return basis.astype(np.float32)


class PhysiologyGuardedFinalSpace(FinalSpaceLongitudinal):
    def __init__(self, variant, decoder_weight, decoder_bias):
        if variant != "masked_scale":
            raise ValueError("Only preregistered masked_scale is enabled")
        super().__init__(variant, decoder_weight, decoder_bias)
        self.encoder.care_head = nn.Linear(160, 82)
        nn.init.zeros_(self.encoder.care_head.weight)
        nn.init.zeros_(self.encoder.care_head.bias)
        self.register_buffer(
            "null_basis",
            torch.from_numpy(current_physiology_null_basis(decoder_weight)),
        )

    def forward(self, daily, stable, reliability, features, support, elapsed, mask):
        memory, aux = self.encoder.encode(
            daily, features, support, elapsed, mask, corrupt=self.training
        )
        update = self.encoder.care_head(memory.flatten(1)) @ self.null_basis.T
        final = stable + reliability[:, None] * update
        final = torch.where(mask.any(1)[:, None], final, daily)
        return final, final @ self.decoder_weight.T + self.decoder_bias, aux
