"""Pack inference-time observation evidence without loading training tools."""

from lingsante_pulse.cgm.models.observation_supported_attention import pack_observations
from lingsante_pulse.cgm.models.final_space_longitudinal import evidence_v2
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    generic_multiscale_features,
)
import numpy as np


def make_inputs(data, daily, contract, control):
    typed, support = evidence_v2(
        data["glucose"], data["observed"], data["cadence"], data["starts"]
    )
    features = (
        generic_multiscale_features(data["glucose"], data["observed"])[0]
        if control
        else typed
    )
    features = (
        (features - contract["features"]["center"]) / contract["features"]["scale"]
    ).astype(np.float32)
    if control:
        support = np.repeat(support[:, :1], 4, axis=1)
    return pack_observations(data, daily, features, support)
