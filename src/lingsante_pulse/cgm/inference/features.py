"""Daily inference transformations shared by the deployed CGM encoder."""

from __future__ import annotations
import numpy as np
import torch
from lingsante_pulse.cgm.models.dynamics import (
    CAREMultiAxisSidecar,
    responsible_dynamics_coordinates,
)


def interpolate_windows(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Match the finite 288-point inputs used by public pretraining."""
    output = np.empty_like(values, dtype=np.float32)
    positions = np.arange(values.shape[1])
    for row in range(len(values)):
        keep = observed[row].astype(bool) & np.isfinite(values[row])
        if not keep.any():
            raise ValueError("window has no physical CGM observation")
        output[row] = np.interp(positions, positions[keep], values[row, keep])
    return output


def affine_gauge_codes(
    common_glucose: np.ndarray,
    common_observed: np.ndarray,
    *,
    center: float,
    scale: float,
) -> np.ndarray:
    values = np.asarray(common_glucose, dtype=np.float64)
    weight = np.asarray(common_observed, dtype=bool) & np.isfinite(values)
    count = weight.sum(axis=1)
    if np.any(count < 2) or scale <= 0:
        raise ValueError(
            "affine gauge needs two canonical observations and positive scale"
        )
    safe = np.nan_to_num(values)
    mean = (safe * weight).sum(axis=1) / count
    variance = (((safe - mean[:, None]) ** 2) * weight).sum(axis=1) / count
    code = np.column_stack(
        [
            (mean - center) / scale,
            np.log(np.maximum(np.sqrt(variance) / scale, 1e-6)),
        ]
    )
    if code.shape != (len(values), 2) or not np.isfinite(code).all():
        raise RuntimeError("invalid affine gauge code")
    return code.astype(np.float32)


def apply_affine_gauge_append_adapter(
    embedding: np.ndarray,
    gauge: np.ndarray,
) -> np.ndarray:
    """Append the 2D gauge without deleting any frozen representation coordinate."""
    embedding = np.asarray(embedding, dtype=np.float32)
    gauge = np.asarray(gauge, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 128:
        raise ValueError("append-only affine gauge expects a 128D embedding")
    if gauge.shape != (len(embedding), 2):
        raise ValueError("append-only affine gauge expects two coordinates per row")
    output = np.concatenate([embedding[:, :96], gauge, embedding[:, 96:]], axis=1)
    if output.shape != (len(embedding), 130) or not np.isfinite(output).all():
        raise RuntimeError("invalid append-only affine-gauge embedding")
    return output.astype(np.float32)


def multi_axis_sidecar_codes(
    common_glucose: np.ndarray,
    common_observed: np.ndarray,
    checkpoint: dict,
    *,
    batch_size: int,
    device: torch.device,
    branches: str = "both",
) -> np.ndarray:
    """Extract the frozen time-even state and time-odd dynamics codes."""
    values = np.asarray(common_glucose, dtype=np.float32)
    observed = np.asarray(common_observed, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 288 or observed.shape != values.shape:
        raise ValueError("multi-axis sidecar expects aligned [N,288] canonical traces")
    center = float(checkpoint["glucose_center"])
    scale = float(checkpoint["glucose_scale"])
    if scale <= 0:
        raise ValueError("multi-axis sidecar glucose scale must be positive")
    weight = observed.astype(np.float64)
    count = weight.sum(axis=1, keepdims=True)
    if np.any(count < 8):
        raise ValueError("multi-axis sidecar requires eight canonical observations")
    mean = (values * weight).sum(axis=1, keepdims=True) / count
    variance = (np.square(values - mean) * weight).sum(axis=1, keepdims=True) / count
    physical = ((values - center) / scale).astype(np.float32)
    relative = ((values - mean) / np.sqrt(variance + 1e-6)).astype(np.float32)
    config = checkpoint["config"]
    model = CAREMultiAxisSidecar(
        state_dimensions=int(config["representation"]["physical_state_dimensions"]),
        dynamics_dimensions=int(config["representation"]["signed_dynamics_dimensions"]),
        hidden_channels=int(config["representation"].get("hidden_channels", 32)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for left in range(0, len(values), batch_size):
            right = left + batch_size
            physical_batch = torch.from_numpy(physical[left:right]).to(device)
            relative_batch = torch.from_numpy(relative[left:right]).to(device)
            observed_batch = torch.from_numpy(observed[left:right]).to(device)
            result = model(
                physical_batch,
                relative_batch,
                observed_batch,
                observed_batch,
            )
            selected = {
                "state": result["state"],
                "dynamics": result["dynamics"],
                "dynamics_readout": result["dynamics_readout"],
                "dynamics_responsible": responsible_dynamics_coordinates(
                    result["dynamics"], model.dynamics_head.weight
                ),
                "both": torch.cat([result["state"], result["dynamics"]], dim=1),
            }
            if branches not in selected:
                raise ValueError(
                    "multi-axis branches must be state, dynamics, dynamics_readout, dynamics_responsible or both"
                )
            outputs.append(selected[branches].cpu().numpy())
    codes = np.concatenate(outputs).astype(np.float32)
    expected_width = {
        "state": 16,
        "dynamics": 16,
        "dynamics_readout": 10,
        "dynamics_responsible": 16,
        "both": 32,
    }[branches]
    if codes.shape != (len(values), expected_width) or not np.isfinite(codes).all():
        raise RuntimeError("invalid frozen multi-axis sidecar codes")
    return codes


def apply_multi_axis_sidecar_adapter(
    affine_embedding: np.ndarray, sidecar: np.ndarray
) -> np.ndarray:
    embedding = np.asarray(affine_embedding, dtype=np.float32)
    code = np.asarray(sidecar, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] not in {130, 137}:
        raise ValueError(
            "multi-axis sidecar requires the frozen 130D Affine CARE base "
            "or its frozen 137D compact-physiology extension"
        )
    if (
        code.ndim != 2
        or len(code) != len(embedding)
        or code.shape[1] not in {10, 16, 32}
    ):
        raise ValueError("multi-axis sidecar requires 10, 16 or 32 learned coordinates")
    output = np.concatenate([embedding, code], axis=1)
    if (
        output.shape
        != (
            len(embedding),
            embedding.shape[1] + code.shape[1],
        )
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid multi-axis embedding")
    return output.astype(np.float32)
