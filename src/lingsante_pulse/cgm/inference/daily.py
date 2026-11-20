#!/usr/bin/env python3
"""Extract fixed 128D TPA-CGM embeddings for the existing 14-task protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.models.dynamics import (
    CAREMultiAxisSidecar,
    responsible_dynamics_coordinates,
)
from lingsante_pulse.cgm.models.multiscale_morphology import (
    CAREMultiscaleMorphologySidecar,
)
from lingsante_pulse.cgm.models.masked_recovery_transport import (
    MaskedRecoveryTransportEncoder,
)
from lingsante_pulse.cgm.models.recovery_transport import normalized_trace
from lingsante_pulse.cgm.models.hierarchical_temporal_contrast import (
    HierarchicalTemporalContrastEncoder,
)
from lingsante_pulse.cgm.diagnostics.spectral_complementarity import (
    regrid_relative_spectrum,
    regrid_relative_trace,
)
from lingsante_pulse.cgm.models.daily import (
    TPACGMAutoencoder,
    deterministic_absolute_cadence_view,
    deterministic_cadence_view,
    masked_window_standardize,
)


from lingsante_pulse.cgm.inference.features import (
    interpolate_windows,
    affine_gauge_codes,
    apply_affine_gauge_append_adapter,
    multi_axis_sidecar_codes,
    apply_multi_axis_sidecar_adapter,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_affine_gauge_adapter(
    embedding: np.ndarray,
    gauge: np.ndarray,
    common_center: np.ndarray,
    common_basis: np.ndarray,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float64)
    gauge = np.asarray(gauge, dtype=np.float64)
    common_center = np.asarray(common_center, dtype=np.float64)
    common_basis = np.asarray(common_basis, dtype=np.float64)
    if (
        embedding.ndim != 2
        or embedding.shape[1] != 128
        or gauge.shape != (len(embedding), 2)
        or common_center.shape != (96,)
        or common_basis.shape != (96, 94)
    ):
        raise ValueError("invalid affine-gauge adapter dimensions")
    with np.errstate(all="ignore"):
        relative = (embedding[:, :96] - common_center) @ common_basis
    output = np.concatenate([relative, gauge, embedding[:, 96:]], axis=1)
    if output.shape != embedding.shape or not np.isfinite(output).all():
        raise RuntimeError("invalid affine-gauge embedding")
    return output.astype(np.float32)


def physio_gauge_codes(
    common_glucose: np.ndarray,
    common_observed: np.ndarray,
) -> np.ndarray:
    """Fixed CGM range-occupancy and robust-variability coordinates."""
    values = np.asarray(common_glucose, dtype=np.float64)
    observed = np.asarray(common_observed, dtype=bool) & np.isfinite(values)
    rows = []
    for row, mask in zip(values, observed):
        x = row[mask]
        if len(x) < 2 or float(np.mean(x)) <= 0:
            raise ValueError("physio gauge needs two positive canonical observations")
        mean = float(np.mean(x))
        q05, q25, q75, q95 = np.quantile(x, [0.05, 0.25, 0.75, 0.95])
        observed_index = np.flatnonzero(mask)
        index_step = np.diff(observed_index)
        canonical_stride = int(np.min(index_step))
        canonical_pairs = index_step == canonical_stride
        observed_difference = np.diff(row[observed_index])
        madiff = float(np.mean(np.abs(observed_difference[canonical_pairs]))) / mean
        rows.append(
            [
                float(np.mean(x < 54.0)),
                float(np.mean(x < 70.0)),
                float(np.mean(x > 180.0)),
                float(np.mean(x > 250.0)),
                float(np.std(x)) / mean,
                madiff,
                float(q75 - q25) / mean,
                float(q95 - q05) / mean,
            ]
        )
    code = np.asarray(rows, dtype=np.float32)
    if code.shape != (len(values), 8) or not np.isfinite(code).all():
        raise RuntimeError("invalid physiology gauge code")
    return code


def apply_affine_physio_gauge_adapter(
    embedding: np.ndarray,
    affine_gauge: np.ndarray,
    physio_gauge: np.ndarray,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    affine_gauge = np.asarray(affine_gauge, dtype=np.float32)
    physio_gauge = np.asarray(physio_gauge, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 128:
        raise ValueError("physiology gauge expects a 128D frozen embedding")
    if affine_gauge.shape != (len(embedding), 2) or physio_gauge.shape != (
        len(embedding),
        8,
    ):
        raise ValueError("invalid physiology gauge dimensions")
    output = np.concatenate(
        [embedding[:, :96], affine_gauge, physio_gauge, embedding[:, 96:]], axis=1
    )
    if output.shape != (len(embedding), 138) or not np.isfinite(output).all():
        raise RuntimeError("invalid affine-physiology gauge embedding")
    return output.astype(np.float32)


def compact_physio_gauge_codes(
    affine_gauge: np.ndarray,
    physio_gauge: np.ndarray,
    ridge_coef: np.ndarray,
    ridge_intercept: np.ndarray,
    residual_center: np.ndarray,
    residual_scale: np.ndarray,
    compact_basis: np.ndarray,
    whitening_scale: np.ndarray,
) -> np.ndarray:
    """Apply a frozen affine-residual PCA whitening map to physiology codes."""
    affine_gauge = np.asarray(affine_gauge, dtype=np.float64)
    physio_gauge = np.asarray(physio_gauge, dtype=np.float64)
    ridge_coef = np.asarray(ridge_coef, dtype=np.float64)
    ridge_intercept = np.asarray(ridge_intercept, dtype=np.float64)
    residual_center = np.asarray(residual_center, dtype=np.float64)
    residual_scale = np.asarray(residual_scale, dtype=np.float64)
    compact_basis = np.asarray(compact_basis, dtype=np.float64)
    whitening_scale = np.asarray(whitening_scale, dtype=np.float64)
    components = compact_basis.shape[1] if compact_basis.ndim == 2 else -1
    if (
        affine_gauge.ndim != 2
        or affine_gauge.shape[1] != 2
        or physio_gauge.shape != (len(affine_gauge), 8)
        or ridge_coef.shape != (8, 2)
        or ridge_intercept.shape != (8,)
        or residual_center.shape != (8,)
        or residual_scale.shape != (8,)
        or compact_basis.shape != (8, components)
        or whitening_scale.shape != (components,)
        or not 1 <= components < 8
        or np.any(residual_scale <= 0)
        or np.any(whitening_scale <= 0)
    ):
        raise ValueError("invalid compact physiology adapter dimensions")
    affine_prediction = affine_gauge @ ridge_coef.T + ridge_intercept
    residual = (physio_gauge - affine_prediction - residual_center) / residual_scale
    compact = (residual @ compact_basis) / whitening_scale
    if (
        compact.shape != (len(affine_gauge), components)
        or not np.isfinite(compact).all()
    ):
        raise RuntimeError("invalid compact physiology code")
    return compact.astype(np.float32)


def apply_affine_compact_physio_gauge_adapter(
    embedding: np.ndarray,
    affine_gauge: np.ndarray,
    compact_physio: np.ndarray,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    affine_gauge = np.asarray(affine_gauge, dtype=np.float32)
    compact_physio = np.asarray(compact_physio, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 128:
        raise ValueError("compact physiology gauge expects a 128D frozen embedding")
    if affine_gauge.shape != (len(embedding), 2):
        raise ValueError("compact physiology gauge expects two affine coordinates")
    if (
        compact_physio.ndim != 2
        or len(compact_physio) != len(embedding)
        or not 1 <= compact_physio.shape[1] < 8
    ):
        raise ValueError("invalid compact physiology coordinates")
    output = np.concatenate(
        [embedding[:, :96], affine_gauge, compact_physio, embedding[:, 96:]], axis=1
    )
    expected_width = 130 + compact_physio.shape[1]
    if (
        output.shape != (len(embedding), expected_width)
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid affine-compact-physiology embedding")
    return output.astype(np.float32)


def first_temporal_moment_codes(tokens: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Mask-centered first physical-time coefficient of frozen common tokens."""
    tokens = np.asarray(tokens, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if tokens.ndim != 3 or observed.shape != tokens.shape[:2]:
        raise ValueError("tokens and observation mask must align")
    time = np.linspace(-1.0, 1.0, tokens.shape[1], dtype=np.float64)[None, :]
    weight = observed.astype(np.float64)
    center = (weight * time).sum(axis=1, keepdims=True) / weight.sum(
        axis=1, keepdims=True
    )
    centered = time - center
    denominator = (weight * centered**2).sum(axis=1, keepdims=True)
    if np.any(denominator <= 1e-12):
        raise ValueError("temporal moment has degenerate time support")
    moment = (tokens * (weight * centered)[:, :, None]).sum(axis=1) / denominator
    if not np.isfinite(moment).all():
        raise RuntimeError("invalid temporal moment code")
    return moment.astype(np.float32)


def compact_moment_pool_codes(
    moment: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    basis: np.ndarray,
    whitening_scale: np.ndarray,
) -> np.ndarray:
    moment = np.asarray(moment, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    whitening_scale = np.asarray(whitening_scale, dtype=np.float64)
    components = basis.shape[1] if basis.ndim == 2 else -1
    if (
        moment.ndim != 2
        or moment.shape[1] != 96
        or center.shape != (96,)
        or scale.shape != (96,)
        or basis.shape != (96, components)
        or whitening_scale.shape != (components,)
        or not 1 <= components <= 16
        or np.any(scale <= 0)
        or np.any(whitening_scale <= 0)
    ):
        raise ValueError("invalid moment-pool adapter dimensions")
    with np.errstate(all="ignore"):
        compact = (((moment - center) / scale) @ basis) / whitening_scale
    if compact.shape != (len(moment), components) or not np.isfinite(compact).all():
        raise RuntimeError("invalid compact moment-pool code")
    return compact.astype(np.float32)


def residual_compact_moment_pool_codes(
    compact_moment: np.ndarray,
    compact_physio_embedding: np.ndarray,
    ridge_coef: np.ndarray,
    ridge_intercept: np.ndarray,
    residual_center: np.ndarray,
    residual_scale: np.ndarray,
    residual_basis: np.ndarray,
    whitening_scale: np.ndarray,
) -> np.ndarray:
    """Remove the frozen Compact137-predictable moment and whiten its residual."""
    compact_moment = np.asarray(compact_moment, dtype=np.float64)
    compact_physio_embedding = np.asarray(compact_physio_embedding, dtype=np.float64)
    ridge_coef = np.asarray(ridge_coef, dtype=np.float64)
    ridge_intercept = np.asarray(ridge_intercept, dtype=np.float64)
    residual_center = np.asarray(residual_center, dtype=np.float64)
    residual_scale = np.asarray(residual_scale, dtype=np.float64)
    residual_basis = np.asarray(residual_basis, dtype=np.float64)
    whitening_scale = np.asarray(whitening_scale, dtype=np.float64)
    source_dimensions = compact_moment.shape[1] if compact_moment.ndim == 2 else -1
    components = residual_basis.shape[1] if residual_basis.ndim == 2 else -1
    if (
        compact_moment.ndim != 2
        or source_dimensions not in {12, 96}
        or compact_physio_embedding.shape != (len(compact_moment), 137)
        or ridge_coef.shape != (source_dimensions, 137)
        or ridge_intercept.shape != (source_dimensions,)
        or residual_center.shape != (source_dimensions,)
        or residual_scale.shape != (source_dimensions,)
        or residual_basis.shape != (source_dimensions, components)
        or whitening_scale.shape != (components,)
        or not 1 <= components <= 16
        or components >= source_dimensions
        or np.any(residual_scale <= 0)
        or np.any(whitening_scale <= 0)
    ):
        raise ValueError("invalid residual moment-pool adapter dimensions")
    with np.errstate(all="ignore"):
        predicted = compact_physio_embedding @ ridge_coef.T + ridge_intercept
        residual = (compact_moment - predicted - residual_center) / residual_scale
        compact = (residual @ residual_basis) / whitening_scale
    if (
        compact.shape != (len(compact_moment), components)
        or not np.isfinite(compact).all()
    ):
        raise RuntimeError("invalid residual moment-pool code")
    return compact.astype(np.float32)


def apply_moment_pool_adapter(
    compact_physio_embedding: np.ndarray, compact_moment: np.ndarray
) -> np.ndarray:
    embedding = np.asarray(compact_physio_embedding, dtype=np.float32)
    moment = np.asarray(compact_moment, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] not in {130, 137}:
        raise ValueError("moment pool requires a frozen 130D or 137D CARE embedding")
    if (
        moment.ndim != 2
        or len(moment) != len(embedding)
        or not 1 <= moment.shape[1] <= 16
    ):
        raise ValueError("moment pool requires 1..16 frozen moment coordinates")
    output = np.concatenate([embedding, moment], axis=1)
    if (
        output.shape
        != (
            len(embedding),
            embedding.shape[1] + moment.shape[1],
        )
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid MomentPool-CGM embedding")
    return output.astype(np.float32)


def select_tpa_export_branch(
    embedding: np.ndarray, branch: str, *, responsibility_partitioned: bool
) -> np.ndarray:
    """Select a frozen TPA responsibility block without changing its coordinates."""
    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 128 or not np.isfinite(values).all():
        raise ValueError("TPA export selection expects finite [N,128] embeddings")
    if branch == "full":
        return values
    if not responsibility_partitioned:
        raise ValueError(
            "bulk/tail export requires a responsibility-partitioned checkpoint"
        )
    if branch == "bulk":
        return values[:, :96]
    if branch == "tail":
        return values[:, 96:128]
    raise ValueError("TPA export branch must be full, bulk or tail")


def morphology_sidecar_codes(
    common_glucose: np.ndarray,
    common_observed: np.ndarray,
    checkpoint: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract the frozen cadence-observable multiscale morphology code."""
    values = np.asarray(common_glucose, dtype=np.float32)
    observed = np.asarray(common_observed, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 288 or observed.shape != values.shape:
        raise ValueError("morphology sidecar expects aligned [N,288] canonical traces")
    weight = observed.astype(np.float64)
    count = weight.sum(axis=1, keepdims=True)
    if np.any(count < 8):
        raise ValueError("morphology sidecar requires eight canonical observations")
    mean = (values * weight).sum(axis=1, keepdims=True) / count
    variance = (np.square(values - mean) * weight).sum(axis=1, keepdims=True) / count
    normalized = ((values - mean) / np.sqrt(variance + 1e-6)).astype(np.float32)
    config = checkpoint["config"]
    if bool(config.get("glucofm_components_used", True)):
        raise ValueError("morphology sidecar must be independent of GlucoFM components")
    model = CAREMultiscaleMorphologySidecar(
        dimensions=int(config["representation"]["dimensions"])
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for left in range(0, len(values), batch_size):
            right = left + batch_size
            result = model(
                torch.from_numpy(normalized[left:right]).to(device),
                torch.from_numpy(observed[left:right]).to(device),
                torch.from_numpy(observed[left:right]).to(device),
            )
            outputs.append(result["code"].cpu().numpy())
    codes = np.concatenate(outputs).astype(np.float32)
    dimensions = int(config["representation"]["dimensions"])
    if codes.shape != (len(values), dimensions) or not np.isfinite(codes).all():
        raise RuntimeError("invalid frozen morphology sidecar codes")
    return codes


def apply_morphology_sidecar_adapter(
    dynamics_embedding: np.ndarray, morphology: np.ndarray
) -> np.ndarray:
    """Append morphology24 to the frozen CARE130+dynamics16 representation."""
    embedding = np.asarray(dynamics_embedding, dtype=np.float32)
    code = np.asarray(morphology, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 146:
        raise ValueError("morphology sidecar requires frozen CARE130+dynamics16")
    if code.shape != (len(embedding), 24):
        raise ValueError("morphology sidecar requires 24 learned coordinates")
    output = np.concatenate([embedding, code], axis=1)
    if output.shape != (len(embedding), 170) or not np.isfinite(output).all():
        raise RuntimeError("invalid CARE morphology embedding")
    return output.astype(np.float32)


def recovery_transport_sidecar_codes(
    glucose: np.ndarray,
    observed: np.ndarray,
    checkpoint: dict,
    *,
    batch_size: int,
    device: torch.device,
    export: str = "code",
) -> np.ndarray:
    """Extract the frozen latent code or its physical recovery coordinates."""
    config = checkpoint["config"]
    boundary = config["method_boundary"]
    if boundary.get("glucofm_role") != "evaluation_only_baseline" or bool(
        checkpoint.get("glucofm_framework_or_components", True)
    ):
        raise ValueError("recovery transport must be independent of GlucoFM")
    values = np.stack(
        [normalized_trace(row, mask) for row, mask in zip(glucose, observed)]
    ).astype(np.float32)
    model = MaskedRecoveryTransportEncoder(**config["representation"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for left in range(0, len(values), batch_size):
            right = left + batch_size
            result = model(
                torch.from_numpy(values[left:right]).to(device),
                torch.from_numpy(observed[left:right]).to(device),
            )
            if export == "code":
                output = result["code"]
            elif export == "target_prediction":
                output = result["prediction"]
            else:
                raise ValueError(f"unknown recovery-transport export: {export}")
            outputs.append(output.cpu().numpy())
    codes = np.concatenate(outputs).astype(np.float32)
    if export == "target_prediction":
        codes = codes * np.asarray(
            checkpoint["target_scale"], dtype=np.float32
        ) + np.asarray(checkpoint["target_mean"], dtype=np.float32)
        dimensions = int(config["representation"]["target_dimensions"])
    else:
        dimensions = int(config["representation"]["output_dimensions"])
    if codes.shape != (len(values), dimensions) or not np.isfinite(codes).all():
        raise RuntimeError("invalid frozen recovery-transport codes")
    return codes


def apply_recovery_transport_sidecar_adapter(
    dynamics_embedding: np.ndarray, recovery_transport: np.ndarray
) -> np.ndarray:
    embedding = np.asarray(dynamics_embedding, dtype=np.float32)
    code = np.asarray(recovery_transport, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 146:
        raise ValueError("recovery transport requires frozen CARE130+dynamics16")
    if (
        code.ndim != 2
        or code.shape[0] != len(embedding)
        or code.shape[1] not in (17, 24)
    ):
        raise ValueError(
            "recovery transport requires 17 target or 24 latent coordinates"
        )
    output = np.concatenate([embedding, code], axis=1)
    if (
        output.shape != (len(embedding), 146 + code.shape[1])
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid CARE recovery-transport embedding")
    return output.astype(np.float32)


def apply_recovery_transport_residual_adapter(
    base_embedding: np.ndarray,
    physical_coordinates: np.ndarray,
    adapter: dict[str, np.ndarray],
) -> np.ndarray:
    """Append the frozen CARE-residual PCA recovery coordinates."""
    base = np.asarray(base_embedding, dtype=np.float64)
    coordinates = np.asarray(physical_coordinates, dtype=np.float64)
    coef = np.asarray(adapter["ridge_coef"], dtype=np.float64)
    intercept = np.asarray(adapter["ridge_intercept"], dtype=np.float64)
    center = np.asarray(adapter["residual_center"], dtype=np.float64)
    scale = np.asarray(adapter["residual_scale"], dtype=np.float64)
    basis = np.asarray(adapter["residual_basis"], dtype=np.float64)
    whitening = np.asarray(adapter["whitening_scale"], dtype=np.float64)
    components = basis.shape[1] if basis.ndim == 2 else -1
    if (
        base.ndim != 2
        or base.shape[1] != 146
        or coordinates.shape != (len(base), 17)
        or coef.shape != (17, 146)
        or intercept.shape != (17,)
        or center.shape != (17,)
        or scale.shape != (17,)
        or basis.shape != (17, components)
        or whitening.shape != (components,)
        or not 1 <= components < 17
        or np.any(scale <= 0)
        or np.any(whitening <= 0)
    ):
        raise ValueError("invalid recovery-transport residual adapter")
    predicted = np.einsum("nd,kd->nk", base, coef, optimize=False) + intercept
    standardized = (coordinates - predicted - center) / scale
    compact = (standardized @ basis) / whitening
    output = np.concatenate([base, compact], axis=1)
    if output.shape != (len(base), 146 + components) or not np.isfinite(output).all():
        raise RuntimeError("invalid residual recovery-transport embedding")
    return output.astype(np.float32)


def hierarchical_temporal_sidecar_codes(
    common_glucose: np.ndarray,
    common_observed: np.ndarray,
    checkpoint: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract the frozen cadence-observable hierarchical temporal code."""
    relative, support = regrid_relative_trace(common_glucose, common_observed)
    config = checkpoint["config"]
    representation = config["representation"]
    if bool(representation.get("glucofm_framework_or_components", True)):
        raise ValueError(
            "hierarchical temporal sidecar must be independent of GlucoFM components"
        )
    if int(support.sum(axis=1).min()) < int(
        config["canary_gates"]["minimum_15min_support"]
    ):
        raise ValueError("hierarchical temporal sidecar has insufficient support")
    model = HierarchicalTemporalContrastEncoder(
        hidden_dimensions=int(representation["hidden_dimensions"]),
        local_dimensions=int(representation["local_dimensions"]),
        output_dimensions=int(representation["output_dimensions"]),
        dilations=tuple(int(value) for value in representation["dilations"]),
        kernel_size=int(representation["kernel_size"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for left in range(0, len(relative), batch_size):
            right = left + batch_size
            result = model(
                torch.from_numpy(relative[left:right]).to(device),
                torch.from_numpy(support[left:right]).to(device),
            )
            outputs.append(result["code"].cpu().numpy())
    codes = np.concatenate(outputs).astype(np.float32)
    dimensions = int(representation["output_dimensions"])
    if codes.shape != (len(relative), dimensions) or not np.isfinite(codes).all():
        raise RuntimeError("invalid frozen hierarchical temporal sidecar codes")
    return codes


def apply_hierarchical_temporal_sidecar_adapter(
    base_embedding: np.ndarray,
    sidecar: np.ndarray,
    *,
    mode: str = "append",
) -> np.ndarray:
    """Append temporal32 or use it as the frozen CARE130 dynamics replacement."""
    embedding = np.asarray(base_embedding, dtype=np.float32)
    code = np.asarray(sidecar, dtype=np.float32)
    if code.shape != (len(embedding), 32):
        raise ValueError(
            "hierarchical temporal sidecar requires 32 learned coordinates"
        )
    expected_base = (
        146 if mode == "append" else 130 if mode == "replace_dynamics" else -1
    )
    if embedding.ndim != 2 or embedding.shape[1] != expected_base:
        raise ValueError(f"hierarchical temporal mode {mode} has invalid base")
    output = np.concatenate([embedding, code], axis=1)
    expected_output = expected_base + 32
    if (
        output.shape != (len(embedding), expected_output)
        or not np.isfinite(output).all()
    ):
        raise RuntimeError("invalid CARE hierarchical temporal embedding")
    return output.astype(np.float32)


def apply_analytic_spectrum_adapter(
    dynamics_embedding: np.ndarray,
    common_glucose: np.ndarray,
    common_observed: np.ndarray,
) -> np.ndarray:
    """Append the fixed full 48-bin cadence-observable relative spectrum."""
    embedding = np.asarray(dynamics_embedding, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 146:
        raise ValueError("analytic spectrum requires frozen CARE130+dynamics16")
    spectrum, support = regrid_relative_spectrum(common_glucose, common_observed)
    if int(support.sum(axis=1).min()) < 90:
        raise ValueError(
            "analytic spectrum requires at least 90 supported 15-minute points"
        )
    output = np.concatenate([embedding, spectrum], axis=1)
    if output.shape != (len(embedding), 194) or not np.isfinite(output).all():
        raise RuntimeError("invalid CARE analytic-spectrum embedding")
    return output.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--affine-gauge-adapter", type=Path)
    parser.add_argument("--moment-pool-adapter", type=Path)
    parser.add_argument("--multi-axis-sidecar", type=Path)
    parser.add_argument("--morphology-sidecar", type=Path)
    parser.add_argument("--hierarchical-temporal-sidecar", type=Path)
    parser.add_argument("--recovery-transport-sidecar", type=Path)
    parser.add_argument("--recovery-transport-residual-adapter", type=Path)
    parser.add_argument(
        "--recovery-transport-export",
        choices=("code", "target_prediction"),
        default="code",
    )
    parser.add_argument(
        "--hierarchical-temporal-mode",
        choices=("append", "replace_dynamics"),
        default="append",
    )
    parser.add_argument("--analytic-spectrum", action="store_true")
    parser.add_argument(
        "--tpa-export-branch", choices=("full", "bulk", "tail"), default="full"
    )
    parser.add_argument(
        "--multi-axis-branches",
        choices=(
            "state",
            "dynamics",
            "dynamics_readout",
            "dynamics_responsible",
            "both",
        ),
        default="both",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    accepted_variants = {
        "tpa_cgm_signed_nested_cadence_full",
        "range_null_generic_transfer",
        "range_null_cgm_signed_tail",
    }
    variant = checkpoint.get("variant")
    if variant not in accepted_variants:
        raise ValueError(
            "the fixed 14-task extractor accepts only frozen full-method checkpoints"
        )
    contract = checkpoint["tail_contract"]
    model = TPACGMAutoencoder(
        profile_width=2 * int(contract["lag"]) + 1,
        profile_input_dim=int(checkpoint.get("profile_input_dim", 128)),
        responsibility_partitioned=bool(
            checkpoint.get("responsibility_partitioned", False)
        ),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    input_decomposition_mode = checkpoint.get("input_decomposition_mode", "none")
    input_normalization = checkpoint.get(
        "input_normalization", "global_train_center_scale"
    )
    cadence_projector_mode = checkpoint.get(
        "cadence_projector_mode", "relative_factor3"
    )

    with np.load(args.npz, allow_pickle=False) as archive:
        observed = archive["observed_mask"].astype(bool)
        glucose = interpolate_windows(
            archive["glucose_mg_dl"].astype(np.float32), observed
        )
        native_cadence_min = (
            archive["native_cadence_min"].astype(np.int16)
            if "native_cadence_min" in archive.files
            else np.full(len(glucose), 5, dtype=np.int16)
        )
        passthrough = {
            key: archive[key]
            for key in archive.files
            if key.startswith("label_")
            or key
            in {
                "subject_id",
                "window_id",
                "sensor_code",
                "sensor_names",
                "split",
                "cohort",
                "native_cadence_min",
                "coverage",
            }
        }
    if input_decomposition_mode in (
        "exact_cadence_range_private",
        "physical_common_relative_private",
    ):
        if cadence_projector_mode == "relative_factor3":
            common_glucose, common_observed = deterministic_cadence_view(
                glucose,
                observed,
                factor=3,
                native_cadence_min=native_cadence_min,
            )
        elif cadence_projector_mode == "absolute_15min":
            common_glucose, common_observed = deterministic_absolute_cadence_view(
                glucose,
                observed,
                target_cadence_min=15,
                native_cadence_min=native_cadence_min,
            )
        else:
            raise ValueError(
                f"unknown cadence projector mode: {cadence_projector_mode}"
            )
    else:
        common_glucose = None
        common_observed = None
    thresholds = {
        "upper_q90": float(contract["upper_thresholds"][0]),
        "upper_q95": float(contract["upper_thresholds"][1]),
        "lower_q90": float(contract["lower_thresholds"][0]),
        "lower_q95": float(contract["lower_thresholds"][1]),
    }
    representations: list[np.ndarray] = []
    bulk_token_captures: list[np.ndarray] = []
    token_hook = None
    if args.moment_pool_adapter is not None:
        if not model.responsibility_partitioned:
            raise ValueError(
                "moment pool requires the frozen partitioned common branch"
            )
        token_hook = model.bulk_token_norm.register_forward_hook(
            lambda _module, _inputs, output: bulk_token_captures.append(
                output.detach().cpu().numpy()
            )
        )
    try:
        with torch.no_grad():
            for left in range(0, len(glucose), args.batch_size):
                raw = torch.from_numpy(glucose[left : left + args.batch_size]).to(
                    device
                )
                mask = torch.from_numpy(observed[left : left + args.batch_size]).to(
                    device
                )
                if input_decomposition_mode in (
                    "exact_cadence_range_private",
                    "physical_common_relative_private",
                ):
                    common_raw = torch.from_numpy(
                        common_glucose[left : left + args.batch_size]
                    ).to(device)
                    common_mask = torch.from_numpy(
                        common_observed[left : left + args.batch_size]
                    ).to(device)
                    normalized = masked_window_standardize(raw, common_mask)
                    common_relative = masked_window_standardize(common_raw, common_mask)
                    common_normalized = (
                        (common_raw - float(contract["center"]))
                        / float(contract["scale"])
                        if input_decomposition_mode
                        == "physical_common_relative_private"
                        else common_relative
                    )
                    common_arguments = {
                        "common_raw_glucose": common_raw,
                        "common_normalized_glucose": common_normalized,
                        "private_common_normalized_glucose": common_relative,
                        "common_observed": common_mask,
                        "common_visible": common_mask,
                    }
                elif input_normalization == "global_train_center_scale":
                    normalized = (raw - float(contract["center"])) / float(
                        contract["scale"]
                    )
                    common_arguments = {}
                elif input_normalization == "window_visible_mean_std":
                    normalized = masked_window_standardize(raw, mask)
                    common_arguments = {}
                else:
                    raise ValueError(
                        f"unknown input normalization: {input_normalization}"
                    )
                representation, _ = model.encode(
                    raw,
                    normalized,
                    mask,
                    mask,
                    variant=variant,
                    center=float(contract["center"]),
                    input_decomposition_mode=input_decomposition_mode,
                    **common_arguments,
                    **thresholds,
                )
                representations.append(representation.cpu().numpy())
    finally:
        if token_hook is not None:
            token_hook.remove()
    embedding = np.concatenate(representations).astype(np.float32)
    if embedding.shape != (len(glucose), 128) or not np.isfinite(embedding).all():
        raise RuntimeError("invalid TPA-CGM embeddings")
    embedding = select_tpa_export_branch(
        embedding,
        args.tpa_export_branch,
        responsibility_partitioned=model.responsibility_partitioned,
    )
    if args.tpa_export_branch != "full" and any(
        value is not None
        for value in (
            args.affine_gauge_adapter,
            args.moment_pool_adapter,
            args.multi_axis_sidecar,
            args.morphology_sidecar,
            args.hierarchical_temporal_sidecar,
            args.recovery_transport_sidecar,
        )
    ):
        raise ValueError("TPA block export cannot be combined with CARE adapters")
    if args.tpa_export_branch != "full" and args.analytic_spectrum:
        raise ValueError("TPA block export cannot be combined with analytic spectrum")
    adapter_sha256 = None
    affine_adapter_mode = None
    if args.affine_gauge_adapter is not None:
        if common_glucose is None or common_observed is None:
            raise ValueError("affine gauge adapter requires an exact common view")
        with np.load(args.affine_gauge_adapter, allow_pickle=False) as adapter:
            expected_checkpoint = str(adapter["checkpoint_sha256"].item())
            if expected_checkpoint != sha256(args.checkpoint):
                raise ValueError("affine gauge adapter checkpoint mismatch")
            gauge = affine_gauge_codes(
                common_glucose,
                common_observed,
                center=float(contract["center"]),
                scale=float(contract["scale"]),
            )
            adapter_mode = (
                str(adapter["adapter_mode"].item())
                if "adapter_mode" in adapter.files
                else "capacity_neutral_pca94"
            )
            affine_adapter_mode = adapter_mode
            if adapter_mode == "append_only":
                embedding = apply_affine_gauge_append_adapter(embedding, gauge)
            elif adapter_mode == "append_affine_physio":
                physio = physio_gauge_codes(common_glucose, common_observed)
                embedding = apply_affine_physio_gauge_adapter(embedding, gauge, physio)
            elif adapter_mode == "append_affine_compact_physio":
                physio = physio_gauge_codes(common_glucose, common_observed)
                compact = compact_physio_gauge_codes(
                    gauge,
                    physio,
                    adapter["ridge_coef"],
                    adapter["ridge_intercept"],
                    adapter["residual_center"],
                    adapter["residual_scale"],
                    adapter["compact_basis"],
                    adapter["whitening_scale"],
                )
                embedding = apply_affine_compact_physio_gauge_adapter(
                    embedding, gauge, compact
                )
            elif adapter_mode == "capacity_neutral_pca94":
                embedding = apply_affine_gauge_adapter(
                    embedding,
                    gauge,
                    adapter["common_center"],
                    adapter["common_basis"],
                )
            else:
                raise ValueError(f"unknown affine gauge adapter mode: {adapter_mode}")
        adapter_sha256 = sha256(args.affine_gauge_adapter)
    moment_adapter_sha256 = None
    if args.moment_pool_adapter is not None:
        if args.affine_gauge_adapter is None or embedding.shape[1] not in {130, 137}:
            raise ValueError("moment pool requires a frozen CARE gauge adapter first")
        if common_observed is None:
            raise ValueError("moment pool requires an exact canonical common mask")
        bulk_tokens = np.concatenate(bulk_token_captures)
        if bulk_tokens.shape != (len(glucose), glucose.shape[1], 96):
            raise RuntimeError("invalid frozen common-token capture")
        moment = first_temporal_moment_codes(bulk_tokens, common_observed)
        with np.load(args.moment_pool_adapter, allow_pickle=False) as adapter:
            adapter_mode = (
                str(adapter["adapter_mode"].item())
                if "adapter_mode" in adapter.files
                else "append_compact_moment"
            )
            if adapter_mode in {
                "append_compact_moment",
                "append_residual_compact_moment",
            }:
                base_moment = compact_moment_pool_codes(
                    moment,
                    adapter["moment_center"],
                    adapter["moment_scale"],
                    adapter["moment_basis"],
                    adapter["whitening_scale"],
                )
            if adapter_mode == "append_compact_moment":
                compact_moment = base_moment
            elif adapter_mode == "append_residual_compact_moment":
                compact_moment = residual_compact_moment_pool_codes(
                    base_moment,
                    embedding,
                    adapter["residual_ridge_coef"],
                    adapter["residual_ridge_intercept"],
                    adapter["residual_center"],
                    adapter["residual_scale"],
                    adapter["residual_basis"],
                    adapter["residual_whitening_scale"],
                )
            elif adapter_mode == "append_raw_residual_moment":
                compact_moment = residual_compact_moment_pool_codes(
                    moment,
                    embedding,
                    adapter["residual_ridge_coef"],
                    adapter["residual_ridge_intercept"],
                    adapter["residual_center"],
                    adapter["residual_scale"],
                    adapter["residual_basis"],
                    adapter["residual_whitening_scale"],
                )
            else:
                raise ValueError(f"unknown moment-pool adapter mode: {adapter_mode}")
        embedding = apply_moment_pool_adapter(embedding, compact_moment)
        moment_adapter_sha256 = sha256(args.moment_pool_adapter)
    multi_axis_sidecar_sha256 = None
    if args.multi_axis_sidecar is not None:
        allowed_base_dimensions = {
            "append_only": 130,
            "append_affine_compact_physio": 137,
        }
        expected_base_dimensions = allowed_base_dimensions.get(affine_adapter_mode)
        if (
            args.affine_gauge_adapter is None
            or expected_base_dimensions is None
            or embedding.shape[1] != expected_base_dimensions
        ):
            raise ValueError(
                "multi-axis sidecar requires the frozen affine CARE base or its "
                "frozen compact-physiology extension"
            )
        if common_glucose is None or common_observed is None:
            raise ValueError(
                "multi-axis sidecar requires an exact canonical common view"
            )
        sidecar_checkpoint = torch.load(
            args.multi_axis_sidecar, map_location="cpu", weights_only=False
        )
        base = sidecar_checkpoint["config"]["base"]
        if str(base["care_checkpoint_sha256"]) != sha256(args.checkpoint):
            raise ValueError("multi-axis sidecar CARE checkpoint mismatch")
        if str(base["affine_adapter_sha256"]) != sha256(args.affine_gauge_adapter):
            raise ValueError("multi-axis sidecar affine adapter mismatch")
        if int(base.get("base_dimensions", 130)) != embedding.shape[1]:
            raise ValueError("multi-axis sidecar base dimension mismatch")
        sidecar_values, sidecar_mask = common_glucose, common_observed
        if (
            sidecar_checkpoint["config"]["representation"].get("input_view")
            == "absolute15"
        ):
            sidecar_values, sidecar_mask = deterministic_absolute_cadence_view(
                glucose,
                observed,
                target_cadence_min=15,
                native_cadence_min=native_cadence_min,
            )
        sidecar = multi_axis_sidecar_codes(
            sidecar_values,
            sidecar_mask,
            sidecar_checkpoint,
            batch_size=args.batch_size,
            device=device,
            branches=args.multi_axis_branches,
        )
        embedding = apply_multi_axis_sidecar_adapter(embedding, sidecar)
        multi_axis_sidecar_sha256 = sha256(args.multi_axis_sidecar)
    if args.analytic_spectrum:
        if args.multi_axis_sidecar is None or args.multi_axis_branches != "dynamics":
            raise ValueError(
                "analytic spectrum requires the frozen dynamics-only sidecar"
            )
        if common_glucose is None or common_observed is None:
            raise ValueError(
                "analytic spectrum requires an exact canonical common view"
            )
        embedding = apply_analytic_spectrum_adapter(
            embedding, common_glucose, common_observed
        )
    morphology_sidecar_sha256 = None
    if args.morphology_sidecar is not None:
        if args.analytic_spectrum:
            raise ValueError("analytic spectrum and morphology cannot be combined")
        if args.multi_axis_sidecar is None or args.multi_axis_branches != "dynamics":
            raise ValueError(
                "morphology sidecar requires the frozen dynamics-only sidecar"
            )
        if common_glucose is None or common_observed is None:
            raise ValueError(
                "morphology sidecar requires an exact canonical common view"
            )
        morphology_checkpoint = torch.load(
            args.morphology_sidecar, map_location="cpu", weights_only=False
        )
        if (
            str(morphology_checkpoint["config"]["base"]["clean_reference"])
            != "care_cgm_multi_axis_dynamics_ablation_seed43_v1b"
        ):
            raise ValueError("morphology sidecar clean-base reference mismatch")
        morphology = morphology_sidecar_codes(
            common_glucose,
            common_observed,
            morphology_checkpoint,
            batch_size=args.batch_size,
            device=device,
        )
        embedding = apply_morphology_sidecar_adapter(embedding, morphology)
        morphology_sidecar_sha256 = sha256(args.morphology_sidecar)
    hierarchical_temporal_sidecar_sha256 = None
    if args.hierarchical_temporal_sidecar is not None:
        if args.analytic_spectrum or args.morphology_sidecar is not None:
            raise ValueError(
                "hierarchical temporal sidecar cannot be combined with closed sidecars"
            )
        if args.hierarchical_temporal_mode == "append":
            if (
                args.multi_axis_sidecar is None
                or args.multi_axis_branches != "dynamics"
            ):
                raise ValueError(
                    "hierarchical append requires the frozen dynamics-only sidecar"
                )
        elif args.multi_axis_sidecar is not None or embedding.shape[1] != 130:
            raise ValueError(
                "hierarchical replacement requires the exact CARE130 base without multi-axis"
            )
        if common_glucose is None or common_observed is None:
            raise ValueError(
                "hierarchical temporal sidecar requires an exact canonical common view"
            )
        temporal_checkpoint = torch.load(
            args.hierarchical_temporal_sidecar,
            map_location="cpu",
            weights_only=False,
        )
        if (
            str(temporal_checkpoint["config"]["reuse_assets"]["base_representation"])
            != "frozen CARE130 plus dynamics16"
        ):
            raise ValueError(
                "hierarchical temporal sidecar clean-base reference mismatch"
            )
        temporal = hierarchical_temporal_sidecar_codes(
            common_glucose,
            common_observed,
            temporal_checkpoint,
            batch_size=args.batch_size,
            device=device,
        )
        embedding = apply_hierarchical_temporal_sidecar_adapter(
            embedding, temporal, mode=args.hierarchical_temporal_mode
        )
        hierarchical_temporal_sidecar_sha256 = sha256(
            args.hierarchical_temporal_sidecar
        )
    recovery_transport_sidecar_sha256 = None
    if args.recovery_transport_sidecar is not None:
        if (
            any(
                value is not None
                for value in (
                    args.morphology_sidecar,
                    args.hierarchical_temporal_sidecar,
                )
            )
            or args.analytic_spectrum
        ):
            raise ValueError(
                "recovery transport cannot be combined with closed sidecars"
            )
        if args.multi_axis_sidecar is None or args.multi_axis_branches != "dynamics":
            raise ValueError(
                "recovery transport requires the frozen dynamics-only sidecar"
            )
        recovery_checkpoint = torch.load(
            args.recovery_transport_sidecar,
            map_location="cpu",
            weights_only=False,
        )
        recovery = recovery_transport_sidecar_codes(
            glucose,
            observed,
            recovery_checkpoint,
            batch_size=args.batch_size,
            device=device,
            export=args.recovery_transport_export,
        )
        if args.recovery_transport_residual_adapter is None:
            embedding = apply_recovery_transport_sidecar_adapter(embedding, recovery)
        else:
            if args.recovery_transport_export != "target_prediction":
                raise ValueError(
                    "recovery residual adapter requires target_prediction export"
                )
            with np.load(
                args.recovery_transport_residual_adapter, allow_pickle=False
            ) as adapter_archive:
                recovery_adapter = {
                    name: adapter_archive[name] for name in adapter_archive.files
                }
            expected_checkpoint = str(
                recovery_adapter["recovery_checkpoint_sha256"].item()
            )
            if expected_checkpoint != sha256(args.recovery_transport_sidecar):
                raise ValueError("recovery residual adapter checkpoint mismatch")
            embedding = apply_recovery_transport_residual_adapter(
                embedding, recovery, recovery_adapter
            )
        recovery_transport_sidecar_sha256 = sha256(args.recovery_transport_sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embedding=embedding, **passthrough)
    report = {
        "acceptance_passed": True,
        "artifact": str(args.output),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dimensions": int(embedding.shape[1]),
        "implementation": "TPA-CGM non-JEPA autoencoder",
        "variant": variant,
        "tpa_export_branch": args.tpa_export_branch,
        "input_decomposition_mode": input_decomposition_mode,
        "input_normalization": input_normalization,
        "cadence_projector_mode": cadence_projector_mode,
        "windows": int(len(embedding)),
        "representation_selection_label_reads": 0,
        "private_test_optimization_reads": 0,
        "affine_gauge_adapter": (
            None
            if args.affine_gauge_adapter is None
            else str(args.affine_gauge_adapter)
        ),
        "affine_gauge_adapter_sha256": adapter_sha256,
        "moment_pool_adapter": (
            None if args.moment_pool_adapter is None else str(args.moment_pool_adapter)
        ),
        "moment_pool_adapter_sha256": moment_adapter_sha256,
        "multi_axis_sidecar": (
            None if args.multi_axis_sidecar is None else str(args.multi_axis_sidecar)
        ),
        "multi_axis_sidecar_sha256": multi_axis_sidecar_sha256,
        "multi_axis_branches": (
            None if args.multi_axis_sidecar is None else args.multi_axis_branches
        ),
        "morphology_sidecar": (
            None if args.morphology_sidecar is None else str(args.morphology_sidecar)
        ),
        "morphology_sidecar_sha256": morphology_sidecar_sha256,
        "hierarchical_temporal_sidecar": (
            None
            if args.hierarchical_temporal_sidecar is None
            else str(args.hierarchical_temporal_sidecar)
        ),
        "hierarchical_temporal_sidecar_sha256": (hierarchical_temporal_sidecar_sha256),
        "hierarchical_temporal_mode": (
            None
            if args.hierarchical_temporal_sidecar is None
            else args.hierarchical_temporal_mode
        ),
        "analytic_spectrum": bool(args.analytic_spectrum),
        "recovery_transport_sidecar": (
            None
            if args.recovery_transport_sidecar is None
            else str(args.recovery_transport_sidecar)
        ),
        "recovery_transport_sidecar_sha256": recovery_transport_sidecar_sha256,
        "recovery_transport_export": (
            None
            if args.recovery_transport_sidecar is None
            else args.recovery_transport_export
        ),
        "recovery_transport_residual_adapter": (
            None
            if args.recovery_transport_residual_adapter is None
            else str(args.recovery_transport_residual_adapter)
        ),
        "recovery_transport_residual_adapter_sha256": (
            None
            if args.recovery_transport_residual_adapter is None
            else sha256(args.recovery_transport_residual_adapter)
        ),
        "glucofm_component_reads": 0,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
