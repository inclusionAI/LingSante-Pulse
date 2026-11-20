#!/usr/bin/env python3
"""Zero-training spectral complementarity screen for CARE-CGM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_rank(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    if variance.sum() <= 0:
        return 0.0
    probability = variance / variance.sum()
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _interpolate(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    positions = np.arange(values.size)
    valid = observed.astype(bool) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError("each spectral window needs two physical observations")
    return np.interp(positions, positions[valid], values[valid]).astype(np.float32)


def deterministic_absolute_cadence_view(
    glucose: np.ndarray,
    observed: np.ndarray,
    native_cadence_min: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy-only copy of the frozen exact 15-minute observation contract."""
    values = np.empty_like(glucose, dtype=np.float32)
    canonical_mask = np.zeros_like(observed, dtype=bool)
    for row in range(len(glucose)):
        source = observed[row].astype(bool) & np.isfinite(glucose[row])
        native = int(native_cadence_min[row])
        if native >= 15:
            canonical = source.copy()
        else:
            if 15 % native:
                raise ValueError("native cadence must divide 15 minutes")
            physical = np.flatnonzero(source)
            canonical = np.zeros(glucose.shape[1], dtype=bool)
            canonical[physical[:: 15 // native]] = True
        values[row] = _interpolate(glucose[row], canonical)
        canonical_mask[row] = canonical
    return values, canonical_mask


def regrid_relative_trace(
    common: np.ndarray, common_observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    common = np.asarray(common, dtype=np.float64)
    common_observed = np.asarray(common_observed, dtype=bool)
    if (
        common.ndim != 2
        or common.shape != common_observed.shape
        or common.shape[1] != 288
    ):
        raise ValueError("spectral audit expects matching [N,288] canonical views")
    grid = np.arange(0, 288, 3)
    values = common[:, grid]
    support = np.zeros_like(values, dtype=bool)
    for row, mask in enumerate(common_observed):
        positions = np.flatnonzero(mask)
        if len(positions) < 2:
            raise ValueError("spectral audit requires two canonical observations")
        support[row] = (grid >= positions[0]) & (grid <= positions[-1])
    count = support.sum(axis=1, keepdims=True)
    mean = (values * support).sum(axis=1, keepdims=True) / count
    variance = (np.square(values - mean) * support).sum(axis=1, keepdims=True) / count
    relative = (values - mean) / np.sqrt(variance + 1e-6)
    relative = np.where(support, relative, 0.0)
    return relative.astype(np.float32), support


def regrid_relative_spectrum(
    common: np.ndarray, common_observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    relative, support = regrid_relative_trace(common, common_observed)
    amplitude = np.abs(np.fft.rfft(relative, axis=1))[:, 1:]
    features = np.log1p(amplitude)
    if features.shape[1] != 48 or not np.isfinite(features).all():
        raise RuntimeError("invalid 48-bin relative spectrum")
    return features.astype(np.float32), support


def fit_audit(
    base: np.ndarray,
    spectrum: np.ndarray,
    split: np.ndarray,
    *,
    ridge_alpha: float,
) -> dict:
    base = np.asarray(base, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    split = np.asarray(split, dtype=np.int64)
    if base.ndim != 2 or base.shape[1] != 146:
        raise ValueError("base must be frozen CARE130+dynamics16")
    if spectrum.shape != (len(base), 48) or split.shape != (len(base),):
        raise ValueError("spectrum or split shape mismatch")
    train = split == 0
    validation = split == 1
    if not train.any() or not validation.any() or set(np.unique(split)) - {0, 1}:
        raise ValueError("spectral audit requires public split0/1")
    base_mean = base[train].mean(axis=0)
    base_scale = np.maximum(base[train].std(axis=0), 1e-6)
    spectrum_mean = spectrum[train].mean(axis=0)
    spectrum_scale = np.maximum(spectrum[train].std(axis=0), 1e-6)
    x = (base - base_mean) / base_scale
    y = (spectrum - spectrum_mean) / spectrum_scale
    with np.errstate(all="ignore"):
        gram = x[train].T @ x[train]
        coefficient = np.linalg.solve(
            gram + ridge_alpha * np.eye(gram.shape[0]), x[train].T @ y[train]
        )
        residual = y - x @ coefficient
    if not np.isfinite(coefficient).all() or not np.isfinite(residual).all():
        raise RuntimeError("non-finite spectral complementarity solution")
    target_variance = np.var(y[validation], axis=0)
    residual_variance = np.var(residual[validation], axis=0)
    explained = 1.0 - residual_variance / np.maximum(target_variance, 1e-12)
    return {
        "train_windows": int(train.sum()),
        "validation_windows": int(validation.sum()),
        "spectral_effective_rank": effective_rank(y[validation]),
        "residual_effective_rank": effective_rank(residual[validation]),
        "validation_predictable_explained_variance_mean": float(explained.mean()),
        "validation_predictable_explained_variance_per_bin": explained.tolist(),
        "validation_residual_energy_fraction": float(
            residual_variance.sum() / max(target_variance.sum(), 1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with np.load(args.npz, allow_pickle=False) as archive:
        glucose = archive["glucose_mg_dl"]
        observed = archive["observed_mask"]
        cadence = archive["native_cadence_min"]
        split = archive["split"]
        window = archive["window_id"]
    with np.load(args.base, allow_pickle=False) as archive:
        base = archive["embedding"]
        base_split = archive["split"]
        base_window = archive["window_id"]
    if not np.array_equal(split, base_split) or not np.array_equal(window, base_window):
        raise ValueError("public corpus and base rows are not exactly aligned")
    common, common_observed = deterministic_absolute_cadence_view(
        glucose, observed, cadence
    )
    spectrum, support = regrid_relative_spectrum(common, common_observed)
    report = fit_audit(
        base,
        spectrum,
        split,
        ridge_alpha=float(config["ridge_alpha"]),
    )
    gates = config["gates"]
    gate_results = {
        "base_hash": sha256(args.base) == str(gates["base_embedding_sha256"]),
        "support": int(support.sum(axis=1).min())
        >= int(gates["minimum_15min_support"]),
        "residual_energy": report["validation_residual_energy_fraction"]
        >= float(gates["validation_residual_energy_fraction_min"]),
        "residual_rank": report["residual_effective_rank"]
        >= float(gates["validation_residual_effective_rank_min"]),
        "isolation": all(
            int(gates[name]) == 0
            for name in (
                "phenotype_label_reads",
                "downstream_identity_reads",
                "private_test_optimization_reads",
                "private_source_optimization_reads",
                "glucofm_component_reads",
            )
        ),
    }
    passed = all(gate_results.values())
    result = {
        "schema_version": 1,
        "acceptance_passed": passed,
        "config_sha256": sha256(args.config),
        "corpus_sha256": sha256(args.npz),
        "base_embedding_sha256": sha256(args.base),
        "minimum_15min_support": int(support.sum(axis=1).min()),
        **report,
        "gate_results": gate_results,
        "isolation": {
            "phenotype_label_reads": 0,
            "downstream_identity_reads": 0,
            "private_test_optimization_reads": 0,
            "private_source_optimization_reads": 0,
            "glucofm_component_reads": 0,
        },
        "next_action": config["if_pass"] if passed else config["if_fail"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
