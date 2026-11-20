#!/usr/bin/env python3
"""Combine a robust causal CARE estimate with reliability-gated innovation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lingsante_pulse.cgm.data.long_term_embedding import (
    _aligned_public_metadata,
    _sha256,
    causal_long_term_mean,
)


def _aligned_embedding(
    reference_ids: np.ndarray, artifact: dict[str, np.ndarray]
) -> np.ndarray:
    reference = np.asarray(reference_ids).astype(str)
    artifact_ids = np.asarray(artifact["window_id"]).astype(str)
    if len(set(reference)) != len(reference) or len(set(artifact_ids)) != len(
        artifact_ids
    ):
        raise ValueError("window ids must be unique")
    if len(reference) != len(artifact_ids) or set(reference) != set(artifact_ids):
        raise ValueError("daily and innovation window-id sets differ")
    lookup = {window_id: row for row, window_id in enumerate(artifact_ids)}
    order = np.asarray([lookup[window_id] for window_id in reference], dtype=np.int64)
    return np.asarray(artifact["embedding"], dtype=np.float32)[order]


def reliability_weight(
    history_days: np.ndarray,
    history_support: np.ndarray,
    *,
    max_days: int,
) -> np.ndarray:
    """Fixed label-free confidence for a learned ordered-history correction."""
    days = np.asarray(history_days, dtype=np.float64)
    support = np.asarray(history_support, dtype=np.float64)
    if max_days < 2 or days.shape != support.shape:
        raise ValueError("invalid reliability inputs")
    if np.any(days < 1) or np.any(support <= 0) or np.any(days > max_days):
        raise ValueError("history support is outside the declared contract")
    prior_fraction = np.clip((days - 1.0) / (max_days - 1.0), 0.0, 1.0)
    mean_coverage = np.clip(support / days, 0.0, 1.0)
    return np.sqrt(prior_fraction * mean_coverage).astype(np.float32)


def build(
    public_path: Path,
    daily_path: Path,
    innovation_path: Path,
    output_path: Path,
    *,
    max_days: int,
) -> dict[str, object]:
    with np.load(public_path, allow_pickle=False) as archive:
        public = {name: archive[name] for name in archive.files}
    with np.load(daily_path, allow_pickle=False) as archive:
        daily_artifact = {name: archive[name] for name in archive.files}
    with np.load(innovation_path, allow_pickle=False) as archive:
        learned_artifact = {name: archive[name] for name in archive.files}

    daily = np.asarray(daily_artifact["embedding"], dtype=np.float32)
    learned = _aligned_embedding(daily_artifact["window_id"], learned_artifact)
    if (
        daily.shape != learned.shape
        or not np.isfinite(daily).all()
        or not np.isfinite(learned).all()
    ):
        raise ValueError("daily and learned embeddings must be aligned finite matrices")
    metadata = _aligned_public_metadata(public, daily_artifact)
    coverage = metadata["observed_mask"].astype(bool).mean(axis=1)
    stable, history_days, history_support = causal_long_term_mean(
        daily,
        metadata["subject_id"],
        metadata["start_time"],
        metadata["episode_index"],
        coverage,
        max_days=max_days,
        sensor_code=metadata.get("sensor_code"),
    )
    reliability = reliability_weight(history_days, history_support, max_days=max_days)
    innovation = learned - daily
    output = stable + reliability[:, None] * innovation
    no_history = history_days == 1
    output[no_history] = daily[no_history]
    if not np.isfinite(output).all() or not np.array_equal(
        output[no_history], daily[no_history]
    ):
        raise RuntimeError("reliability-responsible identity contract failed")

    passthrough = {
        name: value for name, value in daily_artifact.items() if name != "embedding"
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding=output.astype(np.float32),
        history_days=history_days,
        history_support=history_support,
        innovation_reliability=reliability,
        **passthrough,
    )
    return {
        "acceptance_passed": True,
        "schema_version": 1,
        "method": "causal support-weighted CARE estimate plus reliability-gated ordered typed innovation",
        "max_days": max_days,
        "output_dimensions": int(output.shape[1]),
        "windows": int(len(output)),
        "windows_with_prior_context": int(np.sum(history_days > 1)),
        "reliability_mean": float(reliability.mean()),
        "reliability_max": float(reliability.max()),
        "no_history_identity_max_abs": float(
            np.max(np.abs(output[no_history] - daily[no_history]))
        ),
        "phenotype_or_task_routing_reads": 0,
        "future_or_cross_provenance_reads": 0,
        "daily_embedding_sha256": _sha256(daily_path),
        "learned_embedding_sha256": _sha256(innovation_path),
        "public_sha256": _sha256(public_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--innovation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=7)
    args = parser.parse_args()
    report = build(
        args.public,
        args.daily,
        args.innovation,
        args.output,
        max_days=args.max_days,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
