#!/usr/bin/env python3
"""Build a causal multi-day embedding from frozen daily CGM embeddings.

The transform is deliberately label-free.  For every 24-hour window it pools
only the current and preceding contiguous windows from the same subject and
episode.  It exports either the pooled long-term code itself or the daily code
plus the long-minus-daily residual, which lets one shared linear probe ignore
long context when it is not useful without selecting a representation by task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ONE_DAY_MINUTES = 24 * 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_public_metadata(
    public: dict[str, np.ndarray], artifact: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    public_ids = public["window_id"].astype(str)
    artifact_ids = artifact["window_id"].astype(str)
    if len(set(public_ids)) != len(public_ids) or len(set(artifact_ids)) != len(
        artifact_ids
    ):
        raise ValueError("window ids must be unique")
    if len(public_ids) != len(artifact_ids) or set(public_ids) != set(artifact_ids):
        raise ValueError("public and embedding window-id sets differ")
    public_index = {window_id: row for row, window_id in enumerate(public_ids)}
    order = np.asarray(
        [public_index[window_id] for window_id in artifact_ids], dtype=np.int64
    )
    aligned = {
        key: public[key][order]
        for key in ("subject_id", "start_time", "episode_index", "observed_mask")
    }
    if "sensor_code" in public:
        aligned["sensor_code"] = public["sensor_code"][order]
    if "subject_id" in artifact and not np.array_equal(
        artifact["subject_id"].astype(str), aligned["subject_id"].astype(str)
    ):
        raise ValueError("subject ids disagree after exact window-id alignment")
    return aligned


def causal_long_term_mean(
    embedding: np.ndarray,
    subject_id: np.ndarray,
    start_time: np.ndarray,
    episode_index: np.ndarray,
    coverage: np.ndarray,
    *,
    max_days: int,
    sensor_code: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return support-weighted trailing means with no future/cross-episode reads."""
    values = np.asarray(embedding, dtype=np.float64)
    subjects = np.asarray(subject_id).astype(str)
    starts = np.asarray(start_time).astype("datetime64[m]")
    episodes = np.asarray(episode_index)
    support = np.asarray(coverage, dtype=np.float64)
    sensors = (
        np.asarray(sensor_code).astype(str)
        if sensor_code is not None
        else np.full(len(values), "none", dtype="U4")
    )
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("embedding must be a non-empty matrix")
    if not all(
        len(array) == len(values)
        for array in (subjects, starts, episodes, support, sensors)
    ):
        raise ValueError("metadata arrays must match embedding rows")
    if max_days < 1 or not np.isfinite(values).all():
        raise ValueError("invalid max_days or non-finite embedding")
    if np.any(~np.isfinite(support)) or np.any(support <= 0):
        raise ValueError("every window must have positive finite observation support")

    pooled = np.empty_like(values)
    history_days = np.zeros(len(values), dtype=np.int16)
    history_support = np.zeros(len(values), dtype=np.float32)
    keys = np.asarray(
        [
            f"{subject}\x1f{episode}\x1f{sensor}"
            for subject, episode, sensor in zip(subjects, episodes, sensors)
        ]
    )
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        indices = indices[np.argsort(starts[indices], kind="stable")]
        segment: list[int] = []
        previous: np.datetime64 | None = None
        for index in indices:
            if (
                previous is None
                or int((starts[index] - previous).astype("timedelta64[m]").astype(int))
                != ONE_DAY_MINUTES
            ):
                segment = []
            segment.append(int(index))
            segment = segment[-max_days:]
            chosen = np.asarray(segment, dtype=np.int64)
            weights = support[chosen]
            pooled[index] = np.average(values[chosen], axis=0, weights=weights)
            history_days[index] = len(chosen)
            history_support[index] = float(weights.sum())
            previous = starts[index]
    if not np.isfinite(pooled).all() or np.any(history_days < 1):
        raise RuntimeError("invalid causal long-term embedding")
    return pooled.astype(np.float32), history_days, history_support


def build(
    public_path: Path,
    embedding_path: Path,
    output_path: Path,
    *,
    max_days: int,
    mode: str,
) -> dict[str, object]:
    with np.load(public_path, allow_pickle=False) as archive:
        public = {key: archive[key] for key in archive.files}
    with np.load(embedding_path, allow_pickle=False) as archive:
        artifact = {key: archive[key] for key in archive.files}
    daily = np.asarray(artifact["embedding"], dtype=np.float32)
    metadata = _aligned_public_metadata(public, artifact)
    coverage = metadata["observed_mask"].astype(bool).mean(axis=1)
    long_term, history_days, history_support = causal_long_term_mean(
        daily,
        metadata["subject_id"],
        metadata["start_time"],
        metadata["episode_index"],
        coverage,
        max_days=max_days,
        sensor_code=metadata.get("sensor_code"),
    )
    if mode == "long_only":
        output = long_term
    elif mode == "daily_plus_long_residual":
        output = np.concatenate([daily, long_term - daily], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if not np.isfinite(output).all():
        raise RuntimeError("output contains non-finite values")

    passthrough = {key: value for key, value in artifact.items() if key != "embedding"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding=output,
        history_days=history_days,
        history_support=history_support,
        **passthrough,
    )
    report = {
        "acceptance_passed": True,
        "schema_version": 1,
        "mode": mode,
        "max_days": max_days,
        "aggregation": "strictly causal same-subject same-episode support-weighted mean",
        "daily_dimensions": int(daily.shape[1]),
        "output_dimensions": int(output.shape[1]),
        "windows": int(len(output)),
        "history_days_distribution": {
            str(day): int(np.sum(history_days == day))
            for day in np.unique(history_days)
        },
        "windows_with_prior_context": int(np.sum(history_days > 1)),
        "labels_used_by_transform": False,
        "future_windows_read": 0,
        "cross_subject_reads": 0,
        "cross_episode_reads": 0,
        "public_sha256": _sha256(public_path),
        "daily_embedding_sha256": _sha256(embedding_path),
        "output": str(output_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-days", type=int, default=7)
    parser.add_argument(
        "--mode",
        choices=("long_only", "daily_plus_long_residual"),
        required=True,
    )
    args = parser.parse_args()
    report = build(
        args.public,
        args.embedding,
        args.output,
        max_days=args.max_days,
        mode=args.mode,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
