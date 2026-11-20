#!/usr/bin/env python3
"""Build the paper-aligned Hall phenotype windows, labels, and audit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lingsante_pulse.cgm.data.protocol import (
    align_nonoverlapping_windows,
    hall_subject_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    labels = hall_subject_labels(raw)
    eligible = raw[raw["id"].astype(str).isin(labels.index.astype(str))]
    windows = align_nonoverlapping_windows(
        eligible,
        subject_col="id",
        timestamp_col="time",
        value_col="gl",
    )
    window_labels = labels.loc[windows.subject_id]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "hall_glucofm_windows_v1.npz"
    np.savez_compressed(
        npz_path,
        glucose_mg_dl=windows.glucose,
        observed_mask=windows.observed_mask,
        subject_id=windows.subject_id,
        window_id=windows.window_id,
        start_time=windows.start_time,
        circadian_start=windows.circadian_start,
        coverage=windows.coverage,
        episode_index=windows.episode_index,
        label_diabetes=window_labels["diabetes"].to_numpy(dtype=np.int8),
        label_glucotype=window_labels["glucotype"].to_numpy(dtype=np.int8),
        label_insulin_resistance=window_labels["insulin_resistance"].to_numpy(
            dtype=np.int8
        ),
        label_hyperlipidemia=window_labels["hyperlipidemia"].to_numpy(dtype=np.int8),
    )
    subject_counts = {
        task: {
            str(label): int(count)
            for label, count in labels[task].value_counts().sort_index().items()
        }
        for task in labels.columns
    }
    report = {
        "acceptance_passed": bool(
            len(labels) == 56
            and len(windows.glucose) == 250
            and np.all(windows.observed_mask.sum(axis=1) > 0)
            and set(windows.subject_id) == set(labels.index.astype(str))
        ),
        "artifact": str(npz_path),
        "cohort": "Hall",
        "protocol_version": "glucofm-public-window-v1",
        "subjects": int(len(labels)),
        "windows": int(len(windows.glucose)),
        "window_hours": int(len(windows.glucose) * 24),
        "physically_observed_hours_in_windows": float(
            windows.observed_mask.sum() * 5 / 60
        ),
        "coverage_min": float(windows.coverage.min()),
        "coverage_mean": float(windows.coverage.mean()),
        "episodes_with_windows": int(np.unique(windows.episode_index).size),
        "label_counts_by_subject": subject_counts,
        "paper_reported_monitoring_hours": 7090,
        "paper_processed_trace_available": False,
        "known_difference": (
            "The public GlucoBench Hall table yields the exact 56 subjects and label counts, "
            "but Google's processed 7,090-hour trace and fold assignments were not released."
        ),
        "window_rule": (
            "split gaps >60min; downstream non-overlapping 24h time-span windows; "
            "5min floor bins; average collisions; preserve physical mask; no invented coverage cutoff"
        ),
    }
    report_path = args.output_dir / "hall_glucofm_window_audit_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
