#!/usr/bin/env python3
"""Build the exact 65-entry ShanghaiT2DM GlucoFM phenotype cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lingsante_pulse.cgm.data.protocol import (
    align_nonoverlapping_windows,
    shanghai_t2dm_session_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = shanghai_t2dm_session_labels(
        pd.read_excel(args.root / "Shanghai_T2DM_Summary.xlsx")
    )
    recordings = []
    for session_id in labels.index:
        matches = list((args.root / "Shanghai_T2DM").glob(f"{session_id}.xls*"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one CGM file for {session_id}, found {len(matches)}"
            )
        frame = pd.read_excel(matches[0], usecols=["Date", "CGM (mg / dl)"])
        frame.rename(columns={"Date": "time", "CGM (mg / dl)": "gl"}, inplace=True)
        frame["id"] = session_id
        recordings.append(frame)
    raw = pd.concat(recordings, ignore_index=True)
    windows = align_nonoverlapping_windows(
        raw, subject_col="id", timestamp_col="time", value_col="gl"
    )
    window_labels = labels.loc[windows.subject_id]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "shanghai_t2dm_glucofm_windows_v1.npz"
    np.savez_compressed(
        npz_path,
        glucose_mg_dl=windows.glucose,
        observed_mask=windows.observed_mask,
        subject_id=windows.subject_id,
        biological_subject_id=np.asarray(
            [value.split("_")[0] for value in windows.subject_id]
        ),
        window_id=windows.window_id,
        start_time=windows.start_time,
        circadian_start=windows.circadian_start,
        coverage=windows.coverage,
        episode_index=windows.episode_index,
        label_hypoglycemia=window_labels["hypoglycemia"].to_numpy(dtype=np.int8),
        label_insulin_resistance=window_labels["insulin_resistance"].to_numpy(
            dtype=np.int8
        ),
        label_hyperlipidemia=window_labels["hyperlipidemia"].to_numpy(dtype=np.int8),
    )
    report = {
        "acceptance_passed": bool(
            len(labels) == 65
            and labels.index.to_series().str.split("_").str[0].nunique() == 58
            and set(windows.subject_id) == set(labels.index)
        ),
        "artifact": str(npz_path),
        "cohort": "ShanghaiT2DM",
        "protocol_version": "glucofm-public-window-v1",
        "subject_entries": int(len(labels)),
        "biological_subjects": int(
            labels.index.to_series().str.split("_").str[0].nunique()
        ),
        "raw_native_points": int(len(raw)),
        "raw_native_cadence_hours": float(len(raw) * 15 / 60),
        "paper_reported_monitoring_hours": 15634,
        "windows": int(len(windows.glucose)),
        "coverage_min": float(windows.coverage.min()),
        "coverage_mean": float(windows.coverage.mean()),
        "grouping_unit": "recording visit / subject-entry, matching the paper",
        "label_counts_by_subject_entry": {
            task: {
                str(k): int(v)
                for k, v in labels[task].value_counts().sort_index().items()
            }
            for task in labels.columns
        },
    }
    report_path = args.output_dir / "shanghai_t2dm_glucofm_window_audit_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
