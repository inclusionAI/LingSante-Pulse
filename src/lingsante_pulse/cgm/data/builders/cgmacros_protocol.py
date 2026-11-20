#!/usr/bin/env python3
"""Build paired-sensor CGMacros phenotype windows and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lingsante_pulse.cgm.data.protocol import (
    align_nonoverlapping_windows,
    cgmacros_subject_labels,
    recover_linearly_interpolated_native_readings,
)


SENSORS = {"dexcom": ("Dexcom GL", 5), "libre": ("Libre GL", 15)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = cgmacros_subject_labels(pd.read_csv(args.root / "bio.csv"))
    sensor_windows = []
    sensor_reports = {}
    for sensor, (column, cadence) in SENSORS.items():
        recovered_frames = []
        phase_audits = []
        for path in sorted(args.root.glob("CGMacros-*/*.csv")):
            subject_id = str(int(path.parent.name.removeprefix("CGMacros-")))
            raw = pd.read_csv(path, usecols=["Timestamp", column])
            recovered, audit = recover_linearly_interpolated_native_readings(
                raw["Timestamp"], raw[column], cadence
            )
            recovered["id"] = subject_id
            recovered_frames.append(recovered)
            phase_audits.append({"subject_id": subject_id, **audit})
        recovered_all = pd.concat(recovered_frames, ignore_index=True)
        windows = align_nonoverlapping_windows(
            recovered_all, subject_col="id", timestamp_col="time", value_col="gl"
        )
        sensor_windows.append((sensor, windows))
        sensor_reports[sensor] = {
            "cadence_minutes": cadence,
            "native_points_recovered": int(len(recovered_all)),
            "native_cadence_hours": float(len(recovered_all) * cadence / 60),
            "paper_reported_monitoring_hours": 10376 if sensor == "dexcom" else 10998,
            "windows": int(len(windows.glucose)),
            "subjects": int(np.unique(windows.subject_id).size),
            "coverage_min": float(windows.coverage.min()),
            "coverage_mean": float(windows.coverage.mean()),
            "phase_reconstruction_mae_max": float(
                max(row["reconstruction_mae_mg_dl"] for row in phase_audits)
            ),
            "phase_audits": phase_audits,
        }

    fields = {
        name: np.concatenate([getattr(windows, name) for _, windows in sensor_windows])
        for name in (
            "glucose",
            "observed_mask",
            "subject_id",
            "window_id",
            "start_time",
            "circadian_start",
            "coverage",
            "episode_index",
        )
    }
    sensor_code = np.concatenate(
        [
            np.full(len(windows.glucose), index, dtype=np.int8)
            for index, (_, windows) in enumerate(sensor_windows)
        ]
    )
    window_labels = labels.loc[fields["subject_id"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "cgmacros_glucofm_windows_v1.npz"
    np.savez_compressed(
        npz_path,
        glucose_mg_dl=fields["glucose"],
        observed_mask=fields["observed_mask"],
        subject_id=fields["subject_id"],
        window_id=fields["window_id"],
        start_time=fields["start_time"],
        circadian_start=fields["circadian_start"],
        coverage=fields["coverage"],
        episode_index=fields["episode_index"],
        sensor_code=sensor_code,
        sensor_names=np.asarray([name for name, _ in sensor_windows]),
        label_diabetes=window_labels["diabetes"].to_numpy(dtype=np.int8),
        label_insulin_resistance=window_labels["insulin_resistance"].to_numpy(
            dtype=np.int8
        ),
        label_obesity=window_labels["obesity"].to_numpy(dtype=np.int8),
        label_hyperlipidemia=window_labels["hyperlipidemia"].to_numpy(dtype=np.int8),
    )
    report = {
        "acceptance_passed": bool(
            len(labels) == 45
            and len(fields["glucose"]) > 0
            and all(value["subjects"] == 45 for value in sensor_reports.values())
        ),
        "artifact": str(npz_path),
        "cohort": "CGMacros",
        "protocol_version": "glucofm-public-window-v1",
        "subjects": int(len(labels)),
        "windows": int(len(fields["glucose"])),
        "sensor_reports": sensor_reports,
        "paper_processed_trace_available": False,
        "known_difference": (
            "Google used original raw Dexcom/Libre exports, which are absent from the PhysioNet "
            "package. Native-cadence anchors are deterministically recovered from the released "
            "exact 1-minute linear interpolation; reported hours differ slightly from Google. "
            "Standard A1c labels yield 15/16/14, while Table 8 reports 16/14/15 without an ID map."
        ),
        "diabetes_label_rule": "A1c <5.7 / 5.7<=A1c<6.5 / A1c>=6.5",
        "diabetes_label_exact_paper_reproduction": False,
        "paper_diabetes_label_counts": {"0": 16, "1": 14, "2": 15},
        "label_counts_by_subject": {
            task: {
                str(k): int(v)
                for k, v in labels[task].value_counts().sort_index().items()
            }
            for task in labels.columns
        },
    }
    report_path = args.output_dir / "cgmacros_glucofm_window_audit_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **report,
                "sensor_reports": {
                    k: {x: y for x, y in v.items() if x != "phase_audits"}
                    for k, v in sensor_reports.items()
                },
            },
            sort_keys=True,
        )
    )
    if not report["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
