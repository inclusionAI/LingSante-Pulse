#!/usr/bin/env python3
"""Build the exact 19-pretrain/37-downstream Stanford GlucoFM partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lingsante_pulse.cgm.data.protocol import (
    align_nonoverlapping_windows,
    stanford_subject_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cgm = pd.read_csv(args.root / "filtered_cgm_03222026.csv")
    labels = stanford_subject_labels(
        pd.read_csv(args.root / "filtered_metabolic_tests.csv"),
        pd.read_csv(args.root / "filtered_study_participants_characteristics.csv"),
        cgm["subject"].unique(),
    )
    cgm["id"] = cgm["subject"].astype(str).str.replace("43883-", "S", regex=False)
    downstream_raw = cgm[cgm["id"].isin(labels.index)].copy()
    downstream_raw["gl"] = pd.to_numeric(
        downstream_raw["glucose_value"], errors="coerce"
    )
    windows = align_nonoverlapping_windows(
        downstream_raw, subject_col="id", timestamp_col="timestamp", value_col="gl"
    )
    window_labels = labels.loc[windows.subject_id]
    all_normalized_ids = set(cgm["id"].astype(str))
    pretrain_ids = sorted(all_normalized_ids.difference(labels.index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "stanford_glucofm_windows_v1.npz"
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
        label_beta_cell_dysfunction=window_labels["beta_cell_dysfunction"].to_numpy(
            dtype=np.int8
        ),
        label_insulin_resistance=window_labels["insulin_resistance"].to_numpy(
            dtype=np.int8
        ),
    )
    report = {
        "acceptance_passed": bool(
            len(labels) == 37
            and len(pretrain_ids) == 19
            and set(windows.subject_id) == set(labels.index)
        ),
        "artifact": str(npz_path),
        "cohort": "Stanford",
        "protocol_version": "glucofm-public-window-v1",
        "downstream_subjects": int(len(labels)),
        "unlabeled_pretraining_subjects": int(len(pretrain_ids)),
        "unlabeled_pretraining_subject_ids": pretrain_ids,
        "id_normalization": "43883-NNN -> SNNN",
        "duplicate_clinical_row_rule": (
            "prefer venous_without_matching_cgm_and_without_planned_athome_cgm"
        ),
        "raw_downstream_points": int(len(downstream_raw)),
        "raw_downstream_valid_points": int(downstream_raw["gl"].notna().sum()),
        "raw_downstream_native_cadence_hours": float(
            downstream_raw["gl"].notna().sum() * 5 / 60
        ),
        "paper_reported_monitoring_hours": 27571,
        "windows": int(len(windows.glucose)),
        "coverage_min": float(windows.coverage.min()),
        "coverage_mean": float(windows.coverage.mean()),
        "label_counts_by_subject": {
            task: {
                str(k): int(v)
                for k, v in labels[task].value_counts().sort_index().items()
            }
            for task in labels.columns
        },
    }
    report_path = args.output_dir / "stanford_glucofm_window_audit_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
