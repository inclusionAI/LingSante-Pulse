#!/usr/bin/env python3
"""Build the auditable public GlucoFM pretraining corpus and optional PRIVATE mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lingsante_pulse.cgm.data.protocol import (
    shanghai_t2dm_session_labels,
    stanford_subject_labels,
)


GRID_MINUTES = 5
WINDOW_POINTS = 288
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1


@dataclass(frozen=True)
class CorpusPart:
    cohort: str
    native_cadence_min: int
    frame: pd.DataFrame


def stable_validation_subjects(subjects: set[str], fraction: float = 0.2) -> set[str]:
    if len(subjects) < 2:
        raise ValueError("each cohort needs at least two subjects")
    count = min(len(subjects) - 1, max(1, round(len(subjects) * fraction)))
    ranked = sorted(
        subjects,
        key=lambda value: hashlib.sha256(
            f"glucofm-public-validation-v1|{value}".encode()
        ).digest(),
    )
    return set(ranked[:count])


def _window_id(subject: str, episode: int, start_bin: int, start: pd.Timestamp) -> str:
    value = f"glucofm-public-pretrain-v1|{subject}|{episode}|{start_bin}|{start.isoformat()}"
    return hashlib.sha256(value.encode()).hexdigest()


def _interpolate(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    observed = np.flatnonzero(mask)
    if not len(observed):
        raise ValueError("cannot interpolate a window without observations")
    return np.interp(np.arange(len(values)), observed, values[mask]).astype(np.float32)


def extract_random_overlapping_windows(
    parts: list[CorpusPart], *, seed: int
) -> dict[str, np.ndarray]:
    """Split observed traces at >1h gaps and sample deterministic overlapping days."""
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    subjects: list[str] = []
    cohorts: list[str] = []
    cadences: list[int] = []
    starts: list[np.datetime64] = []
    coverages: list[float] = []
    window_ids: list[str] = []
    split_codes: list[int] = []

    for part in parts:
        frame = part.frame[["subject_id", "timestamp", "glucose_mg_dl"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frame["glucose_mg_dl"] = pd.to_numeric(frame["glucose_mg_dl"], errors="coerce")
        frame = frame.dropna(subset=["subject_id", "timestamp", "glucose_mg_dl"])
        frame = frame.sort_values(["subject_id", "timestamp"], kind="stable")
        cohort_subjects = set(frame["subject_id"].astype(str))
        validation = stable_validation_subjects(cohort_subjects)
        episode_index = 0
        for subject_id, subject in frame.groupby("subject_id", sort=True):
            gap = subject["timestamp"].diff().dt.total_seconds().div(60)
            local_episode = (gap.isna() | gap.gt(60)).cumsum()
            for _, episode in subject.groupby(local_episode, sort=True):
                episode_seed = int.from_bytes(
                    hashlib.sha256(
                        f"glucofm-public-overlap-v1|{seed}|{part.cohort}|{subject_id}|{episode_index}".encode()
                    ).digest()[:8],
                    "big",
                )
                episode_rng = np.random.default_rng(episode_seed)
                first = episode["timestamp"].iloc[0]
                elapsed = (
                    (episode["timestamp"] - first)
                    .dt.total_seconds()
                    .div(GRID_MINUTES * 60)
                )
                bins = np.floor(elapsed).astype(np.int64)
                binned = (
                    pd.DataFrame(
                        {"bin": bins, "value": episode["glucose_mg_dl"].to_numpy()}
                    )
                    .groupby("bin", sort=True)["value"]
                    .mean()
                )
                span = int(binned.index.max()) + 1
                left = 0
                while left + WINDOW_POINTS <= span:
                    chosen = binned[
                        (binned.index >= left) & (binned.index < left + WINDOW_POINTS)
                    ]
                    aligned = np.full(WINDOW_POINTS, np.nan, dtype=np.float32)
                    positions = chosen.index.to_numpy(dtype=np.int64) - left
                    aligned[positions] = chosen.to_numpy(dtype=np.float32)
                    mask = np.isfinite(aligned)
                    start = first + pd.Timedelta(minutes=left * GRID_MINUTES)
                    full_subject_id = f"{part.cohort}:{subject_id}"
                    values.append(_interpolate(aligned, mask))
                    masks.append(mask.astype(np.uint8))
                    subjects.append(full_subject_id)
                    cohorts.append(part.cohort)
                    cadences.append(part.native_cadence_min)
                    starts.append(np.datetime64(start))
                    coverages.append(float(mask.mean()))
                    window_ids.append(
                        _window_id(full_subject_id, episode_index, left, start)
                    )
                    split_codes.append(
                        SPLIT_VALIDATION
                        if str(subject_id) in validation
                        else SPLIT_TRAIN
                    )
                    # Advancing 20%-80% of a day gives 20%-80% overlap.
                    left += int(
                        episode_rng.integers(
                            math.ceil(0.2 * WINDOW_POINTS),
                            math.floor(0.8 * WINDOW_POINTS) + 1,
                        )
                    )
                episode_index += 1

    if not values:
        raise ValueError("no public pretraining windows were recovered")
    return {
        "glucose_mg_dl": np.stack(values),
        "observed_mask": np.stack(masks),
        "subject_id": np.asarray(subjects),
        "cohort": np.asarray(cohorts),
        "native_cadence_min": np.asarray(cadences, dtype=np.int16),
        "shifted_window_start": np.asarray(starts, dtype="datetime64[ns]"),
        "coverage": np.asarray(coverages, dtype=np.float32),
        "window_id": np.asarray(window_ids),
        "split": np.asarray(split_codes, dtype=np.int8),
    }


def _load_shanghai(root: Path, policy: str) -> tuple[CorpusPart, dict[str, object]]:
    all_paths = sorted((root / "Shanghai_T2DM").glob("*.xls*"))
    all_visits = {path.name.rsplit(".", 1)[0] for path in all_paths}
    labels = shanghai_t2dm_session_labels(
        pd.read_excel(root / "Shanghai_T2DM_Summary.xlsx")
    )
    downstream_visits = set(labels.index.astype(str))
    paper_pretrain_visits = all_visits.difference(downstream_visits)
    downstream_people = {value.split("_")[0] for value in downstream_visits}
    overlapping_visits = {
        value
        for value in paper_pretrain_visits
        if value.split("_")[0] in downstream_people
    }
    selected = set(paper_pretrain_visits)
    if policy == "subject_disjoint":
        selected.difference_update(overlapping_visits)

    recordings = []
    for path in all_paths:
        visit = path.name.rsplit(".", 1)[0]
        if visit not in selected:
            continue
        frame = pd.read_excel(path)
        cgm_columns = [
            column for column in ("CGM (mg / dl)", "CGM ") if column in frame
        ]
        if len(cgm_columns) != 1 or "Date" not in frame:
            raise ValueError(
                f"unrecognized Shanghai CGM columns for {visit}: {frame.columns.tolist()}"
            )
        frame = frame[["Date", cgm_columns[0]]].rename(
            columns={"Date": "timestamp", cgm_columns[0]: "glucose_mg_dl"}
        )
        frame["subject_id"] = visit
        recordings.append(frame)
    audit = {
        "raw_subject_entries": len(all_visits),
        "paper_pretrain_subject_entries": len(paper_pretrain_visits),
        "downstream_subject_entries": len(downstream_visits),
        "visit_level_overlap": sorted(
            paper_pretrain_visits.intersection(downstream_visits)
        ),
        "biological_overlap_pretrain_visits": sorted(overlapping_visits),
        "selected_policy": policy,
        "selected_pretrain_subject_entries": len(selected),
        "selected_pretrain_subject_ids": sorted(selected),
        "selected_downstream_biological_overlap": sorted(
            {value.split("_")[0] for value in selected}.intersection(downstream_people)
        ),
    }
    return CorpusPart(
        "shanghai_t2dm", 15, pd.concat(recordings, ignore_index=True)
    ), audit


def _load_stanford(root: Path) -> tuple[CorpusPart, dict[str, object]]:
    cgm = pd.read_csv(root / "filtered_cgm_03222026.csv")
    labels = stanford_subject_labels(
        pd.read_csv(root / "filtered_metabolic_tests.csv"),
        pd.read_csv(root / "filtered_study_participants_characteristics.csv"),
        cgm["subject"].unique(),
    )
    cgm["subject_id"] = (
        cgm["subject"].astype(str).str.replace("43883-", "S", regex=False)
    )
    all_subjects = set(cgm["subject_id"].astype(str))
    downstream = set(labels.index.astype(str))
    pretrain = all_subjects.difference(downstream)
    selected = cgm[cgm["subject_id"].isin(pretrain)].rename(
        columns={"timestamp": "timestamp", "glucose_value": "glucose_mg_dl"}
    )
    audit = {
        "raw_subjects": len(all_subjects),
        "pretrain_subjects": len(pretrain),
        "downstream_subjects": len(downstream),
        "identity_overlap": sorted(pretrain.intersection(downstream)),
        "pretrain_subject_ids": sorted(pretrain),
    }
    return CorpusPart("stanford", 5, selected), audit


def _load_big_ideas(root: Path) -> tuple[CorpusPart, dict[str, object]]:
    recordings = []
    subject_ids = []
    for path in sorted(root.glob("*/Dexcom_*.csv")):
        subject = path.parent.name
        frame = pd.read_csv(
            path,
            usecols=["Timestamp (YYYY-MM-DDThh:mm:ss)", "Glucose Value (mg/dL)"],
        ).rename(
            columns={
                "Timestamp (YYYY-MM-DDThh:mm:ss)": "timestamp",
                "Glucose Value (mg/dL)": "glucose_mg_dl",
            }
        )
        frame["subject_id"] = subject
        recordings.append(frame)
        subject_ids.append(subject)
    return CorpusPart("big_ideas", 5, pd.concat(recordings, ignore_index=True)), {
        "pretrain_subjects": len(subject_ids),
        "pretrain_subject_ids": subject_ids,
    }


def _load_colas(root: Path) -> tuple[CorpusPart, dict[str, object]]:
    recordings = []
    raw_subjects = []
    for path in sorted((root / "S1").glob("case*.csv")):
        subject = str(int(path.stem.split()[-1]))
        frame = pd.read_csv(path)
        first_clock = pd.to_timedelta(str(frame["hora"].iloc[0]))
        frame["timestamp"] = (
            pd.Timestamp("2000-01-01")
            + first_clock
            + pd.to_timedelta(np.arange(len(frame)) * GRID_MINUTES, unit="min")
        )
        frame.rename(columns={"glucemia": "glucose_mg_dl"}, inplace=True)
        frame["subject_id"] = subject
        recordings.append(frame)
        raw_subjects.append(subject)
    combined = pd.concat(recordings, ignore_index=True)
    # The window extractor drops missing readings before detecting >1h gaps;
    # case 148 and 150 then have no 24-hour segment, recovering paper N=206.
    return CorpusPart("colas", 5, combined), {
        "raw_subjects": len(raw_subjects),
        "raw_subject_ids": sorted(raw_subjects, key=int),
    }


def _summary(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    rows = {}
    for cohort in np.unique(arrays["cohort"]):
        chosen = arrays["cohort"] == cohort
        identity_units = int(np.unique(arrays["subject_id"][chosen]).size)
        native_cadences = np.unique(arrays["native_cadence_min"][chosen])
        rows[str(cohort)] = {
            "subjects": None if cohort == "private_source_private" else identity_units,
            "identity_units": identity_units,
            "windows": int(chosen.sum()),
            "train_windows": int(np.sum(chosen & (arrays["split"] == SPLIT_TRAIN))),
            "validation_windows": int(
                np.sum(chosen & (arrays["split"] == SPLIT_VALIDATION))
            ),
            "native_cadence_min": (
                int(native_cadences.item())
                if len(native_cadences) == 1
                else [int(value) for value in native_cadences]
            ),
            "physical_coverage_mean": float(arrays["coverage"][chosen].mean()),
        }
        if cohort == "private_source_private":
            rows[str(cohort)]["identity_note"] = (
                "window-level opaque units; private subject IDs are intentionally absent from this NPZ"
            )
    return rows


def _append_private_source(
    public: dict[str, np.ndarray], path: Path
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        count = len(archive["glucose_mg_dl"])
        private_source = {
            "glucose_mg_dl": archive["glucose_mg_dl"].astype(np.float32),
            "observed_mask": archive["observed_mask"].astype(np.uint8),
            "subject_id": np.asarray(
                [f"private_source_private_window:{index}" for index in range(count)]
            ),
            "cohort": np.full(count, "private_source_private"),
            "native_cadence_min": (
                archive["native_cadence_min"].astype(np.int16)
                if "native_cadence_min" in archive.files
                else np.full(count, 5, dtype=np.int16)
            ),
            "shifted_window_start": archive["shifted_window_start"].astype(
                "datetime64[ns]"
            ),
            "coverage": archive["coverage"].astype(np.float32),
            "window_id": (
                archive["window_id"].astype(str)
                if "window_id" in archive.files
                else np.asarray(
                    [
                        hashlib.sha256(
                            f"private_source-mix-v1|{index}".encode()
                        ).hexdigest()
                        for index in range(count)
                    ]
                )
            ),
            "split": archive["split"].astype(np.int8),
        }
        private_source_device_source = archive["device_source_code"].astype(np.int16)
    mixed = {key: np.concatenate([public[key], private_source[key]]) for key in public}
    mixed["device_source_code"] = np.concatenate(
        [
            np.full(len(public["glucose_mg_dl"]), -1, dtype=np.int16),
            private_source_device_source,
        ]
    )
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shanghai-root", type=Path, required=True)
    parser.add_argument("--stanford-root", type=Path, required=True)
    parser.add_argument("--big-ideas-root", type=Path, required=True)
    parser.add_argument("--colas-root", type=Path, required=True)
    parser.add_argument("--private_source-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--shanghai-policy",
        choices=["subject_disjoint", "paper_visit_disjoint"],
        default="subject_disjoint",
    )
    args = parser.parse_args()

    shanghai, shanghai_audit = _load_shanghai(args.shanghai_root, args.shanghai_policy)
    stanford, stanford_audit = _load_stanford(args.stanford_root)
    big_ideas, big_ideas_audit = _load_big_ideas(args.big_ideas_root)
    colas, colas_audit = _load_colas(args.colas_root)
    public = extract_random_overlapping_windows(
        [shanghai, stanford, big_ideas, colas], seed=args.seed
    )
    public_summary = _summary(public)
    eligible_colas = sorted(
        value.split(":", 1)[1]
        for value in np.unique(public["subject_id"])
        if value.startswith("colas:")
    )
    colas_audit["preprocessed_subjects"] = len(eligible_colas)
    colas_audit["preprocessed_subject_ids"] = sorted(eligible_colas, key=int)
    colas_audit["excluded_subject_ids"] = sorted(
        set(colas_audit["raw_subject_ids"]).difference(eligible_colas), key=int
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_path = (
        args.output_dir / f"glucofm_public_pretrain_{args.shanghai_policy}_v1.npz"
    )
    np.savez_compressed(public_path, **public)
    report: dict[str, object] = {
        "acceptance_passed": bool(
            public_summary["shanghai_t2dm"]["subjects"]
            == (43 if args.shanghai_policy == "subject_disjoint" else 44)
            and public_summary["stanford"]["subjects"] == 19
            and public_summary["big_ideas"]["subjects"] == 16
            and public_summary["colas"]["subjects"] == 206
            and not stanford_audit["identity_overlap"]
            and (
                args.shanghai_policy != "subject_disjoint"
                or not shanghai_audit["selected_downstream_biological_overlap"]
            )
        ),
        "protocol": "glucofm-public-pretrain-v1",
        "paper_window_sampling": "fixed-seed random overlapping 24h windows; 20%-80% overlap",
        "gap_rule_minutes": 60,
        "grid_minutes": GRID_MINUTES,
        "window_points": WINDOW_POINTS,
        "seed": args.seed,
        "artifact": str(public_path),
        "cohorts": public_summary,
        "identity_audit": {
            "shanghai_t2dm": shanghai_audit,
            "stanford": stanford_audit,
            "big_ideas": big_ideas_audit,
            "colas": colas_audit,
        },
        "paper_reported": {
            "shanghai_t2dm": {"subjects": 44, "hours": 12414},
            "stanford": {"subjects": 19, "hours": 8761},
            "big_ideas": {"subjects": 16, "hours": 3017},
            "colas": {"subjects": 206, "hours": 9544},
        },
    }
    if args.private_source_npz:
        mixed = _append_private_source(public, args.private_source_npz)
        mixed_path = (
            args.output_dir
            / f"glucofm_private_source_public_mix_{args.shanghai_policy}_source_v2.npz"
        )
        np.savez_compressed(mixed_path, **mixed)
        report["mixed_artifact"] = str(mixed_path)
        report["mixed_cohorts"] = _summary(mixed)
        report["private_source_private_test_windows_available_but_not_trainable"] = int(
            np.sum(
                (mixed["cohort"] == "private_source_private") & (mixed["split"] == 2)
            )
        )
        report["device_source_metadata"] = {
            "scope": "PRIVATE rows only; public rows use -1 and carry no source supervision",
            "is_downstream_phenotype_or_identity": False,
            "private_source_train_counts": {
                str(int(code)): int(count)
                for code, count in zip(
                    *np.unique(
                        mixed["device_source_code"][
                            (mixed["cohort"] == "private_source_private")
                            & (mixed["split"] == SPLIT_TRAIN)
                        ],
                        return_counts=True,
                    )
                )
            },
        }
    report_path = (
        args.output_dir
        / f"glucofm_public_pretrain_{args.shanghai_policy}_audit_v1.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
