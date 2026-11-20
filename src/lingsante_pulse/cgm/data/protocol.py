#!/usr/bin/env python3
"""Shared, auditable preprocessing and linear-probe protocol for GlucoFM tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


GRID_MINUTES = 5
WINDOW_POINTS = 288
PATCH_POINTS = 12


@dataclass(frozen=True)
class AlignedWindows:
    glucose: np.ndarray
    observed_mask: np.ndarray
    subject_id: np.ndarray
    window_id: np.ndarray
    start_time: np.ndarray
    circadian_start: np.ndarray
    coverage: np.ndarray
    episode_index: np.ndarray


def _stable_window_id(
    subject_id: str, episode: int, window: int, start: pd.Timestamp
) -> str:
    raw = f"glucofm-public-v1|{subject_id}|{episode}|{window}|{start.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def align_nonoverlapping_windows(
    frame: pd.DataFrame,
    *,
    subject_col: str,
    timestamp_col: str,
    value_col: str,
    max_within_segment_gap_minutes: int = 60,
) -> AlignedWindows:
    """Split at gaps >1h, then floor-bin non-overlapping 24h windows.

    Missing positions remain NaN and are represented independently by the mask.
    No undocumented coverage threshold is imposed.
    """
    needed = {subject_col, timestamp_col, value_col}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = frame[[subject_col, timestamp_col, value_col]].copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="raise")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[subject_col, timestamp_col]).sort_values(
        [subject_col, timestamp_col], kind="stable"
    )

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    subjects: list[str] = []
    window_ids: list[str] = []
    starts: list[np.datetime64] = []
    circadian_starts: list[int] = []
    coverages: list[float] = []
    episode_indices: list[int] = []

    global_episode = 0
    for subject_id, subject in data.groupby(subject_col, sort=True):
        gap = subject[timestamp_col].diff().dt.total_seconds().div(60)
        local_episode = (gap.isna() | gap.gt(max_within_segment_gap_minutes)).cumsum()
        for _, episode in subject.groupby(local_episode, sort=True):
            episode = episode.dropna(subset=[value_col])
            if episode.empty:
                global_episode += 1
                continue
            first = episode[timestamp_col].iloc[0]
            elapsed = (episode[timestamp_col] - first).dt.total_seconds().div(60)
            bins = np.floor(elapsed / GRID_MINUTES).astype(np.int64)
            binned = (
                pd.DataFrame({"bin": bins, "value": episode[value_col].to_numpy()})
                .query("bin >= 0")
                .groupby("bin", sort=True)["value"]
                .mean()
            )
            num_windows = (int(binned.index.max()) + 1) // WINDOW_POINTS
            for window in range(num_windows):
                left = window * WINDOW_POINTS
                chosen = binned[
                    (binned.index >= left) & (binned.index < left + WINDOW_POINTS)
                ]
                aligned = np.full(WINDOW_POINTS, np.nan, dtype=np.float32)
                positions = chosen.index.to_numpy(dtype=np.int64) - left
                aligned[positions] = chosen.to_numpy(dtype=np.float32)
                mask = np.isfinite(aligned)
                start = first + pd.Timedelta(minutes=left * GRID_MINUTES)
                values.append(aligned)
                masks.append(mask.astype(np.uint8))
                subjects.append(str(subject_id))
                window_ids.append(
                    _stable_window_id(str(subject_id), global_episode, window, start)
                )
                starts.append(np.datetime64(start))
                circadian_starts.append(
                    (start.hour * 60 + start.minute) // GRID_MINUTES
                )
                coverages.append(float(mask.mean()))
                episode_indices.append(global_episode)
            global_episode += 1

    if not values:
        raise ValueError("no complete 24-hour time-span windows were found")
    return AlignedWindows(
        glucose=np.stack(values),
        observed_mask=np.stack(masks),
        subject_id=np.asarray(subjects),
        window_id=np.asarray(window_ids),
        start_time=np.asarray(starts, dtype="datetime64[ns]"),
        circadian_start=np.asarray(circadian_starts, dtype=np.int16),
        coverage=np.asarray(coverages, dtype=np.float32),
        episode_index=np.asarray(episode_indices, dtype=np.int32),
    )


def hall_subject_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the four Hall labels exactly from GlucoFM Appendix A/Table 8."""
    clinical = frame.sort_values("time").groupby("id", sort=True).first()
    required = [
        "FBG",
        "insulin",
        "Tchol",
        "Trg",
        "LDL",
        "diagnosis",
        "glucotype",
        "SSPG",
    ]
    absent = set(required).difference(clinical.columns)
    if absent:
        raise ValueError(f"missing Hall clinical columns: {sorted(absent)}")
    # One public row has derived diagnosis/glucotype values but no laboratory
    # measurements for IR or lipid construction; GlucoFM excludes that subject.
    lab_columns = ["FBG", "insulin", "Tchol", "Trg", "LDL", "SSPG"]
    sufficient = ~clinical[lab_columns].lt(0).all(axis=1)
    clinical = clinical.loc[sufficient].copy()
    homa_ir = clinical["insulin"] * clinical["FBG"] / 405.0
    labels = pd.DataFrame(index=clinical.index)
    labels["diabetes"] = clinical["diagnosis"].gt(0).astype(np.int8)
    labels["glucotype"] = clinical["glucotype"].eq(2).astype(np.int8)
    labels["insulin_resistance"] = np.where(
        clinical["SSPG"].ge(0), clinical["SSPG"].gt(120), homa_ir.gt(2.9)
    ).astype(np.int8)
    labels["hyperlipidemia"] = (
        clinical["Tchol"].ge(240) | clinical["LDL"].ge(160) | clinical["Trg"].ge(200)
    ).astype(np.int8)
    labels.index.name = "subject_id"
    expected = {
        "diabetes": {0: 37, 1: 19},
        "glucotype": {0: 34, 1: 22},
        "insulin_resistance": {0: 35, 1: 21},
        "hyperlipidemia": {0: 48, 1: 8},
    }
    actual = {
        name: labels[name].value_counts().sort_index().to_dict()
        for name in labels.columns
    }
    if len(labels) != 56 or actual != expected:
        raise ValueError(f"Hall label audit failed: N={len(labels)}, counts={actual}")
    return labels


def cgmacros_subject_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct CGMacros labels and assert the GlucoFM Table 8 distribution."""
    data = frame.copy().set_index("subject")
    a1c = data["A1c PDL (Lab)"]
    labels = pd.DataFrame(index=data.index.astype(str))
    # The paper does not disclose its subject-level class mapping. Preserve the
    # standard A1c interpretation instead of manually relabeling IDs to force
    # the paper's 16/14/15 distribution.
    labels["diabetes"] = np.select(
        [a1c.lt(5.7), a1c.lt(6.5)], [0, 1], default=2
    ).astype(np.int8)
    labels["insulin_resistance"] = (
        (data["Insulin "] * data["Fasting GLU - PDL (Lab)"] / 405.0)
        .gt(2.9)
        .to_numpy(dtype=np.int8)
    )
    labels["obesity"] = data["BMI"].ge(30).to_numpy(dtype=np.int8)
    labels["hyperlipidemia"] = (
        data["Cholesterol"].ge(240)
        | data["LDL (Cal)"].ge(160)
        | data["Triglycerides"].ge(200)
    ).to_numpy(dtype=np.int8)
    labels.index.name = "subject_id"
    expected_reconstructable = {
        "diabetes": {0: 15, 1: 16, 2: 14},
        "insulin_resistance": {0: 13, 1: 32},
        "obesity": {0: 22, 1: 23},
        "hyperlipidemia": {0: 33, 1: 12},
    }
    actual = {
        name: labels[name].value_counts().sort_index().to_dict()
        for name in labels.columns
    }
    if len(labels) != 45 or actual != expected_reconstructable:
        raise ValueError(
            f"CGMacros label audit failed: N={len(labels)}, counts={actual}"
        )
    return labels


def recover_linearly_interpolated_native_readings(
    timestamp: pd.Series, values: pd.Series, cadence_minutes: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Recover a sensor's native cadence anchors from CGMacros' 1-min interpolation.

    The public release stores linearly interpolated minute values, not the raw
    exports used by Google. The correct phase minimizes reconstruction error
    when native-cadence anchors are linearly re-interpolated.
    """
    times = pd.to_datetime(timestamp, errors="raise").reset_index(drop=True)
    glucose = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(glucose)
    # pandas 3 may preserve datetime64[us], while pandas 2 commonly stores ns.
    # Converting explicitly avoids interpreting microsecond integers as nanoseconds.
    epoch_minute = (
        times.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 60_000_000_000
    )
    row_index = np.arange(len(times))
    candidates: list[tuple[float, int, np.ndarray]] = []
    for offset in range(cadence_minutes):
        anchors = valid & (epoch_minute % cadence_minutes == offset)
        anchor_index = np.flatnonzero(anchors)
        if len(anchor_index) < 2:
            continue
        inside = (
            valid
            & row_index.__ge__(anchor_index.min())
            & row_index.__le__(anchor_index.max())
        )
        reconstructed = np.interp(
            np.flatnonzero(inside), anchor_index, glucose[anchors]
        )
        error = float(np.mean(np.abs(reconstructed - glucose[inside])))
        candidates.append((error, offset, anchors))
    if not candidates:
        raise ValueError("cannot infer a native sensor phase")
    error, offset, anchors = min(candidates, key=lambda item: (item[0], item[1]))
    recovered = pd.DataFrame(
        {"time": times[anchors].to_numpy(), "gl": glucose[anchors].astype(np.float32)}
    )
    return recovered, {
        "cadence_minutes": cadence_minutes,
        "phase_offset": offset,
        "reconstruction_mae_mg_dl": error,
        "recovered_points": int(len(recovered)),
    }


def align_strictly_causal_window(
    readings: pd.DataFrame,
    onset: pd.Timestamp,
    *,
    grid_minutes: int = GRID_MINUTES,
    window_points: int = WINDOW_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Floor-bin the fixed interval [onset-24h, onset) without future leakage."""
    start = pd.Timestamp(onset) - pd.Timedelta(minutes=grid_minutes * window_points)
    selected = readings.loc[
        readings["time"].ge(start) & readings["time"].lt(onset), ["time", "gl"]
    ].copy()
    elapsed = (selected["time"] - start).dt.total_seconds().div(60)
    selected["bin"] = np.floor(elapsed / grid_minutes).astype(np.int64)
    binned = selected.groupby("bin", sort=True)["gl"].mean()
    binned = binned.loc[(binned.index >= 0) & (binned.index < window_points)]
    glucose = np.full(window_points, np.nan, dtype=np.float32)
    glucose[binned.index.to_numpy(dtype=np.int64)] = binned.to_numpy(dtype=np.float32)
    return glucose, np.isfinite(glucose).astype(np.uint8)


def nearest_reading(
    readings: pd.DataFrame, target: pd.Timestamp, tolerance_minutes: float
) -> float | None:
    """Return the nearest native reading with deterministic earlier-time tie breaking."""
    if readings.empty:
        return None
    delta = (readings["time"] - pd.Timestamp(target)).abs().dt.total_seconds().div(60)
    eligible = readings.loc[delta.le(tolerance_minutes)].copy()
    if eligible.empty:
        return None
    eligible["delta"] = (
        (eligible["time"] - pd.Timestamp(target)).abs().dt.total_seconds().div(60)
    )
    row = eligible.sort_values(["delta", "time"], kind="stable").iloc[0]
    return float(row["gl"])


def ppgr_endpoints(
    observed_change: np.ndarray,
    predicted_change: np.ndarray,
    horizons_minutes: np.ndarray,
) -> dict[str, float]:
    """Compute the four GlucoFM PPGR endpoints from native-cadence changes."""
    observed = np.asarray(observed_change, dtype=np.float64)
    predicted = np.asarray(predicted_change, dtype=np.float64)
    horizons = np.asarray(horizons_minutes, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("observed and predicted changes must be equal 2-D arrays")
    if (
        observed.shape[1] != horizons.size
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("invalid PPGR arrays")
    time_hours = np.r_[0.0, horizons / 60.0]
    observed_with_zero = np.column_stack(
        [np.zeros(len(observed)), np.maximum(observed, 0)]
    )
    predicted_with_zero = np.column_stack(
        [np.zeros(len(predicted)), np.maximum(predicted, 0)]
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    observed_iauc = trapezoid(observed_with_zero, time_hours, axis=1)
    predicted_iauc = trapezoid(predicted_with_zero, time_hours, axis=1)
    observed_peak_index = np.argmax(observed, axis=1)
    predicted_peak_index = np.argmax(predicted, axis=1)
    return {
        "trajectory_mae_mg_dl": float(np.mean(np.abs(predicted - observed))),
        "positive_iauc_mae_mg_dl_h": float(
            np.mean(np.abs(predicted_iauc - observed_iauc))
        ),
        "peak_rise_mae_mg_dl": float(
            np.mean(np.abs(np.max(predicted, axis=1) - np.max(observed, axis=1)))
        ),
        "peak_time_mae_minutes": float(
            np.mean(
                np.abs(horizons[predicted_peak_index] - horizons[observed_peak_index])
            )
        ),
    }


def shanghai_t2dm_session_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the 65 labeled visits and reconstruct all three paper labels."""
    data = frame.copy()
    columns = {
        "fpg": "Fasting Plasma Glucose (mg/dl)",
        "insulin": "Fasting Insulin (pmol/L)",
        "tc": "Total Cholesterol (mmol/L)",
        "tg": "Triglyceride (mmol/L)",
        "ldl": "Low-Density Lipoprotein Cholesterol (mmol/L)",
    }
    for column in columns.values():
        data[column] = pd.to_numeric(data[column], errors="coerce")
    hypoglycemia = data["Hypoglycemia (yes/no)"].astype(str).str.strip().str.lower()
    eligible = data[list(columns.values())].notna().all(axis=1) & hypoglycemia.isin(
        ["yes", "no"]
    )
    data = data.loc[eligible].copy()
    hypoglycemia = hypoglycemia.loc[eligible]
    labels = pd.DataFrame(index=data["Patient Number"].astype(str))
    labels["hypoglycemia"] = hypoglycemia.eq("yes").to_numpy(dtype=np.int8)
    homa_ir = data[columns["fpg"]] * (data[columns["insulin"]] / 6.945) / 405.0
    labels["insulin_resistance"] = homa_ir.gt(2.9).to_numpy(dtype=np.int8)
    labels["hyperlipidemia"] = (
        data[columns["tc"]].mul(38.67).ge(240)
        | data[columns["ldl"]].mul(38.67).ge(160)
        | data[columns["tg"]].mul(88.57).ge(200)
    ).to_numpy(dtype=np.int8)
    labels.index.name = "subject_id"
    expected = {
        "hypoglycemia": {0: 56, 1: 9},
        "insulin_resistance": {0: 24, 1: 41},
        "hyperlipidemia": {0: 45, 1: 20},
    }
    actual = {
        name: labels[name].value_counts().sort_index().to_dict()
        for name in labels.columns
    }
    biological_subjects = labels.index.to_series().str.split("_").str[0].nunique()
    if len(labels) != 65 or biological_subjects != 58 or actual != expected:
        raise ValueError(
            f"Shanghai label audit failed: entries={len(labels)}, people={biological_subjects}, counts={actual}"
        )
    return labels


def stanford_subject_labels(
    metabolic_tests: pd.DataFrame,
    characteristics: pd.DataFrame,
    cgm_subject_ids: np.ndarray,
) -> pd.DataFrame:
    """Recover Google's 37-subject Stanford mapping and three exact labels."""
    main = "venous_without_matching_cgm_and_without_planned_athome_cgm"
    metabolic = (
        metabolic_tests.assign(
            _priority=metabolic_tests["exp_type"].ne(main).astype(np.int8)
        )
        .sort_values(["SubjectID", "_priority"], kind="stable")
        .drop_duplicates("SubjectID")
    )
    clinical = (
        characteristics.assign(
            _priority=characteristics["ExperimentType"].ne(main).astype(np.int8)
        )
        .sort_values(["SubjectID", "_priority"], kind="stable")
        .drop_duplicates("SubjectID")
    )
    joined = metabolic.merge(clinical, on="SubjectID", validate="one_to_one")
    normalized_cgm_ids = {
        str(value).replace("43883-", "S")
        for value in np.asarray(cgm_subject_ids).astype(str)
    }
    joined = joined[joined["SubjectID"].isin(normalized_cgm_ids)].copy()
    labels = pd.DataFrame(index=joined["SubjectID"].astype(str))
    labels["diabetes"] = (
        pd.to_numeric(joined["HbA1c"], errors="raise").ge(5.7).to_numpy(dtype=np.int8)
    )
    labels["beta_cell_dysfunction"] = (
        joined["di_2_classes_median"].eq("Dysfunction").to_numpy(dtype=np.int8)
    )
    labels["insulin_resistance"] = (
        joined["sspg_2_classes"].eq("IR").to_numpy(dtype=np.int8)
    )
    labels.index.name = "subject_id"
    expected = {
        "diabetes": {0: 17, 1: 20},
        "beta_cell_dysfunction": {0: 17, 1: 20},
        "insulin_resistance": {0: 17, 1: 20},
    }
    actual = {
        name: labels[name].value_counts().sort_index().to_dict()
        for name in labels.columns
    }
    if len(labels) != 37 or labels.isna().any().any() or actual != expected:
        raise ValueError(
            f"Stanford label audit failed: N={len(labels)}, counts={actual}"
        )
    return labels.sort_index()


def mask_aware_statistical_features(
    glucose: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Small no-pretraining baseline used to validate the evaluator end to end."""
    if (
        glucose.shape != mask.shape
        or glucose.ndim != 2
        or glucose.shape[1] != WINDOW_POINTS
    ):
        raise ValueError("expected glucose and mask with shape [N, 288]")
    x = np.where(mask.astype(bool), glucose, np.nan)
    indices = np.arange(WINDOW_POINTS, dtype=np.float64)
    rows = []
    for row in x:
        valid = np.isfinite(row)
        observed = row[valid]
        if len(observed) < 2:
            raise ValueError("window has fewer than two observed values")
        slope = np.polyfit(indices[valid], observed, 1)[0]
        diffs = np.diff(row)
        valid_diffs = diffs[np.isfinite(diffs)]
        rows.append(
            [
                np.mean(observed),
                np.std(observed),
                np.min(observed),
                np.max(observed),
                np.quantile(observed, 0.25),
                np.median(observed),
                np.quantile(observed, 0.75),
                np.mean(observed < 70),
                np.mean(observed > 140),
                np.mean(observed > 180),
                slope,
                np.mean(np.abs(valid_diffs)) if len(valid_diffs) else 0.0,
                np.std(valid_diffs) if len(valid_diffs) else 0.0,
                np.mean(valid),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def repeated_subject_folds(
    groups: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int = 10,
    folds: int = 5,
    seed: int = 43,
):
    """Yield deterministic stratified subject splits; windows never split across folds."""
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels)
    unique_groups, first = np.unique(groups, return_index=True)
    group_labels = labels[first]
    for group in unique_groups:
        if np.unique(labels[groups == group]).size != 1:
            raise ValueError(f"group {group} has inconsistent labels")
    for repeat in range(repeats):
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_group_idx, test_group_idx) in enumerate(
            splitter.split(unique_groups, group_labels)
        ):
            train_groups = unique_groups[train_group_idx]
            test_groups = unique_groups[test_group_idx]
            train = np.flatnonzero(np.isin(groups, train_groups))
            test = np.flatnonzero(np.isin(groups, test_groups))
            yield repeat, fold, train, test


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    classes = np.unique(labels)
    predictions = classes[np.argmax(probabilities, axis=1)]
    if len(classes) == 2:
        pr_auc = average_precision_score(labels, probabilities[:, 1])
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
    else:
        binary = label_binarize(labels, classes=classes)
        pr_auc = average_precision_score(binary, probabilities, average="macro")
        roc_auc = roc_auc_score(
            binary, probabilities, average="macro", multi_class="ovr"
        )
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def repeated_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int = 10,
    folds: int = 5,
    seed: int = 43,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run the paper classifier and return fold metrics plus all OOF probabilities."""
    # Older sklearn/scipy LBFGS builds can overflow internally with float32
    # despite standardized inputs. The paper classifier is numerically safer in float64.
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    scores: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for repeat, fold, train, test in repeated_subject_folds(
        groups, labels, repeats=repeats, folds=folds, seed=seed
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed
            ),
        )
        # NumPy 2.0.2 + sklearn 1.6.1 can emit spurious intermediate matmul
        # floating-point warnings from LBFGS even when inputs, coefficients,
        # and probabilities are finite. The explicit finite gates remain active.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            model.fit(features[train], labels[train])
            probabilities = model.predict_proba(features[test])
        if not np.isfinite(probabilities).all():
            raise ValueError("linear probe produced non-finite probabilities")
        model_classes = model[-1].classes_
        if not np.array_equal(model_classes, classes):
            raise ValueError("training fold is missing a class")
        metrics = _classification_metrics(labels[test], probabilities)
        scores.append(
            {
                "repeat": repeat,
                "fold": fold,
                "train_subjects": int(np.unique(np.asarray(groups)[train]).size),
                "test_subjects": int(np.unique(np.asarray(groups)[test]).size),
                "train_windows": int(len(train)),
                "test_windows": int(len(test)),
                **metrics,
            }
        )
        for index, probability in zip(test, probabilities):
            predictions.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "row_index": int(index),
                    "subject_id": str(groups[index]),
                    "label": int(labels[index]),
                    "probabilities": [float(value) for value in probability],
                }
            )
    return scores, predictions


def direct_transfer_probe(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    *,
    seed: int = 43,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit on the full source cohort and evaluate once on the untouched target."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise ValueError("transfer features contain NaN or infinity")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed
        ),
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        model.fit(source_features, source_labels)
        probabilities = model.predict_proba(target_features)
    if not np.array_equal(model[-1].classes_, np.unique(target_labels)):
        raise ValueError("source and target classes are incompatible")
    if not np.isfinite(probabilities).all():
        raise ValueError("transfer probe produced non-finite probabilities")
    return _classification_metrics(target_labels, probabilities), probabilities


def fit_and_score_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    """Fit the fixed paper classifier on explicit indices and score held-out windows."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed
        ),
    )
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        model.fit(x[train], y[train])
        probabilities = model.predict_proba(x[test])
    if not np.array_equal(model[-1].classes_, np.unique(y)):
        raise ValueError("support set is missing a class")
    if not np.isfinite(probabilities).all():
        raise ValueError("few-shot probe produced non-finite probabilities")
    return _classification_metrics(y[test], probabilities)


def sample_support_subjects(
    train: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select exactly K train subjects per class and return all their windows."""
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels)
    selected: list[str] = []
    train_groups = np.unique(groups[train])
    for label in np.unique(labels):
        candidates = np.asarray(
            [
                group
                for group in train_groups
                if labels[np.flatnonzero(groups == group)[0]] == label
            ]
        )
        if len(candidates) < per_class:
            raise ValueError(
                f"class {label} has only {len(candidates)} train subjects for K={per_class}"
            )
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return train[np.isin(groups[train], selected)]


def sample_observation_fraction(
    train: np.ndarray,
    groups: np.ndarray,
    *,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep ceil(fraction*N) windows per training subject, with at least one."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    groups = np.asarray(groups).astype(str)
    chosen: list[int] = []
    for group in np.unique(groups[train]):
        candidates = train[groups[train] == group]
        count = max(1, int(np.ceil(len(candidates) * fraction)))
        chosen.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)
