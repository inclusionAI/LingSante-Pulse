#!/usr/bin/env python3
"""Deterministic sampling helpers for mixed-cadence CGM pretraining."""

from __future__ import annotations

import numpy as np


def cadence_cohort_balanced_indices(
    indices: np.ndarray,
    cohorts: np.ndarray,
    cadences: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
    cadence_15_share: float = 0.5,
    cohort_temperature: float = 0.0,
    group_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Sample cadences equally, then cohorts equally within each cadence.

    Sampling is with replacement so a small real 15-minute cohort is not
    overwhelmed by a much larger 5-minute corpus. The returned order is
    shuffled to avoid cadence-homogeneous mini-batches.
    """
    indices = np.asarray(indices, dtype=np.int64)
    cohorts = np.asarray(cohorts).astype(str)
    cadences = np.asarray(cadences, dtype=np.int64)
    if size <= 0 or indices.ndim != 1:
        raise ValueError("size must be positive and indices must be one-dimensional")
    if not 0.0 <= cadence_15_share <= 1.0 or cohort_temperature < 0.0:
        raise ValueError("invalid cadence share or cohort temperature")
    if len(cohorts) != len(cadences) or len(cohorts) <= int(indices.max(initial=-1)):
        raise ValueError("cohort/cadence arrays do not cover all indices")
    if group_ids is not None:
        group_ids = np.asarray(group_ids).astype(str)
        if len(group_ids) != len(cohorts):
            raise ValueError("group IDs do not cover all indices")

    cadence_values = np.unique(cadences[indices])
    cells: list[np.ndarray] = []
    weights: list[float] = []
    if set(cadence_values).issubset({5, 15}) and len(cadence_values) == 2:
        cadence_weight = {5: 1.0 - cadence_15_share, 15: cadence_15_share}
    else:
        cadence_weight = {
            int(value): 1.0 / len(cadence_values) for value in cadence_values
        }
    for cadence in cadence_values:
        cadence_index = indices[cadences[indices] == cadence]
        cohort_values = np.unique(cohorts[cadence_index])
        cohort_sizes = np.asarray(
            [np.sum(cohorts[cadence_index] == cohort) for cohort in cohort_values],
            dtype=np.float64,
        )
        cohort_weights = np.power(cohort_sizes, cohort_temperature)
        cohort_weights /= cohort_weights.sum()
        for cohort, cohort_weight in zip(cohort_values, cohort_weights):
            cell = cadence_index[cohorts[cadence_index] == cohort]
            if len(cell):
                cells.append(cell)
                weights.append(cadence_weight[int(cadence)] * float(cohort_weight))

    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    cell_counts = rng.multinomial(size, probabilities)
    sampled = []
    for cell, count in zip(cells, cell_counts):
        if group_ids is None:
            sampled.append(rng.choice(cell, count, replace=True))
            continue
        cell_groups = group_ids[cell]
        unique_groups = np.unique(cell_groups)
        group_counts = rng.multinomial(
            count, np.full(len(unique_groups), 1.0 / len(unique_groups))
        )
        sampled.append(
            np.concatenate(
                [
                    rng.choice(cell[cell_groups == group], group_count, replace=True)
                    for group, group_count in zip(unique_groups, group_counts)
                    if group_count
                ]
            )
        )
    result = np.concatenate(sampled).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result
