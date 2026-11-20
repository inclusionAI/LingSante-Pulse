"""Physical cadence and episode identities for CGM input."""

from __future__ import annotations
import numpy as np


def infer_native_cadence(
    observed: np.ndarray,
    sensor_code: np.ndarray | None = None,
    sensor_names: np.ndarray | None = None,
) -> np.ndarray:
    """Infer 5/15-minute native cadence without phenotype information."""
    mask = np.asarray(observed, dtype=bool)
    cadence = np.full(len(mask), 5, dtype=np.int16)
    if sensor_code is not None and sensor_names is not None:
        names = np.asarray(sensor_names).astype(str)
        codes = np.asarray(sensor_code, dtype=np.int64)
        for index, code in enumerate(codes):
            if 0 <= code < len(names) and "libre" in names[code].lower():
                cadence[index] = 15
        return cadence
    for row, values in enumerate(mask):
        modulo_counts = np.asarray([values[offset::3].sum() for offset in range(3)])
        total = int(modulo_counts.sum())
        if total <= 110 and total > 0 and int(modulo_counts.max()) / total >= 0.9:
            cadence[row] = 15
    return cadence


def provenance_subjects(subject: np.ndarray, episode: np.ndarray | None) -> np.ndarray:
    subject = np.asarray(subject).astype(str)
    if episode is None:
        return subject
    return np.char.add(
        np.char.add(subject, "|episode="), np.asarray(episode).astype(str)
    )
