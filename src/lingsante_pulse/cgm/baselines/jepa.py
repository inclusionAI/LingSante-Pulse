#!/usr/bin/env python3
"""Extract frozen official CGM-JEPA embeddings under a train-only scaler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def interpolate_windows(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    positions = np.arange(values.shape[1])
    for row in range(len(values)):
        keep = observed[row].astype(bool) & np.isfinite(values[row])
        if not keep.any():
            raise ValueError("window has no physical CGM observation")
        output[row] = np.interp(positions, positions[keep], values[row, keep])
    return output


def time_features(starts: np.ndarray) -> np.ndarray:
    offsets = np.arange(288, dtype="timedelta64[m]") * 5
    timestamps = (starts[:, None].astype("datetime64[m]") + offsets).reshape(-1)
    index = pd.DatetimeIndex(timestamps)
    features = np.column_stack(
        [
            index.minute / 59.0 - 0.5,
            index.hour / 23.0 - 0.5,
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5,
        ]
    )
    return features.reshape(len(starts), 24, 12, 5).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--reference-corpus", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("cgm_jepa", "x_cgm_jepa"), default="cgm_jepa"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.baseline_root))
    from models.encoder import Encoder

    with np.load(args.reference_corpus, allow_pickle=False) as reference:
        train = reference["split"] == 0
        reference_values = reference["glucose_mg_dl"][train].astype(np.float64)
        mean = float(reference_values.mean())
        std = float(reference_values.std())
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("invalid train-only glucose scaler")

    with np.load(args.npz, allow_pickle=False) as archive:
        values = interpolate_windows(
            archive["glucose_mg_dl"].astype(np.float32),
            archive["observed_mask"].astype(bool),
        )
        start_key = (
            "shifted_window_start"
            if "shifted_window_start" in archive
            else "start_time"
        )
        starts = archive[start_key].astype("datetime64[m]")
        metadata = {
            key: archive[key]
            for key in archive.files
            if key not in {"glucose_mg_dl", "observed_mask"}
        }
    if values.ndim != 2 or values.shape[1] != 288 or not np.isfinite(values).all():
        raise ValueError("expected finite 288-point interpolated windows")

    device = torch.device(args.device)
    model = Encoder.from_pretrained(args.baseline_root / "Output" / args.model).to(
        device
    )
    model.eval()
    embeddings = []
    with torch.no_grad():
        for left in range(0, len(values), args.batch_size):
            right = left + args.batch_size
            patches = torch.from_numpy((values[left:right] - mean) / std).reshape(
                -1, 24, 12
            )
            marks = torch.from_numpy(time_features(starts[left:right]))
            tokens, _ = model(patches.to(device), marks.to(device))
            embeddings.append(tokens.mean(dim=1).cpu().numpy())
    embedding = np.concatenate(embeddings).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite CGM-JEPA embedding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embedding=embedding, **metadata)
    report = {
        "acceptance_passed": True,
        "baseline": "official CGM-JEPA released checkpoint",
        "dimensions": int(embedding.shape[1]),
        "input_preprocessing": "mask-aware linear/edge interpolation to 288 points; train-only global mean/std; 24x12 patches; calendar features",
        "model": args.model,
        "reference_scope": "pretraining split==0 only; no downstream labels/windows",
        "scaler_mean": mean,
        "scaler_std": std,
        "windows": len(embedding),
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
