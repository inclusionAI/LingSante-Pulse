#!/usr/bin/env python3
"""Extract frozen daily embeddings from the PRIVATE-pretrained GlucoFM-like encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.baselines.glucofm_model import GlucoFMLike


def load_online_encoder(
    checkpoint_path: Path,
    device: torch.device,
    *,
    force_relative_auxiliary_features: bool = False,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = GlucoFMLike(
        balance_components=bool(config.get("balance_components", False)),
        calibrate_physiological_units=bool(
            config.get("calibrate_physiological_units", False)
        ),
        relative_auxiliary_features=bool(
            force_relative_auxiliary_features
            or config.get("relative_auxiliary_features", False)
        ),
        dual_reference_features=bool(config.get("dual_reference_features", False)),
        absolute_reference_weight=float(config.get("absolute_reference_weight", 1.0)),
        glucose_center=float(config.get("glucose_center", 0.0)),
        glucose_scale=float(config.get("glucose_scale", 1.0)),
        rate_scale=float(config.get("rate_scale", 1.0)),
        variance_weight=float(config.get("variance_weight", 0.0)),
        covariance_weight=float(config.get("covariance_weight", 0.0)),
        mean_shortcut_weight=float(config.get("mean_shortcut_weight", 0.0)),
        absolute_subspace_dim=int(config.get("absolute_subspace_dim", 8)),
        factorized_output=bool(config.get("factorized_output", False)),
        normalize_dynamic_output=bool(config.get("normalize_dynamic_output", False)),
        cadence_adapter=bool(config.get("cadence_adapter", False)),
        factorized_dynamic_weight=float(config.get("factorized_dynamic_weight", 0.0)),
        factorized_absolute_weight=float(config.get("factorized_absolute_weight", 0.0)),
    )
    model.online.load_state_dict(checkpoint["online"])
    return model.online.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--relative-auxiliary-features", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    encoder = load_online_encoder(
        args.checkpoint,
        device,
        force_relative_auxiliary_features=args.relative_auxiliary_features,
    )
    with np.load(args.npz, allow_pickle=False) as archive:
        glucose = archive["glucose_mg_dl"].astype(np.float32)
        mask = archive["observed_mask"].astype(np.float32)
        if "circadian_start" in archive.files:
            circadian = archive["circadian_start"].astype(np.int64)
        else:
            starts = archive["shifted_window_start"].astype("datetime64[m]")
            circadian = ((starts.astype(np.int64) % (24 * 60)) // 5).astype(np.int64)
        passthrough = {
            key: archive[key]
            for key in archive.files
            if key.startswith("label_")
            or key
            in {
                "subject_id",
                "window_id",
                "sensor_code",
                "sensor_names",
                "split",
                "cohort",
                "native_cadence_min",
                "coverage",
            }
        }
    embeddings = []
    with torch.no_grad():
        for left in range(0, len(glucose), args.batch_size):
            right = left + args.batch_size
            value = torch.from_numpy(glucose[left:right]).to(device)
            observed = torch.from_numpy(mask[left:right]).to(device)
            start = torch.from_numpy(circadian[left:right]).to(device)
            embeddings.append(encoder.embed(value, observed, start).cpu().numpy())
    embedding = np.concatenate(embeddings).astype(np.float32)
    if embedding.shape != (len(glucose), 128) or not np.isfinite(embedding).all():
        raise RuntimeError("invalid frozen embeddings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embedding=embedding, **passthrough)
    report = {
        "acceptance_passed": True,
        "artifact": str(args.output),
        "checkpoint": str(args.checkpoint),
        "dimensions": 128,
        "implementation": "GlucoFM-like paper reimplementation; not official Google code",
        "windows": len(embedding),
    }
    report_path = args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
