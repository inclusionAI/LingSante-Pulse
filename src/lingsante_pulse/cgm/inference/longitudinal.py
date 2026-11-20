#!/usr/bin/env python3
"""Extract the frozen covariance-preserving CARE-CGM longitudinal embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.models.covariance_preserving_innovation import (
    transport_from_state,
)
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
    pack_histories,
    typed_evidence_features,
)
from lingsante_pulse.cgm.models.recurrent_orthogonal_evidence import (
    RecurrentOrthogonalEvidenceEncoder,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
    _align_daily_embedding,
)
from lingsante_pulse.cgm.training.recurrent_orthogonal_canary import extract


from lingsante_pulse.cgm.data.inputs import infer_native_cadence, provenance_subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--daily-embedding", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("glucofm_framework_or_components") is not False:
        raise ValueError("checkpoint does not satisfy the non-GlucoFM boundary")
    with np.load(args.npz, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    glucose = payload["glucose_mg_dl"].astype(np.float32)
    observed = payload["observed_mask"].astype(bool)
    subject = provenance_subjects(payload["subject_id"], payload.get("episode_index"))
    starts = payload["start_time"].astype("datetime64[ns]")
    cadence = infer_native_cadence(
        observed, payload.get("sensor_code"), payload.get("sensor_names")
    )
    source = payload.get("sensor_code", np.full(len(subject), -1, dtype=np.int16))
    split = np.zeros(len(subject), dtype=np.int8)
    cohort = np.full(len(subject), args.npz.parent.name)
    histories = causal_history_indices(
        subject,
        starts,
        cohort,
        split,
        cadence,
        source,
        max_days=int(checkpoint["maximum_days"]),
    )
    raw_features, support = typed_evidence_features(glucose, observed, cadence, starts)
    contract = checkpoint["typed_contract"]
    features = (
        (raw_features - np.asarray(contract["center"], dtype=np.float32))
        / np.asarray(contract["scale"], dtype=np.float32)
    ).astype(np.float32)
    packed_np = pack_histories(
        features,
        support,
        starts,
        histories,
        max_prior_days=int(checkpoint["maximum_days"]) - 1,
    )
    history_count = packed_np["history_mask"].sum(1)
    daily_raw = _align_daily_embedding(
        args.daily_embedding, payload["window_id"].astype(str)
    )
    daily_center = np.asarray(checkpoint["daily_center"], dtype=np.float32)
    daily_scale = np.asarray(checkpoint["daily_scale"], dtype=np.float32)
    daily = ((daily_raw - daily_center) / daily_scale).astype(np.float32)

    device = torch.device(args.device)
    model = RecurrentOrthogonalEvidenceEncoder(
        torch.as_tensor(checkpoint["daily_basis"]), **checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    packed = {
        name: torch.from_numpy(value).to(device) for name, value in packed_np.items()
    }
    output = extract(
        model,
        torch.from_numpy(daily).to(device),
        packed,
        batch_size=int(checkpoint["config"]["training"]["evaluation_batch_size"]),
    )
    memory = output["memory_codes"].reshape(len(daily), -1)
    transport = transport_from_state(checkpoint["candidate_transport"])
    valid_history = history_count > 0
    embedding = transport.embedding(
        daily,
        memory,
        valid_history,
        innovation_ratio=float(checkpoint["candidate_residual_gain"]),
    )
    embedding = (embedding * daily_scale + daily_center).astype(np.float32)
    embedding[~valid_history] = daily_raw[~valid_history]
    if embedding.shape != daily_raw.shape or not np.isfinite(embedding).all():
        raise ValueError("invalid longitudinal embedding")
    if not np.array_equal(embedding[~valid_history], daily_raw[~valid_history]):
        raise ValueError("no-history rows must be exact daily identity")

    output_payload = {
        name: value
        for name, value in payload.items()
        if name not in {"glucose_mg_dl", "observed_mask", "sensor_names"}
    }
    output_payload.update(
        embedding=embedding,
        history_days=(history_count + 1).astype(np.int8),
        inferred_native_cadence_min=cadence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_payload)


if __name__ == "__main__":
    main()
