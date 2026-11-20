"""Label-free support audit for a caller-specified train/validation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lingsante_pulse.cgm.models.cross_day_prediction import (
    prediction_pairs,
    support_summary,
)
from lingsante_pulse.cgm.models.cross_window_nodes import add_cross_window_nodes
from lingsante_pulse.cgm.models.final_space_longitudinal import evidence_v2
from lingsante_pulse.cgm.models.observation_supported_attention import pack_observations
from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
    _load_source,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import sha256


def main(args: argparse.Namespace) -> None:
    source_sha = sha256(args.source)
    if source_sha != args.expected_source_sha:
        raise ValueError("Corpus SHA mismatch")
    if args.output.exists():
        cached = json.loads(args.output.read_text())
        if (
            cached.get("source_sha256") != source_sha
            or any(
                cached.get(k) != 0
                for k in ("optimizer_steps", "model_reads", "phenotype_private_reads")
            )
            or not cached.get("pair_sha256")
        ):
            raise ValueError(
                "Existing support report does not match this label-free corpus"
            )
        # Training independently rebuilds pairs and verifies pair_sha256.
        print(json.dumps(cached), flush=True)
        return
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("Only train/validation splits are allowed")
    if set(data["subject"][data["split"] == 0]) & set(
        data["subject"][data["split"] == 1]
    ):
        raise ValueError("Subject overlap across train/validation")
    features, support = evidence_v2(
        data["glucose"], data["observed"], data["cadence"], data["starts"]
    )
    inputs = pack_observations(
        data, np.zeros((len(features), 146), np.float32), features, support
    )
    trajectory = add_cross_window_nodes(data, inputs)
    pairs, masks = prediction_pairs(data, inputs)
    report = support_summary(data, pairs, masks)
    report.update(
        source_sha256=source_sha,
        optimizer_steps=0,
        model_reads=0,
        phenotype_private_reads=0,
        trajectory_support_fraction=trajectory["trajectory_support_fraction"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
