"""Reproduce nested selection using a separately supplied 24-candidate bundle."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch

import lingsante_pulse.cgm.evaluation.protocol as protocol
from lingsante_pulse.cgm.inference.pipeline import (
    validate_input,
    verify_checkpoints,
    daily_representation,
    longitudinal_representation,
)

ROOT = Path(__file__).resolve().parents[1] / "assets"


def candidate_path(directory, record):
    return directory / Path(record["file"]).relative_to("candidates")


def candidate_records(directory):
    directory = Path(directory)
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(
            "Nested reproduction needs the separate 24-candidate directory "
            "with manifest.json; it is not included in the default package."
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    records = manifest["candidates"]
    if [r["id"] for r in records] != list(protocol.CANDIDATES):
        raise ValueError("Candidate order differs from the evaluation protocol")
    for record in records:
        path = candidate_path(directory, record)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Candidate checksum mismatch: " + record["id"])
    return records


def extract_candidates(
    data_root, output, checkpoint_dir, device, memory_device="cpu", *, candidates_dir
):
    records = candidate_records(candidates_dir)
    verify_checkpoints(checkpoint_dir)
    shared = json.loads((candidates_dir / "manifest.json").read_text())["shared_sha256"]
    for name, expected in shared.items():
        if hashlib.sha256((checkpoint_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Candidate daily component mismatch: " + name)
    torch.set_num_threads(2)
    for cohort in protocol.COHORTS:
        path = data_root / cohort / (cohort + "_glucofm_windows_v1.npz")
        with np.load(path, allow_pickle=False) as archive:
            payload = dict(archive)
        inputs = {k: v for k, v in payload.items() if not k.startswith("label_")}
        validate_input(inputs)
        daily = daily_representation(inputs, checkpoint_dir, device)
        keep = {
            k: v
            for k, v in payload.items()
            if k
            in (
                "subject_id",
                "window_id",
                "start_time",
                "episode_index",
                "sensor_code",
                "biological_subject_id",
            )
            or k.startswith("label_")
        }
        for record in records:
            z, history = longitudinal_representation(
                inputs,
                daily,
                checkpoint_dir,
                cohort,
                memory_device,
                candidate_path(candidates_dir, record),
            )
            target = output / record["id"] / cohort / "embeddings.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, embedding=z, history_days=history, **keep)
        print("Extracted " + cohort, flush=True)


def partition_contract(plan):
    """Only row partitions and selection rules; no source paths or identifiers."""
    return {
        k: plan[k]
        for k in (
            "candidate_ids",
            "reference",
            "score",
            "classifier",
            "outer",
            "inner",
            "counts",
        )
    } | {
        "episodes": [
            {
                "repeat": e["repeat"],
                "fold": e["fold"],
                "cohorts": {
                    c: {k: v for k, v in s.items() if k != "test_people"}
                    for c, s in e["cohorts"].items()
                },
            }
            for e in plan["episodes"]
        ]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--candidates-dir",
        type=Path,
        required=True,
        help="Separate candidate bundle; not included in this package",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--memory-device",
        default="cpu",
        help="Longitudinal inference device; original evaluation used CPU",
    )
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    candidate_records(a.candidates_dir)
    a.output.mkdir(parents=True)
    cache = a.output / "candidate_embeddings"
    extract_candidates(
        a.data_root,
        cache,
        a.checkpoint_dir,
        a.device,
        a.memory_device,
        candidates_dir=a.candidates_dir,
    )
    data, artifacts = protocol.load_data(cache)
    plan = protocol.make_plan(data, artifacts)
    protocol.validate_plan(data, plan)
    expected = json.loads((ROOT / "benchmarks/partitions.json").read_text())
    if partition_contract(plan) != expected:
        raise ValueError(
            "Public data/label order differs from the supplied benchmark partitions"
        )
    metadata = json.loads((ROOT / "benchmarks/input_fingerprints.json").read_text())
    for cohort, d in data.items():
        digest = hashlib.sha256(
            json.dumps(
                {
                    "window_id": d["window_id"].tolist(),
                    "people": d["people"].tolist(),
                    "labels": {k: v.tolist() for k, v in sorted(d["labels"].items())},
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if digest != metadata[cohort]:
            raise ValueError("Public input fingerprint mismatch: " + cohort)
    results = a.output / "evaluation"
    results.mkdir()
    protocol.save_new(results / "plan.json", plan)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lingsante_pulse.cgm.evaluation.protocol",
            "run",
            "--root",
            str(cache),
            "--output",
            str(results),
            "--workers",
            str(a.workers),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
