#!/usr/bin/env python3
"""Create the fit-free two-coordinate CARE physical-gauge adapter."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint["tail_contract"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as existing:
            expected = {
                "adapter_mode": "append_only",
                "checkpoint_sha256": sha256(args.checkpoint),
                "dataset_sha256": sha256(args.corpus),
            }
            for key, value in expected.items():
                if str(existing[key].item()) != value:
                    raise ValueError(f"existing adapter {key} mismatch")
        print(f"ADAPTER_SHA256={sha256(args.output)}")
        return
    np.savez_compressed(
        args.output,
        adapter_mode=np.asarray("append_only"),
        checkpoint_sha256=np.asarray(sha256(args.checkpoint)),
        dataset_sha256=np.asarray(sha256(args.corpus)),
        glucose_center=np.asarray(float(contract["center"]), dtype=np.float32),
        glucose_scale=np.asarray(float(contract["scale"]), dtype=np.float32),
    )
    args.output.chmod(0o600)
    print(f"ADAPTER_SHA256={sha256(args.output)}")


if __name__ == "__main__":
    main()
