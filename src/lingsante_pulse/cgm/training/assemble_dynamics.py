#!/usr/bin/env python3
"""Bind an independently trained CARE sidecar to frozen base artifacts."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-existing-checkpoint-sha256")
    parser.add_argument("--expected-existing-adapter-sha256")
    parser.add_argument("--base-dimensions", type=int, default=130, choices=(130, 137))
    # The sidecar trainer constructs this view independently of the CARE base.
    parser.add_argument("--input-view", choices=("absolute15",), default="absolute15")
    args = parser.parse_args()

    checkpoint_sha = sha256(args.checkpoint)
    adapter_sha = sha256(args.adapter)
    with np.load(args.adapter, allow_pickle=False) as adapter:
        if str(adapter["checkpoint_sha256"].item()) != checkpoint_sha:
            raise ValueError("affine adapter is not bound to the requested checkpoint")
    sidecar = torch.load(args.sidecar, map_location="cpu", weights_only=False)
    bound = copy.deepcopy(sidecar)
    if args.input_view is not None:
        bound["config"]["representation"]["input_view"] = args.input_view
    binding = {
        "care_checkpoint_sha256": checkpoint_sha,
        "affine_adapter_sha256": adapter_sha,
        "base_dimensions": args.base_dimensions,
        "base_frozen": True,
    }
    existing = bound["config"].get("base")
    if existing is not None and existing != binding:
        expected_existing = {
            "care_checkpoint_sha256": args.expected_existing_checkpoint_sha256,
            "affine_adapter_sha256": args.expected_existing_adapter_sha256,
            "base_dimensions": int(existing.get("base_dimensions", 130)),
            "base_frozen": True,
        }
        if (
            args.expected_existing_checkpoint_sha256 is None
            or args.expected_existing_adapter_sha256 is None
            or existing != expected_existing
        ):
            raise ValueError("sidecar already has a different base binding")
        if bound.get("base_binding", existing) != existing:
            raise ValueError("sidecar top-level binding disagrees with its config")
    bound["config"]["base"] = binding
    bound["base_binding"] = binding

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing_bound = torch.load(args.output, map_location="cpu", weights_only=False)
        if existing_bound.get("base_binding") != binding:
            raise ValueError("existing bound sidecar has a different binding")
        if (
            args.input_view is not None
            and existing_bound["config"]["representation"].get("input_view")
            != args.input_view
        ):
            raise ValueError("existing bound sidecar has a different input view")
        print(f"BOUND_SIDECAR_SHA256={sha256(args.output)}")
        return
    torch.save(bound, args.output)
    args.output.chmod(0o600)
    print(f"BOUND_SIDECAR_SHA256={sha256(args.output)}")


if __name__ == "__main__":
    main()
