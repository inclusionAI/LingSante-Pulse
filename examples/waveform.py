"""Encode synthetic ECG windows and print feature shapes."""

import argparse

import numpy as np

from lingsante_pulse import LingSantePulse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="inclusionAI/LingSante-Pulse-1.0",
        help="Local model directory or Hugging Face repo ID (default: %(default)s)",
    )
    parser.add_argument("--revision", help="Hub tag or full commit hash")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    model = LingSantePulse.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    signals = np.random.default_rng(42).normal(size=(2, 2, 1280)).astype(np.float32)
    features = model.encode(
        signals,
        sample_rate=128,
        channels=[
            {"modality": "ECG", "lead": "I"},
            {"modality": "ECG", "lead": "II"},
        ],
        return_tokens=True,
    )
    print("CLS:", features["cls"].shape)  # [2, 512]
    print("PatchMean:", features["patch_mean"].shape)  # [2, 512]
    print("Patch sequence:", features["patch_tokens"].shape)  # [2, 40, 512]


if __name__ == "__main__":
    main()
