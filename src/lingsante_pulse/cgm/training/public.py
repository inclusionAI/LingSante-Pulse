"""Run the paper's public-only multistage pretraining, without downstream labels."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "assets"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(module, *args):
    subprocess.run(
        [sys.executable, "-m", module, *map(str, args)], check=True, cwd=ROOT
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if not a.device.startswith("cuda") or not torch.cuda.is_available():
        p.error(
            "Pretraining requires an available CUDA-compatible accelerator; CPU inference is separate"
        )
    source = a.corpus.resolve()
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as data:
        if any(k.startswith("label_") for k in data.files):
            raise ValueError("Pretraining corpus must not contain phenotype labels")
        if set(np.unique(data["split"])) != {0, 1}:
            raise ValueError("Need pretraining train/validation splits 0/1 only")
        if any("private" in s.lower() for s in data["cohort"].astype(str)):
            raise ValueError("This command is public-only")
        train = set(data["subject_id"][data["split"] == 0])
        val = set(data["subject_id"][data["split"] == 1])
        if train & val:
            raise ValueError("Pretraining train/validation subject overlap")
    output.mkdir(parents=True)
    source_sha = sha(source)
    for kind in ("daily", "dynamics"):
        config = json.loads((ROOT / "configs" / f"{kind}_public.json").read_text())
        config.setdefault("corpus", {})["sha256"] = source_sha
        config["corpus"]["artifact"] = source.name
        (output / f"{kind}.json").write_text(json.dumps(config, indent=2) + "\n")
    run(
        "lingsante_pulse.cgm.training.daily",
        "--config",
        output / "daily.json",
        "--npz",
        source,
        "--output-dir",
        output / "stage_a/daily",
        "--variant",
        "range_null_cgm_signed_tail",
        "--device",
        a.device,
    )
    run(
        "lingsante_pulse.cgm.training.dynamics",
        "--config",
        output / "dynamics.json",
        "--npz",
        source,
        "--output-dir",
        output / "stage_a/dynamics",
        "--steps",
        6960,
        "--device",
        a.device,
    )
    daily_ckpt = output / "stage_a/daily/latest.pt"
    dyn = output / "stage_a/dynamics/best.pt"
    physical = output / "stage_a/physical_scale.npz"
    bound = output / "stage_a/dynamics/bound.pt"
    run(
        "lingsante_pulse.cgm.training.assemble_scale",
        "--checkpoint",
        daily_ckpt,
        "--corpus",
        source,
        "--output",
        physical,
    )
    run(
        "lingsante_pulse.cgm.training.assemble_dynamics",
        "--sidecar",
        dyn,
        "--checkpoint",
        daily_ckpt,
        "--adapter",
        physical,
        "--output",
        bound,
    )
    daily = output / "daily_embeddings.npz"
    run(
        "lingsante_pulse.cgm.inference.daily",
        "--npz",
        source,
        "--checkpoint",
        daily_ckpt,
        "--affine-gauge-adapter",
        physical,
        "--multi-axis-sidecar",
        bound,
        "--multi-axis-branches",
        "dynamics",
        "--output",
        daily,
        "--device",
        a.device,
    )
    run(
        "lingsante_pulse.cgm.diagnostics.cross_day_prediction_support_cli",
        "--source",
        source,
        "--expected-source-sha",
        source_sha,
        "--output",
        output / "stage_b/support.json",
    )
    run(
        "lingsante_pulse.cgm.training.context",
        "--source",
        source,
        "--daily",
        daily,
        "--output",
        output / "stage_b/full",
        "--kind",
        "generic",
        "--cross-day-prediction",
        "--device",
        a.device,
        "--steps",
        6960,
        "--expected-source-sha",
        source_sha,
        "--expected-daily-sha",
        sha(daily),
        "--minimum-fit-endpoints",
        512,
        "--minimum-audit-endpoints",
        64,
    )
    backbone = output / "stage_b/full/checkpoint.pt"
    support = output / "stage_c/support/report.json"
    run(
        "lingsante_pulse.cgm.diagnostics.fourier_clock_transport_memory_support",
        "--source",
        source,
        "--backbone",
        backbone,
        "--expected-source-sha",
        source_sha,
        "--expected-backbone-sha",
        sha(backbone),
        "--output",
        support,
    )
    run(
        "lingsante_pulse.cgm.training.memory",
        "--source",
        source,
        "--daily",
        daily,
        "--backbone",
        backbone,
        "--support",
        support,
        "--output",
        output / "stage_c/full",
        "--kind",
        "candidate",
        "--device",
        a.device,
        "--steps",
        6960,
        "--expected-source-sha",
        source_sha,
        "--expected-daily-sha",
        sha(daily),
        "--expected-backbone-sha",
        sha(backbone),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lingsante_pulse.cgm.checkpoints",
            "--daily",
            str(daily_ckpt),
            "--dynamics",
            str(bound),
            "--physical-scale",
            str(physical),
            "--memory",
            str(output / "stage_c/full/checkpoint.pt"),
            "--output",
            str(output / "checkpoints"),
            "--training-data",
            "public unlabeled CGM",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
