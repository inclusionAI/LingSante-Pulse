"""Extract daily and longitudinal CGM representations."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "assets"

from lingsante_pulse.cgm.inference.features import (
    interpolate_windows,
    affine_gauge_codes,
    apply_affine_gauge_append_adapter,
    multi_axis_sidecar_codes,
    apply_multi_axis_sidecar_adapter,
)
from lingsante_pulse.cgm.models.daily import (
    TPACGMAutoencoder,
    deterministic_cadence_view,
    deterministic_absolute_cadence_view,
    masked_window_standardize,
)
from lingsante_pulse.cgm.models.memory import FourierClockTransportMemory
from lingsante_pulse.cgm.models.cross_window_nodes import add_cross_window_nodes
from lingsante_pulse.cgm.models.hierarchical_continuous_time_day_memory import (
    pack_hctdm,
)
from lingsante_pulse.cgm.data.history import make_inputs
from lingsante_pulse.cgm.data.inputs import infer_native_cadence, provenance_subjects


def verify_checkpoints(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"daily.pt", "dynamics.pt", "memory.pt", "physical_scale.npz"}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete checkpoint bundle")
    for name, record in manifest["files"].items():
        if (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise ValueError("Checkpoint checksum mismatch: " + name)


def validate_input(payload):
    required = {
        "glucose_mg_dl",
        "observed_mask",
        "window_id",
        "subject_id",
        "start_time",
    }
    if not required <= payload.keys():
        raise ValueError(
            "Missing input keys: " + str(sorted(required - payload.keys()))
        )
    x = payload["glucose_mg_dl"]
    mask = payload["observed_mask"].astype(bool)
    if x.ndim != 2 or x.shape[1] != 288 or mask.shape != x.shape or not len(x):
        raise ValueError(
            "Expected nonempty glucose/mask arrays of shape [N,288] on a 5-minute grid"
        )
    if not np.isfinite(x[mask]).all() or np.any(mask.sum(1) < 8):
        raise ValueError(
            "Observed values must be finite; each window needs at least eight observations"
        )
    if len(np.unique(payload["window_id"])) != len(x):
        raise ValueError("window_id must be unique")
    for key in ("window_id", "subject_id", "start_time"):
        if payload[key].shape != (len(x),):
            raise ValueError("Invalid shape: " + key)


def daily_representation(payload, directory, device="cpu", batch_size=256):
    checkpoint = torch.load(
        directory / "daily.pt", map_location="cpu", weights_only=True
    )
    contract = checkpoint["tail_contract"]
    if (
        checkpoint["input_decomposition_mode"] != "exact_cadence_range_private"
        or checkpoint["cadence_projector_mode"] != "relative_factor3"
    ):
        raise ValueError("Unexpected daily encoder configuration")
    model = TPACGMAutoencoder(
        profile_width=2 * int(contract["lag"]) + 1,
        profile_input_dim=int(checkpoint["profile_input_dim"]),
        responsibility_partitioned=bool(checkpoint["responsibility_partitioned"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    observed = payload["observed_mask"].astype(bool)
    glucose = interpolate_windows(payload["glucose_mg_dl"].astype(np.float32), observed)
    # Preserve the original daily extractor's canonical-grid semantics.
    cadence = payload.get(
        "native_cadence_min", np.full(len(glucose), 5, dtype=np.int16)
    )
    common, common_mask = deterministic_cadence_view(
        glucose, observed, factor=3, native_cadence_min=cadence
    )
    thresholds = {
        f"{side}_q{quantile}": float(contract[f"{side}_thresholds"][i])
        for side in ("upper", "lower")
        for i, quantile in enumerate((90, 95))
    }
    output = []
    with torch.inference_mode():
        for left in range(0, len(glucose), batch_size):
            sl = slice(left, left + batch_size)
            raw, mask, shared, shared_mask = [
                torch.from_numpy(a[sl]).to(device)
                for a in (glucose, observed, common, common_mask)
            ]
            relative = masked_window_standardize(shared, shared_mask)
            z, _ = model.encode(
                raw,
                masked_window_standardize(raw, shared_mask),
                mask,
                mask,
                variant=checkpoint["variant"],
                center=float(contract["center"]),
                input_decomposition_mode=checkpoint["input_decomposition_mode"],
                common_raw_glucose=shared,
                common_normalized_glucose=relative,
                private_common_normalized_glucose=relative,
                common_observed=shared_mask,
                common_visible=shared_mask,
                **thresholds,
            )
            output.append(z.cpu().numpy())
    with np.load(directory / "physical_scale.npz", allow_pickle=False) as scale:
        if str(scale["adapter_mode"]) != "append_only":
            raise ValueError("Wrong physical scale file")
        for key in ("center", "scale"):
            if not np.isclose(float(scale["glucose_" + key]), float(contract[key])):
                raise ValueError("Physical scale does not match daily encoder")
    gauge = affine_gauge_codes(
        common,
        common_mask,
        center=float(contract["center"]),
        scale=float(contract["scale"]),
    )
    daily = apply_affine_gauge_append_adapter(np.concatenate(output), gauge)
    dynamics = torch.load(
        directory / "dynamics.pt", map_location="cpu", weights_only=True
    )
    if dynamics["config"]["representation"].get("input_view") == "absolute15":
        common, common_mask = deterministic_absolute_cadence_view(
            glucose, observed, target_cadence_min=15, native_cadence_min=cadence
        )
    code = multi_axis_sidecar_codes(
        common,
        common_mask,
        dynamics,
        batch_size=batch_size,
        device=torch.device(device),
        branches="dynamics",
    )
    return apply_multi_axis_sidecar_adapter(daily, code)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, dict):
        return {k: as_numpy(v) for k, v in value.items()}
    return value


def longitudinal_representation(
    payload, daily, directory, cohort, device="cpu", memory_checkpoint=None
):
    checkpoint = torch.load(
        memory_checkpoint or directory / "memory.pt",
        map_location="cpu",
        weights_only=True,
    )
    contract = as_numpy(checkpoint["contract"])
    n = len(daily)
    data = {
        "glucose": payload["glucose_mg_dl"],
        "observed": payload["observed_mask"].astype(bool),
        "subject": provenance_subjects(
            payload["subject_id"], payload.get("episode_index")
        ),
        "starts": payload["start_time"],
        "cohort": np.full(n, cohort),
        "split": np.zeros(n, np.int8),
        "cadence": infer_native_cadence(
            payload["observed_mask"],
            payload.get("sensor_code"),
            payload.get("sensor_names"),
        ),
        "source": payload.get("sensor_code", np.full(n, -1)),
    }
    packed = make_inputs(
        data,
        ((daily - contract["daily_center"]) / contract["daily_scale"]).astype(
            np.float32
        ),
        contract,
        False,
    )
    add_cross_window_nodes(data, packed)
    packed.update(
        pack_hctdm(
            data,
            observation_policy=contract.get("memory_observation_policy", "fixed15_v1"),
        )
    )
    memory_width = int(checkpoint["model"]["fctm_lift.weight"].shape[0])
    model = FourierClockTransportMemory(False, memory_width=memory_width).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    tensors = {k: torch.from_numpy(v).to(device) for k, v in packed.items()}
    with torch.inference_mode():
        z = (
            torch.cat(
                [
                    model(**{k: v[left : left + 32] for k, v in tensors.items()})[0]
                    for left in range(0, n, 32)
                ]
            )
            .cpu()
            .numpy()
        )
    final = z * contract["daily_scale"] + contract["daily_center"]
    valid = packed["mask"] & (packed["support"][:, :, 0] > 0)
    no_history = ~valid[:, 1:].any(1)
    final[no_history] = daily[no_history]
    if (
        final.shape != (n, 146)
        or not np.isfinite(final).all()
        or packed["elapsed_days"].max() + 1 > 7
    ):
        raise ValueError("Invalid longitudinal representation")
    return final.astype(np.float32), valid.sum(1)


def extract(
    payload,
    checkpoint_dir=ROOT / "checkpoints",
    cohort="external",
    device="cpu",
    memory_device="cpu",
):
    # Exclude phenotype labels before entering either representation stage.
    inputs = {k: v for k, v in payload.items() if not k.startswith("label_")}
    validate_input(inputs)
    verify_checkpoints(Path(checkpoint_dir))
    torch.set_num_threads(2)
    daily = daily_representation(inputs, Path(checkpoint_dir), device)
    return longitudinal_representation(
        inputs, daily, Path(checkpoint_dir), cohort, memory_device
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--cohort", default="external")
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-device", default="cpu")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.input, allow_pickle=False) as archive:
        payload = dict(archive)
    embedding, history = extract(
        payload, a.checkpoint_dir, a.cohort, a.device, a.memory_device
    )
    keep = {
        k: v
        for k, v in payload.items()
        if k
        in ("subject_id", "window_id", "start_time", "episode_index", "sensor_code")
        or k.startswith("label_")
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, embedding=embedding, history_days=history, **keep)
    print(
        json.dumps(
            {
                "windows": len(embedding),
                "features": embedding.shape[1],
                "history_windows": int((history > 1).sum()),
                "label_fitting": False,
            }
        )
    )


if __name__ == "__main__":
    main()
