"""Export tensor-only inference weights from your own trusted training outputs."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch


def safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy())
    if isinstance(x, dict):
        return {k: safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(safe(v) for v in x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    raise TypeError(type(x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("daily", "dynamics", "physical-scale", "memory", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--training-data", default="public unlabeled CGM")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ("daily", "dynamics", "memory"):
        # This utility is for locally produced/trusted training files, not unknown uploads.
        original = torch.load(getattr(a, name), map_location="cpu", weights_only=False)
        if name == "daily":
            fields = [
                "model",
                "variant",
                "tail_contract",
                "input_normalization",
                "profile_input_dim",
                "responsibility_partitioned",
                "input_decomposition_mode",
                "cadence_projector_mode",
            ]
            out = {k: original[k] for k in fields}
        elif name == "dynamics":
            out = {k: original[k] for k in ("model", "glucose_center", "glucose_scale")}
            out["config"] = {"representation": original["config"]["representation"]}
        else:
            out = {k: original[k] for k in ("model", "contract")}
        path = a.output / (name + ".pt")
        torch.save(safe(out), path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        assert all(
            torch.equal(v, restored["model"][k]) for k, v in original["model"].items()
        )
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "state_dict_exact": True,
        }
    path = a.output / "physical_scale.npz"
    with np.load(a.physical_scale, allow_pickle=False) as scale:
        np.savez(
            path,
            **{
                k: scale[k] for k in ("adapter_mode", "glucose_center", "glucose_scale")
            },
        )
    files[path.name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    (a.output / "manifest.json").write_text(
        json.dumps({"training_data": a.training_data, "files": files}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
