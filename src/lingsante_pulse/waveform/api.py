"""Pretrained feature extraction from arrays and channel metadata."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .model import LingSantePulseEncoder
from .preprocessing import prepare
from .channels import resolve_metadata, unknown_message
from .hub import resolve_model_dir


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LingSantePulse:
    """Load local pretrained weights once, then call :meth:`encode`.

    ``device='auto'`` selects CUDA when available, otherwise CPU. All returned
    tensors are float32 on CPU. ``from_pretrained`` also accepts a Hub repo ID.
    ``encoder`` exposes the underlying PyTorch module for tensor-level use.
    """

    def __init__(self, model_dir, *, device="auto"):
        directory = resolve_model_dir(Path(model_dir), local_files_only=True)
        config = json.loads((directory / "config.json").read_text())
        if (
            config.get("model") != "LingSante-Pulse"
            or config.get("format_version") != 1
        ):
            raise ValueError("Unsupported model package")
        for filename in ("model.safetensors", "channels.npz", "channels.json"):
            if _sha256(directory / filename) != config["files"][filename]:
                raise ValueError(f"Model asset checksum mismatch: {filename}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Preserve the caller's random-number state during module construction.
        with torch.random.fork_rng(devices=[]):
            self.encoder = LingSantePulseEncoder()
        state = load_file(directory / "model.safetensors", device="cpu")
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.eval().requires_grad_(False).to(self.device)
        with np.load(directory / "channels.npz", allow_pickle=False) as table:
            self.channel_names = tuple(table["names"].tolist())
            vectors = table["vectors"].copy()
        if (
            vectors.shape != (len(self.channel_names), 384)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid channel vector table")
        self._vectors = torch.from_numpy(vectors.astype(np.float32)).to(self.device)
        self._indices = {name: i for i, name in enumerate(self.channel_names)}
        self.channel_descriptions = json.loads(
            (directory / "channels.json").read_text()
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path="inclusionAI/LingSante-Pulse-1.0",
        *,
        device="auto",
        revision=None,
        local_files_only=False,
    ):
        """Load the official Hub model by default, or a local model directory.

        Hub downloads use the standard local cache. Pin a full commit hash
        with ``revision`` for reproducibility. ``local_files_only=True`` uses
        only local files or an existing Hub cache. No remote Python is loaded.
        """
        directory = resolve_model_dir(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(directory, device=device)

    def _condition_vectors(self, names):
        unresolved = [name for name in names if name not in self._indices]
        if unresolved:
            raise ValueError(unknown_message(unresolved, self.channel_names))
        return self._vectors[[self._indices[name] for name in names]]

    def _prepare_mapping(self, signals, sample_rate, observed, sample_present, names):
        if not signals:
            raise ValueError("signals cannot be empty")
        if len(names) != len(signals):
            raise ValueError("Provide exactly one metadata entry per dict channel")
        arrays = []
        batch_size, length = None, None
        for (key, values), name in zip(signals.items(), names):
            if hasattr(values, "detach"):
                values = values.detach().cpu().float().numpy()
            values = np.asarray(values)
            if values.ndim == 1:
                values = values[None, None]
            elif values.ndim == 2:
                values = values[:, None]
            else:
                raise ValueError("Dict values must have shape [T] or [B,T]")
            rate = sample_rate[key] if isinstance(sample_rate, Mapping) else sample_rate
            masks = {}
            for mask_name, masks_by_channel in (
                ("observed", observed),
                ("sample_present", sample_present),
            ):
                if masks_by_channel is not None:
                    if (
                        not isinstance(masks_by_channel, Mapping)
                        or key not in masks_by_channel
                    ):
                        raise ValueError(
                            f"{mask_name} must be a dict with all input channel keys"
                        )
                    mask = masks_by_channel[key]
                    if hasattr(mask, "detach"):
                        mask = mask.detach().cpu().numpy()
                    mask = np.asarray(mask)
                    masks[mask_name] = (
                        mask[None, None] if mask.ndim == 1 else mask[:, None]
                    )
            prepared = prepare(values, rate, [name], **masks)
            # Compare duration before patch padding, so unlike windows cannot align accidentally.
            current_length = int(round(values.shape[-1] * 128 / float(rate)))
            if batch_size is None:
                batch_size, length = len(values), current_length
            if len(values) != batch_size or current_length != length:
                raise ValueError(
                    "All dict channels must share batch size and physical duration"
                )
            arrays.append(prepared)
        return names, tuple(
            np.concatenate([a[i] for a in arrays], axis=1) for i in range(3)
        )

    @torch.inference_mode()
    def encode(
        self,
        signals,
        sample_rate=128,
        channel_names=None,
        *,
        observed=None,
        sample_present=None,
        batch_size=16,
        return_tokens=False,
        channels=None,
    ):
        """Return CLS / PatchMean and optionally the ordered patch sequence.

        Arrays: [T], [C,T], [B,C,T]. Dict: channel name -> [T] or [B,T];
        sample_rate may then be a matching dict. Channel order is preserved.
        Global features have shape [B,C*256]. Patch features, when requested,
        have shape [B,P,C*256], with a boolean [B,P,C] observed-valid mask.
        ``channels`` accepts modality and optional lead. Only the fixed
        checkpoint's channel identities and preprocessing are supported.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if isinstance(signals, Mapping):
            if channel_names is not None:
                raise ValueError("Dict input already specifies channel names")
            names = resolve_metadata(
                channels, None if channels is not None else list(signals)
            )
            names, prepared = self._prepare_mapping(
                signals, sample_rate, observed, sample_present, names
            )
        else:
            names = resolve_metadata(channels, channel_names)
            prepared = prepare(
                signals,
                sample_rate,
                names,
                observed=observed,
                sample_present=sample_present,
            )
        x, obs, pm = prepared
        batch, channels, samples = x.shape
        vectors = self._condition_vectors(names)
        # Flatten channels into independent encoder calls, as in the main table.
        vectors = vectors.expand(batch, -1, -1).reshape(batch * channels, 384)
        flat = [array.reshape(batch * channels, samples) for array in (x, obs, pm)]
        cls_rows, mean_rows, token_rows, valid_rows = [], [], [], []
        self.encoder.eval()
        for start in range(0, batch * channels, batch_size):
            stop = start + batch_size
            xx, oo, pp = [torch.from_numpy(a[start:stop]).to(self.device) for a in flat]
            with torch.autocast(device_type=self.device.type, enabled=False):
                tokens = self.encoder(xx, oo, pp, vectors[start:stop])
            patch = tokens[:, self.encoder.num_prefix_tokens :]
            valid = (oo & pp).reshape(len(xx), -1, 32).any(-1)
            mean = (patch * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            cls_rows.append(tokens[:, 0].float().cpu())
            mean_rows.append(mean.float().cpu())
            if return_tokens:
                token_rows.append(
                    torch.where(valid[..., None], patch, 0.0).float().cpu()
                )
                valid_rows.append(valid.cpu())
        output = {
            "cls": torch.cat(cls_rows).reshape(batch, channels * 256),
            "patch_mean": torch.cat(mean_rows).reshape(batch, channels * 256),
            "channel_names": tuple(names),
        }
        if return_tokens:
            patches = samples // 32
            output["patch_tokens"] = (
                torch.cat(token_rows)
                .reshape(batch, channels, patches, 256)
                .permute(0, 2, 1, 3)
                .reshape(batch, patches, channels * 256)
            )
            output["patch_valid"] = (
                torch.cat(valid_rows).reshape(batch, channels, patches).permute(0, 2, 1)
            )
            output["patch_times"] = torch.arange(patches, dtype=torch.float32) * 0.25
        if any(
            isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all())
            for v in output.values()
        ):
            raise RuntimeError("Encoder produced non-finite output")
        return output
