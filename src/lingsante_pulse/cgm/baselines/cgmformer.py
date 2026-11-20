#!/usr/bin/env python3
"""Extract frozen embeddings from the released CGMformer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inputs(
    values: np.ndarray,
    observed_mask: np.ndarray,
    token_dictionary: dict,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        values.shape != observed_mask.shape
        or values.ndim != 2
        or values.shape[1] != 288
    ):
        raise ValueError("expected values and observed_mask with shape [N, 288]")
    if protocol not in {"released_author_code", "paper_faithful"}:
        raise ValueError(f"unsupported protocol: {protocol}")

    pad_id = int(token_dictionary["<pad>"])
    cls_id = int(token_dictionary["<cls>"])
    numeric = {
        int(key): int(value)
        for key, value in token_dictionary.items()
        if isinstance(key, int)
    }
    usable = observed_mask.astype(bool) & np.isfinite(values)
    token_ids = np.full(values.shape, pad_id, dtype=np.int64)
    clipped = np.clip(values[usable], 39, 301).astype(np.int64)
    token_ids[usable] = np.asarray(
        [numeric[int(value)] for value in clipped], dtype=np.int64
    )
    input_ids = np.concatenate(
        [np.full((len(values), 1), cls_id, dtype=np.int64), token_ids], axis=1
    )

    if protocol == "released_author_code":
        # This exactly matches gen_attention_mask() and mean_nonpadding_embs()
        # in the released repository: every one of the 289 positions participates.
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        pooling_mask = np.ones_like(input_ids, dtype=np.float32)
    else:
        attention_mask = np.concatenate(
            [np.ones((len(values), 1), dtype=np.int64), usable.astype(np.int64)], axis=1
        )
        pooling_mask = attention_mask.astype(np.float32)
    return input_ids, attention_mask, pooling_mask


def load_released_model(checkpoint: Path):
    import torch
    from transformers import BertConfig, BertForMaskedLM

    class SinusoidalPositionEmbedding(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, position_ids):
            batch_size, sequence_length = position_ids.shape
            output = torch.zeros(
                batch_size,
                sequence_length,
                self.hidden_size,
                device=position_ids.device,
                dtype=torch.float32,
            )
            frequencies = torch.exp(
                torch.arange(0, self.hidden_size, 2, device=position_ids.device)
                * -(math.log(10000.0) / self.hidden_size)
            )
            positions = torch.arange(
                0, sequence_length, device=position_ids.device, dtype=torch.float32
            ).unsqueeze(1)
            angles = positions * frequencies
            output[:, :, 0::2] = torch.sin(angles).expand(batch_size, -1, -1)
            output[:, :, 1::2] = torch.cos(angles).expand(batch_size, -1, -1)
            return output

    config = BertConfig.from_pretrained(checkpoint)
    config._attn_implementation = "eager"
    config.output_hidden_states = True
    model = BertForMaskedLM(config)
    model.bert.embeddings.position_embeddings = SinusoidalPositionEmbedding(
        config.hidden_size
    )
    state = torch.load(
        checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("released_author_code", "paper_faithful"),
        default="released_author_code",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    with args.token_dictionary.open("rb") as handle:
        token_dictionary = pickle.load(handle)
    with np.load(args.npz, allow_pickle=False) as archive:
        values = archive["glucose_mg_dl"].astype(np.float32)
        observed = archive["observed_mask"].astype(bool)
        metadata = {
            key: archive[key]
            for key in archive.files
            if key not in {"glucose_mg_dl", "observed_mask"}
        }
    input_ids, attention_mask, pooling_mask = build_inputs(
        values, observed, token_dictionary, args.protocol
    )

    # Keep the released repository source in the audit trail.  Inference itself
    # uses the released Transformers checkpoint, as in load_released_model().
    model_file = args.baseline_root / "CGMFormer" / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    model = load_released_model(args.checkpoint).eval().to(args.device)
    embeddings = []
    with torch.no_grad():
        for left in range(0, len(input_ids), args.batch_size):
            right = min(left + args.batch_size, len(input_ids))
            ids = torch.from_numpy(input_ids[left:right]).to(args.device)
            attention = torch.from_numpy(attention_mask[left:right]).to(args.device)
            pool = torch.from_numpy(pooling_mask[left:right]).to(args.device)
            hidden = model(
                input_ids=ids, attention_mask=attention, output_hidden_states=True
            ).hidden_states[-1]
            pooled = (hidden * pool.unsqueeze(-1)).sum(dim=1) / pool.sum(
                dim=1, keepdim=True
            )
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
    embedding = np.concatenate(embeddings, axis=0)
    if embedding.shape != (len(values), 128) or not np.isfinite(embedding).all():
        raise RuntimeError(f"unexpected embedding shape or values: {embedding.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embedding=embedding, **metadata)
    audit = {
        "implementation": "released CGMformer checkpoint; strict standard-BERT state load with released sinusoidal position formula",
        "protocol": args.protocol,
        "input": str(args.npz),
        "windows": int(len(values)),
        "embedding_shape": list(embedding.shape),
        "observed_fraction": float(observed.mean()),
        "checkpoint_sha256": sha256(args.checkpoint / "pytorch_model.bin"),
        "token_dictionary_sha256": sha256(args.token_dictionary),
        "official_source_commit": "715c5055ab4501b4e9eed0f4050b237723bcd468",
        "official_model_source_sha256": sha256(model_file),
        "downstream_label_optimization_reads": 0,
        "private_test_reads": 0,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
