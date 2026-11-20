#!/usr/bin/env python3
"""Train and audit the single CARE-CGM longitudinal evidence canary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score

try:
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        CausalLongitudinalEvidenceEncoder,
        RESPONSIBILITIES,
        causal_history_indices,
        generic_multiscale_features,
        pack_histories,
        standardize_evidence,
        typed_evidence_features,
    )
except ModuleNotFoundError:  # package import in local unit tests
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        CausalLongitudinalEvidenceEncoder,
        RESPONSIBILITIES,
        causal_history_indices,
        generic_multiscale_features,
        pack_histories,
        standardize_evidence,
        typed_evidence_features,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ev(target: np.ndarray, prediction: np.ndarray) -> float:
    scores = [
        explained_variance_score(target[:, index], prediction[:, index])
        for index in range(target.shape[1])
        if float(np.var(target[:, index])) > 1e-10
    ]
    return float(np.mean(scores)) if scores else 0.0


def effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(0), compute_uv=False)
    probability = np.square(singular)
    probability /= max(float(probability.sum()), 1e-12)
    probability = probability[probability > 0]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def ridge_ev(
    code: np.ndarray,
    target: np.ndarray,
    fit_rows: np.ndarray,
    audit_rows: np.ndarray,
) -> float:
    center = code[fit_rows].mean(0)
    scale = np.maximum(code[fit_rows].std(0), 1e-6)
    model = Ridge(alpha=max(len(fit_rows), 1), fit_intercept=True)
    model.fit((code[fit_rows] - center) / scale, target[fit_rows])
    prediction = model.predict((code[audit_rows] - center) / scale)
    return mean_ev(target[audit_rows], prediction)


def _align_daily_embedding(
    daily_path: Path, source_window_id: np.ndarray
) -> np.ndarray:
    with np.load(daily_path, allow_pickle=False) as archive:
        daily = archive["embedding"].astype(np.float32)
        window_id = archive["window_id"].astype(str)
        label_keys = [key for key in archive.files if key.startswith("label_")]
    if label_keys:
        # Passthrough labels may exist in an extractor artifact, but this trainer
        # refuses to read or align them.
        label_keys.clear()
    source_id = np.asarray(source_window_id).astype(str)
    if len(np.unique(window_id)) != len(window_id):
        raise ValueError("daily embedding window_id values are not unique")
    lookup = {value: index for index, value in enumerate(window_id)}
    if set(lookup) != set(source_id):
        raise ValueError("daily embedding and source window sets differ")
    aligned = daily[np.asarray([lookup[value] for value in source_id], dtype=np.int64)]
    if aligned.shape != (len(source_id), 146) or not np.isfinite(aligned).all():
        raise ValueError("expected a finite frozen CARE146 matrix")
    return aligned


def _to_device(
    arrays: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}


def _batch_forward(
    model: CausalLongitudinalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    rows: torch.Tensor,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    return model(
        daily[rows],
        packed["features"][rows],
        packed["support"][rows],
        packed["elapsed_days"][rows],
        packed["history_mask"][rows],
    )


def _dispersion_loss(values: torch.Tensor) -> torch.Tensor:
    flattened = values.flatten(1)
    centered = flattened - flattened.mean(0, keepdim=True)
    deviation = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = torch.relu(0.5 - deviation).mean()
    covariance = centered.T @ centered / max(len(centered), 1)
    covariance = covariance - torch.diag(torch.diag(covariance))
    return variance + covariance.square().mean()


def _loss(
    output: dict[str, torch.Tensor | list[torch.Tensor]],
    daily_target: torch.Tensor,
    evidence_target: torch.Tensor,
    target_support: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    prediction = output["typed_predictions"]
    squared = torch.square(prediction - evidence_target).mean(dim=2)
    evidence = torch.sum(squared * target_support) / torch.clamp(
        target_support.sum(), min=1.0
    )
    daily = torch.mean(torch.square(output["long_prediction"] - daily_target))
    dispersion = _dispersion_loss(output["memory_codes"])
    weights = config["training"]["loss_weights"]
    return (
        float(weights["current_care_prediction"]) * daily
        + float(weights["responsibility_prediction"]) * evidence
        + float(weights["memory_dispersion"]) * dispersion
    )


def _validation_loss(
    model: CausalLongitudinalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_support: torch.Tensor,
    rows: np.ndarray,
    config: dict,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    batch_size = int(config["training"]["evaluation_batch_size"])
    with torch.no_grad():
        for left in range(0, len(rows), batch_size):
            batch = torch.as_tensor(rows[left : left + batch_size], device=daily.device)
            output = _batch_forward(model, daily, packed, batch)
            loss = _loss(
                output, daily[batch], target[batch], target_support[batch], config
            )
            total += float(loss) * len(batch)
            count += len(batch)
    return total / max(count, 1)


def _train(
    model: CausalLongitudinalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_support: torch.Tensor,
    fit_rows: np.ndarray,
    audit_rows: np.ndarray,
    config: dict,
    *,
    seed_offset: int,
) -> list[dict[str, float | int]]:
    settings = config["training"]
    seed = int(settings["seed"]) + seed_offset
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = None
    history = []
    for step in range(1, int(settings["canary_optimizer_steps"]) + 1):
        rows = torch.as_tensor(
            rng.choice(fit_rows, size=int(settings["batch_size"]), replace=True),
            device=daily.device,
        )
        model.train()
        output = _batch_forward(model, daily, packed, rows)
        loss = _loss(output, daily[rows], target[rows], target_support[rows], config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(settings["checkpoint_interval"]) == 0:
            audit_loss = _validation_loss(
                model, daily, packed, target, target_support, audit_rows, config
            )
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "audit_loss": audit_loss,
                }
            )
            if audit_loss < best_loss:
                best_loss = audit_loss
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no label-free checkpoint was selected")
    model.load_state_dict(best_state)
    return history


def _extract(
    model: CausalLongitudinalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for left in range(0, len(daily), batch_size):
            rows = torch.arange(
                left, min(left + batch_size, len(daily)), device=daily.device
            )
            output = _batch_forward(model, daily, packed, rows)
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    collected.setdefault(key, []).append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def _permuted_loss(
    model: CausalLongitudinalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_support: torch.Tensor,
    rows: np.ndarray,
    config: dict,
    *,
    reverse: bool,
) -> float:
    selected = torch.as_tensor(rows, device=daily.device)
    modified = {key: value.clone() for key, value in packed.items()}
    if reverse:
        for row in selected.tolist():
            valid = torch.nonzero(packed["history_mask"][row], as_tuple=False).flatten()
            modified["features"][row, valid] = packed["features"][row, valid.flip(0)]
            modified["support"][row, valid] = packed["support"][row, valid.flip(0)]
    else:
        permutation = selected.roll(1)
        for key in ("features", "support", "elapsed_days", "history_mask"):
            modified[key][selected] = packed[key][permutation]
    return _validation_loss(
        model, daily, modified, target, target_support, rows, config
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--daily-embedding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sha256(args.source) != config["data"]["corpus_sha256"]:
        raise ValueError("corpus SHA mismatch")

    with np.load(args.source, allow_pickle=False) as archive:
        required = {
            "glucose_mg_dl",
            "observed_mask",
            "subject_id",
            "cohort",
            "native_cadence_min",
            "shifted_window_start",
            "window_id",
            "split",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing source fields: {sorted(missing)}")
        glucose = archive["glucose_mg_dl"].astype(np.float32)
        observed = archive["observed_mask"].astype(bool)
        subject = archive["subject_id"].astype(str)
        cohort = archive["cohort"].astype(str)
        cadence = archive["native_cadence_min"].astype(np.int16)
        starts = archive["shifted_window_start"].astype("datetime64[ns]")
        split = archive["split"].astype(np.int8)
        window_id = archive["window_id"].astype(str)
        source = (
            archive["device_source_code"].astype(np.int16)
            if "device_source_code" in archive.files
            else np.full(len(split), -1, dtype=np.int16)
        )
        label_keys_present = sorted(
            key for key in archive.files if key.startswith("label_")
        )
    if label_keys_present:
        raise ValueError("train/validation pretraining corpus must not contain labels")
    daily_raw = _align_daily_embedding(args.daily_embedding, window_id)
    fit_mask = split == int(config["data"]["fit_split"])
    audit_mask = split == int(config["data"]["audit_split"])
    daily_center = daily_raw[fit_mask].mean(0, keepdims=True)
    daily_scale = np.maximum(daily_raw[fit_mask].std(0, keepdims=True), 1e-5)
    daily = ((daily_raw - daily_center) / daily_scale).astype(np.float32)

    histories = causal_history_indices(
        subject,
        starts,
        cohort,
        split,
        cadence,
        source,
        max_days=int(config["method"]["maximum_days"]),
    )
    typed_raw, typed_support = typed_evidence_features(
        glucose, observed, cadence, starts
    )
    generic_raw, generic_support = generic_multiscale_features(glucose, observed)
    typed, typed_contract = standardize_evidence(typed_raw, np.flatnonzero(fit_mask))
    generic, generic_contract = standardize_evidence(
        generic_raw, np.flatnonzero(fit_mask)
    )
    candidate_pack_np = pack_histories(
        typed,
        typed_support,
        starts,
        histories,
        max_prior_days=int(config["method"]["maximum_days"]) - 1,
    )
    control_pack_np = pack_histories(
        generic,
        generic_support,
        starts,
        histories,
        max_prior_days=int(config["method"]["maximum_days"]) - 1,
    )
    history_count = candidate_pack_np["history_mask"].sum(axis=1)
    fit_rows = np.flatnonzero(fit_mask & (history_count > 0))
    audit_rows = np.flatnonzero(audit_mask & (history_count > 0))
    deep_audit_rows = np.flatnonzero(audit_mask & (history_count >= 2))
    if not len(deep_audit_rows):
        raise ValueError("reversal audit requires at least two prior days")

    device = torch.device(args.device)
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    daily_tensor = torch.from_numpy(daily).to(device)
    typed_tensor = torch.from_numpy(typed).to(device)
    typed_support_tensor = torch.from_numpy(typed_support).to(device)
    generic_tensor = torch.from_numpy(generic).to(device)
    generic_support_tensor = torch.from_numpy(generic_support).to(device)
    candidate_pack = _to_device(candidate_pack_np, device)
    control_pack = _to_device(control_pack_np, device)
    model_config = config["method"]["model"]
    candidate = CausalLongitudinalEvidenceEncoder(**model_config).to(device)
    candidate_history = _train(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_support_tensor,
        fit_rows,
        audit_rows,
        config,
        seed_offset=0,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    control = CausalLongitudinalEvidenceEncoder(**model_config).to(device)
    control_history = _train(
        control,
        daily_tensor,
        control_pack,
        generic_tensor,
        generic_support_tensor,
        fit_rows,
        audit_rows,
        config,
        seed_offset=100003,
    )

    batch_size = int(config["training"]["evaluation_batch_size"])
    candidate_output = _extract(candidate, daily_tensor, candidate_pack, batch_size)
    control_output = _extract(control, daily_tensor, control_pack, batch_size)
    candidate_memory = candidate_output["memory_codes"].reshape(len(daily), -1)
    control_memory = control_output["memory_codes"].reshape(len(daily), -1)
    branch_ev = {
        name: mean_ev(
            typed[audit_rows, index],
            candidate_output["typed_predictions"][audit_rows, index],
        )
        for index, name in enumerate(RESPONSIBILITIES)
    }
    correct_loss = _validation_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_support_tensor,
        audit_rows,
        config,
    )
    shuffled_loss = _permuted_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_support_tensor,
        audit_rows,
        config,
        reverse=False,
    )
    reversed_loss = _permuted_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_support_tensor,
        deep_audit_rows,
        config,
        reverse=True,
    )
    correct_deep_loss = _validation_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_support_tensor,
        deep_audit_rows,
        config,
    )
    no_history = np.flatnonzero(history_count == 0)
    identity_error = float(
        np.max(np.abs(candidate_output["embedding"][no_history] - daily[no_history]))
    )
    candidate_parameters = sum(value.numel() for value in candidate.parameters())
    control_parameters = sum(value.numel() for value in control.parameters())
    metrics = {
        "fit_longitudinal_endpoints": int(len(fit_rows)),
        "audit_longitudinal_endpoints": int(len(audit_rows)),
        "deep_audit_endpoints": int(len(deep_audit_rows)),
        "candidate_parameters": candidate_parameters,
        "control_parameters": control_parameters,
        "candidate_branch_target_ev": branch_ev,
        "candidate_memory_typed_ev": ridge_ev(
            candidate_memory, typed.reshape(len(typed), -1), fit_rows, audit_rows
        ),
        "control_memory_typed_ev": ridge_ev(
            control_memory, typed.reshape(len(typed), -1), fit_rows, audit_rows
        ),
        "candidate_current_care_prediction_ev": mean_ev(
            daily[audit_rows], candidate_output["long_prediction"][audit_rows]
        ),
        "control_current_care_prediction_ev": mean_ev(
            daily[audit_rows], control_output["long_prediction"][audit_rows]
        ),
        "candidate_signed_low_high_memory_ev": ridge_ev(
            candidate_output["memory_codes"][:, 2],
            typed[:, 2, [0, 1, 2, 3, 8, 9]],
            fit_rows,
            audit_rows,
        ),
        "history_shuffle_loss_ratio": shuffled_loss / max(correct_loss, 1e-12),
        "history_reversal_loss_ratio": reversed_loss / max(correct_deep_loss, 1e-12),
        "daily_effective_rank": effective_rank(daily[audit_rows]),
        "candidate_effective_rank": effective_rank(
            candidate_output["embedding"][audit_rows]
        ),
        "control_effective_rank": effective_rank(
            control_output["embedding"][audit_rows]
        ),
        "no_history_identity_max_abs": identity_error,
    }
    metrics["candidate_memory_typed_advantage"] = (
        metrics["candidate_memory_typed_ev"] - metrics["control_memory_typed_ev"]
    )
    gates = config["canary_gates"]
    gate_results = {
        "support": len(fit_rows) >= int(gates["fit_endpoints_min"])
        and len(audit_rows) >= int(gates["audit_endpoints_min"]),
        "all_responsibilities_positive": all(
            value >= float(gates["each_responsibility_ev_min"])
            for value in branch_ev.values()
        ),
        "typed_advantage": metrics["candidate_memory_typed_advantage"]
        >= float(gates["typed_memory_advantage_min"]),
        "current_care_prediction": metrics["candidate_current_care_prediction_ev"]
        >= float(gates["current_care_prediction_ev_min"]),
        "history_shuffle": metrics["history_shuffle_loss_ratio"]
        >= float(gates["shuffle_loss_ratio_min"]),
        "history_reversal": metrics["history_reversal_loss_ratio"]
        >= float(gates["reversal_loss_ratio_min"]),
        "signed_extremes": metrics["candidate_signed_low_high_memory_ev"]
        >= float(gates["signed_low_high_ev_min"]),
        "rank": metrics["candidate_effective_rank"]
        >= float(gates["daily_rank_retention_min"]) * metrics["daily_effective_rank"],
        "identity": identity_error <= float(gates["identity_max_abs"]),
        "equal_budget": candidate_parameters == control_parameters,
        "isolation": all(int(value) == 0 for value in config["isolation"].values()),
    }
    accepted = all(gate_results.values())
    args.output_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_root / "checkpoint.pt"
    embeddings_path = args.output_root / "embeddings.npz"
    report_path = args.output_root / "mechanism_report.json"
    torch.save(
        {
            "model": candidate.state_dict(),
            "model_config": model_config,
            "daily_center": daily_center,
            "daily_scale": daily_scale,
            "typed_contract": typed_contract,
            "generic_contract": generic_contract,
            "corpus_sha256": config["data"]["corpus_sha256"],
            "daily_embedding_sha256": sha256(args.daily_embedding),
            "glucofm_framework_or_components": False,
        },
        checkpoint_path,
    )
    candidate_raw = (candidate_output["embedding"] * daily_scale + daily_center).astype(
        np.float32
    )
    candidate_raw[history_count == 0] = daily_raw[history_count == 0]
    control_raw = (control_output["embedding"] * daily_scale + daily_center).astype(
        np.float32
    )
    control_raw[history_count == 0] = daily_raw[history_count == 0]
    np.savez_compressed(
        embeddings_path,
        embedding=candidate_raw,
        control_embedding=control_raw,
        history_days=(history_count + 1).astype(np.int8),
        split=split,
        window_id=window_id,
    )
    report = {
        "schema_version": 1,
        "candidate": config["candidate"],
        "acceptance_passed": accepted,
        "config_sha256": sha256(args.config),
        "corpus_sha256": sha256(args.source),
        "daily_embedding_sha256": sha256(args.daily_embedding),
        **metrics,
        "gate_results": gate_results,
        "candidate_history": candidate_history,
        "control_history": control_history,
        "isolation": config["isolation"],
        "glucofm_role": "evaluation_only_baseline",
        "next_action": config["if_canary_passes"]
        if accepted
        else config["if_canary_fails"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest_path = args.output_root / "artifacts.sha256"
    manifest_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in (checkpoint_path, embeddings_path, report_path)
        )
        + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
