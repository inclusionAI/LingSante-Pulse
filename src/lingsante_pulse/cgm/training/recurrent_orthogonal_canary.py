#!/usr/bin/env python3
"""Train/audit the recurrent orthogonal CARE-CGM longitudinal canary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

try:
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        RESPONSIBILITIES,
        causal_history_indices,
        generic_multiscale_features,
        pack_histories,
        standardize_evidence,
        typed_evidence_features,
    )
    from lingsante_pulse.cgm.models.recurrent_orthogonal_evidence import (
        RecurrentOrthogonalEvidenceEncoder,
    )
    from lingsante_pulse.cgm.models.covariance_preserving_innovation import (
        calibrate_innovation_gain,
        fit_covariance_transport,
        transport_state,
    )
    from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
        _align_daily_embedding,
        _dispersion_loss,
        _to_device,
        effective_rank,
        mean_ev,
        ridge_ev,
        sha256,
    )
except ModuleNotFoundError:  # package import in local unit tests
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        RESPONSIBILITIES,
        causal_history_indices,
        generic_multiscale_features,
        pack_histories,
        standardize_evidence,
        typed_evidence_features,
    )
    from lingsante_pulse.cgm.models.recurrent_orthogonal_evidence import (
        RecurrentOrthogonalEvidenceEncoder,
    )
    from lingsante_pulse.cgm.models.covariance_preserving_innovation import (
        calibrate_innovation_gain,
        fit_covariance_transport,
        transport_state,
    )
    from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
        _align_daily_embedding,
        _dispersion_loss,
        _to_device,
        effective_rank,
        mean_ev,
        ridge_ev,
        sha256,
    )


def principal_basis(values: np.ndarray, explained_fraction: float) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    cumulative = np.cumsum(np.square(singular)) / np.sum(np.square(singular))
    components = int(np.searchsorted(cumulative, explained_fraction) + 1)
    return right[:components].T.astype(np.float32)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(0, keepdims=True)
    right = right - right.mean(0, keepdims=True)
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    return float(numerator / max(float(denominator), 1e-12))


def batch_forward(
    model: RecurrentOrthogonalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return model(
        daily[rows],
        packed["features"][rows],
        packed["support"][rows],
        packed["elapsed_days"][rows],
        packed["history_mask"][rows],
    )


def loss_for(
    output: dict[str, torch.Tensor],
    daily_target: torch.Tensor,
    evidence_target: torch.Tensor,
    transition_target: torch.Tensor,
    target_support: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    evidence_error = torch.square(output["typed_predictions"] - evidence_target).mean(
        dim=2
    )
    transition_error = torch.square(
        output["transition_predictions"] - transition_target
    ).mean(dim=2)
    denominator = torch.clamp(target_support.sum(), min=1.0)
    weights = config["training"]["loss_weights"]
    return (
        float(weights["current_care_prediction"])
        * torch.mean(torch.square(output["long_prediction"] - daily_target))
        + float(weights["responsibility_prediction"])
        * torch.sum(evidence_error * target_support)
        / denominator
        + float(weights["signed_transition_prediction"])
        * torch.sum(transition_error * target_support)
        / denominator
        + float(weights["memory_dispersion"]) * _dispersion_loss(output["memory_codes"])
    )


def validation_loss(
    model: RecurrentOrthogonalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    target: torch.Tensor,
    transition_target: torch.Tensor,
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
            output = batch_forward(model, daily, packed, batch)
            loss = loss_for(
                output,
                daily[batch],
                target[batch],
                transition_target[batch],
                target_support[batch],
                config,
            )
            total += float(loss) * len(batch)
            count += len(batch)
    return total / max(count, 1)


def train_variant(
    model: RecurrentOrthogonalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    target: torch.Tensor,
    transition_target: torch.Tensor,
    target_support: torch.Tensor,
    fit_rows: np.ndarray,
    audit_rows: np.ndarray,
    config: dict,
    *,
    optimizer_steps: int | None = None,
) -> list[dict[str, float | int]]:
    settings = config["training"]
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = None
    history = []
    steps = int(
        settings["canary_optimizer_steps"]
        if optimizer_steps is None
        else optimizer_steps
    )
    for step in range(1, steps + 1):
        rows = torch.as_tensor(
            rng.choice(fit_rows, size=int(settings["batch_size"]), replace=True),
            device=daily.device,
        )
        model.train()
        output = batch_forward(model, daily, packed, rows)
        loss = loss_for(
            output,
            daily[rows],
            target[rows],
            transition_target[rows],
            target_support[rows],
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % int(settings["checkpoint_interval"]) == 0:
            audit_loss = validation_loss(
                model,
                daily,
                packed,
                target,
                transition_target,
                target_support,
                audit_rows,
                config,
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
        raise RuntimeError("no label-free checkpoint selected")
    model.load_state_dict(best_state)
    return history


def extract(
    model: RecurrentOrthogonalEvidenceEncoder,
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
            output = batch_forward(model, daily, packed, rows)
            for key, value in output.items():
                collected.setdefault(key, []).append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def calibrate_gain(
    model: RecurrentOrthogonalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    fit_rows: np.ndarray,
    energy_ratio: float,
    batch_size: int,
) -> float:
    output = extract(model, daily, packed, batch_size)
    residual_rms = float(
        np.sqrt(np.mean(np.square(output["orthogonal_residual"][fit_rows])))
    )
    daily_rms = float(np.sqrt(np.mean(np.square(daily[fit_rows].cpu().numpy()))))
    if residual_rms <= 1e-8:
        raise ValueError("cannot calibrate a collapsed residual")
    gain = energy_ratio * daily_rms / residual_rms
    model.set_residual_gain(gain)
    return gain


def perturbed_loss(
    model: RecurrentOrthogonalEvidenceEncoder,
    daily: torch.Tensor,
    packed: dict[str, torch.Tensor],
    target: torch.Tensor,
    transition_target: torch.Tensor,
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
    return validation_loss(
        model,
        daily,
        modified,
        target,
        transition_target,
        target_support,
        rows,
        config,
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
        if any(key.startswith("label_") for key in archive.files):
            raise ValueError("pretraining corpus must not contain labels")
    daily_raw = _align_daily_embedding(args.daily_embedding, window_id)
    fit_mask = split == int(config["data"]["fit_split"])
    audit_mask = split == int(config["data"]["audit_split"])
    fit_all = np.flatnonzero(fit_mask)
    daily_center = daily_raw[fit_mask].mean(0, keepdims=True)
    daily_scale = np.maximum(daily_raw[fit_mask].std(0, keepdims=True), 1e-5)
    daily = ((daily_raw - daily_center) / daily_scale).astype(np.float32)
    basis = principal_basis(
        daily[fit_mask], float(config["method"]["daily_subspace_explained_fraction"])
    )
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
    typed, typed_contract = standardize_evidence(typed_raw, fit_all)
    generic, generic_contract = standardize_evidence(generic_raw, fit_all)
    maximum_prior = int(config["method"]["maximum_days"]) - 1
    candidate_pack_np = pack_histories(
        typed, typed_support, starts, histories, max_prior_days=maximum_prior
    )
    control_pack_np = pack_histories(
        generic, generic_support, starts, histories, max_prior_days=maximum_prior
    )
    history_count = candidate_pack_np["history_mask"].sum(1)
    fit_rows = np.flatnonzero(fit_mask & (history_count > 0))
    audit_rows = np.flatnonzero(audit_mask & (history_count > 0))
    deep_audit_rows = np.flatnonzero(audit_mask & (history_count >= 2))
    typed_transition = typed - candidate_pack_np["features"][:, -1]
    generic_transition = generic - control_pack_np["features"][:, -1]

    device = torch.device(args.device)
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    daily_tensor = torch.from_numpy(daily).to(device)
    typed_tensor = torch.from_numpy(typed).to(device)
    typed_transition_tensor = torch.from_numpy(typed_transition).to(device)
    typed_support_tensor = torch.from_numpy(typed_support).to(device)
    generic_tensor = torch.from_numpy(generic).to(device)
    generic_transition_tensor = torch.from_numpy(generic_transition).to(device)
    generic_support_tensor = torch.from_numpy(generic_support).to(device)
    candidate_pack = _to_device(candidate_pack_np, device)
    control_pack = _to_device(control_pack_np, device)
    model_config = config["method"]["model"]
    candidate = RecurrentOrthogonalEvidenceEncoder(
        torch.from_numpy(basis), **model_config
    ).to(device)
    candidate_history = train_variant(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_transition_tensor,
        typed_support_tensor,
        fit_rows,
        audit_rows,
        config,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    control = RecurrentOrthogonalEvidenceEncoder(
        torch.from_numpy(basis), **model_config
    ).to(device)
    control_history = train_variant(
        control,
        daily_tensor,
        control_pack,
        generic_tensor,
        generic_transition_tensor,
        generic_support_tensor,
        fit_rows,
        audit_rows,
        config,
    )
    batch_size = int(config["training"]["evaluation_batch_size"])
    energy_ratio = float(config["method"]["residual_energy_ratio"])
    candidate_output = extract(candidate, daily_tensor, candidate_pack, batch_size)
    control_output = extract(control, daily_tensor, control_pack, batch_size)
    candidate_memory = candidate_output["memory_codes"].reshape(len(daily), -1)
    control_memory = control_output["memory_codes"].reshape(len(daily), -1)
    output_contract = config["method"].get("output_contract", "pointwise_orthogonal")
    candidate_transport_state = None
    control_transport_state = None
    transport_diagnostics = None
    control_transport_diagnostics = None
    if output_contract == "covariance_preserving_innovation":
        candidate_transport, transport_diagnostics = fit_covariance_transport(
            daily, candidate_memory, fit_rows
        )
        control_transport, control_transport_diagnostics = fit_covariance_transport(
            daily, control_memory, fit_rows
        )
        valid_history = history_count > 0
        candidate_fit_innovation = candidate_transport.innovation(
            daily[fit_rows], candidate_memory[fit_rows]
        )
        control_fit_innovation = control_transport.innovation(
            daily[fit_rows], control_memory[fit_rows]
        )
        candidate_gain = calibrate_innovation_gain(
            daily[fit_rows], candidate_fit_innovation, energy_ratio
        )
        control_gain = calibrate_innovation_gain(
            daily[fit_rows], control_fit_innovation, energy_ratio
        )
        candidate_embedding = candidate_transport.embedding(
            daily,
            candidate_memory,
            valid_history,
            innovation_ratio=candidate_gain,
        )
        control_embedding = control_transport.embedding(
            daily,
            control_memory,
            valid_history,
            innovation_ratio=control_gain,
        )
        candidate_residual = candidate_fit_innovation * candidate_gain
        candidate_transport_state = transport_state(candidate_transport)
        control_transport_state = transport_state(control_transport)
        orthogonality = None
    elif output_contract == "pointwise_orthogonal":
        candidate_gain = calibrate_gain(
            candidate, daily_tensor, candidate_pack, fit_rows, energy_ratio, batch_size
        )
        control_gain = calibrate_gain(
            control, daily_tensor, control_pack, fit_rows, energy_ratio, batch_size
        )
        candidate_output = extract(candidate, daily_tensor, candidate_pack, batch_size)
        control_output = extract(control, daily_tensor, control_pack, batch_size)
        candidate_memory = candidate_output["memory_codes"].reshape(len(daily), -1)
        control_memory = control_output["memory_codes"].reshape(len(daily), -1)
        candidate_embedding = candidate_output["embedding"]
        control_embedding = control_output["embedding"]
        candidate_residual = (
            candidate_output["orthogonal_residual"][fit_rows] * candidate_gain
        )
        orthogonality = float(
            np.max(np.abs(candidate_output["orthogonal_residual"] @ basis))
        )
    else:
        raise ValueError(f"unknown output contract: {output_contract}")
    branch_ev = {
        name: mean_ev(
            typed[audit_rows, index],
            candidate_output["typed_predictions"][audit_rows, index],
        )
        for index, name in enumerate(RESPONSIBILITIES)
    }
    correct_loss = validation_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_transition_tensor,
        typed_support_tensor,
        audit_rows,
        config,
    )
    shuffled_loss = perturbed_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_transition_tensor,
        typed_support_tensor,
        audit_rows,
        config,
        reverse=False,
    )
    correct_deep_loss = validation_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_transition_tensor,
        typed_support_tensor,
        deep_audit_rows,
        config,
    )
    reversed_loss = perturbed_loss(
        candidate,
        daily_tensor,
        candidate_pack,
        typed_tensor,
        typed_transition_tensor,
        typed_support_tensor,
        deep_audit_rows,
        config,
        reverse=True,
    )
    measured_energy_ratio = float(
        np.sqrt(np.mean(np.square(candidate_residual)))
        / np.sqrt(np.mean(np.square(daily[fit_rows])))
    )
    no_history = np.flatnonzero(history_count == 0)
    identity_error = float(
        np.max(np.abs(candidate_embedding[no_history] - daily[no_history]))
    )
    candidate_parameters = sum(value.numel() for value in candidate.parameters())
    control_parameters = sum(value.numel() for value in control.parameters())
    metrics = {
        "fit_longitudinal_endpoints": int(len(fit_rows)),
        "audit_longitudinal_endpoints": int(len(audit_rows)),
        "deep_audit_endpoints": int(len(deep_audit_rows)),
        "daily_subspace_components": int(basis.shape[1]),
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
        "candidate_signed_low_high_memory_ev": ridge_ev(
            candidate_output["memory_codes"][:, 2],
            typed[:, 2, [0, 1, 2, 3, 8, 9]],
            fit_rows,
            audit_rows,
        ),
        "history_shuffle_loss_ratio": shuffled_loss / max(correct_loss, 1e-12),
        "history_reversal_loss_ratio": reversed_loss / max(correct_deep_loss, 1e-12),
        "daily_effective_rank": effective_rank(daily[audit_rows]),
        "candidate_effective_rank": effective_rank(candidate_embedding[audit_rows]),
        "control_effective_rank": effective_rank(control_embedding[audit_rows]),
        "daily_candidate_linear_cka": linear_cka(
            daily[audit_rows], candidate_embedding[audit_rows]
        ),
        "output_contract": output_contract,
        "candidate_residual_gain": candidate_gain,
        "control_residual_gain": control_gain,
        "measured_residual_energy_ratio": measured_energy_ratio,
        "no_history_identity_max_abs": identity_error,
    }
    if orthogonality is not None:
        metrics["orthogonal_residual_basis_overlap_max_abs"] = orthogonality
    if transport_diagnostics is not None:
        metrics["covariance_transport"] = transport_diagnostics
        metrics["control_covariance_transport"] = control_transport_diagnostics
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
        "daily_geometry": metrics["daily_candidate_linear_cka"]
        >= float(gates["daily_candidate_cka_min"]),
        "energy": abs(measured_energy_ratio - energy_ratio)
        <= float(gates["residual_energy_ratio_tolerance"]),
        "identity": identity_error <= float(gates["identity_max_abs"]),
        "equal_budget": candidate_parameters == control_parameters,
        "isolation": all(int(value) == 0 for value in config["isolation"].values()),
    }
    if output_contract == "pointwise_orthogonal":
        gate_results["orthogonality"] = orthogonality <= float(
            gates["orthogonality_max_abs"]
        )
    else:
        gate_results["fit_covariance_shape"] = transport_diagnostics[
            "fit_covariance_relative_frobenius_error"
        ] <= float(gates["fit_covariance_relative_error_max"])
        gate_results["fit_daily_innovation_independence"] = transport_diagnostics[
            "fit_daily_innovation_cross_covariance_relative_frobenius"
        ] <= float(gates["fit_cross_covariance_relative_error_max"])
        gate_results["memory_support"] = transport_diagnostics[
            "retained_memory_singular_ratio"
        ] >= float(gates["retained_memory_singular_ratio_min"])
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
            "daily_basis": basis,
            "output_contract": output_contract,
            "candidate_transport": candidate_transport_state,
            "control_transport": control_transport_state,
            "typed_contract": typed_contract,
            "generic_contract": generic_contract,
            "corpus_sha256": config["data"]["corpus_sha256"],
            "daily_embedding_sha256": sha256(args.daily_embedding),
            "glucofm_framework_or_components": False,
        },
        checkpoint_path,
    )
    candidate_raw = (candidate_embedding * daily_scale + daily_center).astype(
        np.float32
    )
    control_raw = (control_embedding * daily_scale + daily_center).astype(np.float32)
    candidate_raw[history_count == 0] = daily_raw[history_count == 0]
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
