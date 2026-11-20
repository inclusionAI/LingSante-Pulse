#!/usr/bin/env python3
"""Run the single frozen 6,960-step CARE-CGM longitudinal candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.models.covariance_preserving_innovation import (
    calibrate_innovation_gain,
    fit_covariance_transport,
    transport_state,
)
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    RESPONSIBILITIES,
    causal_history_indices,
    pack_histories,
    standardize_evidence,
    typed_evidence_features,
)
from lingsante_pulse.cgm.models.recurrent_orthogonal_evidence import (
    RecurrentOrthogonalEvidenceEncoder,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
    _align_daily_embedding,
    _to_device,
    effective_rank,
    mean_ev,
    ridge_ev,
    sha256,
)
from lingsante_pulse.cgm.training.recurrent_orthogonal_canary import (
    extract,
    linear_cka,
    perturbed_loss,
    principal_basis,
    train_variant,
    validation_loss,
)


def _load_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if any(key.startswith("label_") for key in archive.files):
            raise ValueError("pretraining corpus must not contain labels")
        return {
            "glucose": archive["glucose_mg_dl"].astype(np.float32),
            "observed": archive["observed_mask"].astype(bool),
            "subject": archive["subject_id"].astype(str),
            "cohort": archive["cohort"].astype(str),
            "cadence": archive["native_cadence_min"].astype(np.int16),
            "starts": archive["shifted_window_start"].astype("datetime64[ns]"),
            "split": archive["split"].astype(np.int8),
            "window_id": archive["window_id"].astype(str),
            "source": (
                archive["device_source_code"].astype(np.int16)
                if "device_source_code" in archive.files
                else np.full(len(archive["split"]), -1, dtype=np.int16)
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--daily-embedding", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    canary = json.loads(args.canary_report.read_text())
    if not canary.get("acceptance_passed") or not all(canary["gate_results"].values()):
        raise ValueError("full run requires the exact passing canary")
    if sha256(args.source) != config["data"]["corpus_sha256"]:
        raise ValueError("corpus SHA mismatch")
    if sha256(args.daily_embedding) != config["data"]["daily_embedding_sha256"]:
        raise ValueError("daily embedding SHA mismatch")
    if config["method"]["output_contract"] != "covariance_preserving_innovation":
        raise ValueError("unexpected output contract")

    data = _load_source(args.source)
    daily_raw = _align_daily_embedding(args.daily_embedding, data["window_id"])
    fit_mask = data["split"] == int(config["data"]["fit_split"])
    audit_mask = data["split"] == int(config["data"]["audit_split"])
    fit_all = np.flatnonzero(fit_mask)
    daily_center = daily_raw[fit_mask].mean(0, keepdims=True)
    daily_scale = np.maximum(daily_raw[fit_mask].std(0, keepdims=True), 1e-5)
    daily = ((daily_raw - daily_center) / daily_scale).astype(np.float32)
    basis = principal_basis(
        daily[fit_mask], float(config["method"]["daily_subspace_explained_fraction"])
    )
    histories = causal_history_indices(
        data["subject"],
        data["starts"],
        data["cohort"],
        data["split"],
        data["cadence"],
        data["source"],
        max_days=int(config["method"]["maximum_days"]),
    )
    typed_raw, typed_support = typed_evidence_features(
        data["glucose"], data["observed"], data["cadence"], data["starts"]
    )
    typed, typed_contract = standardize_evidence(typed_raw, fit_all)
    packed_np = pack_histories(
        typed,
        typed_support,
        data["starts"],
        histories,
        max_prior_days=int(config["method"]["maximum_days"]) - 1,
    )
    history_count = packed_np["history_mask"].sum(1)
    fit_rows = np.flatnonzero(fit_mask & (history_count > 0))
    audit_rows = np.flatnonzero(audit_mask & (history_count > 0))
    deep_audit_rows = np.flatnonzero(audit_mask & (history_count >= 2))
    transition = typed - packed_np["features"][:, -1]

    device = torch.device(args.device)
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    daily_tensor = torch.from_numpy(daily).to(device)
    typed_tensor = torch.from_numpy(typed).to(device)
    transition_tensor = torch.from_numpy(transition).to(device)
    support_tensor = torch.from_numpy(typed_support).to(device)
    packed = _to_device(packed_np, device)
    model = RecurrentOrthogonalEvidenceEncoder(
        torch.from_numpy(basis), **config["method"]["model"]
    ).to(device)
    steps = int(config["training"]["full_optimizer_steps_if_canary_passes"])
    history = train_variant(
        model,
        daily_tensor,
        packed,
        typed_tensor,
        transition_tensor,
        support_tensor,
        fit_rows,
        audit_rows,
        config,
        optimizer_steps=steps,
    )
    batch_size = int(config["training"]["evaluation_batch_size"])
    output = extract(model, daily_tensor, packed, batch_size)
    memory = output["memory_codes"].reshape(len(daily), -1)
    transport, diagnostics = fit_covariance_transport(daily, memory, fit_rows)
    fit_innovation = transport.innovation(daily[fit_rows], memory[fit_rows])
    gain = calibrate_innovation_gain(
        daily[fit_rows],
        fit_innovation,
        float(config["method"]["residual_energy_ratio"]),
    )
    valid_history = history_count > 0
    embedding = transport.embedding(daily, memory, valid_history, innovation_ratio=gain)
    embedding[~valid_history] = daily[~valid_history]

    correct_loss = validation_loss(
        model,
        daily_tensor,
        packed,
        typed_tensor,
        transition_tensor,
        support_tensor,
        audit_rows,
        config,
    )
    shuffled_loss = perturbed_loss(
        model,
        daily_tensor,
        packed,
        typed_tensor,
        transition_tensor,
        support_tensor,
        audit_rows,
        config,
        reverse=False,
    )
    correct_deep_loss = validation_loss(
        model,
        daily_tensor,
        packed,
        typed_tensor,
        transition_tensor,
        support_tensor,
        deep_audit_rows,
        config,
    )
    reversed_loss = perturbed_loss(
        model,
        daily_tensor,
        packed,
        typed_tensor,
        transition_tensor,
        support_tensor,
        deep_audit_rows,
        config,
        reverse=True,
    )
    branch_ev = {
        name: mean_ev(
            typed[audit_rows, index], output["typed_predictions"][audit_rows, index]
        )
        for index, name in enumerate(RESPONSIBILITIES)
    }
    measured_ratio = float(
        np.sqrt(np.mean(np.square(fit_innovation * gain)))
        / np.sqrt(np.mean(np.square(daily[fit_rows])))
    )
    metrics = {
        "optimizer_steps": steps,
        "fit_longitudinal_endpoints": int(len(fit_rows)),
        "audit_longitudinal_endpoints": int(len(audit_rows)),
        "candidate_branch_target_ev": branch_ev,
        "candidate_memory_typed_ev": ridge_ev(
            memory, typed.reshape(len(typed), -1), fit_rows, audit_rows
        ),
        "candidate_current_care_prediction_ev": mean_ev(
            daily[audit_rows], output["long_prediction"][audit_rows]
        ),
        "candidate_signed_low_high_memory_ev": ridge_ev(
            output["memory_codes"][:, 2],
            typed[:, 2, [0, 1, 2, 3, 8, 9]],
            fit_rows,
            audit_rows,
        ),
        "history_shuffle_loss_ratio": shuffled_loss / max(correct_loss, 1e-12),
        "history_reversal_loss_ratio": reversed_loss / max(correct_deep_loss, 1e-12),
        "daily_effective_rank": effective_rank(daily[audit_rows]),
        "candidate_effective_rank": effective_rank(embedding[audit_rows]),
        "daily_candidate_linear_cka": linear_cka(
            daily[audit_rows], embedding[audit_rows]
        ),
        "candidate_residual_gain": gain,
        "measured_residual_energy_ratio": measured_ratio,
        "no_history_identity_max_abs": float(
            np.max(np.abs(embedding[~valid_history] - daily[~valid_history]))
        ),
        "covariance_transport": diagnostics,
    }
    gates = config["canary_gates"]
    gate_results = {
        "all_responsibilities_positive": all(
            value >= float(gates["each_responsibility_ev_min"])
            for value in branch_ev.values()
        ),
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
        "energy": abs(measured_ratio - float(config["method"]["residual_energy_ratio"]))
        <= float(gates["residual_energy_ratio_tolerance"]),
        "identity": metrics["no_history_identity_max_abs"]
        <= float(gates["identity_max_abs"]),
        "fit_covariance_shape": diagnostics["fit_covariance_relative_frobenius_error"]
        <= float(gates["fit_covariance_relative_error_max"]),
        "fit_daily_innovation_independence": diagnostics[
            "fit_daily_innovation_cross_covariance_relative_frobenius"
        ]
        <= float(gates["fit_cross_covariance_relative_error_max"]),
        "memory_support": diagnostics["retained_memory_singular_ratio"]
        >= float(gates["retained_memory_singular_ratio_min"]),
        "isolation": all(int(value) == 0 for value in config["isolation"].values()),
    }
    accepted = all(gate_results.values())
    args.output_root.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_root / "checkpoint.pt"
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "model_config": config["method"]["model"],
            "daily_center": daily_center,
            "daily_scale": daily_scale,
            "daily_basis": basis,
            "typed_contract": typed_contract,
            "candidate_transport": transport_state(transport),
            "candidate_residual_gain": gain,
            "maximum_days": int(config["method"]["maximum_days"]),
            "history": history,
            "config": config,
            "glucofm_framework_or_components": False,
        },
        checkpoint,
    )
    trainval_embedding = (embedding * daily_scale + daily_center).astype(np.float32)
    trainval_embedding[~valid_history] = daily_raw[~valid_history]
    embeddings = args.output_root / "trainval_embeddings.npz"
    np.savez_compressed(
        embeddings,
        embedding=trainval_embedding,
        history_days=(history_count + 1).astype(np.int8),
        split=data["split"],
        window_id=data["window_id"],
    )
    report = args.output_root / "mechanism_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate": config["candidate"],
                "acceptance_passed": accepted,
                "canary_config_sha256": sha256(args.config),
                "canary_report_sha256": sha256(args.canary_report),
                **metrics,
                "gate_results": gate_results,
                "isolation": config["isolation"],
                "next_action": "run the unchanged fixed public 14-task protocol once"
                if accepted
                else "stop before public evaluation",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = args.output_root / "artifacts.sha256"
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (checkpoint, embeddings, report)
        )
    )
    print(report.read_text(), end="")
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
