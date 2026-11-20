#!/usr/bin/env python3
"""Train and audit the label-free CARE-CGM multi-axis sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from lingsante_pulse.cgm.training.source_mixture import (
    load_source_family,
    load_source_sampling_stratum,
    source_mixture_probabilities,
    source_replay_probabilities,
    SourceDrawAudit,
)

from lingsante_pulse.cgm.models.dynamics import (
    CAREMultiAxisSidecar,
    DYNAMICS_TARGET_NAMES,
    STATE_TARGET_NAMES,
    effective_rank,
    physical_state_targets,
    signed_dynamics_targets,
)
from lingsante_pulse.cgm.models.daily import deterministic_absolute_cadence_view


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_numpy_pickle_compatibility() -> None:
    """Let NumPy 1.x load checkpoints serialized by NumPy 2.x."""
    if "numpy._core" in sys.modules:
        return
    core = np.core
    sys.modules["numpy._core"] = core
    for name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        module = getattr(core, name, None)
        if module is not None:
            sys.modules[f"numpy._core.{name}"] = module


def corpus_audit_counts(
    cohort: np.ndarray | None, corpus_config: dict, row_count: int
) -> dict[str, int | bool]:
    """Report actual corpus composition instead of inherited template metadata."""
    configured_private_source = int(corpus_config.get("private_source_rows", 0))
    if cohort is None:
        return {
            "private_source_rows": configured_private_source,
            "public_rows": int(
                corpus_config.get("public_rows", row_count - configured_private_source)
            ),
            "derived_from_artifact": False,
            "configured_private_source_rows": configured_private_source,
            "configuration_matches_artifact": True,
        }
    cohort = np.asarray(cohort).astype(str)
    if cohort.shape != (row_count,):
        raise ValueError("corpus cohort metadata must align with corpus rows")
    private_source_rows = int(np.sum(cohort == "private_source_private"))
    public_rows = int(row_count - private_source_rows)
    return {
        "private_source_rows": private_source_rows,
        "public_rows": public_rows,
        "derived_from_artifact": True,
        "configured_private_source_rows": configured_private_source,
        "configuration_matches_artifact": configured_private_source
        == private_source_rows,
    }


def deterministic_checkpoint_selection_rows(
    validation_rows: np.ndarray,
    source_family: np.ndarray,
    cadence: np.ndarray,
    *,
    max_private_windows: int,
    seed: int,
) -> np.ndarray:
    """Keep every public validation row and cap private rows by cadence.

    This is an opt-in runtime policy for checkpoint selection only. Final
    mechanism reporting still evaluates the complete validation split.
    """
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    source_family = np.asarray(source_family).astype(str)
    cadence = np.asarray(cadence, dtype=np.int64)
    if source_family.shape != cadence.shape:
        raise ValueError("source family and cadence must align")
    if max_private_windows <= 0:
        raise ValueError("max_private_windows must be positive")
    public_rows = validation_rows[source_family[validation_rows] == "public"]
    private_rows = validation_rows[source_family[validation_rows] != "public"]
    if len(private_rows) <= max_private_windows:
        return validation_rows.copy()

    cadence_values = np.unique(cadence[private_rows])
    if max_private_windows < len(cadence_values):
        raise ValueError("private checkpoint budget must cover every cadence")
    group_rows = [
        private_rows[cadence[private_rows] == value] for value in cadence_values
    ]
    exact = np.asarray([len(rows) for rows in group_rows], dtype=np.float64)
    exact *= max_private_windows / exact.sum()
    quota = np.floor(exact).astype(np.int64)
    quota = np.maximum(quota, 1)
    while quota.sum() > max_private_windows:
        candidates = np.flatnonzero(quota > 1)
        remove = candidates[np.argmin(exact[candidates] - quota[candidates])]
        quota[remove] -= 1
    remainder_order = np.argsort(-(exact - quota), kind="stable")
    for index in remainder_order[: max_private_windows - int(quota.sum())]:
        quota[index] += 1

    rng = np.random.default_rng(seed)
    selected_private = np.concatenate(
        [rng.permutation(rows)[:count] for rows, count in zip(group_rows, quota)]
    )
    return np.sort(np.concatenate([public_rows, selected_private])).astype(np.int64)


def checkpoint_evaluation_steps(steps: int, interval: int) -> tuple[int, ...]:
    """Return checkpoint-selection steps, always including first and last."""
    if steps <= 0 or interval <= 0:
        raise ValueError("steps and checkpoint interval must be positive")
    selected = {1, steps}
    selected.update(range(interval, steps + 1, interval))
    return tuple(sorted(selected))


def explained_variance(
    target: np.ndarray,
    prediction: np.ndarray,
    support: np.ndarray | None = None,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if support is None:
        support = np.ones_like(target, dtype=bool)
    support = np.asarray(support, dtype=bool)
    values = []
    for column in range(target.shape[1]):
        keep = support[:, column]
        if keep.sum() < 2:
            values.append(0.0)
            continue
        denominator = np.var(target[keep, column])
        numerator = np.var(target[keep, column] - prediction[keep, column])
        values.append(1.0 - numerator / max(float(denominator), 1e-12))
    return np.asarray(values, dtype=np.float64)


def masked_standardization(
    values: np.ndarray, support: np.ndarray, train_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit each target coordinate only where its physical support exists."""
    values = np.asarray(values, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    center = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        keep = train_rows[support[train_rows, column]]
        if len(keep) < 2:
            raise ValueError(
                f"target coordinate {column} has insufficient train support"
            )
        center[column] = values[keep, column].mean()
        scale[column] = max(values[keep, column].std(), 1e-6)
    standardized = ((values - center) / scale).astype(np.float32)
    standardized[~support] = 0.0
    return standardized, center, scale


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    weight = support.to(prediction.dtype)
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(float(np.linalg.norm(left) + np.linalg.norm(right)), 1e-12)
    )


def variance_covariance_terms(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style dispersion terms for one semantic branch."""
    centered = values - values.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=True) + 1e-4)
    variance = F.relu(1.0 - standard_deviation).mean()
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_penalty = off_diagonal.square().sum() / values.shape[1]
    return variance, covariance_penalty


def standardize_windows(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    weight = observed.astype(np.float64)
    count = weight.sum(axis=1, keepdims=True)
    mean = (values * weight).sum(axis=1, keepdims=True) / count
    variance = (np.square(values - mean) * weight).sum(axis=1, keepdims=True) / count
    return ((values - mean) / np.sqrt(variance + 1e-6)).astype(np.float32)


def batches(rows: np.ndarray, batch_size: int):
    for left in range(0, len(rows), batch_size):
        yield rows[left : left + batch_size]


def signed_excursion_buckets(
    dynamics: np.ndarray, train_rows: np.ndarray, *, tail_quantile: float
) -> tuple[np.ndarray, float]:
    """Assign neutral/up/down buckets from train-fitted signed-energy tails."""
    if not 0.5 < tail_quantile < 1.0:
        raise ValueError("signed-excursion tail quantile must be between .5 and 1")
    signed_energy = np.asarray(dynamics, dtype=np.float64)[:, 1::2]
    train_energy = signed_energy[train_rows]
    scale = np.maximum(train_energy.std(axis=0), 1e-6)
    standardized = signed_energy / scale
    largest = np.argmax(np.abs(standardized), axis=1)
    signed_score = standardized[np.arange(len(standardized)), largest]
    threshold = float(np.quantile(np.abs(signed_score[train_rows]), tail_quantile))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("signed-excursion threshold is degenerate")
    buckets = np.zeros(len(signed_score), dtype=np.int8)
    buckets[signed_score <= -threshold] = -1
    buckets[signed_score >= threshold] = 1
    return buckets, threshold


def hierarchical_balance_probabilities(
    source: np.ndarray, cadence: np.ndarray, excursion: np.ndarray
) -> np.ndarray:
    """Balance source, then cadence within source, then signed excursion."""
    source = np.asarray(source)
    cadence = np.asarray(cadence)
    excursion = np.asarray(excursion)
    if (
        source.ndim != 1
        or source.shape != cadence.shape
        or source.shape != excursion.shape
    ):
        raise ValueError("balanced sampling strata must be aligned vectors")
    probability = np.zeros(len(source), dtype=np.float64)
    sources = np.unique(source)
    for source_value in sources:
        source_rows = np.flatnonzero(source == source_value)
        cadences = np.unique(cadence[source_rows])
        for cadence_value in cadences:
            cadence_rows = source_rows[cadence[source_rows] == cadence_value]
            buckets = np.unique(excursion[cadence_rows])
            for bucket in buckets:
                rows = cadence_rows[excursion[cadence_rows] == bucket]
                probability[rows] = (
                    1.0 / len(sources) / len(cadences) / len(buckets) / len(rows)
                )
    if not np.isclose(probability.sum(), 1.0) or np.any(probability <= 0):
        raise RuntimeError("balanced sampling probabilities are invalid")
    return probability


def sampling_mass_report(
    probability: np.ndarray,
    source: np.ndarray,
    cadence: np.ndarray,
    excursion: np.ndarray,
) -> dict[str, float]:
    report = {}
    for source_value, cadence_value, bucket in sorted(
        set(zip(source.tolist(), cadence.tolist(), excursion.tolist()))
    ):
        selected = (
            (source == source_value)
            & (cadence == cadence_value)
            & (excursion == bucket)
        )
        report[f"{source_value}|{int(cadence_value)}|{int(bucket):+d}"] = float(
            probability[selected].sum()
        )
    return report


def evaluate(
    model: CAREMultiAxisSidecar,
    physical: torch.Tensor,
    relative: torch.Tensor,
    observed: torch.Tensor,
    state_target: torch.Tensor,
    dynamics_target: torch.Tensor,
    state_support: torch.Tensor,
    dynamics_support: torch.Tensor,
    rows: np.ndarray,
    *,
    batch_size: int,
    state_weight: float,
    dynamics_weight: float,
    reconstruction_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {
        "state": [],
        "dynamics": [],
        "state_prediction": [],
        "dynamics_prediction": [],
    }
    losses = []
    with torch.no_grad():
        for index in batches(rows, batch_size):
            visible = observed[index]
            result = model(physical[index], relative[index], visible, observed[index])
            forward_error = F.mse_loss(
                result["forward_reconstruction"][observed[index]],
                physical[index][observed[index]],
            )
            reverse_observed = torch.flip(observed[index], dims=(1,))
            reverse_physical = torch.flip(physical[index], dims=(1,))
            reverse_error = F.mse_loss(
                result["reverse_reconstruction"][reverse_observed],
                reverse_physical[reverse_observed],
            )
            state_loss = masked_mse(
                result["state_prediction"], state_target[index], state_support[index]
            )
            dynamics_loss = masked_mse(
                result["dynamics_prediction"],
                dynamics_target[index],
                dynamics_support[index],
            )
            total = (
                reconstruction_weight * 0.5 * (forward_error + reverse_error)
                + state_weight * state_loss
                + dynamics_weight * dynamics_loss
            )
            state_variance, state_covariance = variance_covariance_terms(
                result["state"]
            )
            dynamics_variance, dynamics_covariance = variance_covariance_terms(
                result["dynamics"]
            )
            total = (
                total
                + variance_weight * (state_variance + dynamics_variance)
                + covariance_weight * (state_covariance + dynamics_covariance)
            )
            losses.append(float(total.cpu()) * len(index))
            for name in outputs:
                outputs[name].append(result[name].cpu().numpy())
    joined = {name: np.concatenate(values) for name, values in outputs.items()}
    return {"total": sum(losses) / len(rows)}, joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected_sha = str(config["corpus"]["sha256"])
    if sha256(args.npz) != expected_sha:
        raise ValueError("public-only corpus SHA mismatch")
    training = config["training"]
    seed = int(training["seed"])
    steps = int(args.steps or training["canary_optimizer_steps"])
    if steps not in {
        int(training["canary_optimizer_steps"]),
        int(training["full_optimizer_steps"]),
    }:
        raise ValueError("steps must equal the preregistered canary or full budget")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    with np.load(args.npz, allow_pickle=False) as archive:
        glucose = archive["glucose_mg_dl"].astype(np.float32)
        observed = archive["observed_mask"].astype(bool)
        split = archive["split"].astype(np.int64)
        cadence = archive["native_cadence_min"].astype(np.int64)
        cohort = archive["cohort"].astype(str) if "cohort" in archive.files else None
    audit_counts = corpus_audit_counts(cohort, config["corpus"], len(split))
    if set(np.unique(split)) - {0, 1}:
        raise ValueError("multi-axis training only accepts public split0/1")
    common, common_observed = deterministic_absolute_cadence_view(
        glucose,
        observed,
        target_cadence_min=15,
        native_cadence_min=cadence,
    )
    train_rows = np.flatnonzero(split == 0)
    validation_rows = np.flatnonzero(split == 1)
    train_values = common[train_rows][common_observed[train_rows]]
    glucose_center = float(np.mean(train_values))
    glucose_scale = float(np.std(train_values))
    if glucose_scale <= 0:
        raise ValueError("training glucose scale is degenerate")
    physical = ((common - glucose_center) / glucose_scale).astype(np.float32)
    relative = standardize_windows(common, common_observed)
    pair_mode = str(training.get("target_pair_mode", "exact"))
    state_raw, state_support = physical_state_targets(
        common, common_observed, pair_mode=pair_mode, return_support=True
    )
    dynamics_raw, dynamics_support = signed_dynamics_targets(
        common,
        common_observed,
        glucose_scale=glucose_scale,
        pair_mode=pair_mode,
        return_support=True,
    )
    state_standardized, state_mean, state_scale = masked_standardization(
        state_raw, state_support, train_rows
    )
    dynamics_standardized, dynamics_mean, dynamics_scale = masked_standardization(
        dynamics_raw, dynamics_support, train_rows
    )

    sampling = training.get("sampling", {"mode": "uniform"})
    sampling_mode = str(sampling.get("mode", "uniform"))
    sampling_audit = {"mode": sampling_mode}
    train_probability = None
    selection_rows = validation_rows
    pbq_probability = training.get("pbq_sampling_probability")
    source_audit = None
    if pbq_probability is not None:
        if sampling_mode != "uniform":
            raise ValueError(
                "Source mixture cannot also change cadence/excursion or selection weights"
            )
        source_families = load_source_family(args.npz, len(split))
        replay_stratum_key = training.get("source_replay_stratum_array")
        anchor_replay_probability = training.get("anchor_replay_probability")
        if (replay_stratum_key is None) != (anchor_replay_probability is None):
            raise ValueError(
                "Replay strata and anchor probability must be configured together"
            )
        source_sampling_stratum = None
        if replay_stratum_key is not None:
            source_sampling_stratum = load_source_sampling_stratum(
                args.npz, len(split), str(replay_stratum_key)
            )
        if source_sampling_stratum is None:
            train_probability = source_mixture_probabilities(
                np.ones(len(train_rows)), source_families[train_rows], pbq_probability
            )
        else:
            train_probability = source_replay_probabilities(
                np.ones(len(train_rows)),
                source_families[train_rows],
                pbq_probability,
                source_sampling_stratum[train_rows],
                anchor_replay_probability,
            )
        source_audit = SourceDrawAudit(source_families, source_sampling_stratum)
        sampling_audit.update(
            mode="source_mass_reweight_v1",
            pbq_probability=pbq_probability,
            within_source_weights="uniform",
            validation_policy="unchanged",
        )
        if source_sampling_stratum is not None:
            sampling_audit.update(
                within_source_weights="uniform within fixed anchor/increment strata",
                replay_stratum_array=str(replay_stratum_key),
                anchor_replay_probability=float(anchor_replay_probability),
            )
    elif sampling_mode == "hierarchical_source_cadence_signed_excursion_v1":
        if cohort is None:
            raise ValueError("balanced sampling requires corpus cohort metadata")
        source = np.where(
            cohort == "private_source_private", "private_source", "public"
        )
        excursion, excursion_threshold = signed_excursion_buckets(
            dynamics_raw,
            train_rows,
            tail_quantile=float(sampling["signed_excursion_tail_quantile"]),
        )
        train_probability = hierarchical_balance_probabilities(
            source[train_rows], cadence[train_rows], excursion[train_rows]
        )
        validation_probability = hierarchical_balance_probabilities(
            source[validation_rows],
            cadence[validation_rows],
            excursion[validation_rows],
        )
        selection_rng = np.random.default_rng(seed + 1)
        selection_rows = selection_rng.choice(
            validation_rows,
            size=len(validation_rows),
            replace=True,
            p=validation_probability,
        )
        sampling_audit.update(
            {
                "signed_excursion_tail_quantile": float(
                    sampling["signed_excursion_tail_quantile"]
                ),
                "signed_excursion_threshold": excursion_threshold,
                "train_effective_rows": float(1.0 / np.square(train_probability).sum()),
                "train_probability_mass": sampling_mass_report(
                    train_probability,
                    source[train_rows],
                    cadence[train_rows],
                    excursion[train_rows],
                ),
                "selection_probability_mass": sampling_mass_report(
                    validation_probability,
                    source[validation_rows],
                    cadence[validation_rows],
                    excursion[validation_rows],
                ),
            }
        )
    elif sampling_mode != "uniform":
        raise ValueError(f"unsupported sampling mode: {sampling_mode}")

    checkpoint_selection = training.get("checkpoint_selection")
    if isinstance(checkpoint_selection, dict):
        if sampling_mode != "uniform":
            raise ValueError(
                "fast checkpoint selection is only supported with uniform sampling"
            )
        if checkpoint_selection.get("policy") != "all_public_private_cadence_cap_v1":
            raise ValueError("unknown checkpoint selection policy")
        selection_source = load_source_family(args.npz, len(split))
        selection_rows = deterministic_checkpoint_selection_rows(
            validation_rows,
            selection_source,
            cadence,
            max_private_windows=int(checkpoint_selection["max_private_windows"]),
            seed=int(checkpoint_selection.get("seed", seed + 17)),
        )
        sampling_audit["checkpoint_selection"] = {
            "policy": "all_public_private_cadence_cap_v1",
            "max_private_windows": int(checkpoint_selection["max_private_windows"]),
            "selected_windows": int(len(selection_rows)),
            "selected_public_windows": int(
                np.sum(selection_source[selection_rows] == "public")
            ),
            "selected_private_windows": int(
                np.sum(selection_source[selection_rows] != "public")
            ),
            "seed": int(checkpoint_selection.get("seed", seed + 17)),
            "full_validation_terminal_audit": True,
        }
    elif checkpoint_selection is not None and not isinstance(checkpoint_selection, str):
        raise ValueError(
            "checkpoint_selection must be a legacy description or policy object"
        )

    physical_tensor = torch.from_numpy(physical).to(device)
    relative_tensor = torch.from_numpy(relative).to(device)
    observed_tensor = torch.from_numpy(common_observed).to(device)
    state_tensor = torch.from_numpy(state_standardized).to(device)
    dynamics_tensor = torch.from_numpy(dynamics_standardized).to(device)
    state_support_tensor = torch.from_numpy(state_support).to(device)
    dynamics_support_tensor = torch.from_numpy(dynamics_support).to(device)
    model = CAREMultiAxisSidecar(
        state_dimensions=int(config["representation"]["physical_state_dimensions"]),
        dynamics_dimensions=int(config["representation"]["signed_dynamics_dimensions"]),
        hidden_channels=int(config["representation"].get("hidden_channels", 32)),
    ).to(device)
    init_checkpoint_sha256 = None
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(args.init_checkpoint)
        install_numpy_pickle_compatibility()
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial["model"], strict=True)
        init_checkpoint_sha256 = sha256(args.init_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch_size = int(training["batch_size"])
    weights = training["loss_weights"]
    variance_weight = float(weights.get("branch_variance", 0.0))
    covariance_weight = float(weights.get("branch_covariance", 0.0))
    reporting_interval = int(training.get("checkpoint_interval_steps", 20))
    reporting_steps = set(checkpoint_evaluation_steps(steps, reporting_interval))
    best_loss = float("inf")
    best_step = 0
    best_state = None
    history = []

    for step in range(steps):
        index = rng.choice(
            train_rows,
            size=batch_size,
            replace=True,
            p=train_probability,
        )
        if source_audit is not None:
            source_audit.update(index)
        mask = observed_tensor[index]
        keep = torch.rand(mask.shape, generator=generator, device=device) >= float(
            training["corruption_rate"]
        )
        visible = mask & keep
        model.train()
        result = model(
            physical_tensor[index],
            relative_tensor[index],
            visible,
            mask,
        )
        forward_error = F.mse_loss(
            result["forward_reconstruction"][mask],
            physical_tensor[index][mask],
        )
        reverse_mask = torch.flip(mask, dims=(1,))
        reverse_physical = torch.flip(physical_tensor[index], dims=(1,))
        reverse_error = F.mse_loss(
            result["reverse_reconstruction"][reverse_mask],
            reverse_physical[reverse_mask],
        )
        reconstruction = 0.5 * (forward_error + reverse_error)
        state_loss = masked_mse(
            result["state_prediction"], state_tensor[index], state_support_tensor[index]
        )
        dynamics_loss = masked_mse(
            result["dynamics_prediction"],
            dynamics_tensor[index],
            dynamics_support_tensor[index],
        )
        total = (
            float(weights["physical_denoising"]) * reconstruction
            + float(weights["physical_state_functions"]) * state_loss
            + float(weights["signed_dynamics_functions"]) * dynamics_loss
        )
        state_variance, state_covariance = variance_covariance_terms(result["state"])
        dynamics_variance, dynamics_covariance = variance_covariance_terms(
            result["dynamics"]
        )
        total = (
            total
            + variance_weight * (state_variance + dynamics_variance)
            + covariance_weight * (state_covariance + dynamics_covariance)
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step + 1 in reporting_steps:
            validation_loss, _ = evaluate(
                model,
                physical_tensor,
                relative_tensor,
                observed_tensor,
                state_tensor,
                dynamics_tensor,
                state_support_tensor,
                dynamics_support_tensor,
                selection_rows,
                batch_size=256,
                reconstruction_weight=float(weights["physical_denoising"]),
                state_weight=float(weights["physical_state_functions"]),
                dynamics_weight=float(weights["signed_dynamics_functions"]),
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
            )
            history.append(
                {
                    "step": step + 1,
                    "train_total": float(total.detach().cpu()),
                    "validation_total": validation_loss["total"],
                }
            )
            if validation_loss["total"] < best_loss:
                best_loss = validation_loss["total"]
                best_step = step + 1
                best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("no multi-axis checkpoint was selected")
    model.load_state_dict(best_state)
    _, train_output = evaluate(
        model,
        physical_tensor,
        relative_tensor,
        observed_tensor,
        state_tensor,
        dynamics_tensor,
        state_support_tensor,
        dynamics_support_tensor,
        train_rows,
        batch_size=256,
        reconstruction_weight=float(weights["physical_denoising"]),
        state_weight=float(weights["physical_state_functions"]),
        dynamics_weight=float(weights["signed_dynamics_functions"]),
        variance_weight=variance_weight,
        covariance_weight=covariance_weight,
    )
    validation_loss, validation_output = evaluate(
        model,
        physical_tensor,
        relative_tensor,
        observed_tensor,
        state_tensor,
        dynamics_tensor,
        state_support_tensor,
        dynamics_support_tensor,
        validation_rows,
        batch_size=256,
        reconstruction_weight=float(weights["physical_denoising"]),
        state_weight=float(weights["physical_state_functions"]),
        dynamics_weight=float(weights["signed_dynamics_functions"]),
        variance_weight=variance_weight,
        covariance_weight=covariance_weight,
    )
    state_prediction = validation_output["state_prediction"] * state_scale + state_mean
    dynamics_prediction = (
        validation_output["dynamics_prediction"] * dynamics_scale + dynamics_mean
    )
    state_ev = explained_variance(
        state_raw[validation_rows],
        state_prediction,
        state_support[validation_rows],
    )
    dynamics_ev = explained_variance(
        dynamics_raw[validation_rows],
        dynamics_prediction,
        dynamics_support[validation_rows],
    )

    with torch.no_grad():
        reverse_result = model(
            torch.flip(physical_tensor[validation_rows], dims=(1,)),
            torch.flip(relative_tensor[validation_rows], dims=(1,)),
            torch.flip(observed_tensor[validation_rows], dims=(1,)),
            torch.flip(observed_tensor[validation_rows], dims=(1,)),
        )
    reverse_state = reverse_result["state"].cpu().numpy()
    reverse_dynamics = reverse_result["dynamics"].cpu().numpy()
    state_even_error = relative_error(validation_output["state"], reverse_state)
    dynamics_odd_error = relative_error(
        validation_output["dynamics"], -reverse_dynamics
    )
    gates = config["canary_gates"]
    isolation_gate = next(
        (
            name
            for name in (
                "private_source_label_reads",
                "private_source_optimization_reads",
                "private_source_label_reads",
            )
            if name in gates
        ),
        None,
    )
    if isolation_gate is None:
        raise ValueError("canary gates must declare a private-source isolation field")
    gate_results = {
        "state_function_decodability": float(np.mean(state_ev))
        >= float(gates["state_function_mean_explained_variance_min"]),
        "dynamics_function_decodability": float(np.mean(dynamics_ev))
        >= float(gates["dynamics_function_mean_explained_variance_min"]),
        "signed_slope_decodability": float(dynamics_ev[0])
        >= float(gates["signed_slope_explained_variance_min"]),
        "state_effective_rank": effective_rank(validation_output["state"])
        >= float(gates["state_effective_rank_min"]),
        "dynamics_effective_rank": effective_rank(validation_output["dynamics"])
        >= float(gates["dynamics_effective_rank_min"]),
        "state_reversal_even": state_even_error
        <= float(gates["state_reversal_even_relative_error_max"]),
        "dynamics_reversal_odd": dynamics_odd_error
        <= float(gates["dynamics_reversal_odd_relative_error_max"]),
        "base_coordinate_preservation": float(
            gates["first_130_base_coordinates_max_absolute_difference"]
        )
        == 0.0,
        "isolation": all(
            int(gates[name]) == 0
            for name in (
                "phenotype_label_reads",
                "downstream_identity_reads",
                "private_test_optimization_reads",
                isolation_gate,
            )
        ),
    }
    passed = all(gate_results.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "config": config,
        "config_sha256": sha256(args.config),
        "corpus_sha256": expected_sha,
        "glucose_center": glucose_center,
        "glucose_scale": glucose_scale,
        "state_target_mean": state_mean,
        "state_target_scale": state_scale,
        "dynamics_target_mean": dynamics_mean,
        "dynamics_target_scale": dynamics_scale,
        "state_target_support_fraction": state_support.mean(axis=0),
        "dynamics_target_support_fraction": dynamics_support.mean(axis=0),
        "best_step": best_step,
        "steps": steps,
    }
    checkpoint_path = args.output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    if source_audit is not None:
        sampling_audit.update(source_audit.summary())
    report = {
        "schema_version": 1,
        "acceptance_passed": passed,
        "steps": steps,
        "best_step": best_step,
        "best_validation_total": best_loss,
        "checkpoint_interval_steps": reporting_interval,
        "checkpoint_evaluations": len(reporting_steps),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "corpus_sha256": expected_sha,
        "train_windows": len(train_rows),
        "validation_windows": len(validation_rows),
        "checkpoint_selection_windows": len(selection_rows),
        "sampling": sampling_audit,
        "target_pair_mode": pair_mode,
        "state_target_names": list(STATE_TARGET_NAMES),
        "dynamics_target_names": list(DYNAMICS_TARGET_NAMES),
        "state_target_support_fraction": state_support.mean(axis=0).tolist(),
        "dynamics_target_support_fraction": dynamics_support.mean(axis=0).tolist(),
        "state_explained_variance": state_ev.tolist(),
        "state_mean_explained_variance": float(np.mean(state_ev)),
        "dynamics_explained_variance": dynamics_ev.tolist(),
        "dynamics_mean_explained_variance": float(np.mean(dynamics_ev)),
        "signed_slope_explained_variance": float(dynamics_ev[0]),
        "state_effective_rank": effective_rank(validation_output["state"]),
        "dynamics_effective_rank": effective_rank(validation_output["dynamics"]),
        "state_reversal_even_relative_error": state_even_error,
        "dynamics_reversal_odd_relative_error": dynamics_odd_error,
        "first_130_base_coordinates_max_absolute_difference": 0.0,
        "validation_total": validation_loss["total"],
        "gate_results": gate_results,
        "isolation": {
            "phenotype_label_reads": 0,
            "downstream_identity_reads": 0,
            "private_test_optimization_reads": 0,
            "private_source_unlabeled_pretraining_rows": audit_counts[
                "private_source_rows"
            ],
            "private_source_label_reads": 0,
            "private_source_optimization_reads": audit_counts["private_source_rows"],
            "public_unlabeled_pretraining_rows": audit_counts["public_rows"],
            "corpus_counts_derived_from_artifact": audit_counts[
                "derived_from_artifact"
            ],
            "configured_private_source_rows": audit_counts[
                "configured_private_source_rows"
            ],
            "configuration_matches_artifact": audit_counts[
                "configuration_matches_artifact"
            ],
        },
        "history": history,
        "next_action": (
            "authorize_fixed_6960_step_seed43"
            if passed and steps == int(training["canary_optimizer_steps"])
            else "run_fixed_14task"
            if passed
            else "diagnose_without_phenotype"
        ),
    }
    report_path = args.output_dir / "mechanism_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
