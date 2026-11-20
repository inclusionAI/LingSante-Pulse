#!/usr/bin/env python3
"""Exploratory GlucoFM-like pretraining on PRIVATE or mixed CGM windows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.baselines.glucofm_model import (
    GlucoFMLike,
    augment_cgm,
    counterfactual_cadence_view,
    counterfactual_sensor_view,
    reverse_gradient,
    scale_invariant_dynamics_targets,
)
from lingsante_pulse.cgm.baselines.absolute_bottleneck import absolute_patch_features
from lingsante_pulse.cgm.training.sampling import cadence_cohort_balanced_indices


def merge_pcgrad(
    base_gradients: tuple[torch.Tensor | None, ...],
    auxiliary_gradients: tuple[torch.Tensor | None, ...],
    auxiliary_weight: float,
) -> tuple[list[torch.Tensor | None], float, bool]:
    """Project a conflicting auxiliary gradient off the base objective gradient."""
    shared = [
        (base, auxiliary)
        for base, auxiliary in zip(base_gradients, auxiliary_gradients)
        if base is not None and auxiliary is not None
    ]
    if not shared:
        return (
            [
                base if auxiliary is None else auxiliary_weight * auxiliary
                for base, auxiliary in zip(base_gradients, auxiliary_gradients)
            ],
            float("nan"),
            False,
        )
    dot = sum((base * auxiliary).sum() for base, auxiliary in shared)
    base_norm_sq = sum(base.square().sum() for base, _ in shared)
    auxiliary_norm_sq = sum(auxiliary.square().sum() for _, auxiliary in shared)
    denominator = torch.sqrt(base_norm_sq * auxiliary_norm_sq).clamp_min(1e-12)
    cosine = float((dot / denominator).detach().cpu())
    conflicted = bool(float(dot.detach().cpu()) < 0)
    projection = (
        dot / base_norm_sq.clamp_min(1e-12) if conflicted else dot.new_zeros(())
    )
    merged: list[torch.Tensor | None] = []
    for base, auxiliary in zip(base_gradients, auxiliary_gradients):
        if base is None:
            merged.append(None if auxiliary is None else auxiliary_weight * auxiliary)
        elif auxiliary is None:
            merged.append(base)
        else:
            merged.append(base + auxiliary_weight * (auxiliary - projection * base))
    return merged, cosine, conflicted


def standardized_embedding_anchor_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Penalize drift from a frozen teacher in train-standardized coordinates."""
    standardized_student = (student - center) / scale
    standardized_teacher = (teacher - center) / scale
    if mode == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(
            standardized_student, standardized_teacher
        )
    if mode == "mse":
        return torch.nn.functional.mse_loss(standardized_student, standardized_teacher)
    raise ValueError(f"unsupported embedding anchor loss: {mode}")


def cross_day_same_source_partners(
    index: np.ndarray,
    subject_id: np.ndarray,
    start: np.ndarray,
    source: np.ndarray,
    cohort: np.ndarray,
) -> np.ndarray:
    """Choose one deterministic next-day, same-source PRIVATE partner per anchor."""
    partners = np.full(len(subject_id), -1, dtype=np.int64)
    selected = np.asarray(index, dtype=np.int64)
    if not (len(subject_id) == len(start) == len(source) == len(cohort)):
        raise ValueError("trait pairing metadata arrays must have equal length")
    selected = selected[
        (cohort[selected].astype(str) == "private_source_private")
        & (source[selected] >= 0)
    ]
    groups: dict[tuple[str, int], list[int]] = {}
    for row in selected:
        groups.setdefault((str(subject_id[row]), int(source[row])), []).append(int(row))
    day = start.astype("datetime64[D]")
    for rows in groups.values():
        rows_array = np.asarray(rows, dtype=np.int64)
        group_days = np.unique(day[rows_array])
        if len(group_days) < 2:
            continue
        for position, anchor_day in enumerate(group_days):
            anchor = rows_array[day[rows_array] == anchor_day]
            partner = rows_array[
                day[rows_array] == group_days[(position + 1) % len(group_days)]
            ]
            anchor = anchor[np.argsort(start[anchor], kind="stable")]
            partner = partner[np.argsort(start[partner], kind="stable")]
            partners[anchor] = np.resize(partner, len(anchor))
    return partners


def ema_cross_day_trait_loss(
    model: GlucoFMLike,
    anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    partner: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Match the online 1D trait to an EMA target from another day."""
    online_trait = model.online.embed(*anchor)[:, -1]
    with torch.no_grad():
        target_trait = model.target.embed(*partner)[:, -1]
    return torch.nn.functional.smooth_l1_loss(online_trait, target_trait)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--balance-cohort-cadence", action="store_true")
    parser.add_argument("--cadence-15-share", type=float, default=0.5)
    parser.add_argument("--cohort-sampling-temperature", type=float, default=0.0)
    parser.add_argument("--balance-subject-within-cohort", action="store_true")
    parser.add_argument("--balance-components", action="store_true")
    parser.add_argument("--calibrate-physiological-units", action="store_true")
    parser.add_argument("--relative-auxiliary-features", action="store_true")
    parser.add_argument(
        "--sparse-aware-masking",
        action="store_true",
        help="For <=200-point windows, mask at most 20%% of observed points",
    )
    parser.add_argument("--dual-reference-features", action="store_true")
    parser.add_argument("--absolute-reference-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=0.0)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--mean-shortcut-weight", type=float, default=0.0)
    parser.add_argument("--absolute-subspace-dim", type=int, default=8)
    parser.add_argument("--factorized-output", action="store_true")
    parser.add_argument("--normalize-dynamic-output", action="store_true")
    parser.add_argument("--cadence-adapter", action="store_true")
    parser.add_argument("--factorized-dynamic-weight", type=float, default=0.0)
    parser.add_argument("--factorized-absolute-weight", type=float, default=0.0)
    parser.add_argument("--source-adversarial-weight", type=float, default=0.0)
    parser.add_argument(
        "--source-adversarial-mode",
        choices=("joint", "alternating"),
        default="joint",
    )
    parser.add_argument("--source-classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--balance-source-classes", action="store_true")
    parser.add_argument("--condition-source-on-absolute-features", action="store_true")
    parser.add_argument("--sensor-counterfactual-weight", type=float, default=0.0)
    parser.add_argument(
        "--sensor-counterfactual-target-mode",
        choices=("symmetric", "ema_dense"),
        default="symmetric",
    )
    parser.add_argument(
        "--sensor-counterfactual-augmentation",
        choices=("mixed_sensor", "cadence_only"),
        default="mixed_sensor",
    )
    parser.add_argument("--minimum-source-train-windows", type=int, default=100)
    parser.add_argument(
        "--minimum-validation-native5-dynamic-effective-rank",
        type=float,
        default=0.0,
    )
    parser.add_argument("--dynamic-teacher-checkpoint", type=Path)
    parser.add_argument(
        "--dynamic-teacher-relative-auxiliary-features", action="store_true"
    )
    parser.add_argument("--dynamic-distillation-weight", type=float, default=0.0)
    parser.add_argument("--dynamics-target-weight", type=float, default=0.0)
    parser.add_argument("--dynamics-target-normalize-input", action="store_true")
    parser.add_argument(
        "--maximum-validation-dynamics-target", type=float, default=float("inf")
    )
    parser.add_argument("--maximum-validation-mcr-td", type=float, default=float("inf"))
    parser.add_argument(
        "--dynamics-target-gradient-mode",
        choices=("joint", "pcgrad"),
        default="joint",
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--embedding-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--embedding-anchor-loss",
        choices=("smooth_l1", "mse"),
        default="smooth_l1",
    )
    parser.add_argument(
        "--maximum-validation-embedding-anchor", type=float, default=float("inf")
    )
    parser.add_argument(
        "--maximum-validation-embedding-anchor-max-dimension-rms",
        type=float,
        default=float("inf"),
    )
    parser.add_argument("--trait-consistency-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.dynamic_distillation_weight < 0:
        raise ValueError("dynamic distillation weight must be non-negative")
    if args.dynamics_target_weight < 0:
        raise ValueError("dynamics target weight must be non-negative")
    if args.maximum_validation_dynamics_target <= 0:
        raise ValueError("maximum validation dynamics target must be positive")
    if args.maximum_validation_mcr_td <= 0:
        raise ValueError("maximum validation MCR+TD must be positive")
    if (
        math.isfinite(args.maximum_validation_dynamics_target)
        and args.dynamics_target_weight == 0
    ):
        raise ValueError("finite dynamics target guard requires a positive weight")
    if args.dynamics_target_normalize_input and args.dynamics_target_weight == 0:
        raise ValueError(
            "dynamics target input normalization requires a positive weight"
        )
    if (
        args.dynamics_target_gradient_mode != "joint"
        and args.dynamics_target_weight == 0
    ):
        raise ValueError("dynamics target gradient mode requires a positive weight")
    if args.embedding_anchor_weight < 0:
        raise ValueError("embedding anchor weight must be non-negative")
    if args.trait_consistency_weight < 0:
        raise ValueError("trait consistency weight must be non-negative")
    if args.trait_consistency_weight > 0 and not args.factorized_output:
        raise ValueError("trait consistency requires factorized output")
    if args.embedding_anchor_weight > 0 and args.initial_checkpoint is None:
        raise ValueError("embedding anchor requires an initial checkpoint")
    if args.maximum_validation_embedding_anchor <= 0:
        raise ValueError("maximum validation embedding anchor must be positive")
    if args.maximum_validation_embedding_anchor_max_dimension_rms <= 0:
        raise ValueError("maximum anchor dimension RMS must be positive")
    if (
        math.isfinite(args.maximum_validation_embedding_anchor)
        and args.embedding_anchor_weight == 0
    ):
        raise ValueError(
            "finite embedding anchor guard requires a positive anchor weight"
        )
    if (
        math.isfinite(args.maximum_validation_embedding_anchor_max_dimension_rms)
        and args.embedding_anchor_weight == 0
    ):
        raise ValueError(
            "finite anchor dimension guard requires a positive anchor weight"
        )
    if args.dynamic_distillation_weight > 0 and args.dynamic_teacher_checkpoint is None:
        raise ValueError(
            "positive dynamic distillation weight requires a teacher checkpoint"
        )
    if args.source_adversarial_weight < 0:
        raise ValueError("source adversarial weight must be non-negative")
    if args.source_classifier_learning_rate <= 0:
        raise ValueError("source classifier learning rate must be positive")
    if args.condition_source_on_absolute_features and not args.factorized_output:
        raise ValueError("conditional source adversary requires factorized output")
    if (
        args.condition_source_on_absolute_features
        and args.source_adversarial_weight == 0
    ):
        raise ValueError("source conditioning requires a positive adversarial weight")
    if args.sensor_counterfactual_weight < 0:
        raise ValueError("sensor counterfactual weight must be non-negative")
    if args.sensor_counterfactual_weight > 0 and not args.factorized_output:
        raise ValueError("sensor counterfactual consistency requires factorized output")
    if (
        args.sensor_counterfactual_target_mode != "symmetric"
        and args.sensor_counterfactual_weight == 0
    ):
        raise ValueError("counterfactual target mode requires a positive weight")
    if (
        args.sensor_counterfactual_augmentation != "mixed_sensor"
        and args.sensor_counterfactual_weight == 0
    ):
        raise ValueError("counterfactual augmentation requires a positive weight")
    if args.normalize_dynamic_output and not args.factorized_output:
        raise ValueError("dynamic output normalization requires factorized output")
    if args.cadence_adapter and not (
        args.factorized_output
        and args.sensor_counterfactual_weight > 0
        and args.sensor_counterfactual_target_mode == "ema_dense"
    ):
        raise ValueError(
            "cadence adapter requires factorized EMA-dense counterfactual training"
        )
    if args.minimum_source_train_windows < 1:
        raise ValueError("minimum source train windows must be positive")
    if args.minimum_validation_native5_dynamic_effective_rank < 0:
        raise ValueError(
            "minimum validation dynamic effective rank must be non-negative"
        )
    if args.balance_subject_within_cohort and not args.balance_cohort_cadence:
        raise ValueError("subject balancing requires cohort/cadence balancing")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    with np.load(args.npz, allow_pickle=False) as archive:
        glucose = archive["glucose_mg_dl"].astype(np.float32)
        observed = archive["observed_mask"].astype(np.float32)
        split = archive["split"]
        starts = archive["shifted_window_start"].astype("datetime64[m]")
        cohorts = (
            archive["cohort"].astype(str)
            if "cohort" in archive.files
            else np.full(len(glucose), "private_source_private")
        )
        subject_ids = (
            archive["subject_id"].astype(str) if "subject_id" in archive.files else None
        )
        cadences = (
            archive["native_cadence_min"].astype(np.int64)
            if "native_cadence_min" in archive.files
            else np.full(len(glucose), 5, dtype=np.int64)
        )
        device_sources = (
            archive["device_source_code"].astype(np.int64)
            if "device_source_code" in archive.files
            else np.full(len(glucose), -1, dtype=np.int64)
        )
    if glucose.shape[1:] != (288,) or glucose.shape != observed.shape:
        raise ValueError("expected matching [N,288] glucose and physical masks")
    if not np.isfinite(glucose).all() or not np.isin(observed, [0, 1]).all():
        raise ValueError("invalid PRIVATE input")
    minute_of_day = starts.astype("datetime64[m]").astype(np.int64) % (24 * 60)
    circadian = (minute_of_day // 5).astype(np.int64)
    train_index = np.flatnonzero(split == 0)
    validation_index = np.flatnonzero(split == 1)
    test_windows = int(np.sum(split == 2))
    if args.balance_subject_within_cohort and subject_ids is None:
        raise ValueError("subject balancing requires subject_id metadata")
    if args.trait_consistency_weight > 0 and subject_ids is None:
        raise ValueError("trait consistency requires subject_id metadata")

    trait_partner = np.full(len(glucose), -1, dtype=np.int64)
    if args.trait_consistency_weight > 0:
        train_trait_partner = cross_day_same_source_partners(
            train_index, subject_ids, starts, device_sources, cohorts
        )
        validation_trait_partner = cross_day_same_source_partners(
            validation_index, subject_ids, starts, device_sources, cohorts
        )
        trait_partner[train_index] = train_trait_partner[train_index]
        trait_partner[validation_index] = validation_trait_partner[validation_index]
        if not np.any(trait_partner[train_index] >= 0):
            raise ValueError("trait consistency requires train cross-day pairs")
        if not np.any(trait_partner[validation_index] >= 0):
            raise ValueError("trait consistency requires validation cross-day pairs")
    train_trait_pair_anchors = int(np.sum(trait_partner[train_index] >= 0))
    validation_trait_pair_anchors = int(np.sum(trait_partner[validation_index] >= 0))

    source_target = np.full(len(glucose), -1, dtype=np.int64)
    source_train_counts: dict[int, int] = {}
    eligible_source_codes: list[int] = []
    if args.source_adversarial_weight > 0:
        codes, counts = np.unique(
            device_sources[train_index][device_sources[train_index] >= 0],
            return_counts=True,
        )
        source_train_counts = {
            int(code): int(count) for code, count in zip(codes, counts)
        }
        eligible_source_codes = [
            code
            for code, count in source_train_counts.items()
            if count >= args.minimum_source_train_windows
        ]
        if len(eligible_source_codes) < 2:
            raise ValueError("source adversary requires at least two eligible sources")
        for target, code in enumerate(eligible_source_codes):
            source_target[device_sources == code] = target

    absolute_feature_center = None
    absolute_feature_scale = None
    if args.factorized_output:
        train_absolute_features = absolute_patch_features(
            glucose[train_index], observed[train_index]
        )
        absolute_feature_center = torch.from_numpy(train_absolute_features.mean(axis=0))
        absolute_feature_scale = torch.from_numpy(
            np.maximum(train_absolute_features.std(axis=0), 1e-6)
        )

    glucose_center = 0.0
    glucose_scale = 1.0
    rate_scale = 1.0
    calibrate_physiological_units = (
        args.calibrate_physiological_units or args.dual_reference_features
    )
    if calibrate_physiological_units:
        train_values = glucose[train_index]
        train_observed = observed[train_index].astype(bool)
        observed_values = train_values[train_observed]
        glucose_center = float(np.mean(observed_values))
        glucose_scale = float(np.std(observed_values))
        rate = np.zeros_like(train_values)
        rate_valid = np.zeros_like(train_observed)
        for lag in range(1, 10):
            eligible = (
                train_observed[:, lag:]
                & train_observed[:, :-lag]
                & ~rate_valid[:, lag:]
            )
            current = (train_values[:, lag:] - train_values[:, :-lag]) / float(lag)
            rate[:, lag:] = np.where(eligible, current, rate[:, lag:])
            rate_valid[:, lag:] |= eligible
        rate_scale = float(np.std(rate[rate_valid]))
        if not all(
            math.isfinite(value) and value > 0 for value in (glucose_scale, rate_scale)
        ):
            raise ValueError("invalid train-only physiological calibration")

    def tensor_batch(
        index: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(glucose[index]).to(device),
            torch.from_numpy(observed[index]).to(device),
            torch.from_numpy(circadian[index]).to(device),
        )

    dynamics_target_center = None
    dynamics_target_scale = None
    if args.dynamics_target_weight > 0:
        with torch.no_grad():
            train_dynamics_targets = scale_invariant_dynamics_targets(
                torch.from_numpy(glucose[train_index]),
                torch.from_numpy(observed[train_index]),
            )
        dynamics_target_center = train_dynamics_targets.mean(dim=0).to(device)
        dynamics_target_scale = (
            train_dynamics_targets.std(dim=0, unbiased=False).clamp_min(1e-3).to(device)
        )

    model = GlucoFMLike(
        sparse_aware_masking=args.sparse_aware_masking,
        balance_components=args.balance_components,
        calibrate_physiological_units=calibrate_physiological_units,
        relative_auxiliary_features=args.relative_auxiliary_features,
        dual_reference_features=args.dual_reference_features,
        absolute_reference_weight=args.absolute_reference_weight,
        glucose_center=glucose_center,
        glucose_scale=glucose_scale,
        rate_scale=rate_scale,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        mean_shortcut_weight=args.mean_shortcut_weight,
        absolute_subspace_dim=args.absolute_subspace_dim,
        factorized_output=args.factorized_output,
        normalize_dynamic_output=args.normalize_dynamic_output,
        cadence_adapter=args.cadence_adapter,
        factorized_dynamic_weight=args.factorized_dynamic_weight,
        factorized_absolute_weight=args.factorized_absolute_weight,
        absolute_feature_center=absolute_feature_center,
        absolute_feature_scale=absolute_feature_scale,
    ).to(device)
    initial_checkpoint_sha256 = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial_checkpoint["model"])
        initial_checkpoint_sha256 = hashlib.sha256(
            args.initial_checkpoint.read_bytes()
        ).hexdigest()
    embedding_anchor = None
    embedding_anchor_mean = None
    embedding_anchor_scale = None
    if args.embedding_anchor_weight > 0:
        embedding_anchor = copy.deepcopy(model.online).eval().requires_grad_(False)
        anchor_embeddings = []
        with torch.no_grad():
            for left in range(0, len(train_index), args.batch_size):
                index = train_index[left : left + args.batch_size]
                anchor_embeddings.append(
                    embedding_anchor.embed(*tensor_batch(index)).cpu()
                )
        anchor_embedding = torch.cat(anchor_embeddings).to(torch.float32)
        embedding_anchor_mean = anchor_embedding.mean(dim=0).to(device)
        embedding_anchor_scale = (
            anchor_embedding.std(dim=0, unbiased=False).clamp_min(1e-3).to(device)
        )
    dynamic_teacher = None
    dynamic_projector = None
    dynamic_teacher_mean = None
    dynamic_teacher_std = None
    dynamic_teacher_sha256 = None
    dynamics_predictor = None
    if args.dynamics_target_weight > 0:
        dynamics_predictor = torch.nn.Linear(128, 10).to(device)
    source_classifier = None
    source_class_weights = None
    if args.source_adversarial_weight > 0:
        source_input_dimensions = 127 if args.factorized_output else 128
        if args.condition_source_on_absolute_features:
            source_input_dimensions += 48
        source_classifier = torch.nn.Linear(
            source_input_dimensions, len(eligible_source_codes)
        ).to(device)
        if args.balance_source_classes:
            counts = torch.tensor(
                [source_train_counts[code] for code in eligible_source_codes],
                dtype=torch.float32,
                device=device,
            )
            source_class_weights = counts.sum() / (len(counts) * counts)

    def build_source_features(
        embedding: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        dynamic_dimensions = 127 if args.factorized_output else 128
        dynamic = torch.nn.functional.layer_norm(
            embedding[eligible, :dynamic_dimensions],
            (dynamic_dimensions,),
        )
        if not args.condition_source_on_absolute_features:
            return dynamic
        absolute = model.online.absolute_features(
            values[eligible], mask[eligible]
        ).detach()
        return torch.cat([dynamic, absolute], dim=1)

    if args.dynamic_distillation_weight > 0:
        teacher_checkpoint = torch.load(
            args.dynamic_teacher_checkpoint, map_location="cpu", weights_only=False
        )
        teacher_config = teacher_checkpoint.get("config", {})
        dynamic_teacher_model = GlucoFMLike(
            balance_components=bool(teacher_config.get("balance_components", False)),
            calibrate_physiological_units=bool(
                teacher_config.get("calibrate_physiological_units", False)
            ),
            relative_auxiliary_features=bool(
                args.dynamic_teacher_relative_auxiliary_features
                or teacher_config.get("relative_auxiliary_features", False)
            ),
            dual_reference_features=bool(
                teacher_config.get("dual_reference_features", False)
            ),
            factorized_output=bool(teacher_config.get("factorized_output", False)),
            normalize_dynamic_output=bool(
                teacher_config.get("normalize_dynamic_output", False)
            ),
            cadence_adapter=bool(teacher_config.get("cadence_adapter", False)),
            factorized_dynamic_weight=float(
                teacher_config.get("factorized_dynamic_weight", 0.0)
            ),
            factorized_absolute_weight=float(
                teacher_config.get("factorized_absolute_weight", 0.0)
            ),
            absolute_reference_weight=float(
                teacher_config.get("absolute_reference_weight", 1.0)
            ),
            glucose_center=float(teacher_config.get("glucose_center", 0.0)),
            glucose_scale=float(teacher_config.get("glucose_scale", 1.0)),
            rate_scale=float(teacher_config.get("rate_scale", 1.0)),
        )
        dynamic_teacher_model.online.load_state_dict(teacher_checkpoint["online"])
        dynamic_teacher = (
            dynamic_teacher_model.online.to(device).eval().requires_grad_(False)
        )
        dynamic_projector = torch.nn.Linear(128, 128).to(device)
        torch.nn.init.eye_(dynamic_projector.weight)
        torch.nn.init.zeros_(dynamic_projector.bias)
        digest = hashlib.sha256()
        with args.dynamic_teacher_checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        dynamic_teacher_sha256 = digest.hexdigest()

        teacher_embeddings = []
        with torch.no_grad():
            for left in range(0, len(train_index), args.batch_size):
                index = train_index[left : left + args.batch_size]
                teacher_embeddings.append(
                    dynamic_teacher.embed(*tensor_batch(index)).cpu()
                )
        teacher_embedding = torch.cat(teacher_embeddings).to(torch.float32)
        dynamic_teacher_mean = teacher_embedding.mean(dim=0).to(device)
        dynamic_teacher_std = (
            teacher_embedding.std(dim=0, unbiased=False).clamp_min(1e-3).to(device)
        )
    sigma_parameter = model.online.embedder.sigma_rho
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter is not sigma_parameter
    ]
    if dynamic_projector is not None:
        regular_parameters.extend(dynamic_projector.parameters())
    if dynamics_predictor is not None:
        regular_parameters.extend(dynamics_predictor.parameters())
    if source_classifier is not None and args.source_adversarial_mode == "joint":
        regular_parameters.extend(source_classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": regular_parameters, "lr": 1e-4, "weight_decay": 1e-2},
            {"params": [sigma_parameter], "lr": 1e-3, "weight_decay": 0.0},
        ]
    )
    source_optimizer = None
    if source_classifier is not None and args.source_adversarial_mode == "alternating":
        source_optimizer = torch.optim.AdamW(
            source_classifier.parameters(),
            lr=args.source_classifier_learning_rate,
            weight_decay=1e-2,
        )
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_epoch = -1
    global_step = 0
    total_steps = args.epochs * math.ceil(len(train_index) / args.batch_size)
    if args.max_train_steps is not None:
        total_steps = min(total_steps, args.max_train_steps)
    started = time.monotonic()

    stop = False
    for epoch in range(args.epochs):
        model.train()
        train_total: list[float] = []
        train_mcr: list[float] = []
        train_td: list[float] = []
        train_variance: list[float] = []
        train_covariance: list[float] = []
        train_mean_shortcut: list[float] = []
        train_factorized_dynamic: list[float] = []
        train_factorized_absolute: list[float] = []
        train_dynamic_distillation: list[float] = []
        train_dynamics_target: list[float] = []
        train_embedding_anchor: list[float] = []
        train_dynamics_gradient_cosine: list[float] = []
        train_dynamics_gradient_conflict: list[float] = []
        train_source_adversarial: list[float] = []
        train_source_accuracy: list[float] = []
        train_sensor_counterfactual: list[float] = []
        train_trait_consistency: list[float] = []
        order = (
            cadence_cohort_balanced_indices(
                train_index,
                cohorts,
                cadences,
                size=len(train_index),
                rng=rng,
                cadence_15_share=args.cadence_15_share,
                cohort_temperature=args.cohort_sampling_temperature,
                group_ids=subject_ids if args.balance_subject_within_cohort else None,
            )
            if args.balance_cohort_cadence
            else rng.permutation(train_index)
        )
        for left in range(0, len(order), args.batch_size):
            index = order[left : left + args.batch_size]
            values, mask, start = tensor_batch(index)
            values, mask = augment_cgm(values, mask)
            loss, parts = model.pretraining_loss(values, mask, start)
            dynamic_distillation = torch.zeros((), device=device)
            dynamics_target_loss = torch.zeros((), device=device)
            embedding_anchor_loss = torch.zeros((), device=device)
            source_adversarial = torch.zeros((), device=device)
            source_accuracy = float("nan")
            sensor_counterfactual = torch.zeros((), device=device)
            trait_consistency = torch.zeros((), device=device)
            if dynamic_teacher is not None and dynamic_projector is not None:
                student_embedding = model.online.embed(values, mask, start)
                with torch.no_grad():
                    teacher_embedding = dynamic_teacher.embed(values, mask, start)
                    teacher_embedding = (
                        teacher_embedding - dynamic_teacher_mean
                    ) / dynamic_teacher_std
                dynamic_distillation = torch.nn.functional.smooth_l1_loss(
                    dynamic_projector(student_embedding), teacher_embedding
                )
                loss = loss + args.dynamic_distillation_weight * dynamic_distillation
            if dynamics_predictor is not None:
                dynamics_embedding = model.online.embed(values, mask, start)
                if args.dynamics_target_normalize_input:
                    dynamics_embedding = torch.nn.functional.layer_norm(
                        dynamics_embedding, (dynamics_embedding.shape[1],)
                    )
                dynamics_target = scale_invariant_dynamics_targets(values, mask)
                standardized_target = (
                    dynamics_target - dynamics_target_center
                ) / dynamics_target_scale
                dynamics_target_loss = torch.nn.functional.smooth_l1_loss(
                    dynamics_predictor(dynamics_embedding), standardized_target
                )
                loss = loss + args.dynamics_target_weight * dynamics_target_loss
            if embedding_anchor is not None:
                student_embedding = model.online.embed(values, mask, start)
                with torch.no_grad():
                    anchor_embedding = embedding_anchor.embed(values, mask, start)
                embedding_anchor_loss = standardized_embedding_anchor_loss(
                    student_embedding,
                    anchor_embedding,
                    embedding_anchor_mean,
                    embedding_anchor_scale,
                    args.embedding_anchor_loss,
                )
                loss = loss + args.embedding_anchor_weight * embedding_anchor_loss
            if source_classifier is not None:
                target = torch.from_numpy(source_target[index]).to(device)
                eligible = target.ge(0)
                if eligible.any():
                    if source_optimizer is not None:
                        with torch.no_grad():
                            classifier_embedding = model.online.embed(
                                values, mask, start
                            )
                            classifier_features = build_source_features(
                                classifier_embedding, values, mask, eligible
                            )
                        source_optimizer.zero_grad(set_to_none=True)
                        classifier_logits = source_classifier(classifier_features)
                        classifier_loss = torch.nn.functional.cross_entropy(
                            classifier_logits,
                            target[eligible],
                            weight=source_class_weights,
                        )
                        classifier_loss.backward()
                        classifier_grad_norm = torch.nn.utils.clip_grad_norm_(
                            list(source_classifier.parameters()), 1.0
                        )
                        if not math.isfinite(float(classifier_grad_norm)):
                            raise RuntimeError(
                                "non-finite source classifier gradient norm"
                            )
                        source_optimizer.step()
                        source_classifier.requires_grad_(False)
                    student_embedding = model.online.embed(values, mask, start)
                    classifier_features = build_source_features(
                        student_embedding, values, mask, eligible
                    )
                    logits = source_classifier(reverse_gradient(classifier_features))
                    source_adversarial = torch.nn.functional.cross_entropy(
                        logits,
                        target[eligible],
                        weight=source_class_weights,
                    )
                    source_accuracy = float(
                        logits.argmax(dim=1)
                        .eq(target[eligible])
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                    loss = loss + args.source_adversarial_weight * source_adversarial
            if args.sensor_counterfactual_weight > 0:
                if args.sensor_counterfactual_target_mode == "ema_dense":
                    with torch.no_grad():
                        reference_embedding = model.target.embed(values, mask, start)[
                            :, :127
                        ]
                else:
                    reference_embedding = model.online.embed(values, mask, start)[
                        :, :127
                    ]
                counterfactual_view = (
                    counterfactual_cadence_view
                    if args.sensor_counterfactual_augmentation == "cadence_only"
                    else counterfactual_sensor_view
                )
                counterfactual_values, counterfactual_mask = counterfactual_view(
                    values, mask
                )
                counterfactual_embedding = model.online.embed(
                    counterfactual_values, counterfactual_mask, start
                )[:, :127]
                sensor_counterfactual = torch.nn.functional.smooth_l1_loss(
                    counterfactual_embedding, reference_embedding
                )
                loss = loss + (
                    args.sensor_counterfactual_weight * sensor_counterfactual
                )
            if args.trait_consistency_weight > 0:
                partner = trait_partner[index]
                eligible = partner >= 0
                if np.any(eligible):
                    eligible_tensor = torch.from_numpy(eligible).to(device)
                    anchor = (
                        values[eligible_tensor],
                        mask[eligible_tensor],
                        start[eligible_tensor],
                    )
                    trait_consistency = ema_cross_day_trait_loss(
                        model, anchor, tensor_batch(partner[eligible])
                    )
                    loss = loss + args.trait_consistency_weight * trait_consistency
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pretraining loss")
            optimizer.zero_grad(set_to_none=True)
            optimization_parameters = regular_parameters + [sigma_parameter]
            if (
                dynamics_predictor is not None
                and args.dynamics_target_gradient_mode == "pcgrad"
            ):
                base_loss = loss - args.dynamics_target_weight * dynamics_target_loss
                base_gradients = torch.autograd.grad(
                    base_loss,
                    optimization_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                auxiliary_gradients = torch.autograd.grad(
                    dynamics_target_loss,
                    optimization_parameters,
                    allow_unused=True,
                )
                merged, gradient_cosine, gradient_conflict = merge_pcgrad(
                    base_gradients,
                    auxiliary_gradients,
                    args.dynamics_target_weight,
                )
                for parameter, gradient in zip(optimization_parameters, merged):
                    parameter.grad = gradient
                train_dynamics_gradient_cosine.append(gradient_cosine)
                train_dynamics_gradient_conflict.append(float(gradient_conflict))
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(optimization_parameters, 1.0)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            if source_optimizer is not None:
                source_classifier.requires_grad_(True)
            progress = min(1.0, global_step / max(1, total_steps - 1))
            momentum = 1.0 - (1.0 - 0.997) * (math.cos(math.pi * progress) + 1.0) / 2.0
            model.update_target(momentum)
            global_step += 1
            train_total.append(float(loss.detach().cpu()))
            train_mcr.append(float(parts["mcr"].cpu()))
            train_td.append(float(parts["td"].cpu()))
            train_variance.append(float(parts["variance"].cpu()))
            train_covariance.append(float(parts["covariance"].cpu()))
            train_mean_shortcut.append(float(parts["mean_shortcut"].cpu()))
            train_factorized_dynamic.append(float(parts["factorized_dynamic"].cpu()))
            train_factorized_absolute.append(float(parts["factorized_absolute"].cpu()))
            train_dynamic_distillation.append(
                float(dynamic_distillation.detach().cpu())
            )
            train_dynamics_target.append(float(dynamics_target_loss.detach().cpu()))
            train_embedding_anchor.append(float(embedding_anchor_loss.detach().cpu()))
            train_source_adversarial.append(float(source_adversarial.detach().cpu()))
            train_source_accuracy.append(source_accuracy)
            train_sensor_counterfactual.append(
                float(sensor_counterfactual.detach().cpu())
            )
            train_trait_consistency.append(float(trait_consistency.detach().cpu()))
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop = True
                break

        model.eval()
        validation_total: list[float] = []
        validation_mcr: list[float] = []
        validation_td: list[float] = []
        validation_variance: list[float] = []
        validation_covariance: list[float] = []
        validation_mean_shortcut: list[float] = []
        validation_factorized_dynamic: list[float] = []
        validation_factorized_absolute: list[float] = []
        validation_dynamic_distillation: list[float] = []
        validation_dynamics_target: list[float] = []
        validation_embedding_anchor: list[float] = []
        validation_embedding_anchor_squared_error_sum = None
        validation_embedding_anchor_count = 0
        validation_source_adversarial: list[float] = []
        validation_source_accuracy: list[float] = []
        validation_sensor_counterfactual: list[float] = []
        validation_trait_consistency: list[float] = []
        validation_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            torch.manual_seed(args.seed + 100_000)
            for left in range(0, len(validation_index), args.batch_size):
                index = validation_index[left : left + args.batch_size]
                loss, parts = model.pretraining_loss(*tensor_batch(index))
                dynamic_distillation = torch.zeros((), device=device)
                dynamics_target_loss = torch.zeros((), device=device)
                embedding_anchor_loss = torch.zeros((), device=device)
                source_adversarial = torch.zeros((), device=device)
                source_accuracy = float("nan")
                sensor_counterfactual = torch.zeros((), device=device)
                trait_consistency = torch.zeros((), device=device)
                if dynamic_teacher is not None and dynamic_projector is not None:
                    batch = tensor_batch(index)
                    student_embedding = model.online.embed(*batch)
                    teacher_embedding = dynamic_teacher.embed(*batch)
                    teacher_embedding = (
                        teacher_embedding - dynamic_teacher_mean
                    ) / dynamic_teacher_std
                    dynamic_distillation = torch.nn.functional.smooth_l1_loss(
                        dynamic_projector(student_embedding), teacher_embedding
                    )
                    loss = (
                        loss + args.dynamic_distillation_weight * dynamic_distillation
                    )
                if dynamics_predictor is not None:
                    batch = tensor_batch(index)
                    dynamics_embedding = model.online.embed(*batch)
                    if args.dynamics_target_normalize_input:
                        dynamics_embedding = torch.nn.functional.layer_norm(
                            dynamics_embedding, (dynamics_embedding.shape[1],)
                        )
                    dynamics_target = scale_invariant_dynamics_targets(
                        batch[0], batch[1]
                    )
                    standardized_target = (
                        dynamics_target - dynamics_target_center
                    ) / dynamics_target_scale
                    dynamics_target_loss = torch.nn.functional.smooth_l1_loss(
                        dynamics_predictor(dynamics_embedding), standardized_target
                    )
                    loss = loss + args.dynamics_target_weight * dynamics_target_loss
                if embedding_anchor is not None:
                    batch = tensor_batch(index)
                    student_embedding = model.online.embed(*batch)
                    anchor_embedding = embedding_anchor.embed(*batch)
                    embedding_anchor_loss = standardized_embedding_anchor_loss(
                        student_embedding,
                        anchor_embedding,
                        embedding_anchor_mean,
                        embedding_anchor_scale,
                        args.embedding_anchor_loss,
                    )
                    standardized_difference = (
                        student_embedding - anchor_embedding
                    ) / embedding_anchor_scale
                    batch_squared_error_sum = standardized_difference.square().sum(
                        dim=0
                    )
                    if validation_embedding_anchor_squared_error_sum is None:
                        validation_embedding_anchor_squared_error_sum = (
                            batch_squared_error_sum.cpu()
                        )
                    else:
                        validation_embedding_anchor_squared_error_sum += (
                            batch_squared_error_sum.cpu()
                        )
                    validation_embedding_anchor_count += len(index)
                    loss = loss + args.embedding_anchor_weight * embedding_anchor_loss
                if source_classifier is not None:
                    target = torch.from_numpy(source_target[index]).to(device)
                    eligible = target.ge(0)
                    if eligible.any():
                        batch = tensor_batch(index)
                        student_embedding = model.online.embed(*batch)
                        classifier_features = build_source_features(
                            student_embedding, batch[0], batch[1], eligible
                        )
                        logits = source_classifier(classifier_features)
                        source_adversarial = torch.nn.functional.cross_entropy(
                            logits,
                            target[eligible],
                            weight=source_class_weights,
                        )
                        source_accuracy = float(
                            logits.argmax(dim=1)
                            .eq(target[eligible])
                            .float()
                            .mean()
                            .cpu()
                        )
                if args.sensor_counterfactual_weight > 0:
                    batch = tensor_batch(index)
                    if args.sensor_counterfactual_target_mode == "ema_dense":
                        reference_embedding = model.target.embed(*batch)[:, :127]
                    else:
                        reference_embedding = model.online.embed(*batch)[:, :127]
                    counterfactual_view = (
                        counterfactual_cadence_view
                        if args.sensor_counterfactual_augmentation == "cadence_only"
                        else counterfactual_sensor_view
                    )
                    counterfactual_values, counterfactual_mask = counterfactual_view(
                        batch[0], batch[1]
                    )
                    counterfactual_embedding = model.online.embed(
                        counterfactual_values, counterfactual_mask, batch[2]
                    )[:, :127]
                    sensor_counterfactual = torch.nn.functional.smooth_l1_loss(
                        counterfactual_embedding, reference_embedding
                    )
                    loss = loss + (
                        args.sensor_counterfactual_weight * sensor_counterfactual
                    )
                if args.trait_consistency_weight > 0:
                    partner = trait_partner[index]
                    eligible = partner >= 0
                    if np.any(eligible):
                        trait_consistency = ema_cross_day_trait_loss(
                            model,
                            tensor_batch(index[eligible]),
                            tensor_batch(partner[eligible]),
                        )
                        loss = loss + args.trait_consistency_weight * trait_consistency
                validation_total.append(float(loss.cpu()))
                validation_mcr.append(float(parts["mcr"].cpu()))
                validation_td.append(float(parts["td"].cpu()))
                validation_variance.append(float(parts["variance"].cpu()))
                validation_covariance.append(float(parts["covariance"].cpu()))
                validation_mean_shortcut.append(float(parts["mean_shortcut"].cpu()))
                validation_factorized_dynamic.append(
                    float(parts["factorized_dynamic"].cpu())
                )
                validation_factorized_absolute.append(
                    float(parts["factorized_absolute"].cpu())
                )
                validation_dynamic_distillation.append(
                    float(dynamic_distillation.cpu())
                )
                validation_dynamics_target.append(float(dynamics_target_loss.cpu()))
                validation_embedding_anchor.append(float(embedding_anchor_loss.cpu()))
                validation_source_adversarial.append(float(source_adversarial.cpu()))
                validation_source_accuracy.append(source_accuracy)
                validation_sensor_counterfactual.append(
                    float(sensor_counterfactual.cpu())
                )
                validation_trait_consistency.append(float(trait_consistency.cpu()))
                validation_embeddings.append(
                    model.online.embed(*tensor_batch(index)).cpu().numpy()
                )
        embedding = np.concatenate(validation_embeddings).astype(np.float64)
        singular_values = np.linalg.svd(
            embedding - embedding.mean(axis=0), compute_uv=False
        )
        explained = np.square(singular_values)
        explained /= explained.sum()
        effective_rank = float(np.exp(-np.sum(explained * np.log(explained + 1e-12))))
        native5_dynamic = embedding[cadences[validation_index] == 5, :127]
        dynamic_singular_values = np.linalg.svd(
            native5_dynamic - native5_dynamic.mean(axis=0), compute_uv=False
        )
        dynamic_explained = np.square(dynamic_singular_values)
        dynamic_explained /= dynamic_explained.sum()
        native5_dynamic_effective_rank = float(
            np.exp(-np.sum(dynamic_explained * np.log(dynamic_explained + 1e-12)))
        )
        validation_embedding_anchor_max_dimension_rms = 0.0
        if validation_embedding_anchor_squared_error_sum is not None:
            validation_embedding_anchor_max_dimension_rms = float(
                (
                    validation_embedding_anchor_squared_error_sum
                    / validation_embedding_anchor_count
                )
                .sqrt()
                .max()
            )
        row = {
            "epoch": epoch,
            "step": global_step,
            "sigma_grid_steps": float(model.online.embedder.sigma.detach().cpu()),
            "train_loss": float(np.mean(train_total)),
            "train_mcr": float(np.mean(train_mcr)),
            "train_td": float(np.mean(train_td)),
            "train_variance": float(np.mean(train_variance)),
            "train_covariance": float(np.mean(train_covariance)),
            "train_mean_shortcut": float(np.mean(train_mean_shortcut)),
            "train_factorized_dynamic": float(np.mean(train_factorized_dynamic)),
            "train_factorized_absolute": float(np.mean(train_factorized_absolute)),
            "train_dynamic_distillation": float(np.mean(train_dynamic_distillation)),
            "train_dynamics_target": float(np.mean(train_dynamics_target)),
            "train_embedding_anchor": float(np.mean(train_embedding_anchor)),
            "train_dynamics_gradient_cosine": (
                float(np.mean(train_dynamics_gradient_cosine))
                if train_dynamics_gradient_cosine
                else float("nan")
            ),
            "train_dynamics_gradient_conflict_fraction": (
                float(np.mean(train_dynamics_gradient_conflict))
                if train_dynamics_gradient_conflict
                else float("nan")
            ),
            "train_source_adversarial": float(np.mean(train_source_adversarial)),
            "train_source_accuracy": float(np.nanmean(train_source_accuracy)),
            "train_sensor_counterfactual": float(np.mean(train_sensor_counterfactual)),
            "train_trait_consistency": float(np.mean(train_trait_consistency)),
            "validation_loss": float(np.mean(validation_total)),
            "validation_mcr": float(np.mean(validation_mcr)),
            "validation_td": float(np.mean(validation_td)),
            "validation_variance": float(np.mean(validation_variance)),
            "validation_covariance": float(np.mean(validation_covariance)),
            "validation_mean_shortcut": float(np.mean(validation_mean_shortcut)),
            "validation_factorized_dynamic": float(
                np.mean(validation_factorized_dynamic)
            ),
            "validation_factorized_absolute": float(
                np.mean(validation_factorized_absolute)
            ),
            "validation_dynamic_distillation": float(
                np.mean(validation_dynamic_distillation)
            ),
            "validation_dynamics_target": float(np.mean(validation_dynamics_target)),
            "validation_embedding_anchor": float(np.mean(validation_embedding_anchor)),
            "validation_embedding_anchor_max_dimension_rms": (
                validation_embedding_anchor_max_dimension_rms
            ),
            "validation_source_adversarial": float(
                np.mean(validation_source_adversarial)
            ),
            "validation_source_accuracy": float(np.nanmean(validation_source_accuracy)),
            "validation_sensor_counterfactual": float(
                np.mean(validation_sensor_counterfactual)
            ),
            "validation_trait_consistency": float(
                np.mean(validation_trait_consistency)
            ),
            "validation_effective_rank": effective_rank,
            "validation_top_pc_share": float(explained[0]),
            "validation_native5_dynamic_effective_rank": (
                native5_dynamic_effective_rank
            ),
            "validation_native5_dynamic_top_pc_share": float(dynamic_explained[0]),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint = {
            "implementation": "GlucoFM-like paper reimplementation; not official Google code",
            "model": model.state_dict(),
            "online": model.online.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "validation_loss": row["validation_loss"],
            "config": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "mask_ratio": [0.5, 0.6],
                "sparse_aware_masking": args.sparse_aware_masking,
                "patch_mask_policy": "sparse_point_budget_v1"
                if args.sparse_aware_masking
                else "fixed_patch_ratio_v1",
                "mcr_weight": 1.0,
                "td_weight": 1.0,
                "seed": args.seed,
                "balance_components": args.balance_components,
                "calibrate_physiological_units": calibrate_physiological_units,
                "relative_auxiliary_features": args.relative_auxiliary_features,
                "dual_reference_features": args.dual_reference_features,
                "absolute_reference_weight": args.absolute_reference_weight,
                "glucose_center": glucose_center,
                "glucose_scale": glucose_scale,
                "rate_scale": rate_scale,
                "variance_weight": args.variance_weight,
                "covariance_weight": args.covariance_weight,
                "mean_shortcut_weight": args.mean_shortcut_weight,
                "absolute_subspace_dim": args.absolute_subspace_dim,
                "factorized_output": args.factorized_output,
                "normalize_dynamic_output": args.normalize_dynamic_output,
                "cadence_adapter": args.cadence_adapter,
                "factorized_dynamic_weight": args.factorized_dynamic_weight,
                "factorized_absolute_weight": args.factorized_absolute_weight,
                "dynamic_distillation_weight": args.dynamic_distillation_weight,
                "dynamics_target_weight": args.dynamics_target_weight,
                "dynamics_target_normalize_input": (
                    args.dynamics_target_normalize_input
                ),
                "dynamics_target_gradient_mode": args.dynamics_target_gradient_mode,
                "maximum_validation_dynamics_target": (
                    args.maximum_validation_dynamics_target
                ),
                "maximum_validation_mcr_td": args.maximum_validation_mcr_td,
                "embedding_anchor_weight": args.embedding_anchor_weight,
                "embedding_anchor_loss": args.embedding_anchor_loss,
                "maximum_validation_embedding_anchor": (
                    args.maximum_validation_embedding_anchor
                ),
                "maximum_validation_embedding_anchor_max_dimension_rms": (
                    args.maximum_validation_embedding_anchor_max_dimension_rms
                ),
                "initial_checkpoint_sha256": initial_checkpoint_sha256,
                "dynamic_teacher_sha256": dynamic_teacher_sha256,
                "source_adversarial_weight": args.source_adversarial_weight,
                "source_adversarial_mode": args.source_adversarial_mode,
                "source_classifier_learning_rate": (
                    args.source_classifier_learning_rate
                ),
                "balance_source_classes": args.balance_source_classes,
                "condition_source_on_absolute_features": (
                    args.condition_source_on_absolute_features
                ),
                "sensor_counterfactual_weight": args.sensor_counterfactual_weight,
                "sensor_counterfactual_target_mode": args.sensor_counterfactual_target_mode,
                "sensor_counterfactual_augmentation": args.sensor_counterfactual_augmentation,
                "minimum_source_train_windows": args.minimum_source_train_windows,
                "minimum_validation_native5_dynamic_effective_rank": (
                    args.minimum_validation_native5_dynamic_effective_rank
                ),
                "trait_consistency_weight": args.trait_consistency_weight,
                "train_trait_pair_anchors": train_trait_pair_anchors,
                "validation_trait_pair_anchors": validation_trait_pair_anchors,
                "eligible_source_codes": eligible_source_codes,
                "source_train_counts": source_train_counts,
                "dynamic_teacher_relative_auxiliary_features": (
                    args.dynamic_teacher_relative_auxiliary_features
                ),
                "balance_cohort_cadence": args.balance_cohort_cadence,
                "cadence_15_share": args.cadence_15_share,
                "cohort_sampling_temperature": args.cohort_sampling_temperature,
                "balance_subject_within_cohort": args.balance_subject_within_cohort,
            },
        }
        if dynamic_projector is not None:
            checkpoint["dynamic_projector"] = dynamic_projector.state_dict()
        if dynamics_predictor is not None:
            checkpoint["dynamics_predictor"] = dynamics_predictor.state_dict()
            checkpoint["dynamics_target_center"] = dynamics_target_center.cpu()
            checkpoint["dynamics_target_scale"] = dynamics_target_scale.cpu()
        if source_classifier is not None:
            checkpoint["source_classifier"] = source_classifier.state_dict()
        checkpoint_geometry_passed = (
            native5_dynamic_effective_rank
            >= args.minimum_validation_native5_dynamic_effective_rank
            and row["validation_mcr"] + row["validation_td"]
            <= args.maximum_validation_mcr_td
            and row["validation_dynamics_target"]
            <= args.maximum_validation_dynamics_target
            and row["validation_embedding_anchor"]
            <= args.maximum_validation_embedding_anchor
            and row["validation_embedding_anchor_max_dimension_rms"]
            <= args.maximum_validation_embedding_anchor_max_dimension_rms
        )
        if checkpoint_geometry_passed and row["validation_loss"] < best_validation:
            best_validation = row["validation_loss"]
            best_epoch = epoch
            torch.save(checkpoint, args.output_dir / "best.pt")
            (args.output_dir / "best.pt").chmod(0o600)
        if epoch % 10 == 0 or epoch == args.epochs - 1 or stop:
            torch.save(checkpoint, args.output_dir / "latest.pt")
            (args.output_dir / "latest.pt").chmod(0o600)
        if stop:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "acceptance_passed": bool(best_epoch >= 0 and math.isfinite(best_validation)),
        "implementation": "GlucoFM-like paper reimplementation; not official Google code",
        "paper_architecture_alignment": {
            "dual_state_event_stream": True,
            "causal_mask_aware_gaussian": True,
            "context_layers": 3,
            "hidden_dimension": 128,
            "heads": 4,
            "predictor_layers": 1,
            "mcr_and_td_objectives": True,
            "physical_mask_preserved": True,
        },
        "representation_regularization": {
            "balance_components": args.balance_components,
            "calibrate_physiological_units": calibrate_physiological_units,
            "relative_auxiliary_features": args.relative_auxiliary_features,
            "dual_reference_features": args.dual_reference_features,
            "absolute_reference_weight": args.absolute_reference_weight,
            "glucose_center": glucose_center,
            "glucose_scale": glucose_scale,
            "rate_scale": rate_scale,
            "variance_weight": args.variance_weight,
            "covariance_weight": args.covariance_weight,
            "mean_shortcut_weight": args.mean_shortcut_weight,
            "absolute_subspace_dim": args.absolute_subspace_dim,
            "factorized_output": args.factorized_output,
            "normalize_dynamic_output": args.normalize_dynamic_output,
            "cadence_adapter": args.cadence_adapter,
            "factorized_dynamic_weight": args.factorized_dynamic_weight,
            "factorized_absolute_weight": args.factorized_absolute_weight,
            "dynamic_distillation_weight": args.dynamic_distillation_weight,
            "dynamics_target_weight": args.dynamics_target_weight,
            "dynamics_target_normalize_input": (args.dynamics_target_normalize_input),
            "dynamics_target_gradient_mode": args.dynamics_target_gradient_mode,
            "maximum_validation_dynamics_target": (
                args.maximum_validation_dynamics_target
            ),
            "maximum_validation_mcr_td": args.maximum_validation_mcr_td,
            "embedding_anchor_weight": args.embedding_anchor_weight,
            "embedding_anchor_loss": args.embedding_anchor_loss,
            "maximum_validation_embedding_anchor": (
                args.maximum_validation_embedding_anchor
            ),
            "maximum_validation_embedding_anchor_max_dimension_rms": (
                args.maximum_validation_embedding_anchor_max_dimension_rms
            ),
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "dynamic_teacher_sha256": dynamic_teacher_sha256,
            "source_adversarial_weight": args.source_adversarial_weight,
            "source_adversarial_mode": args.source_adversarial_mode,
            "source_classifier_learning_rate": args.source_classifier_learning_rate,
            "balance_source_classes": args.balance_source_classes,
            "condition_source_on_absolute_features": (
                args.condition_source_on_absolute_features
            ),
            "sensor_counterfactual_weight": args.sensor_counterfactual_weight,
            "sensor_counterfactual_target_mode": args.sensor_counterfactual_target_mode,
            "sensor_counterfactual_augmentation": args.sensor_counterfactual_augmentation,
            "minimum_source_train_windows": args.minimum_source_train_windows,
            "minimum_validation_native5_dynamic_effective_rank": (
                args.minimum_validation_native5_dynamic_effective_rank
            ),
            "trait_consistency_weight": args.trait_consistency_weight,
            "train_trait_pair_anchors": train_trait_pair_anchors,
            "validation_trait_pair_anchors": validation_trait_pair_anchors,
            "eligible_source_codes": eligible_source_codes,
            "source_train_counts": source_train_counts,
            "dynamic_teacher_relative_auxiliary_features": (
                args.dynamic_teacher_relative_auxiliary_features
            ),
            "balance_cohort_cadence": args.balance_cohort_cadence,
            "cadence_15_share": args.cadence_15_share,
            "cohort_sampling_temperature": args.cohort_sampling_temperature,
            "balance_subject_within_cohort": args.balance_subject_within_cohort,
            "status": "experimental extension; not part of GlucoFM",
        },
        "known_implementation_uncertainty": (
            "The paper does not publish code, exact layer ordering, normalization placement, "
            "initialization, augmentation view pairing, or optimizer schedule."
        ),
        "best_epoch": best_epoch,
        "sparse_aware_masking": args.sparse_aware_masking,
        "patch_mask_policy": "sparse_point_budget_v1"
        if args.sparse_aware_masking
        else "fixed_patch_ratio_v1",
        "best_validation_loss": best_validation,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "input_snapshot": str(args.npz),
        "snapshot_is_complete_history": False,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        + (
            sum(p.numel() for p in dynamic_projector.parameters())
            if dynamic_projector
            else 0
        )
        + (
            sum(p.numel() for p in dynamics_predictor.parameters())
            if dynamics_predictor
            else 0
        )
        + (
            sum(p.numel() for p in source_classifier.parameters())
            if source_classifier
            else 0
        ),
        "total_pretraining_parameters": sum(p.numel() for p in model.parameters())
        + (
            sum(p.numel() for p in dynamic_projector.parameters())
            if dynamic_projector
            else 0
        )
        + (
            sum(p.numel() for p in dynamics_predictor.parameters())
            if dynamics_predictor
            else 0
        )
        + (
            sum(p.numel() for p in source_classifier.parameters())
            if source_classifier
            else 0
        ),
        "train_windows": int(len(train_index)),
        "validation_windows": int(len(validation_index)),
        "test_windows_available_but_not_used": test_windows,
        "test_windows_read_for_optimization": 0,
        "training_cohorts": {
            str(cohort): int(np.sum((cohorts == cohort) & (split == 0)))
            for cohort in np.unique(cohorts)
        },
        "training_native_cadences": {
            str(int(cadence)): int(np.sum((cadences == cadence) & (split == 0)))
            for cadence in np.unique(cadences)
        },
        "torch_version": torch.__version__,
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path.chmod(0o600)
    print(
        json.dumps({k: v for k, v in result.items() if k != "history"}, sort_keys=True)
    )
    if not result["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
