"""Frozen seed43 FCTM candidate/control training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.models.cross_window_nodes import add_cross_window_nodes
from lingsante_pulse.cgm.models.distribution_phase import DistributionBarycenter
from lingsante_pulse.cgm.models.episodic_evidence import (
    RandomRowRepartitionEpisodes,
    SubjectEpisodes,
    episodic_loss,
)
from lingsante_pulse.cgm.training.source_mixture import (
    SourceMixtureEpisodes,
    load_source_family,
)
from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import statistics_tokens
from lingsante_pulse.cgm.models.memory import (
    ARCHITECTURE,
    PREREGISTERED_STEPS,
    FourierClockTransportMemory,
    freeze_observed_state_backbone,
    parameter_budget,
    spectral_rows,
)
from lingsante_pulse.cgm.models.hierarchical_continuous_time_day_memory import (
    pack_hctdm,
)
from lingsante_pulse.cgm.training.support import validate_memory_support
from lingsante_pulse.cgm.training.final_space_longitudinal import prepare, readout
from lingsante_pulse.cgm.training.longitudinal_statistics_tokens import (
    add_events,
    event_readout,
)
from lingsante_pulse.cgm.training.observation_supported_attention import make_inputs
from lingsante_pulse.cgm.training.parallel_longitudinal import CORPUS_SHA, DAILY_SHA
from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
    _load_source,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
    _align_daily_embedding,
    effective_rank,
    sha256,
)
from lingsante_pulse.cgm.training.targets import raw_hourly_history_targets


BASE_CHECKPOINT_SHA = "3a397cb17de31ce7927f79ea1c87e68e0dfd8d442f6b32b2d40f1c9a09ee9721"
POLICY = "fourier_clock_transport_memory_support_v1"
RAW_ARCHITECTURE = "fourier_clock_transport_memory_raw_hourly_v1"


def target_loss(
    support, support_targets, query, query_targets, main_target, stop_through_fit=False
):
    return episodic_loss(
        support,
        support_targets,
        query,
        query_targets,
        stop_through_fit=stop_through_fit,
        target_split=64 if main_target == "structured_evidence" else 168,
    )


def apply_frozen_daily_normalization(raw, aligned, contract, frozen_contract):
    """Place new rows in the exact daily coordinate system expected by a frozen backbone."""
    values = np.asarray(raw, dtype=np.float32)
    center = np.asarray(frozen_contract["daily_center"])
    scale = np.asarray(frozen_contract["daily_scale"])
    if (
        values.ndim != 2
        or center.shape != (values.shape[1],)
        or scale.shape != center.shape
    ):
        raise ValueError("Frozen daily normalization dimensions do not match")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Frozen daily normalization is not finite and positive")
    output = dict(aligned)
    output["daily"] = ((values - center) / scale).astype(np.float32)
    updated = dict(contract, daily_center=center.copy(), daily_scale=scale.copy())
    return output, updated


def load_memory_initialization(
    model,
    base,
    checkpoint_path,
    expected_sha,
    expected_steps,
    expected_corpus_sha,
    expected_daily_sha,
    expected_backbone_sha,
):
    """Load only a provenance-locked FCTM state on the identical frozen backbone."""
    if sha256(checkpoint_path) != expected_sha:
        raise ValueError("Memory initialization checksum mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = (
        ARCHITECTURE,
        "candidate",
        43,
        expected_steps,
        expected_corpus_sha,
        expected_daily_sha,
        expected_backbone_sha,
    )
    actual = (
        payload.get("architecture"),
        payload.get("kind"),
        payload.get("seed"),
        payload.get("steps"),
        payload.get("corpus_sha256"),
        payload.get("daily_sha256"),
        payload.get("backbone_sha256"),
    )
    if actual != expected:
        raise ValueError("Memory initialization provenance mismatch")
    state = payload.get("model", {})
    if set(state) != set(model.state_dict()):
        raise ValueError("Memory initialization architecture mismatch")
    for name, value in base["model"].items():
        if (
            name.startswith("fctm_")
            or name not in state
            or not torch.equal(value, state[name])
        ):
            raise ValueError("Memory initialization changed the frozen backbone")
    for key in ("daily_center", "daily_scale"):
        np.testing.assert_allclose(
            payload["contract"][key], base["contract"][key], rtol=0, atol=0
        )
    model.load_state_dict(state, strict=True)
    return payload


def load_fctm_initialization(
    model,
    checkpoint_path,
    expected_sha,
    expected_steps,
    expected_corpus_sha,
    expected_daily_sha,
    expected_backbone_sha,
):
    """Load only the FCTM parameters while retaining the current Stage-B backbone."""
    if sha256(checkpoint_path) != expected_sha:
        raise ValueError("FCTM initialization checksum mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual = (
        payload.get("architecture"),
        payload.get("kind"),
        payload.get("steps"),
        payload.get("corpus_sha256"),
        payload.get("daily_sha256"),
        payload.get("backbone_sha256"),
    )
    expected = (
        ARCHITECTURE,
        "candidate",
        expected_steps,
        expected_corpus_sha,
        expected_daily_sha,
        expected_backbone_sha,
    )
    if actual != expected:
        raise ValueError("FCTM initialization provenance mismatch")
    source_state = payload.get("model", {})
    target_state = model.state_dict()
    source_names = {name for name in source_state if name.startswith("fctm_")}
    target_names = {name for name in target_state if name.startswith("fctm_")}
    if source_names != target_names:
        raise ValueError("FCTM initialization architecture mismatch")
    target_state.update({name: source_state[name] for name in source_names})
    model.load_state_dict(target_state, strict=True)
    return payload


def encode(model, packed, rows, track_input_key=None):
    rows = np.asarray(rows, dtype=np.int64)
    device = next(model.parameters()).device
    outputs = []
    tracked = []
    for start in range(0, len(rows), 32):
        selected = {
            key: value[rows[start : start + 32]].to(device, non_blocking=True)
            for key, value in packed.items()
        }
        if track_input_key is not None:
            selected[track_input_key] = (
                selected[track_input_key].detach().requires_grad_(True)
            )
            tracked.append(selected[track_input_key])
        outputs.append(model(**selected)[0])
    output = torch.cat(outputs)
    return (output, tracked) if track_input_key is not None else output


def pack_fourier_memory(data, observation_policy="fixed15_v1"):
    """Return frozen model inputs and the true adjacent-day spectral transport target."""
    memory = pack_hctdm(data, observation_policy=observation_policy)
    with torch.no_grad():
        spectral = spectral_rows(
            torch.from_numpy(memory["hctdm_base"]),
            torch.from_numpy(memory["hctdm_clock"]),
            torch.from_numpy(memory["hctdm_point_mask"]),
            False,
        ).numpy()
    target = np.zeros((len(spectral), 64), np.float32)
    for endpoint in np.flatnonzero(memory["hctdm_active"]):
        days = np.flatnonzero(memory["hctdm_day_mask"][endpoint])
        target[endpoint] = (
            (spectral[endpoint, days[1:]] * spectral[endpoint, days[:-1]])
            .mean(0)
            .reshape(-1)
        )
    return memory, target


def train(args):
    episode_grouping = getattr(args, "episode_grouping", "recorded_identity")
    average_two = getattr(args, "average_two_episodes", False)
    if average_two and episode_grouping != "recorded_identity":
        raise ValueError("Two-episode averaging requires subject-disjoint TER episodes")
    ridge_gradient = getattr(args, "ridge_gradient", "full")
    stop_through_fit = ridge_gradient == "stop_fit"
    main_target = getattr(args, "main_target", "structured_evidence")
    initialize_fctm = getattr(args, "initialize_fctm", None)
    if args.initialize_memory is not None and initialize_fctm is not None:
        raise ValueError("Choose either full-memory or FCTM-only initialization")
    continuing = args.initialize_memory is not None or initialize_fctm is not None
    scaled_budget = bool(args.allow_scaled_budget)
    valid_budget = (
        continuing
        and scaled_budget
        and args.steps > 0
        and args.expected_total_steps == args.expected_memory_steps + args.steps
        and args.expected_total_steps >= max(PREREGISTERED_STEPS)
    ) or (
        not continuing
        and (
            args.steps in PREREGISTERED_STEPS
            or (scaled_budget and args.steps >= max(PREREGISTERED_STEPS))
        )
    )
    if not args.device.startswith("cuda") or not valid_budget:
        raise ValueError(
            "Use a preregistered 600/6960 schedule, or explicitly authorize a "
            "data-scaled full budget of at least 6960 steps"
        )
    source_sha = sha256(args.source)
    daily_sha = sha256(args.daily)
    backbone_sha = sha256(args.backbone)
    if (
        source_sha != args.expected_source_sha
        or daily_sha != args.expected_daily_sha
        or backbone_sha != args.expected_backbone_sha
    ):
        raise ValueError("Frozen input mismatch")
    support_report = json.loads(args.support.read_text())
    observation_policy = getattr(args, "memory_observation_policy", "fixed15_v1")
    validate_memory_support(support_report, allow_sparse=scaled_budget)
    if (
        support_report.get("memory_observation_policy", "fixed15_v1")
        != observation_policy
    ):
        raise ValueError("Memory support/input policy mismatch")
    if scaled_budget and (
        support_report.get("source_sha256") != source_sha
        or support_report.get("backbone_sha256") != backbone_sha
    ):
        raise ValueError("Memory support provenance mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1} or set(
        data["subject"][data["split"] == 0]
    ) & set(data["subject"][data["split"] == 1]):
        raise ValueError("Split contract failure")
    base = torch.load(args.backbone, map_location="cpu", weights_only=False)
    expected_backbone_architecture = (
        "cross_day_prediction_raw_hourly_v1"
        if main_target == "raw_hourly7d"
        else "cross_day_prediction_v1"
    )
    if (base["architecture"], base["steps"], base["seed"], base["kind"]) != (
        expected_backbone_architecture,
        args.expected_backbone_steps,
        args.seed,
        "generic",
    ):
        raise ValueError("Backbone contract mismatch")
    if (
        base["contract"].get("episode_grouping", "recorded_identity")
        != episode_grouping
    ):
        raise ValueError("Stage B/C episode grouping mismatch")
    if base["contract"].get("ridge_gradient", "full") != ridge_gradient:
        raise ValueError("Stage B/C Ridge gradient mismatch")
    if base["contract"].get("reader_objective", "episodic_ridge") != "episodic_ridge":
        raise ValueError(
            "Standard Stage C requires the episodic Ridge Stage-B objective"
        )
    if base["contract"].get("main_target", "structured_evidence") != main_target:
        raise ValueError("Stage-B/C target mismatch")
    raw = _align_daily_embedding(args.daily, data["window_id"])
    aligned, fit, audit, contract = prepare(data, raw, False)
    if args.reuse_backbone_daily_normalization:
        aligned, contract = apply_frozen_daily_normalization(
            raw, aligned, contract, base["contract"]
        )
    for key in ("daily_center", "daily_scale"):
        np.testing.assert_allclose(contract[key], base["contract"][key], rtol=0, atol=0)
    inputs = make_inputs(data, aligned["daily"], base["contract"], False)
    clinical = add_events(data, aligned["daily"], inputs, False)
    event_inputs = {
        key: torch.from_numpy(value) for key, value in inputs.items() if key != "values"
    }
    event_inputs["events"] = torch.from_numpy(clinical)
    event_raw = statistics_tokens(**event_inputs)[0][:, 28:, :8].flatten(1).numpy()
    events = (
        (event_raw - np.asarray(base["contract"]["event_center"]))
        / np.asarray(base["contract"]["event_scale"])
    ).astype(np.float32)
    targets = np.concatenate([aligned["target"], events], 1).astype(np.float32)
    if main_target == "raw_hourly7d":
        targets, raw_target_contract = raw_hourly_history_targets(
            data,
            fit,
            {
                "center": base["contract"]["raw_target_center"],
                "scale": base["contract"]["raw_target_scale"],
            },
        )
        if raw_target_contract["target_dimension"] != 336:
            raise ValueError("Raw target dimension drift")
    inputs.pop("events")
    trajectory_support = add_cross_window_nodes(data, inputs)
    if base["contract"].get("trajectory_support_policy") == "masked_sparse_v1":
        from lingsante_pulse.cgm.models.local_trajectory_occupation import (
            validate_masked_sparse_support,
        )

        validate_masked_sparse_support(trajectory_support)
    elif trajectory_support["retained_observed_fraction"] < 0.8:
        raise ValueError("Insufficient trajectory support")
    memory, memory_target = pack_fourier_memory(data, observation_policy)
    memory_active = memory["hctdm_active"].copy()
    for key in (
        "hctdm_base",
        "hctdm_clock",
        "hctdm_point_mask",
        "hctdm_day_age",
        "hctdm_day_mask",
        "hctdm_active",
    ):
        inputs[key] = memory[key]

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reservation = torch.empty(1, device=args.device)
    model = FourierClockTransportMemory(
        args.kind == "generic", memory_width=args.memory_width
    ).to(args.device)
    initialization = None
    if args.initialize_memory is not None:
        if args.kind != "candidate" or args.memory_width != 16:
            raise ValueError(
                "Memory continuation requires the released candidate architecture"
            )
        initialization = load_memory_initialization(
            model,
            base,
            args.initialize_memory,
            args.expected_memory_sha,
            args.expected_memory_steps,
            args.expected_memory_corpus_sha,
            args.expected_memory_daily_sha,
            backbone_sha,
        )
    else:
        loaded = model.load_state_dict(base["model"], strict=False)
        expected_missing = {
            name for name in model.state_dict() if name.startswith("fctm_")
        }
        if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
            raise ValueError("Unexpected warm-start state")
        if initialize_fctm is not None:
            initialization = load_fctm_initialization(
                model,
                initialize_fctm,
                args.expected_memory_sha,
                args.expected_memory_steps,
                args.expected_memory_corpus_sha,
                args.expected_memory_daily_sha,
                args.expected_memory_backbone_sha,
            )
    trainable_names = freeze_observed_state_backbone(model)
    budget = parameter_budget(args.memory_width)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (total_parameters, trainable_parameters, 146) != (
        budget["total"],
        budget["trainable"],
        budget["final_dimension"],
    ):
        raise ValueError("FCTM budget drift")
    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("fctm_")
    }
    packed_device = "cpu" if args.cpu_packed_inputs else args.device
    packed = {
        key: torch.from_numpy(value).to(packed_device) for key, value in inputs.items()
    }
    base_packed = {
        key: value for key, value in packed.items() if not key.startswith("hctdm_")
    }
    if not args.cpu_packed_inputs:
        packed["hctdm_base"].requires_grad_(True)
    target = torch.from_numpy(targets).to(args.device)
    all_rows = np.arange(len(target), dtype=np.int64)
    frozen_model = DistributionBarycenter(control=False).to(args.device)
    frozen_model.load_state_dict(base["model"])
    frozen_model.eval()
    with torch.no_grad():
        initial = torch.cat(
            [
                encode(model, packed, all_rows[start : start + 256])
                for start in range(0, len(all_rows), 256)
            ]
        )
        frozen = torch.cat(
            [
                encode(frozen_model, base_packed, all_rows[start : start + 256])
                for start in range(0, len(all_rows), 256)
            ]
        )
    initial_np = initial.cpu().numpy()
    initial_error = float((initial - frozen).abs().max())
    del frozen_model, frozen
    fixed_support, _ = SubjectEpisodes(data["subject"], fit).next()
    base_loss = float(
        target_loss(
            initial[fixed_support],
            target[fixed_support],
            initial[audit],
            target[audit],
            main_target,
        )
    )
    base_readout = readout(initial_np, aligned["target"], fit, audit)
    base_event = event_readout(initial_np, events, fit, audit)
    memory_fit = np.asarray(fit)[memory_active[fit]]
    memory_audit = np.asarray(audit)[memory_active[audit]]
    if len(memory_fit) < 64 or len(memory_audit) < 16:
        raise ValueError("Insufficient active FCTM readout support")
    base_memory_ev = event_readout(initial_np, memory_target, memory_fit, memory_audit)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4
    )
    pbq_probability = getattr(args, "pbq_sampling_probability", None)
    source_families = (
        None
        if pbq_probability is None
        else load_source_family(args.source, len(data["subject"]))
    )
    sampler = (
        SubjectEpisodes(data["subject"], fit, seed=args.seed)
        if source_families is None
        else SourceMixtureEpisodes(
            data["subject"], fit, source_families, pbq_probability, seed=args.seed
        )
    )
    if episode_grouping == "random_row":
        sampler = RandomRowRepartitionEpisodes(
            sampler, data["subject"], source_families
        )
    episode_hash = hashlib.sha256()
    episode_union_hash = hashlib.sha256()
    trace = []
    identity_overlap_total = 0
    max_mixer_gradient = 0.0
    max_writeback_gradient = 0.0
    max_spectral_input_gradient = 0.0
    for step in range(1, args.steps + 1):
        support_rows, query = sampler.next()
        identity_overlap = set(data["subject"][support_rows]) & set(
            data["subject"][query]
        )
        if episode_grouping == "recorded_identity" and identity_overlap:
            raise ValueError("Episode subject overlap")
        if episode_grouping == "random_row" and set(support_rows) & set(query):
            raise ValueError("Random-row grouping contract failure")
        identity_overlap_total += len(identity_overlap)
        episode_hash.update(support_rows.astype("<i8").tobytes())
        episode_hash.update(query.astype("<i8").tobytes())
        episode_union_hash.update(
            np.sort(np.concatenate([support_rows, query])).astype("<i8").tobytes()
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if not args.cpu_packed_inputs:
            packed["hctdm_base"].grad = None
        if step <= 8 and args.cpu_packed_inputs:
            support_z, support_inputs = encode(
                model, packed, support_rows, "hctdm_base"
            )
            query_z, query_inputs = encode(model, packed, query, "hctdm_base")
            tracked_inputs = support_inputs + query_inputs
        else:
            support_z = encode(model, packed, support_rows)
            query_z = encode(model, packed, query)
            tracked_inputs = []
        loss = target_loss(
            support_z,
            target[support_rows],
            query_z,
            target[query],
            main_target,
            stop_through_fit=stop_through_fit,
        )
        if average_two:
            support_rows2, query2 = sampler.next()
            identity_overlap2 = set(data["subject"][support_rows2]) & set(
                data["subject"][query2]
            )
            if identity_overlap2:
                raise ValueError("Second episode subject overlap")
            identity_overlap_total += len(identity_overlap2)
            episode_hash.update(support_rows2.astype("<i8").tobytes())
            episode_hash.update(query2.astype("<i8").tobytes())
            episode_union_hash.update(
                np.sort(np.concatenate([support_rows2, query2])).astype("<i8").tobytes()
            )
            loss2 = target_loss(
                encode(model, packed, support_rows2),
                target[support_rows2],
                encode(model, packed, query2),
                target[query2],
                main_target,
                stop_through_fit=stop_through_fit,
            )
            loss = (loss + loss2) / 2
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite objective")
        loss.backward()
        writeback_gradient = model.fctm_writeback.weight.grad
        if writeback_gradient is None or not bool(
            torch.isfinite(writeback_gradient).all()
        ):
            raise ValueError("Invalid writeback gradient")
        max_writeback_gradient = max(
            max_writeback_gradient, float(writeback_gradient.norm())
        )
        mixer_gradients = [
            p.grad
            for name, p in model.named_parameters()
            if name.startswith("fctm_") and not name.startswith("fctm_writeback")
        ]
        finite = [gradient for gradient in mixer_gradients if gradient is not None]
        if finite:
            if not all(bool(torch.isfinite(gradient).all()) for gradient in finite):
                raise ValueError("Invalid mixer gradient")
            max_mixer_gradient = max(
                max_mixer_gradient, sum(float(gradient.norm()) for gradient in finite)
            )
        if step <= 8:
            input_gradients = [
                value.grad for value in tracked_inputs if value.grad is not None
            ]
            if not args.cpu_packed_inputs and packed["hctdm_base"].grad is not None:
                input_gradients.append(packed["hctdm_base"].grad)
            if input_gradients:
                max_spectral_input_gradient = max(
                    max_spectral_input_gradient,
                    sum(float(value.norm()) for value in input_gradients),
                )
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                validation = float(
                    target_loss(
                        encode(model, packed, fixed_support),
                        target[fixed_support],
                        encode(model, packed, audit),
                        target[audit],
                        main_target,
                        stop_through_fit=stop_through_fit,
                    )
                )
            trace.append(
                {"step": step, "train": float(loss.detach()), "validation": validation}
            )
            print(json.dumps(trace[-1]), flush=True)

    model.eval()
    with torch.no_grad():
        final = (
            torch.cat(
                [
                    encode(model, packed, all_rows[start : start + 256])
                    for start in range(0, len(all_rows), 256)
                ]
            )
            .cpu()
            .numpy()
        )
    final_loss = float(
        target_loss(
            torch.from_numpy(final[fixed_support]).to(args.device),
            target[fixed_support],
            torch.from_numpy(final[audit]).to(args.device),
            target[audit],
            main_target,
            stop_through_fit=stop_through_fit,
        )
    )
    learned = readout(final, aligned["target"], fit, audit)
    learned_event = event_readout(final, events, fit, audit)
    learned_memory_ev = event_readout(final, memory_target, memory_fit, memory_audit)
    inactive = ~memory_active
    no_history = inputs["mask"].sum(1) == 1
    frozen_unchanged = all(
        torch.equal(value, model.state_dict()[name].detach().cpu())
        for name, value in frozen_before.items()
    )
    checkpoint = args.output / "checkpoint.pt"
    effective_steps = args.expected_total_steps if continuing else args.steps
    architecture = RAW_ARCHITECTURE if main_target == "raw_hourly7d" else ARCHITECTURE
    payload = {
        "model": model.cpu().state_dict(),
        "contract": base["contract"],
        "kind": args.kind,
        "steps": effective_steps,
        "seed": args.seed,
        "architecture": architecture,
        "corpus_sha256": source_sha,
        "daily_sha256": daily_sha,
        "backbone_sha256": backbone_sha,
        "support_sha256": sha256(args.support),
    }
    if continuing:
        payload["optimizer_steps_this_run"] = args.steps
        payload["initial_memory_sha256"] = args.expected_memory_sha
    payload["contract"] = dict(
        base["contract"],
        memory_observation_policy=observation_policy,
        stage_c_episode_grouping=episode_grouping,
        stage_c_ridge_gradient=ridge_gradient,
        stage_c_reader_objective="episodic_ridge",
        stage_c_episodes_per_optimizer_step=2 if average_two else 1,
        main_target=main_target,
    )
    if pbq_probability is not None or episode_grouping == "random_row":
        payload["contract"]["stage_c_training_sampling"] = sampler.summary()
    torch.save(payload, checkpoint)
    report = {
        "kind": args.kind,
        "seed": args.seed,
        "steps": args.steps,
        "architecture": architecture,
        "memory_width": args.memory_width,
        "packed_input_device": packed_device,
        "memory_observation_policy": observation_policy,
        "memory_support_policy": "available_active_endpoints_v1"
        if scaled_budget
        else "legacy_density_v1",
        "legacy_density_gate_passed": support_report["supported_for_canary"],
        "training_budget_policy": (
            "data_scaled_subject_exposure"
            if scaled_budget
            else "preregistered_fixed_steps"
        ),
        "daily_normalization_source": (
            "frozen_backbone_contract"
            if args.reuse_backbone_daily_normalization
            else "current_training_split"
        ),
        "completed_optimizer_steps": args.steps,
        "effective_total_steps": effective_steps,
        "initial_memory_checkpoint_sha256": (
            args.expected_memory_sha if continuing else None
        ),
        "initial_memory_steps": args.expected_memory_steps if continuing else 0,
        "clock_permutation_control": args.kind == "generic",
        "final_dimension": 146,
        "backbone_sha256": backbone_sha,
        "checkpoint_sha256": sha256(checkpoint),
        "support_sha256": sha256(args.support),
        "corpus_sha256": source_sha,
        "daily_sha256": daily_sha,
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_names": trainable_names,
        "initial_backbone_max_error": initial_error,
        "frozen_backbone_unchanged": frozen_unchanged,
        "inactive_identity_max_error": float(
            np.max(np.abs(final[inactive] - initial_np[inactive]))
        )
        if inactive.any()
        else 0.0,
        "no_history_identity_max_error": float(
            np.max(np.abs(final[no_history] - initial_np[no_history]))
        )
        if no_history.any()
        else 0.0,
        "max_fctm_mixer_gradient": max_mixer_gradient,
        "max_fctm_writeback_gradient": max_writeback_gradient,
        "max_fctm_spectral_input_gradient": max_spectral_input_gradient,
        "base_episode_validation_loss": base_loss,
        "episode_validation_loss": final_loss,
        "base_readout": base_readout,
        "learned": learned,
        "base_event_ev": base_event,
        "event_ev": learned_event,
        "base_fctm_ev": base_memory_ev,
        "fctm_ev": learned_memory_ev,
        "fctm_fit_endpoints": len(memory_fit),
        "fctm_audit_endpoints": len(memory_audit),
        "output_rank": effective_rank(final[audit]),
        "episode_sequence_sha256": episode_hash.hexdigest(),
        "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        "episode_union_sha256": episode_union_hash.hexdigest(),
        "episode_grouping": episode_grouping,
        "ridge_gradient": ridge_gradient,
        "reader_objective": "episodic_ridge",
        "main_target": main_target,
        "mean_identity_overlap": identity_overlap_total / args.steps,
        "control_integrity_passed": (
            (episode_grouping == "recorded_identity" and identity_overlap_total == 0)
            or (episode_grouping == "random_row" and identity_overlap_total > 0)
        ),
        "fctm_target_sha256": hashlib.sha256(memory_target.tobytes()).hexdigest(),
        "trajectory_support": trajectory_support,
        "phenotype_or_private_test_reads": 0,
        "private_source_label_reads": 0,
        "identity_feature_reads": 0,
        "glucofm_components": 0,
        "task_routing": 0,
        "checkpoint_selection": "last_step",
        "training": trace,
    }
    if pbq_probability is not None or episode_grouping == "random_row":
        report["training_sampling"] = sampler.summary()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "training"}),
        flush=True,
    )
    del reservation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "daily", "backbone", "support", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kind", choices=("candidate", "generic"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--allow-scaled-budget", action="store_true")
    parser.add_argument("--memory-width", type=int, default=16)
    parser.add_argument("--cpu-packed-inputs", action="store_true")
    parser.add_argument("--reuse-backbone-daily-normalization", action="store_true")
    parser.add_argument("--initialize-memory", type=Path)
    parser.add_argument("--initialize-fctm", type=Path)
    parser.add_argument("--expected-memory-sha", default="")
    parser.add_argument("--expected-memory-steps", type=int, default=0)
    parser.add_argument("--expected-memory-corpus-sha", default="")
    parser.add_argument("--expected-memory-daily-sha", default="")
    parser.add_argument("--expected-memory-backbone-sha", default="")
    parser.add_argument("--expected-total-steps", type=int, default=0)
    parser.add_argument("--pbq-sampling-probability", type=float, default=None)
    parser.add_argument(
        "--episode-grouping",
        choices=("recorded_identity", "random_row"),
        default="recorded_identity",
    )
    parser.add_argument(
        "--ridge-gradient", choices=("full", "stop_fit"), default="full"
    )
    parser.add_argument("--average-two-episodes", action="store_true")
    parser.add_argument(
        "--main-target",
        choices=("structured_evidence", "raw_hourly7d"),
        default="structured_evidence",
    )
    parser.add_argument(
        "--memory-observation-policy",
        choices=("fixed15_v1", "observed_interval_v1"),
        default="fixed15_v1",
    )
    parser.add_argument("--expected-backbone-steps", type=int, default=6960)
    parser.add_argument("--expected-source-sha", default=CORPUS_SHA)
    parser.add_argument("--expected-daily-sha", default=DAILY_SHA)
    parser.add_argument("--expected-backbone-sha", default=BASE_CHECKPOINT_SHA)
    train(parser.parse_args())
