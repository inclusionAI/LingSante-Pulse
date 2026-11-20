"""One subject-disjoint differentiable-readout canary; no persistent training head."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from lingsante_pulse.cgm.models.episodic_evidence import (
    RandomRowRepartitionEpisodes,
    SubjectEpisodes,
    episodic_loss,
)
from lingsante_pulse.cgm.training.source_mixture import (
    SourceMixtureEpisodes,
    load_source_family,
)
from lingsante_pulse.cgm.models.observation_supported_attention import (
    ObservationSupportedAttention,
)
from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import statistics_tokens
from lingsante_pulse.cgm.training.longitudinal_statistics_tokens import (
    add_events,
    event_readout,
)
from lingsante_pulse.cgm.training.observation_supported_attention import make_inputs
from lingsante_pulse.cgm.training.final_space_longitudinal import prepare, readout
from lingsante_pulse.cgm.training.parallel_longitudinal import CORPUS_SHA, DAILY_SHA
from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
    _load_source,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
    _align_daily_embedding,
    sha256,
    effective_rank,
)
from lingsante_pulse.cgm.training.recurrent_orthogonal_canary import linear_cka
from lingsante_pulse.cgm.training.targets import raw_hourly_history_targets


def persistent_reconstruction_loss(prediction, target, main_target_mode):
    error = (prediction - target).square()
    split = 64 if main_target_mode == "structured_evidence" else 168
    return error[:, :split].mean() + error[:, split:].mean()


def persistent_state_loss(prediction, target, groups):
    error = (prediction - target).square()
    return torch.stack([error[:, idx].mean() for idx in groups.values()]).mean()


def episodic_target_loss(
    support,
    support_targets,
    query,
    query_targets,
    main_target_mode,
    stop_through_fit=False,
):
    return episodic_loss(
        support,
        support_targets,
        query,
        query_targets,
        stop_through_fit=stop_through_fit,
        target_split=64 if main_target_mode == "structured_evidence" else 168,
    )


def encode(model, packed, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if getattr(model, "pooled_query_control", None) is not None and len(rows) > 32:
        return torch.cat(
            [encode(model, packed, rows[i : i + 32]) for i in range(0, len(rows), 32)]
        )
    device = next(model.parameters()).device
    selected = {k: v[rows].to(device, non_blocking=True) for k, v in packed.items()}
    return model(**selected)[0]


def slot_permuted_batch(packed, rows, device):
    """Select rows, reverse day slots, then move one diagnostic batch."""
    rows = np.asarray(rows, dtype=np.int64)
    output = {}
    for key, value in packed.items():
        selected = value[rows]
        if not key.startswith("union_"):
            permutation = torch.arange(
                selected.shape[1] - 1, -1, -1, device=selected.device
            )
            selected = selected.index_select(1, permutation)
        output[key] = selected.to(device, non_blocking=True)
    return output


def train(args):
    if args.steps < 600:
        raise ValueError("At least 600 optimizer steps are required")
    calibrated = getattr(args, "moment_calibrated", False)
    exposure = getattr(args, "exposure_pair_measure", False)
    balanced = getattr(args, "cohort_balanced_episodes", False)
    flux = getattr(args, "state_flux_occupation", False)
    graph = getattr(args, "day_graph_occupation", False)
    union = getattr(args, "observed_union_occupation", False)
    boundary = getattr(args, "boundary_trajectory", False)
    intraday = getattr(args, "intraday_memory", False)
    residence = getattr(args, "residence_clock", False)
    capacity = getattr(args, "capacity_matching", False)
    edge_value = getattr(args, "edge_conditioned_value", False)
    route = getattr(args, "distribution_phase", None)
    antecedent = getattr(args, "antecedent_match", False)
    slow_context = getattr(args, "slow_context", False)
    cross_day = getattr(args, "cross_day_prediction", False)
    episode_grouping = getattr(args, "episode_grouping", "recorded_identity")
    ridge_gradient = getattr(args, "ridge_gradient", "full")
    reader_objective = getattr(args, "reader_objective", "episodic_ridge")
    main_target_mode = getattr(args, "main_target", "structured_evidence")
    stop_through_fit = ridge_gradient == "stop_fit"
    if (episode_grouping != "recorded_identity" or stop_through_fit) and not (
        cross_day and args.kind == "generic"
    ):
        raise ValueError("Reader controls are restricted to fixed TER Stage B")
    if reader_objective == "persistent" and (
        not (cross_day and args.kind == "generic")
        or episode_grouping != "recorded_identity"
        or ridge_gradient != "full"
    ):
        raise ValueError(
            "Persistent control changes only the fixed Stage-B reader lifecycle"
        )
    if main_target_mode != "structured_evidence" and (
        not cross_day
        or args.kind != "generic"
        or reader_objective not in ("persistent", "episodic_ridge")
    ):
        raise ValueError(
            "Raw recovery target is restricted to a matched Stage-B reader-lifecycle control"
        )
    sparse_support = getattr(args, "allow_masked_sparse_trajectories", False)
    if sparse_support and not (cross_day and args.kind == "generic"):
        raise ValueError(
            "Masked sparse support is scoped to observed-state pretraining"
        )
    double_main = getattr(args, "double_main_control", False)
    average_two = getattr(args, "average_two_episodes", False)
    if average_two and not (
        cross_day
        and args.kind == "generic"
        and reader_objective == "episodic_ridge"
        and episode_grouping == "recorded_identity"
    ):
        raise ValueError("Two-episode averaging is restricted to standard TER Stage B")
    if double_main:
        if (
            cross_day
            or route
            or antecedent
            or slow_context
            or args.kind != "candidate"
            or args.steps != 600
        ):
            raise ValueError("Extra-main control is a fixed Distribution600 experiment")
        route = "distribution"
    if cross_day:
        if route or antecedent or slow_context:
            raise ValueError("Cross-day objective uses original Distribution only")
        route = "distribution"
    if sum((antecedent, slow_context, bool(route))) > 1:
        raise ValueError("One correspondence extension only")
    if antecedent:
        route = "antecedent_match"
    if slow_context:
        route = "slow_context"
    if sum((residence, intraday, capacity, edge_value, bool(route))) > 1:
        raise ValueError("One correspondence extension only")
    unmatched = (
        getattr(args, "unmatched_correspondence", False)
        or intraday
        or residence
        or capacity
        or edge_value
        or bool(route)
    )
    correspondence = getattr(args, "cross_window_correspondence", False) or unmatched
    if correspondence and boundary:
        raise ValueError("No boundary variant in cross-window correspondence")
    anchored = getattr(args, "anchored_trajectory", False) or boundary or correspondence
    if anchored and args.kind != "candidate" and not correspondence:
        raise ValueError("Reuse frozen absolute-input control")
    gap_recurrent = getattr(args, "gap_recurrent_occupation", False) or anchored
    trajectory = getattr(args, "local_trajectory_occupation", False) or gap_recurrent
    responsibility = getattr(args, "responsibility_gradient", False)
    if gap_recurrent and responsibility:
        raise ValueError("No gradient projection in gap recurrent model")
    if responsibility and not trajectory:
        raise ValueError("Responsibility update requires frozen local trajectory model")
    if trajectory and (
        union or graph or flux or getattr(args, "occupation_path", False)
    ):
        raise ValueError("One local trajectory model only")
    if union and (
        graph
        or flux
        or getattr(args, "occupation_path", False)
        or args.kind != "candidate"
    ):
        raise ValueError("One fixed union candidate only")
    if graph and (flux or getattr(args, "occupation_path", False)):
        raise ValueError("One occupation model only")
    if flux and getattr(args, "occupation_path", False):
        raise ValueError("One occupation model only")
    path = (
        getattr(args, "occupation_path", False) or flux or graph or union or trajectory
    )
    relational = (
        getattr(args, "shared_pair_lift", False) or exposure or balanced or path
    )
    if path and (balanced or exposure or getattr(args, "shared_pair_lift", False)):
        raise ValueError("One path architecture only")
    if balanced and (exposure or args.kind != "candidate"):
        raise ValueError(
            "Balanced run uses frozen SharedPair candidate architecture only"
        )
    if exposure and getattr(args, "shared_pair_lift", False):
        raise ValueError("One pair architecture only")
    coherent = getattr(args, "coherent", False) or calibrated
    if relational and coherent:
        raise ValueError("One architecture only")
    if not args.device.startswith("cuda"):
        raise ValueError("Remote CUDA training only")
    reservation = torch.empty(1, device=args.device)
    if slow_context or cross_day or double_main:
        print("CGM_CUDA_RESERVED", args.device, flush=True)
    source_sha = sha256(args.source)
    daily_sha = sha256(args.daily)
    if source_sha != args.expected_source_sha or daily_sha != args.expected_daily_sha:
        raise ValueError("Frozen source mismatch")
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("Train/validation only")
    if set(data["subject"][data["split"] == 0]) & set(
        data["subject"][data["split"] == 1]
    ):
        raise ValueError("Subject overlap")
    generic = args.kind == "generic" and not (
        calibrated or relational
    )  # New controls keep identical typed inputs.
    a, fit, audit, contract = prepare(
        data, _align_daily_embedding(args.daily, data["window_id"]), generic
    )
    if (
        len(fit) < args.minimum_fit_endpoints
        or len(audit) < args.minimum_audit_endpoints
    ):
        raise ValueError("Insufficient supported endpoints")
    inputs = make_inputs(data, a["daily"], contract, generic)
    clinical = add_events(data, a["daily"], inputs, generic)
    target_inputs = {k: torch.from_numpy(v) for k, v in inputs.items() if k != "values"}
    target_inputs["events"] = torch.from_numpy(clinical)
    event_raw = statistics_tokens(**target_inputs)[0][:, 28:, :8].flatten(1).numpy()
    ec, es = event_raw[fit].mean(0), np.maximum(event_raw[fit].std(0), 1e-5)
    events = ((event_raw - ec) / es).astype(np.float32)
    targets = np.concatenate([a["target"], events], 1)
    if main_target_mode == "raw_hourly7d":
        targets, raw_target_contract = raw_hourly_history_targets(data, fit)
        contract.update(
            main_target=main_target_mode,
            raw_target_center=raw_target_contract["center"],
            raw_target_scale=raw_target_contract["scale"],
            raw_target_contract={
                key: value
                for key, value in raw_target_contract.items()
                if key not in ("center", "scale")
            },
        )
    else:
        contract["main_target"] = main_target_mode
    inputs.pop(
        "events"
    )  # Existing attention reads four responsibility evidence, not labels.
    path_report = None
    if path:
        from lingsante_pulse.cgm.models.occupation_path import (
            add_occupation_path,
            OccupationPath,
        )

        if trajectory:
            from lingsante_pulse.cgm.models.local_trajectory_occupation import (
                add_local_trajectory,
                LocalTrajectoryOccupation,
                trajectory_support_valid,
            )

            if correspondence:
                from lingsante_pulse.cgm.models.cross_window_nodes import (
                    add_cross_window_nodes,
                )

                path_report = add_cross_window_nodes(data, inputs)
                if slow_context:
                    from lingsante_pulse.cgm.models.slow_context import add_slow_context

                    add_slow_context(data, inputs)
            elif boundary:
                from lingsante_pulse.cgm.models.boundary_trajectory import (
                    add_boundary_trajectory,
                )

                path_report = add_boundary_trajectory(data, inputs)
            else:
                path_report = add_local_trajectory(
                    data,
                    inputs,
                    sorted_control=args.kind == "generic"
                    and not (responsibility or gap_recurrent),
                )
            if sparse_support:
                from lingsante_pulse.cgm.models.local_trajectory_occupation import (
                    validate_masked_sparse_support,
                )

                validate_masked_sparse_support(path_report)
                contract["trajectory_support_policy"] = "masked_sparse_v1"
                print("MASKED_SPARSE_SUPPORT", json.dumps(path_report), flush=True)
            elif not trajectory_support_valid(path_report):
                raise ValueError("Insufficient complete local trajectory support")
        elif union:
            from lingsante_pulse.cgm.models.observed_union_occupation import (
                add_observed_union,
                ObservedUnionOccupation,
            )

            path_report = add_observed_union(data, inputs)
        elif flux:
            from lingsante_pulse.cgm.models.state_flux_occupation import (
                add_state_flux,
                StateFluxOccupation,
            )

            path_report = add_state_flux(data, inputs, unsigned=args.kind == "generic")
            if path_report["flow_support_fraction"] < 0.7:
                raise ValueError("Insufficient supported flow")
        else:
            path_report = add_occupation_path(data, inputs)
        if graph:
            from lingsante_pulse.cgm.models.day_graph_occupation import (
                DayGraphOccupation,
                day_relations,
                graph_collision,
            )
        if not sparse_support and path_report["retained_observed_fraction"] < 0.8:
            raise ValueError("Insufficient observed path support")
    if calibrated:
        from lingsante_pulse.cgm.models.moment_calibrated_days import (
            add_calibration,
            MomentCalibratedDayAttention,
            moment_projectors,
        )

        add_calibration(data, inputs, physical=args.kind == "candidate")
    contract.update(event_center=ec, event_scale=es)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if relational:
        from lingsante_pulse.cgm.models.shared_pair_lift import SharedPairLift

        if exposure:
            from lingsante_pulse.cgm.models.exposure_pair_measure import (
                ExposurePairMeasure,
                exposure_collision_check,
            )
        if gap_recurrent:
            from lingsante_pulse.cgm.models.gap_recurrent_occupation import (
                GapRecurrentOccupation,
                temporal_collision,
            )

            if correspondence:
                from lingsante_pulse.cgm.models.cross_window_correspondence import (
                    CrossWindowCorrespondence,
                    cross_window_temporal_collision,
                    query_collision,
                )

                if route:
                    from lingsante_pulse.cgm.models.distribution_phase import (
                        DistributionBarycenter,
                        PhaseCoupling,
                        branch_modules,
                        route_diagnostics,
                    )

                    if slow_context:
                        from lingsante_pulse.cgm.models.slow_context import (
                            SlowContext,
                            branch_modules,
                            context_diagnostics,
                            synthetic_context_inputs,
                        )

                        model = SlowContext(control=args.kind == "generic").to(
                            args.device
                        )
                    elif antecedent:
                        from lingsante_pulse.cgm.models.antecedent_match import (
                            AntecedentMatch,
                            matching_diagnostics,
                        )

                        model = AntecedentMatch(control=args.kind == "generic").to(
                            args.device
                        )
                    else:
                        model = (
                            DistributionBarycenter
                            if route == "distribution"
                            else PhaseCoupling
                        )(control=args.kind == "generic" and not cross_day).to(
                            args.device
                        )
                elif edge_value:
                    from lingsante_pulse.cgm.models.edge_conditioned_value import (
                        EdgeConditionedValue,
                        edge_diagnostics,
                    )

                    model = EdgeConditionedValue(control=args.kind == "generic").to(
                        args.device
                    )
                elif capacity:
                    from lingsante_pulse.cgm.models.capacity_matching import (
                        CapacityMatching,
                        capacity_diagnostics,
                    )

                    model = CapacityMatching(control=args.kind == "generic").to(
                        args.device
                    )
                elif residence:
                    from lingsante_pulse.cgm.models.residence_clock import (
                        ResidenceClock,
                        clock_diagnostics,
                    )

                    model = ResidenceClock(control=args.kind == "generic").to(
                        args.device
                    )
                elif intraday:
                    from lingsante_pulse.cgm.models.intraday_memory import (
                        IntradayMemory,
                        intraday_diagnostics,
                    )

                    model = IntradayMemory(control=args.kind == "generic").to(
                        args.device
                    )
                elif unmatched:
                    from lingsante_pulse.cgm.models.unmatched_correspondence import (
                        UnmatchedCorrespondence,
                    )

                    model = UnmatchedCorrespondence(control=args.kind == "generic").to(
                        args.device
                    )
                else:
                    model = CrossWindowCorrespondence(
                        control=args.kind == "generic"
                    ).to(args.device)
                temporal_collision = cross_window_temporal_collision
                if slow_context:
                    temporal_collision = lambda m: cross_window_temporal_collision(
                        m, synthetic_context_inputs
                    )
            elif anchored:
                from lingsante_pulse.cgm.models.anchored_trajectory import (
                    AnchoredTrajectory,
                )

                model = AnchoredTrajectory().to(args.device)
            else:
                model = GapRecurrentOccupation(control=args.kind == "generic").to(
                    args.device
                )
        elif trajectory:
            model = LocalTrajectoryOccupation(
                control=args.kind == "generic" and not responsibility
            ).to(args.device)
        else:
            model = (
                ObservedUnionOccupation()
                if union
                else (
                    DayGraphOccupation
                    if graph
                    else (
                        StateFluxOccupation
                        if flux
                        else (
                            OccupationPath
                            if path
                            else (ExposurePairMeasure if exposure else SharedPairLift)
                        )
                    )
                )(control=args.kind == "generic")
            ).to(args.device)
    else:
        model = (
            MomentCalibratedDayAttention()
            if calibrated
            else ObservationSupportedAttention(coherent=coherent)
        ).to(args.device)
    initial_checkpoint_sha = None
    initialize_checkpoint = getattr(args, "initialize_checkpoint", None)
    if initialize_checkpoint is not None:
        initial_checkpoint_sha = sha256(initialize_checkpoint)
        if initial_checkpoint_sha != args.expected_initialize_sha:
            raise ValueError("Stage-B initialization checksum mismatch")
        initial_payload = torch.load(
            initialize_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            initial_payload.get("architecture"),
            initial_payload.get("kind"),
            initial_payload.get("steps"),
            initial_payload.get("corpus_sha256"),
        ) != ("cross_day_prediction_v1", "generic", 6960, source_sha):
            raise ValueError("Stage-B initialization provenance mismatch")
        model.load_state_dict(initial_payload["model"], strict=True)
    persistent_main = None
    persistent_aux = None
    max_reader_gradient = 0.0
    if reader_objective == "persistent":
        persistent_main = torch.nn.Linear(146, targets.shape[1]).to(args.device)
        persistent_aux = torch.nn.Linear(146, 16).to(args.device)
    packed_device = "cpu" if args.cpu_packed_inputs else args.device
    packed = {k: torch.from_numpy(v).to(packed_device) for k, v in inputs.items()}
    target = torch.from_numpy(targets).to(args.device)
    stable = torch.from_numpy(a["stable"]).to(args.device)
    if balanced:
        from lingsante_pulse.cgm.models.cohort_balanced_episodes import (
            CohortBalancedEpisodes,
        )
    pbq_probability = getattr(args, "pbq_sampling_probability", None)
    source_families = None
    if pbq_probability is not None:
        if balanced or double_main or not cross_day or args.kind != "generic":
            raise ValueError(
                "Source mixture is restricted to the fixed TER StageB configuration"
            )
        source_families = load_source_family(args.source, len(data["subject"]))
        sampler = SourceMixtureEpisodes(
            data["subject"], fit, source_families, pbq_probability, seed=args.seed
        )
    else:
        sampler = (
            CohortBalancedEpisodes(data["subject"], data["cohort"], fit)
            if balanced
            else SubjectEpisodes(data["subject"], fit, seed=args.seed)
        )
    if episode_grouping == "random_row":
        sampler = RandomRowRepartitionEpisodes(
            sampler, data["subject"], source_families
        )
    if double_main:
        from lingsante_pulse.cgm.models.double_episode_control import extra_sampler

        extra = extra_sampler(data["subject"], fit)
        extra_hash = hashlib.sha256()
        extra_max_gradient = 0.0
        extra_trace = []
    if cross_day:
        from lingsante_pulse.cgm.models.cross_day_prediction import (
            STATE_GROUPS,
            prediction_pairs,
            prediction_inputs,
            support_summary,
            state_loss,
            state_readout,
        )

        aux_pairs, aux_masks = prediction_pairs(data, inputs)
        aux_support = support_summary(data, aux_pairs, aux_masks)
        if not aux_support["supported_for_canary"]:
            raise ValueError("Insufficient cross-day input/target support")
        saved_support = json.loads((args.output.parent / "support.json").read_text())
        if saved_support["pair_sha256"] != aux_support["pair_sha256"]:
            raise ValueError("Cross-day pairs changed after support check")
        aux_inputs = prediction_inputs(inputs, aux_pairs, aux_masks)
        aux_packed = {
            k: torch.from_numpy(v).to(packed_device) for k, v in aux_inputs.items()
        }
        aux_subject = data["subject"][aux_pairs[:, 0]]
        aux_fit = np.flatnonzero(data["split"][aux_pairs[:, 0]] == 0)
        aux_audit = np.flatnonzero(data["split"][aux_pairs[:, 0]] == 1)
        aux_future = a["target"][aux_pairs[:, 1], :16].copy()
        aux_observed = a["target"][aux_pairs[:, 0], :16].copy()
        aux_target = torch.from_numpy(
            aux_future if args.kind == "candidate" else aux_observed
        ).to(args.device)
        aux_sampler = (
            SubjectEpisodes(aux_subject, aux_fit, seed=args.seed)
            if source_families is None
            else SourceMixtureEpisodes(
                aux_subject,
                aux_fit,
                source_families[aux_pairs[:, 0]],
                pbq_probability,
                seed=args.seed,
            )
        )
        if episode_grouping == "random_row":
            aux_sampler = RandomRowRepartitionEpisodes(
                aux_sampler,
                aux_subject,
                None if source_families is None else source_families[aux_pairs[:, 0]],
                seed=10000 + args.seed,
            )
        aux_hash = hashlib.sha256()
        aux_max_gradient = 0.0
        with torch.no_grad():
            aux_initial = (
                encode(model, aux_packed, np.arange(len(aux_fit) + len(aux_audit)))
                .cpu()
                .numpy()
            )
        aux_trace = []
    optimization_parameters = list(model.parameters())
    if reader_objective == "persistent":
        from sklearn.linear_model import Ridge

        main_initializer = Ridge(alpha=1.0).fit(a["stable"][fit], targets[fit])
        aux_initializer = Ridge(alpha=1.0).fit(
            aux_initial[aux_fit], aux_target[aux_fit].cpu().numpy()
        )
        with torch.no_grad():
            persistent_main.weight.copy_(
                torch.from_numpy(main_initializer.coef_).to(args.device)
            )
            persistent_main.bias.copy_(
                torch.from_numpy(main_initializer.intercept_).to(args.device)
            )
            persistent_aux.weight.copy_(
                torch.from_numpy(aux_initializer.coef_).to(args.device)
            )
            persistent_aux.bias.copy_(
                torch.from_numpy(aux_initializer.intercept_).to(args.device)
            )
        optimization_parameters += list(persistent_main.parameters()) + list(
            persistent_aux.parameters()
        )
    optimizer = torch.optim.AdamW(
        optimization_parameters, lr=0.0005, weight_decay=0.0001
    )
    source_counts = {str(c): 0 for c in np.unique(data["cohort"][fit])}
    fixed_support, _ = SubjectEpisodes(data["subject"], fit).next()
    with torch.no_grad():
        initial = torch.cat(
            [
                encode(model, packed, np.arange(i, min(i + 256, len(target))))
                for i in range(0, len(target), 256)
            ]
        )
        initial_error = float((initial - stable).abs().max())
        stable_loss = (
            float(
                persistent_reconstruction_loss(
                    persistent_main(stable[audit]), target[audit], main_target_mode
                )
            )
            if reader_objective == "persistent"
            else float(
                episodic_target_loss(
                    stable[fixed_support],
                    target[fixed_support],
                    stable[audit],
                    target[audit],
                    main_target_mode,
                )
            )
        )
    args.output.mkdir(parents=True, exist_ok=False)
    episode_hash = hashlib.sha256()
    episode_union_hash = hashlib.sha256()
    trace = []
    max_gradient = 0.0
    max_lift_gradient = 0.0
    max_edge_gradient = 0.0
    identity_overlap_total = 0
    if cross_day:
        aux_union_hash = hashlib.sha256()
        aux_identity_overlap_total = 0
    if gap_recurrent:
        temporal_gradients = dict(memory=0.0, bridge=0.0, decay=0.0)
    if correspondence:
        correspondence_gradients = dict(query=0.0, key=0.0, message=0.0)
    if unmatched:
        unmatched_gradient = 0.0
    if intraday:
        intraday_gradients = dict(intraday=0.0, intraday_writeback=0.0)
    if residence:
        clock_gradients = dict(clock_lift=0.0, clock_writeback=0.0)
    if route:
        route_gradients = {k: 0.0 for k in branch_modules(model)}
    if edge_value:
        import time

        edge_gradients = dict(edge_filter=0.0, edge_writeback=0.0)
        torch.cuda.reset_peak_memory_stats(args.device)
        edge_start = time.perf_counter()
    if responsibility:
        from lingsante_pulse.cgm.models.responsibility_gradient import (
            family_losses,
            responsibility_backward,
        )
        from lingsante_pulse.cgm.models.episodic_evidence import ridge_predict

        responsibility_report = dict(
            scope="readout" if args.kind == "candidate" else "all",
            conflict_steps=0,
            maximum_correction_norm=0.0,
            reverse_passes_per_step=3,
        )
    for step in range(1, args.steps + 1):
        support, query = sampler.next()
        identity_overlap = set(data["subject"][support]) & set(data["subject"][query])
        if episode_grouping == "recorded_identity" and identity_overlap:
            raise ValueError("Episode subject overlap")
        if episode_grouping == "random_row" and set(support) & set(query):
            raise ValueError("Random-row grouping contract failure")
        identity_overlap_total += len(identity_overlap)
        if balanced:
            for c in source_counts:
                for rows in (support, query):
                    n = int(np.count_nonzero(data["cohort"][rows] == c))
                    if n != 16:
                        raise ValueError("Unbalanced cohort exposure")
                    source_counts[c] += n
        episode_hash.update(support.astype("<i8").tobytes())
        episode_hash.update(query.astype("<i8").tobytes())
        episode_union_hash.update(
            np.sort(np.concatenate([support, query])).astype("<i8").tobytes()
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        support_embedding = encode(model, packed, support)
        query_embedding = encode(model, packed, query)
        if reader_objective == "persistent":
            main_embedding = torch.cat([support_embedding, query_embedding])
            main_target = torch.cat([target[support], target[query]])
            loss = persistent_reconstruction_loss(
                persistent_main(main_embedding), main_target, main_target_mode
            )
        elif responsibility:
            error = (
                ridge_predict(support_embedding, target[support], query_embedding)
                - target[query]
            ).square()
            losses = family_losses(error)
            loss = sum(losses.values())
        else:
            loss = episodic_target_loss(
                support_embedding,
                target[support],
                query_embedding,
                target[query],
                main_target_mode,
                stop_through_fit=stop_through_fit,
            )
        if cross_day:
            sa, qa = aux_sampler.next()
            aux_identity_overlap = set(aux_subject[sa]) & set(aux_subject[qa])
            if episode_grouping == "recorded_identity" and aux_identity_overlap:
                raise ValueError("Auxiliary subject overlap")
            if episode_grouping == "random_row" and set(sa) & set(qa):
                raise ValueError("Auxiliary random-row grouping contract failure")
            aux_identity_overlap_total += len(aux_identity_overlap)
            aux_hash.update(sa.astype("<i8").tobytes())
            aux_hash.update(qa.astype("<i8").tobytes())
            aux_union_hash.update(
                np.sort(np.concatenate([sa, qa])).astype("<i8").tobytes()
            )
            aux_support_embedding = encode(model, aux_packed, sa)
            aux_query_embedding = encode(model, aux_packed, qa)
            if reader_objective == "persistent":
                aux_loss = persistent_state_loss(
                    persistent_aux(
                        torch.cat([aux_support_embedding, aux_query_embedding])
                    ),
                    torch.cat([aux_target[sa], aux_target[qa]]),
                    STATE_GROUPS,
                )
            else:
                aux_loss = state_loss(
                    aux_support_embedding,
                    aux_target[sa],
                    aux_query_embedding,
                    aux_target[qa],
                    stop_through_fit=stop_through_fit,
                )
            aux_gradient = torch.autograd.grad(
                aux_loss, model.readout.weight, retain_graph=True
            )[0]
            if not bool(torch.isfinite(aux_gradient).all()):
                raise ValueError("Invalid auxiliary gradient")
            aux_max_gradient = max(aux_max_gradient, float(aux_gradient.norm()))
            loss = loss + aux_loss
        if average_two:
            support2, query2 = sampler.next()
            identity_overlap2 = set(data["subject"][support2]) & set(
                data["subject"][query2]
            )
            if identity_overlap2:
                raise ValueError("Second episode subject overlap")
            identity_overlap_total += len(identity_overlap2)
            episode_hash.update(support2.astype("<i8").tobytes())
            episode_hash.update(query2.astype("<i8").tobytes())
            episode_union_hash.update(
                np.sort(np.concatenate([support2, query2])).astype("<i8").tobytes()
            )
            loss2 = episodic_target_loss(
                encode(model, packed, support2),
                target[support2],
                encode(model, packed, query2),
                target[query2],
                main_target_mode,
                stop_through_fit=stop_through_fit,
            )
            sa2, qa2 = aux_sampler.next()
            aux_identity_overlap2 = set(aux_subject[sa2]) & set(aux_subject[qa2])
            if aux_identity_overlap2:
                raise ValueError("Second auxiliary episode subject overlap")
            aux_identity_overlap_total += len(aux_identity_overlap2)
            aux_hash.update(sa2.astype("<i8").tobytes())
            aux_hash.update(qa2.astype("<i8").tobytes())
            aux_union_hash.update(
                np.sort(np.concatenate([sa2, qa2])).astype("<i8").tobytes()
            )
            aux_loss2 = state_loss(
                encode(model, aux_packed, sa2),
                aux_target[sa2],
                encode(model, aux_packed, qa2),
                aux_target[qa2],
                stop_through_fit=stop_through_fit,
            )
            aux_gradient2 = torch.autograd.grad(
                aux_loss2, model.readout.weight, retain_graph=True
            )[0]
            if not bool(torch.isfinite(aux_gradient2).all()):
                raise ValueError("Invalid second auxiliary gradient")
            aux_max_gradient = max(aux_max_gradient, float(aux_gradient2.norm()))
            loss = (loss + loss2 + aux_loss2) / 2
        if double_main:
            se, qe = extra.next()
            if set(data["subject"][se]) & set(data["subject"][qe]):
                raise ValueError("Extra episode subject overlap")
            extra_hash.update(se.astype("<i8").tobytes())
            extra_hash.update(qe.astype("<i8").tobytes())
            extra_loss = episodic_loss(
                encode(model, packed, se),
                target[se],
                encode(model, packed, qe),
                target[qe],
            )
            extra_gradient = torch.autograd.grad(
                extra_loss, model.readout.weight, retain_graph=True
            )[0]
            if not bool(torch.isfinite(extra_gradient).all()):
                raise ValueError("Invalid extra gradient")
            extra_max_gradient = max(extra_max_gradient, float(extra_gradient.norm()))
            loss = loss + extra_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite episodic objective")
        if responsibility:
            update = responsibility_backward(
                model, losses, responsibility_report["scope"]
            )
            responsibility_report["conflict_steps"] += int(update["conflict"])
            responsibility_report["maximum_correction_norm"] = max(
                responsibility_report["maximum_correction_norm"],
                update["correction_norm"],
            )
        else:
            loss.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("Invalid outer gradient")
        if reader_objective == "persistent":
            reader_gradients = [
                p.grad
                for p in (*persistent_main.parameters(), *persistent_aux.parameters())
            ]
            if any(
                g is None or not bool(torch.isfinite(g).all()) for g in reader_gradients
            ):
                raise ValueError("Invalid persistent reader gradient")
            max_reader_gradient = max(
                max_reader_gradient, max(float(g.norm()) for g in reader_gradients)
            )
        final_weight = model.readout.weight if relational else model.score[-1].weight
        max_gradient = max(max_gradient, float(final_weight.grad.norm()))
        if relational:
            max_lift_gradient = max(
                max_lift_gradient,
                sum(float(p.grad.norm()) for p in model.lifts.parameters()),
            )
        if graph:
            max_edge_gradient = max(
                max_edge_gradient,
                sum(float(p.grad.norm()) for p in model.edge_lift.parameters()),
            )
        if gap_recurrent:
            for name in temporal_gradients:
                norm = (
                    float(model.decay.grad.norm())
                    if name == "decay"
                    else sum(
                        float(p.grad.norm()) for p in getattr(model, name).parameters()
                    )
                )
                temporal_gradients[name] = max(temporal_gradients[name], norm)
        if correspondence:
            for name in correspondence_gradients:
                correspondence_gradients[name] = max(
                    correspondence_gradients[name],
                    sum(
                        float(p.grad.norm()) for p in getattr(model, name).parameters()
                    ),
                )
        if unmatched:
            unmatched_gradient = max(
                unmatched_gradient, float(model.unmatched_logit.grad.abs())
            )
        if intraday:
            for name in intraday_gradients:
                intraday_gradients[name] = max(
                    intraday_gradients[name],
                    sum(
                        float(p.grad.norm()) for p in getattr(model, name).parameters()
                    ),
                )
        if residence:
            for name in clock_gradients:
                clock_gradients[name] = max(
                    clock_gradients[name],
                    sum(
                        float(p.grad.norm()) for p in getattr(model, name).parameters()
                    ),
                )
        if edge_value:
            for name in edge_gradients:
                edge_gradients[name] = max(
                    edge_gradients[name],
                    sum(
                        float(p.grad.norm()) for p in getattr(model, name).parameters()
                    ),
                )
        if route:
            for name, module in branch_modules(model).items():
                route_gradients[name] = max(
                    route_gradients[name],
                    sum(float(p.grad.norm()) for p in module.parameters()),
                )
        torch.nn.utils.clip_grad_norm_(optimization_parameters, 1.0)
        optimizer.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                audit_embedding = encode(model, packed, audit)
                validation = (
                    float(
                        persistent_reconstruction_loss(
                            persistent_main(audit_embedding),
                            target[audit],
                            main_target_mode,
                        )
                    )
                    if reader_objective == "persistent"
                    else float(
                        episodic_target_loss(
                            encode(model, packed, fixed_support),
                            target[fixed_support],
                            audit_embedding,
                            target[audit],
                            main_target_mode,
                            stop_through_fit=stop_through_fit,
                        )
                    )
                )
            row = dict(step=step, train=float(loss), validation=validation)
            trace.append(row)
            print(json.dumps(row), flush=True)
            if cross_day:
                aux_trace.append(dict(step=step, aux_loss=float(aux_loss)))
            if double_main:
                extra_trace.append(dict(step=step, extra_loss=float(extra_loss)))
    model.eval()
    with torch.no_grad():
        z = torch.cat(
            [
                encode(model, packed, np.arange(i, min(i + 256, len(target))))
                for i in range(0, len(target), 256)
            ]
        )
        base_support = z[fixed_support]
        loss = (
            float(
                persistent_reconstruction_loss(
                    persistent_main(z[audit]), target[audit], main_target_mode
                )
            )
            if reader_objective == "persistent"
            else float(
                episodic_target_loss(
                    base_support,
                    target[fixed_support],
                    z[audit],
                    target[audit],
                    main_target_mode,
                    stop_through_fit=stop_through_fit,
                )
            )
        )
        # Only the audit rows are consumed by this control.  Materializing a
        # full-corpus clone is equivalent but becomes prohibitive for scaled
        # unlabeled corpora.
        shuffled = {k: v[audit].clone() for k, v in packed.items()}
        donors = np.roll(audit, 1)
        for k in shuffled:
            if k == "union_has_history":
                continue
            if k == "union_path":
                shuffled[k][:, 288:] = packed[k][donors, 288:]
            elif k != "projector":
                shuffled[k][:, 1:] = packed[k][donors, 1:]
        if calibrated:
            shuffled_valid = (
                (shuffled["mask"] & (shuffled["support"][:, :, 0] > 0)).cpu().numpy()
            )
            shuffled["projector"] = torch.from_numpy(
                moment_projectors(
                    shuffled["moments"].cpu().numpy(),
                    shuffled_valid,
                    args.kind == "candidate",
                )
            ).to(packed_device)
        shuffle_rows = np.arange(len(audit), dtype=np.int64)
        shuffled_embedding = encode(model, shuffled, shuffle_rows)
        shuffle_loss = (
            float(
                persistent_reconstruction_loss(
                    persistent_main(shuffled_embedding), target[audit], main_target_mode
                )
            )
            if reader_objective == "persistent"
            else float(
                episodic_target_loss(
                    base_support,
                    target[fixed_support],
                    shuffled_embedding,
                    target[audit],
                    main_target_mode,
                )
            )
        )
    final = z.cpu().numpy()
    baseline = readout(a["stable"], a["target"], fit, audit)
    learned = readout(final, a["target"], fit, audit)
    event_ev = event_readout(final, events, fit, audit)
    base_event = event_readout(a["stable"], events, fit, audit)
    valid = inputs["mask"] & (inputs["support"][:, :, 0] > 0)
    no_history = ~valid[:, 1:].any(1)
    low = np.where(valid[:, :, None], inputs["values"], np.inf).min(1)
    high = np.where(valid[:, :, None], inputs["values"], -np.inf).max(1)
    bound_error = float(max(0, np.max(low - final), np.max(final - high)))
    cka = linear_cka(a["stable"][audit], final[audit])
    ratio = shuffle_loss / max(loss, 1e-12)
    gates = dict(
        finite=bool(np.isfinite(final).all()),
        initial_stable=initial_error <= 1e-5,
        value_bounds=bound_error <= 1e-5,
        current_mean=learned["current_mean"] >= baseline["current_mean"] - 0.01,
        current_branches=bool(
            np.all(
                np.asarray(learned["current"]) >= np.asarray(baseline["current"]) - 0.02
            )
        ),
        longitudinal_gain=learned["longitudinal_mean"]
        >= baseline["longitudinal_mean"] + 0.02,
        longitudinal_branches=bool(
            np.all(
                np.asarray(learned["longitudinal"])
                >= np.asarray(baseline["longitudinal"]) - 0.01
            )
        ),
        recurrence_gain=event_ev >= base_event + 0.02,
        geometry=cka >= 0.90,
        history_used=ratio >= 1.01,
        no_history_identity=bool(
            np.array_equal(final[no_history], a["daily"][no_history])
        ),
        outer_gradient=max_gradient > 0,
        seven_day_span=bool(inputs["elapsed_days"].max() + 1 <= 7),
        subject_split_isolation=episode_grouping == "recorded_identity",
    )
    coherent_report = None
    if coherent:
        with torch.no_grad():
            check, weights = model(**{k: v[audit] for k, v in packed.items()})
            clean = torch.where(
                packed["mask"][audit, :, None]
                & (packed["support"][audit, :, 0, None] > 0),
                packed["values"][audit],
                0.0,
            )
            joint_error = float((check - (weights * clean).sum(1)).abs().max())
            sum_error = float((weights.sum(1) - 1).abs().max())
            minimum = float(weights.min())
        gates["joint_convex"] = (
            weights.shape[-1] == 1
            and joint_error <= 1e-5
            and sum_error <= 1e-5
            and minimum >= 0
        )
        coherent_report = dict(
            joint_reconstruction_max_error=joint_error,
            weight_sum_max_error=sum_error,
            minimum_weight=minimum,
        )
    pair_report = None
    if relational:
        two = audit[valid[audit].sum(1) == 2]
        two_update = (
            float(np.max(np.abs(final[two] - a["stable"][two]))) if len(two) else 0.0
        )
        with torch.no_grad():
            permuted = slot_permuted_batch(packed, audit, args.device)
            error = float((model(**permuted)[0] - z[audit]).abs().max())
        gates.update(
            shared_lift_gradient=max_lift_gradient > 0,
            two_day_update=two_update > 1e-6,
            slot_permutation=error <= 1e-5,
        )
        pair_report = dict(
            max_shared_lift_gradient=max_lift_gradient,
            two_day_audit_count=len(two),
            two_day_max_update=two_update,
            slot_permutation_max_error=error,
            control=args.kind == "generic",
            value_bounds_is_diagnostic_not_contract=True,
        )
        if exposure:
            collision = exposure_collision_check(model, packed, int(audit[0]))
            gates["exposure_response"] = (
                collision["output_max_difference"] > 1e-6
                if args.kind == "candidate"
                else collision["output_max_difference"] <= 1e-5
            )
            pair_report["exposure_collision"] = collision
    moment_report = None
    graph_report = None
    if graph:
        collision = graph_collision(model)
        gates["day_edge_gradient"] = max_edge_gradient > 0
        gates["day_topology_response"] = (
            (
                collision["embedding_max_difference"] > 1e-6
                and collision["edge_moment_max_difference"] > 1e-6
            )
            if args.kind == "candidate"
            else collision["embedding_max_difference"] <= 1e-5
        )
        with torch.no_grad():
            mass = (
                packed["path"][audit, :, :, -1].sum(2)
                * packed["mask"][audit]
                * (packed["support"][audit, :, 0] > 0)
            )
            _, edge_weight = day_relations(
                torch.zeros(len(audit), 7, 36, device=args.device),
                mass,
                packed["elapsed_days"][audit],
                False,
            )
            edge_fraction = float((edge_weight.sum(1) > 0).float().mean())
        graph_report = dict(
            complete_graph_control=args.kind == "generic",
            max_edge_gradient=max_edge_gradient,
            audit_adjacent_edge_fraction=edge_fraction,
            collision=collision,
        )
    if calibrated:
        with torch.no_grad():
            base = torch.where(
                packed["mask"][audit] & (packed["support"][audit, :, 0] > 0),
                packed["support"][audit, :, 0],
                0.0,
            )
            base = base / base.sum(1, keepdim=True)
            delta = weights[:, :, 0] - base
            moments = packed["moments"][audit].double()
            old = (base.double()[:, :, None] * moments).sum(1)
            error = (delta.double()[:, :, None] * moments).sum(
                1
            ).abs() / old.abs().clamp_min(1.0)
            maximum_update = float(delta.abs().max())
        gates["physical_moments"] = (
            args.kind != "candidate" or float(error.max()) <= 1e-5
        )
        gates["nonzero_day_update"] = maximum_update > 1e-6
        moment_report = dict(
            relative_physical_error=float(error.max()),
            max_weight_update=maximum_update,
            audit_update_fraction=float(
                (delta.abs().max(1).values > 1e-6).float().mean()
            ),
            audit_mean_free_directions=float(
                np.trace(inputs["projector"][audit], axis1=1, axis2=2).mean()
            ),
            fit_mean_free_directions=float(
                np.trace(inputs["projector"][fit], axis1=1, axis2=2).mean()
            ),
            physical_constraints=args.kind == "candidate",
            control_definition="identical typed input, remove only two physical moment constraints",
        )
    checkpoint = args.output / "checkpoint.pt"
    architecture = (
        "day_graph_occupation_v1"
        if graph
        else (
            "state_flux_occupation_v1"
            if flux
            else (
                "occupation_path_v1"
                if path
                else (
                    "exposure_pair_measure_v1"
                    if exposure
                    else (
                        "shared_pair_lift_v1"
                        if relational
                        else (
                            "moment_calibrated_days_v1"
                            if calibrated
                            else (
                                "coherent_episodic_v1"
                                if coherent
                                else "episodic_evidence_v1"
                            )
                        )
                    )
                )
            )
        )
    )
    if union:
        architecture = "observed_union_occupation_v1"
    if trajectory:
        architecture = "local_trajectory_occupation_v1"
    if responsibility:
        architecture = "responsibility_gradient_occupation_v1"
    if gap_recurrent:
        architecture = "gap_recurrent_occupation_v1"
        collision = temporal_collision(model)
        gates["temporal_gradient"] = all(v > 0 for v in temporal_gradients.values())
        gates["temporal_order_response"] = (
            collision["day_order_embedding_difference"] > 1e-6
        )
        gates["temporal_gap_response"] = (
            collision["gap_embedding_difference"] > 1e-6
            if args.kind == "candidate" or correspondence
            else collision["gap_embedding_difference"] <= 1e-5
        )
    if anchored:
        architecture = "anchored_trajectory_v1"
    if boundary:
        architecture = "boundary_trajectory_v1"
    if correspondence:
        architecture = "cross_window_correspondence_v1"
        query_difference = query_collision(model)
        gates["correspondence_gradient"] = all(
            v > 0 for v in correspondence_gradients.values()
        )
        gates["query_specific_attention"] = (
            query_difference > 1e-6
            if args.kind == "candidate" or unmatched
            else query_difference <= 1e-6
        )
    if unmatched:
        architecture = "unmatched_correspondence_v1"
        gates["unmatched_gradient"] = unmatched_gradient > 0
    if intraday:
        architecture = "intraday_memory_v1"
        intraday_checks = intraday_diagnostics(model)
        gates["intraday_gradient"] = all(v > 0 for v in intraday_gradients.values())
        gates["intraday_order_response"] = (
            intraday_checks["order_writeback_difference"] > 1e-6
            if args.kind == "candidate"
            else intraday_checks["order_writeback_difference"] <= 1e-5
        )
        gates["intraday_gap_reset"] = intraday_checks["post_gap_max_error"] <= 1e-6
    if residence:
        architecture = "residence_clock_v1"
        clock_checks = clock_diagnostics(model)
        gates["clock_gradient"] = all(v > 0 for v in clock_gradients.values())
        gates["clock_response"] = clock_checks["clock_writeback_difference"] > 1e-6
        gates["clock_gap_reset"] = clock_checks["post_gap_max_error"] <= 1e-6
        gates["clock_prefix_causal"] = clock_checks["prefix_max_error"] <= 1e-6
        gates["clock_private_isolation"] = clock_checks["private_max_error"] <= 1e-6
    if capacity:
        architecture = "capacity_matching_v1"
        capacity_checks = capacity_diagnostics(
            model,
            packed["nodes"][audit],
            packed["mask"][audit] & (packed["support"][audit, :, 0] > 0),
            packed["elapsed_days"][audit],
        )
        gates["capacity_columns"] = capacity_checks["column_excess"] <= 1e-6
        gates["capacity_rows"] = capacity_checks["row_excess"] <= 1e-6
        gates["capacity_equal_mass"] = capacity_checks["equal_mass_error"] <= 1e-6
        gates["capacity_selective_response"] = (
            capacity_checks["selective_difference"] > 1e-6
        )
    if edge_value:
        architecture = "edge_conditioned_value_v1"
        edge_checks = edge_diagnostics(model)
        gates["edge_gradient"] = all(v > 0 for v in edge_gradients.values())
        gates["edge_association_response"] = (
            edge_checks["association_writeback_difference"] > 1e-6
        )
        gates["edge_future_isolation"] = edge_checks["future_message_max_error"] <= 1e-6
        gates["edge_private_filter_isolation"] = (
            edge_checks["private_filter_max_error"] <= 1e-6
        )
        edge_runtime = dict(
            training_and_final_diagnostics_seconds=time.perf_counter() - edge_start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
        )
    if route:
        architecture = route + "_v1"
        route_checks = route_diagnostics(model)
        gates["route_gradient"] = all(v > 0 for v in route_gradients.values())
        gates["route_response"] = route_checks["branch_response"] > 1e-6
        gates["route_mask_isolation"] = route_checks["mask_max_error"] <= 1e-6
        gates["route_future_isolation"] = (
            route_checks["future_message_max_error"] <= 1e-6
        )
        if antecedent:
            matching_checks = matching_diagnostics(model)
            gates["matching_condition_isolation"] = (
                matching_checks["irrelevant_phase_key_error"] <= 1e-6
            )
            gates["matching_condition_response"] = (
                matching_checks["relevant_phase_key_difference"] > 1e-6
            )
            gates["matching_private_isolation"] = (
                matching_checks["private_attention_error"] <= 1e-6
            )
        if slow_context:
            context_checks = context_diagnostics(model)
            gates["context_response"] = context_checks["slow_change_writeback"] > 1e-6
            gates["context_private_isolation"] = context_checks["private_error"] <= 1e-6
            gates["context_gap_isolation"] = context_checks["gap_error"] <= 1e-6
            gates["context_conditioning"] = (
                context_checks["pairing_difference"] > 1e-6
                if args.kind == "candidate"
                else context_checks["pairing_difference"] <= 1e-6
            )
        if route == "phase":
            gates["route_private_isolation"] = route_checks["private_max_error"] <= 1e-6
            gates["route_additive_control"] = (
                route_checks["additive_pairing_max_error"] <= 1e-6
            )
    if cross_day:
        with torch.no_grad():
            aux_final = (
                encode(model, aux_packed, np.arange(len(aux_pairs))).cpu().numpy()
            )
        cross_day_report = dict(
            target_kind="unseen_future_day"
            if args.kind == "candidate"
            else "observed_last_day",
            distribution_control=model.distribution_control,
            support=aux_support,
            episode_sha256=aux_hash.hexdigest(),
            input_sha256=hashlib.sha256(
                b"".join(aux_inputs[k].tobytes() for k in sorted(aux_inputs))
            ).hexdigest(),
            future_target_sha256=hashlib.sha256(aux_future.tobytes()).hexdigest(),
            observed_target_sha256=hashlib.sha256(aux_observed.tobytes()).hexdigest(),
            max_aux_readout_gradient=aux_max_gradient,
            future_readout=state_readout(aux_final, aux_future, aux_fit, aux_audit),
            observed_readout=state_readout(aux_final, aux_observed, aux_fit, aux_audit),
            stable_future_readout=state_readout(
                aux_initial, aux_future, aux_fit, aux_audit
            ),
            training=aux_trace,
        )
        architecture = (
            "cross_day_prediction_persistent_raw_hourly_v1"
            if reader_objective == "persistent" and main_target_mode == "raw_hourly7d"
            else (
                "cross_day_prediction_persistent_evidence_v1"
                if reader_objective == "persistent"
                else (
                    "cross_day_prediction_raw_hourly_v1"
                    if main_target_mode == "raw_hourly7d"
                    else "cross_day_prediction_v1"
                )
            )
        )
    if double_main:
        architecture = "double_main_control_v1"
    contract.update(
        episode_grouping=episode_grouping,
        ridge_gradient=ridge_gradient,
        reader_objective=reader_objective,
        main_target=main_target_mode,
        episodes_per_optimizer_step=2 if average_two else 1,
    )
    if source_families is not None or episode_grouping == "random_row":
        contract["stage_b_training_sampling"] = sampler.summary()
        contract["stage_b_aux_training_sampling"] = aux_sampler.summary()
    checkpoint_payload = dict(
        model=model.cpu().state_dict(),
        contract=contract,
        kind=args.kind,
        steps=args.steps,
        seed=args.seed,
        architecture=architecture,
        corpus_sha256=source_sha,
        daily_sha256=daily_sha,
    )
    if initial_checkpoint_sha is not None:
        checkpoint_payload["initial_checkpoint_sha256"] = initial_checkpoint_sha
    if reader_objective == "persistent":
        checkpoint_payload["training_only_readers"] = {
            "main": persistent_main.cpu().state_dict(),
            "auxiliary": persistent_aux.cpu().state_dict(),
        }
    torch.save(checkpoint_payload, checkpoint)
    report = dict(
        kind=args.kind,
        seed=args.seed,
        steps=args.steps,
        baseline=baseline,
        learned=learned,
        event_ev=event_ev,
        stable_event_ev=base_event,
        gates=gates,
        acceptance_passed=all(gates.values()),
        rank_policy="diagnostic_only",
        output_rank=effective_rank(final[audit]),
        initial_stable_error=initial_error,
        value_bound_error=bound_error,
        cka=cka,
        history_shuffle_ratio=ratio,
        episode_validation_loss=loss,
        stable_episode_validation_loss=stable_loss,
        max_outer_gradient=max_gradient,
        fit_endpoints=len(fit),
        audit_endpoints=len(audit),
        fit_subjects=len(sampler.subjects),
        parameters=sum(p.numel() for p in model.parameters()),
        inference_parameters=sum(p.numel() for p in model.parameters()),
        episode_sequence_sha256=episode_hash.hexdigest(),
        episode_union_sha256=episode_union_hash.hexdigest(),
        episode_grouping=episode_grouping,
        ridge_gradient=ridge_gradient,
        reader_objective=reader_objective,
        main_target=main_target_mode,
        training_only_reader_parameters=(
            sum(
                p.numel()
                for p in (*persistent_main.parameters(), *persistent_aux.parameters())
            )
            if reader_objective == "persistent"
            else 0
        ),
        max_training_only_reader_gradient=max_reader_gradient,
        mean_identity_overlap=identity_overlap_total / args.steps,
        target_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
        checkpoint_sha256=sha256(checkpoint),
        corpus_sha256=source_sha,
        daily_sha256=daily_sha,
        phenotype_or_private_test_reads=0,
        glucofm_components=0,
        final_dimension=146,
        control_integrity_passed=(
            (episode_grouping == "recorded_identity" and identity_overlap_total == 0)
            or (episode_grouping == "random_row" and identity_overlap_total > 0)
        ),
        packed_input_device=packed_device,
        checkpoint_selection="last_step",
        training=trace,
        initial_checkpoint_sha256=initial_checkpoint_sha,
    )
    if source_families is not None or episode_grouping == "random_row":
        report["training_sampling"] = sampler.summary()
        report["aux_training_sampling"] = aux_sampler.summary()
    if coherent:
        report["coherent"] = coherent_report
    if calibrated:
        report["moment_calibration"] = moment_report
    if relational:
        report["shared_pair_lift"] = pair_report
    if exposure:
        report["exposure_pair_measure"] = True
    if balanced:
        report["cohort_balanced_episodes"] = dict(
            gates_passed=True,
            per_source_sampled_rows=source_counts,
            cohort_id_encoder_inputs=0,
        )
    if path:
        report["occupation_path"] = path_report
    if flux:
        report["state_flux_occupation"] = True
    if graph:
        report["day_graph"] = graph_report
    if union:
        report["observed_union"] = True
    if trajectory:
        report["local_trajectory_occupation"] = True
    if responsibility:
        report["responsibility_gradient"] = responsibility_report
    if gap_recurrent:
        report["gap_recurrent"] = dict(
            regular_control=args.kind == "generic" and not correspondence,
            max_gradients=temporal_gradients,
            collision=collision,
        )
    if anchored:
        report["anchored_trajectory"] = True
    if boundary:
        report["boundary_trajectory"] = True
    if correspondence:
        report["cross_window_correspondence"] = dict(
            pooled_query_control=args.kind == "generic" and not unmatched,
            max_gradients=correspondence_gradients,
            query_collision_difference=query_difference,
        )
    if unmatched:
        report["unmatched_correspondence"] = dict(
            global_gate_control=args.kind == "generic"
            and not (intraday or residence or capacity or edge_value or route),
            max_unmatched_gradient=unmatched_gradient,
            unmatched_logit=float(model.unmatched_logit.detach()),
        )
    if route:
        report["distribution_phase"] = dict(
            route=route,
            control=args.kind == "generic",
            max_gradients=route_gradients,
            diagnostics=route_checks,
        )
    if antecedent:
        report["antecedent_match"] = dict(
            tail_control=args.kind == "generic",
            distribution_control=model.distribution_control,
            diagnostics=matching_checks,
        )
    if slow_context:
        report["slow_context"] = dict(
            additive_control=model.additive_context_control,
            distribution_control=model.distribution_control,
            diagnostics=context_checks,
        )
    if cross_day:
        cross_day_report["episode_union_sha256"] = aux_union_hash.hexdigest()
        cross_day_report["mean_identity_overlap"] = (
            aux_identity_overlap_total / args.steps
        )
        report["cross_day_prediction"] = cross_day_report
    if double_main:
        report["double_main_control"] = dict(
            episode_sha256=extra_hash.hexdigest(),
            skipped_episodes=600,
            training_encoder_calls=4 * args.steps,
            max_extra_gradient=extra_max_gradient,
            distribution_control=model.distribution_control,
            training=extra_trace,
        )
    if edge_value:
        report["edge_conditioned_value"] = dict(
            mean_field_control=args.kind == "generic",
            max_gradients=edge_gradients,
            diagnostics=edge_checks,
            runtime=edge_runtime,
        )
    if capacity:
        report["capacity_matching"] = dict(
            uniform_control=args.kind == "generic",
            diagnostics=capacity_checks,
            mass_equality_scope="same input weights;not equality across independently trained arms",
            causal_boundary="observed window endpoint",
        )
    if intraday:
        report["intraday_memory"] = dict(
            sorted_control=args.kind == "generic",
            max_gradients=intraday_gradients,
            diagnostics=intraday_checks,
        )
    if residence:
        report["residence_clock"] = dict(
            accumulated_control=args.kind == "generic",
            max_gradients=clock_gradients,
            diagnostics=clock_checks,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("source", "daily", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--kind", choices=["candidate", "generic"], default="candidate")
    p.add_argument("--coherent", action="store_true")
    p.add_argument("--moment-calibrated", action="store_true")
    p.add_argument("--shared-pair-lift", action="store_true")
    p.add_argument("--exposure-pair-measure", action="store_true")
    p.add_argument("--cohort-balanced-episodes", action="store_true")
    p.add_argument("--pbq-sampling-probability", type=float, default=None)
    p.add_argument("--occupation-path", action="store_true")
    p.add_argument("--state-flux-occupation", action="store_true")
    p.add_argument("--day-graph-occupation", action="store_true")
    p.add_argument("--observed-union-occupation", action="store_true")
    p.add_argument("--local-trajectory-occupation", action="store_true")
    p.add_argument("--responsibility-gradient", action="store_true")
    p.add_argument("--gap-recurrent-occupation", action="store_true")
    p.add_argument("--anchored-trajectory", action="store_true")
    p.add_argument("--boundary-trajectory", action="store_true")
    p.add_argument("--cross-window-correspondence", action="store_true")
    p.add_argument("--unmatched-correspondence", action="store_true")
    p.add_argument("--intraday-memory", action="store_true")
    p.add_argument("--residence-clock", action="store_true")
    p.add_argument("--capacity-matching", action="store_true")
    p.add_argument("--edge-conditioned-value", action="store_true")
    p.add_argument("--distribution-phase", choices=["distribution", "phase"])
    p.add_argument("--antecedent-match", action="store_true")
    p.add_argument("--slow-context", action="store_true")
    p.add_argument("--cross-day-prediction", action="store_true")
    p.add_argument(
        "--episode-grouping",
        choices=["recorded_identity", "random_row"],
        default="recorded_identity",
    )
    p.add_argument("--ridge-gradient", choices=["full", "stop_fit"], default="full")
    p.add_argument(
        "--reader-objective",
        choices=["episodic_ridge", "persistent"],
        default="episodic_ridge",
    )
    p.add_argument(
        "--main-target",
        choices=["structured_evidence", "raw_hourly7d"],
        default="structured_evidence",
    )
    p.add_argument("--allow-masked-sparse-trajectories", action="store_true")
    p.add_argument("--double-main-control", action="store_true")
    p.add_argument("--average-two-episodes", action="store_true")
    p.add_argument("--expected-source-sha", default=CORPUS_SHA)
    p.add_argument("--expected-daily-sha", default=DAILY_SHA)
    p.add_argument("--minimum-fit-endpoints", type=int, default=2400)
    p.add_argument("--minimum-audit-endpoints", type=int, default=300)
    p.add_argument("--cpu-packed-inputs", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--initialize-checkpoint", type=Path)
    p.add_argument("--expected-initialize-sha", default="")
    train(p.parse_args())
