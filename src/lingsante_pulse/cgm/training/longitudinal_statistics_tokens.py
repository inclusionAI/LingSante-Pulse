"""Single preregistered statistics-token candidate and two fixed controls."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import (
    LongitudinalStatisticsEncoder,
    statistics_tokens,
)
from lingsante_pulse.cgm.models.observation_supported_attention import pack_observations
from lingsante_pulse.cgm.models.final_space_longitudinal import evidence_v2
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
    mean_ev,
)
from lingsante_pulse.cgm.training.recurrent_orthogonal_canary import linear_cka
from lingsante_pulse.cgm.models.statistics_acceptance import POLICY, mechanism_passed


def add_events(data, daily, packed, generic=False):
    typed, support = evidence_v2(
        data["glucose"], data["observed"], data["cadence"], data["starts"]
    )
    holder = np.zeros_like(typed)
    holder[:, 0, :2] = typed[:, 2, [0, 2]]
    clinical = pack_observations(data, daily, holder, support)["features"][:, :, 0, :2]
    packed["events"] = (
        np.maximum(packed["features"][:, :, 0, :2], 0) if generic else clinical
    )
    return clinical


def call(model, packed, rows):
    return model(**{k: v[rows] for k, v in packed.items()})


def objective(z, head, target, stable, coordinate_penalty=True):
    error = (head(z) - target).square()
    loss = error[:, :64].mean() + error[:, 64:].mean()
    return loss + (z - stable).square().mean() if coordinate_penalty else loss


@torch.no_grad()
def outputs(model, packed):
    model.eval()
    return torch.cat(
        [
            call(model, packed, r)
            for r in np.array_split(
                np.arange(len(packed["values"])),
                max(1, int(np.ceil(len(packed["values"]) / 256))),
            )
        ]
    )


def event_readout(z, target, fit, audit):
    pred = (
        make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        .fit(z[fit], target[fit])
        .predict(z[audit])
    )
    return float(mean_ev(target[audit], pred))


def train(args):
    certified = getattr(args, "certified_challenges", False)
    cross_response = getattr(args, "cross_day_response", False)
    density_cache = getattr(args, "glucodensity_cache", None)
    reversible = getattr(args, "reversible_evidence", False)
    cross_moment = getattr(args, "responsibility_cross_moment", False)
    if cross_moment and (
        reversible
        or certified
        or cross_response
        or density_cache
        or args.kind not in ("candidate", "generic")
    ):
        raise ValueError(
            "Cross moment requires candidate or generic only, no other branch"
        )
    if reversible and (
        certified
        or cross_response
        or density_cache
        or args.kind not in ("candidate", "generic")
    ):
        raise ValueError(
            "Reversible evidence requires one candidate or generic control, no other branch"
        )
    loss_function = lambda z, h, t, s: objective(
        z, h, t, s, coordinate_penalty=not (reversible or cross_moment)
    )
    if density_cache and (
        certified or cross_response or args.kind not in ("candidate", "gaussian")
    ):
        raise ValueError(
            "Glucodensity uses the statistics parent and a Gaussian control only"
        )
    if args.kind == "gaussian" and not density_cache:
        raise ValueError("Gaussian control requires the fixed density cache")
    if cross_response and (
        not certified or args.kind not in ("candidate", "no_context")
    ):
        raise ValueError(
            "Cross-day response requires certified candidate or no-context control"
        )
    if args.kind == "no_context" and not cross_response:
        raise ValueError("No-context control requires cross-day response training")
    if certified and args.kind not in ("candidate", "pooled_challenges", "no_context"):
        raise ValueError("Certified candidate or fixed pooled-event control only")
    if not certified and args.kind == "pooled_challenges":
        raise ValueError("Pooled challenge control requires certified input")
    if not args.device.startswith("cuda"):
        raise ValueError("Remote CUDA training only")
    reservation = torch.empty(1, device=args.device)
    if sha256(args.source) != CORPUS_SHA or sha256(args.daily) != DAILY_SHA:
        raise ValueError("Frozen source mismatch")
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("Only train/validation may be loaded")
    if set(data["subject"][data["split"] == 0]) & set(
        data["subject"][data["split"] == 1]
    ):
        raise ValueError("Subject overlap")
    a, fit, audit, contract = prepare(
        data,
        _align_daily_embedding(args.daily, data["window_id"]),
        args.kind == "generic",
    )
    if len(fit) < 2400 or len(audit) < 300:
        raise ValueError("Insufficient supported endpoints")
    inputs = make_inputs(data, a["daily"], contract, args.kind == "generic")
    clinical = add_events(data, a["daily"], inputs, args.kind == "generic")
    # Event targets always use observed clinical thresholds, including controls.
    target_inputs = {k: torch.from_numpy(v) for k, v in inputs.items() if k != "values"}
    target_inputs["events"] = torch.from_numpy(clinical)
    event_raw = statistics_tokens(**target_inputs)[0][:, 28:, :8].flatten(1).numpy()
    ec, es = event_raw[fit].mean(0), np.maximum(event_raw[fit].std(0), 1e-5)
    event_target = ((event_raw - ec) / es).astype(np.float32)
    target_np = np.concatenate([a["target"], event_target], axis=1)
    # All fit-only initializers are identical because targets/stable do not depend on kind.
    decoder = Ridge(alpha=1.0).fit(a["stable"][fit], target_np[fit])
    contract.update(
        event_center=ec,
        event_scale=es,
        decoder_weight=decoder.coef_,
        decoder_bias=decoder.intercept_,
    )
    torch.set_num_threads(2)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    if cross_moment:
        from lingsante_pulse.cgm.models.responsibility_cross_moment import (
            ResponsibilityCrossMomentEncoder,
        )

        model = ResponsibilityCrossMomentEncoder().to(args.device)
    elif reversible:
        from lingsante_pulse.cgm.models.reversible_evidence import (
            ReversibleEvidenceEncoder,
        )

        model = ReversibleEvidenceEncoder().to(args.device)
    elif density_cache:
        from lingsante_pulse.cgm.models.glucodensity_encoder import (
            GlucodensityEncoder,
            CACHE_SHA,
        )

        if sha256(density_cache) != CACHE_SHA:
            raise ValueError("Frozen glucodensity cache mismatch")
        with np.load(density_cache, allow_pickle=False) as cache:
            if (
                not np.array_equal(cache["window_id"], data["window_id"])
                or str(cache["corpus_sha256"]) != CORPUS_SHA
            ):
                raise ValueError("Density source/order mismatch")
            inputs.update(
                density_tokens=cache[
                    "gaussian_tokens" if args.kind == "gaussian" else "density_tokens"
                ],
                density_mask=cache["density_mask"],
            )
        model = GlucodensityEncoder().to(args.device)
    elif certified:
        from lingsante_pulse.cgm.models.certified_challenges import (
            pack_challenge_tokens,
        )
        from lingsante_pulse.cgm.models.challenge_set_encoder import ChallengeSetEncoder

        inputs.update(pack_challenge_tokens(data))
        model = ChallengeSetEncoder(pooled_control=args.kind == "pooled_challenges").to(
            args.device
        )
    else:
        model = LongitudinalStatisticsEncoder(mean_only=args.kind == "mean_only").to(
            args.device
        )
    dimension = 730 if cross_moment else (194 if reversible else 146)
    head = torch.nn.Linear(dimension, 336).to(args.device)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, :146].copy_(torch.from_numpy(decoder.coef_).to(args.device))
        head.bias.copy_(torch.from_numpy(decoder.intercept_).to(args.device))
    params = list(model.parameters()) + list(head.parameters())
    response_decoder = None
    if cross_response:
        from lingsante_pulse.cgm.models.cross_day_response import (
            ConditionalRecoveryDecoder,
            eligible_responses,
            choose_response_events,
            recovery_loss,
            evaluate_recovery,
        )

        response_decoder = ConditionalRecoveryDecoder().to(args.device)
        params += list(response_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.0001)
    packed = {k: torch.from_numpy(v).to(args.device) for k, v in inputs.items()}
    target = torch.from_numpy(target_np).to(args.device)
    stable = torch.from_numpy(a["stable"]).to(args.device)
    if cross_response:
        eligible = eligible_responses(packed).cpu().numpy()
        if eligible[fit].any(1).sum() < 200 or eligible[audit].any(1).sum() < 50:
            raise ValueError("Insufficient independent days for response pretraining")
        response_rng = np.random.default_rng(43)
        response_batch_hash = hashlib.sha256()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(43)
    batch_hash = hashlib.sha256()
    trace, max_gradient, max_challenge_gradient, max_density_gradient = (
        [],
        0.0,
        0.0,
        0.0,
    )
    coupling_gradients = dict(shift=0.0, log_scale=0.0, query=0.0)
    moment_gradients = dict(day_projection=0.0, day_score=0.0, query=0.0)
    for step in range(1, args.steps + 1):
        rows = rng.choice(fit, 128, replace=True)
        batch_hash.update(rows.astype("<i8").tobytes())
        model.train()
        loss = loss_function(
            call(model, packed, rows), head, target[rows], stable[rows]
        )
        if cross_response:
            chosen = choose_response_events(eligible, rows, response_rng)
            response_batch_hash.update(chosen.astype("<i8").tobytes())
            loss = loss + recovery_loss(
                model, response_decoder, packed, rows, chosen, args.kind == "no_context"
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_gradient = max(
            max_gradient, float(model.stat_projection.weight.grad.norm())
        )
        if certified:
            max_challenge_gradient = max(
                max_challenge_gradient,
                float(model.challenge_projection[0].weight.grad.norm()),
            )
        if density_cache:
            max_density_gradient = max(
                max_density_gradient,
                float(model.density_projection[0].weight.grad.norm()),
            )
        if reversible:
            for name, p in [
                ("shift", model.shift[-1].weight),
                ("log_scale", model.log_scale[-1].weight),
                ("query", model.query),
            ]:
                coupling_gradients[name] = max(
                    coupling_gradients[name], float(p.grad.norm())
                )
        if cross_moment:
            for name, p in [
                ("day_projection", model.day_projection.weight),
                ("day_score", model.day_score.weight),
                ("query", model.query),
            ]:
                moment_gradients[name] = max(
                    moment_gradients[name], float(p.grad.norm())
                )
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                validation = float(
                    loss_function(
                        call(model, packed, audit), head, target[audit], stable[audit]
                    )
                )
            row = dict(step=step, train=float(loss), validation=validation)
            if cross_response:
                row["validation_main_only"] = validation
                row["validation_recovery"] = evaluate_recovery(
                    model,
                    response_decoder,
                    packed,
                    audit,
                    eligible,
                    args.kind == "no_context",
                )
            trace.append(row)
            print(json.dumps(row), flush=True)
    z = outputs(model, packed)
    with torch.no_grad():
        loss = float(loss_function(z[audit], head, target[audit], stable[audit]))
        shuffled = {k: v.clone() for k, v in packed.items()}
        donors = np.roll(audit, 1)
        for key in shuffled:
            first_history = 4 if certified and key.startswith("challenge_") else 1
            if density_cache and key.startswith("density_"):
                first_history = 7
            shuffled[key][audit, first_history:] = packed[key][donors, first_history:]
        shuffle_loss = float(
            loss_function(
                call(model, shuffled, audit), head, target[audit], stable[audit]
            )
        )
    final = z.cpu().numpy()
    baseline = readout(a["stable"], a["target"], fit, audit)
    learned = readout(final, a["target"], fit, audit)
    event_ev = event_readout(final, event_target, fit, audit)
    base_event_ev = event_readout(a["stable"], event_target, fit, audit)
    rank, base_rank = effective_rank(final[audit]), effective_rank(a["stable"][audit])
    cka = linear_cka(a["stable"][audit], final[audit])
    no_history = ~(inputs["mask"][:, 1:] & (inputs["support"][:, 1:, 0] > 0)).any(1)
    no_history_target = (
        np.pad(a["daily"][no_history], ((0, 0), (0, dimension - 146)))
        if dimension > 146
        else a["daily"][no_history]
    )
    ratio = shuffle_loss / max(loss, 1e-12)
    gates = dict(
        finite=bool(np.isfinite(final).all()),
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
        recurrence_gain=event_ev >= base_event_ev + 0.02,
        rank=rank >= 0.95 * base_rank,
        geometry=cka >= 0.90,
        history_used=ratio >= 1.01,
        no_history_identity=bool(np.array_equal(final[no_history], no_history_target)),
        statistics_gradient=max_gradient > 0,
        seven_day_span=bool(inputs["elapsed_days"].max() + 1 <= 7),
        subject_split_isolation=True,
    )
    checkpoint = args.output / "checkpoint.pt"
    response_report = None
    if cross_response:
        response_report = evaluate_recovery(
            model, response_decoder, packed, audit, eligible, args.kind == "no_context"
        )
        gates["recovery_finite"] = bool(np.isfinite(response_report["mse"]))
    if certified:
        gates["certified_challenge_gradient"] = max_challenge_gradient > 0
    if density_cache:
        gates["density_gradient"] = max_density_gradient > 0
    reversible_report = None
    moment_report = None
    if cross_moment:
        stable_error = float(np.max(np.abs(final[:, :146] - a["stable"])))
        moment_rms = float(np.sqrt(np.mean(final[audit, 146:] ** 2)))
        gates.update(
            stable_linear_recovery=stable_error <= 1e-5,
            moment_gradient=all(
                np.isfinite(v) and v > 0 for v in moment_gradients.values()
            ),
            nontrivial_moment=moment_rms > 0,
        )
        moment_report = dict(
            stable_max_error=stable_error,
            interaction_rms=moment_rms,
            max_gradients=moment_gradients,
            stable_selector_readout=readout(final[:, :146], a["target"], fit, audit),
        )
    if reversible:
        with torch.no_grad():
            restored_s, restored_h = model.inverse(z)
        restored_s, restored_h = restored_s.cpu().numpy(), restored_h.cpu().numpy()
        inverse_error = float(np.max(np.abs(restored_s - a["stable"])))
        shift_rms = float(
            np.sqrt(np.mean((final[audit, :146] - restored_s[audit]) ** 2))
        )
        scale_rms = float(
            np.sqrt(np.mean((final[audit, 146:] - restored_h[audit]) ** 2))
        )
        gates.update(
            stable_inverse=inverse_error <= 1e-4,
            coupling_gradient=all(
                np.isfinite(v) and v > 0 for v in coupling_gradients.values()
            ),
            nontrivial_coupling=shift_rms > 0 and scale_rms > 0,
        )
        reversible_report = dict(
            inverse_max_error=inverse_error,
            shift_rms=shift_rms,
            scale_rms=scale_rms,
            max_gradients=coupling_gradients,
            uncoupled_readout=readout(
                np.concatenate([restored_s, restored_h], 1), a["target"], fit, audit
            ),
        )
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            contract=contract,
            training_head=head.cpu().state_dict(),
            kind=args.kind,
            steps=args.steps,
            seed=43,
            response_decoder=None
            if response_decoder is None
            else response_decoder.cpu().state_dict(),
            architecture="responsibility_cross_moment_v1"
            if cross_moment
            else (
                "reversible_evidence_v1"
                if reversible
                else (
                    "phase_glucodensity_v1"
                    if density_cache
                    else (
                        "cross_day_response_v1"
                        if cross_response
                        else (
                            "certified_challenge_set_v1"
                            if certified
                            else "longitudinal_statistics_tokens_v1"
                        )
                    )
                )
            ),
            corpus_sha256=CORPUS_SHA,
            daily_sha256=DAILY_SHA,
        ),
        checkpoint,
    )
    report = dict(
        kind=args.kind,
        seed=43,
        steps=args.steps,
        baseline=baseline,
        learned=learned,
        event_ev=event_ev,
        stable_event_ev=base_event_ev,
        gates=gates,
        acceptance_passed=mechanism_passed(gates)
        and (not certified or gates["certified_challenge_gradient"]),
        legacy_all_gates_passed=all(gates.values()),
        acceptance_policy=POLICY,
        fit_endpoints=len(fit),
        audit_endpoints=len(audit),
        max_statistics_gradient=max_gradient,
        output_rank=rank,
        stable_rank=base_rank,
        cka=cka,
        history_shuffle_ratio=ratio,
        stable_coordinate_mse=float(
            np.mean((final[audit, :146] - a["stable"][audit]) ** 2)
        ),
        parameters=sum(p.numel() for p in params),
        inference_parameters=sum(p.numel() for p in model.parameters()),
        batch_sequence_sha256=batch_hash.hexdigest(),
        checkpoint_sha256=sha256(checkpoint),
        corpus_sha256=CORPUS_SHA,
        daily_sha256=DAILY_SHA,
        maximum_context_days=float(inputs["elapsed_days"].max() + 1),
        phenotype_or_private_test_reads=0,
        glucofm_components=0,
        final_dimension=dimension,
        checkpoint_selection="last_step",
        training=trace,
    )
    if certified:
        report.update(
            max_challenge_gradient=max_challenge_gradient,
            certified_challenge_counts={
                split: dict(
                    endpoints=len(rows),
                    with_challenges=int(inputs["challenge_mask"][rows].any(1).sum()),
                    valid_events=int(inputs["challenge_mask"][rows].sum()),
                )
                for split, rows in (("fit", fit), ("audit", audit))
            },
            challenge_variant="mean_event_control"
            if args.kind == "pooled_challenges"
            else "individual_events",
        )
    if cross_response:
        report.update(
            response=response_report,
            response_batch_sequence_sha256=response_batch_hash.hexdigest(),
            response_training_eligible_endpoints=int(eligible[fit].any(1).sum()),
            response_role="condition_only_control"
            if args.kind == "no_context"
            else "cross_day_context",
            acceptance_passed=report["acceptance_passed"] and gates["recovery_finite"],
        )
    if density_cache:
        report.update(
            density_cache_sha256=CACHE_SHA,
            max_density_gradient=max_density_gradient,
            density_variant="gaussian" if args.kind == "gaussian" else "empirical",
            acceptance_passed=report["acceptance_passed"] and gates["density_gradient"],
        )
    if reversible:
        report.update(
            reversible=reversible_report,
            target_sha256=hashlib.sha256(target_np.tobytes()).hexdigest(),
            acceptance_passed=report["acceptance_passed"]
            and all(
                gates[k]
                for k in ("stable_inverse", "coupling_gradient", "nontrivial_coupling")
            ),
        )
    if cross_moment:
        report.update(
            cross_moment=moment_report,
            target_sha256=hashlib.sha256(target_np.tobytes()).hexdigest(),
            acceptance_passed=report["acceptance_passed"]
            and all(
                gates[k]
                for k in (
                    "stable_linear_recovery",
                    "moment_gradient",
                    "nontrivial_moment",
                )
            ),
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "daily", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=[
            "candidate",
            "generic",
            "mean_only",
            "pooled_challenges",
            "no_context",
            "gaussian",
        ],
        default="candidate",
    )
    parser.add_argument("--glucodensity-cache", type=Path)
    parser.add_argument("--reversible-evidence", action="store_true")
    parser.add_argument("--responsibility-cross-moment", action="store_true")
    parser.add_argument("--certified-challenges", action="store_true")
    parser.add_argument("--cross-day-response", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, choices=[600, 6960], default=600)
    train(parser.parse_args())
