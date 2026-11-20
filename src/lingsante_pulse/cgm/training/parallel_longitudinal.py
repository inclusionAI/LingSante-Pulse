#!/usr/bin/env python3
"""Frozen four-direction longitudinal training, canary decisions and extraction."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from lingsante_pulse.cgm.data.long_term_embedding import causal_long_term_mean
from lingsante_pulse.cgm.data.reliability_responsible_longitudinal_embedding import (
    reliability_weight,
)
from lingsante_pulse.cgm.models.covariance_preserving_innovation import (
    fit_covariance_transport,
    calibrate_innovation_gain,
    transport_state,
    transport_from_state,
)
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    typed_evidence_features,
    generic_multiscale_features,
    causal_history_indices,
    standardize_evidence,
    pack_histories,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import (
    LongitudinalIdea,
    VARIANTS,
    bounded_histories,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
    _align_daily_embedding,
    _dispersion_loss,
    sha256,
    mean_ev,
    effective_rank,
)
from lingsante_pulse.cgm.training.recurrent_orthogonal_canary import linear_cka
from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
    _load_source,
)


CORPUS_SHA = "52a64594f9c438a930de9247e0875903efa8cf91292a13f00b03401e95190ad2"
DAILY_SHA = "5d02f2f52bd418202a17822a3bcceb93d264cc6cfea4793237b56c915e74ab9c"


def assemble(data, daily_raw, *, control=False, contract=None, strict_seven_days=False):
    fit = data["split"] == 0
    if contract is None:
        center = daily_raw[fit].mean(0, keepdims=True)
        scale = np.maximum(daily_raw[fit].std(0, keepdims=True), 1e-5)
    else:
        center, scale = contract["daily_center"], contract["daily_scale"]
    daily = ((daily_raw - center) / scale).astype(np.float32)
    histories = causal_history_indices(
        data["subject"],
        data["starts"],
        data["cohort"],
        data["split"],
        data["cadence"],
        data["source"],
        max_days=7,
    )
    if strict_seven_days:
        # A 24h endpoint consumes one of the seven days; earliest start is t-6d.
        histories = bounded_histories(histories, data["starts"])
    target_raw, support = typed_evidence_features(
        data["glucose"], data["observed"], data["cadence"], data["starts"]
    )
    input_raw = (
        generic_multiscale_features(data["glucose"], data["observed"])[0]
        if control
        else target_raw
    )
    if contract is None:
        features, feature_contract = standardize_evidence(
            input_raw, np.flatnonzero(fit)
        )
        target, target_contract = standardize_evidence(target_raw, np.flatnonzero(fit))
        contract = dict(
            daily_center=center,
            daily_scale=scale,
            features=feature_contract,
            targets=target_contract,
        )
    else:
        features = (
            (input_raw - contract["features"]["center"]) / contract["features"]["scale"]
        ).astype(np.float32)
        target = (
            (target_raw - contract["targets"]["center"]) / contract["targets"]["scale"]
        ).astype(np.float32)
    input_support = np.repeat(support[:, :1], 4, axis=1) if control else support
    packed = pack_histories(features, input_support, data["starts"], histories)
    target_packed = pack_histories(target, support, data["starts"], histories)
    transition = target - target_packed["features"][:, -1]
    # Encode provenance in grouping only; opaque identities never enter the model.
    keys = np.asarray(
        [
            json.dumps([s, c, int(p), int(k), int(d)])
            for s, c, p, k, d in zip(
                data["subject"],
                data["cohort"],
                data["split"],
                data["cadence"],
                data["source"],
            )
        ]
    )
    groups = np.unique(keys, return_inverse=True)[1]
    stable, days, coverage = causal_long_term_mean(
        daily,
        keys,
        data["starts"],
        np.zeros(len(keys)),
        data["observed"].mean(1),
        max_days=7,
    )
    reliability = reliability_weight(days, coverage, max_days=7)
    return dict(
        daily=daily,
        stable=stable,
        target=target,
        support=support,
        transition=transition,
        groups=groups,
        packed=packed,
        reliability=reliability,
        days=days,
    ), contract


def forward(model, tensors, rows, packed=None):
    p = tensors["packed"] if packed is None else packed
    return model(
        tensors["daily"][rows],
        p["features"][rows],
        p["support"][rows],
        p["elapsed_days"][rows],
        p["history_mask"][rows],
    )


def loss_for(out, t, rows):
    s = t["support"][rows]
    typed = (
        (out["typed_predictions"] - t["target"][rows]).square().mean(-1) * s
    ).sum() / s.sum().clamp(min=1)
    delta = (
        (out["transition_predictions"] - t["transition"][rows]).square().mean(-1) * s
    ).sum() / s.sum().clamp(min=1)
    return (
        (out["long_prediction"] - t["daily"][rows]).square().mean()
        + typed
        + 0.5 * delta
        + 0.05 * _dispersion_loss(out["memory_codes"])
        + out["auxiliary"]
    )


def outputs(model, t, rows, packed=None):
    model.eval()
    parts = {k: [] for k in ("memory_codes", "typed_predictions", "long_prediction")}
    with torch.no_grad():
        for j in range(0, len(rows), 256):
            out = forward(model, t, rows[j : j + 256], packed)
            for k in parts:
                parts[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def validation(model, t, rows, packed=None):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for j in range(0, len(rows), 256):
            r = rows[j : j + 256]
            total += float(loss_for(forward(model, t, r, packed), t, r)) * len(r)
    return total / len(rows)


def tensorize(a, device):
    return {
        k: (
            {p: torch.from_numpy(v).to(device) for p, v in x.items()}
            if k == "packed"
            else torch.from_numpy(x).to(device)
        )
        for k, x in a.items()
    }


def train(args):
    if sha256(args.source) != CORPUS_SHA or sha256(args.daily) != DAILY_SHA:
        raise ValueError("frozen input SHA mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("only train/validation corpus may enter this experiment")
    daily_raw = _align_daily_embedding(args.daily, data["window_id"])
    a, contract = assemble(
        data, daily_raw, control=args.control, strict_seven_days=args.strict_seven_days
    )
    history = a["packed"]["history_mask"].sum(1)
    fit = np.flatnonzero((data["split"] == 0) & (history > 0))
    audit = np.flatnonzero((data["split"] == 1) & (history > 0))
    if len(fit) < 2400 or len(audit) < 300:
        raise ValueError("insufficient frozen longitudinal support")
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    torch.set_num_threads(2)
    model = LongitudinalIdea(args.variant).to(args.device)
    t = tensorize(a, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    rng = np.random.default_rng(43)
    best = float("inf")
    best_state = None
    training_log = []
    for step in range(1, args.steps + 1):
        rows = torch.as_tensor(rng.choice(fit, 128, replace=True), device=args.device)
        model.train()
        out = forward(model, t, rows)
        loss = loss_for(out, t, rows)
        if args.variant == "day_contrast":
            p = t["packed"]
            loss = loss + 0.1 * model.contrast_loss(
                t["daily"][rows],
                p["features"][rows],
                p["support"][rows],
                p["elapsed_days"][rows],
                p["history_mask"][rows],
                t["groups"][rows],
            )
        if not torch.isfinite(loss):
            raise ValueError("non-finite train loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            v = validation(model, t, audit)
            if v < best:
                best = v
                best_state = copy.deepcopy(model.state_dict())
            row = dict(step=step, train=float(loss), validation=v)
            training_log.append(row)
            if step % 100 == 0:
                print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    out = outputs(model, t, np.arange(len(history)))
    memory = out["memory_codes"].reshape(len(history), 160)
    transported = None
    diagnostics = {}
    fit_stable = fit[a["days"][fit] > 1]
    try:
        transported, diagnostics = fit_covariance_transport(
            a["stable"], memory, fit_stable
        )
        gain = calibrate_innovation_gain(
            a["stable"][fit_stable],
            transported.innovation(a["stable"][fit_stable], memory[fit_stable]),
            0.25,
        )
        final = a["stable"] + gain * a["reliability"][:, None] * transported.innovation(
            a["stable"], memory
        )
    except ValueError as exc:
        diagnostics["failure"] = str(exc)
        gain = 0.0
        final = a["stable"].copy()
    final[a["days"] == 1] = a["daily"][a["days"] == 1]
    branches = {
        name: mean_ev(a["target"][audit, k], out["typed_predictions"][audit, k])
        for k, name in enumerate(("state", "challenge", "extreme", "rhythm"))
    }
    shuffled = {k: v.clone() for k, v in t["packed"].items()}
    for k in shuffled:
        shuffled[k][audit] = t["packed"][k][np.roll(audit, 1)]
    history_ratio = validation(model, t, audit, shuffled) / max(
        validation(model, t, audit), 1e-12
    )
    native = effective_rank(a["stable"][audit])
    rank = effective_rank(final[audit])
    cka = linear_cka(a["stable"][audit], final[audit])
    target_ev = float(np.mean(list(branches.values())))
    gates = dict(
        finite=bool(np.isfinite(final).all()),
        history_used=history_ratio >= 1.01,
        branches_positive=all(v > 0 for v in branches.values()),
        typed_ev=target_ev >= 0.1,
        rank=rank >= 0.95 * native,
        stable_geometry=cka >= 0.9,
        transport_supported=transported is not None,
        no_history_identity=bool(
            np.array_equal(final[a["days"] == 1], a["daily"][a["days"] == 1])
        ),
    )
    maximum_context_days = float(a["packed"]["elapsed_days"].max() + 1.0)
    if args.strict_seven_days:
        gates["seven_day_span"] = maximum_context_days <= 7.0
    report = dict(
        variant=args.variant,
        control=args.control,
        steps=args.steps,
        seed=43,
        parameters=sum(p.numel() for p in model.parameters()),
        branch_ev=branches,
        typed_ev=target_ev,
        history_shuffle_ratio=history_ratio,
        stable_rank=native,
        output_rank=rank,
        stable_cka=cka,
        gates=gates,
        acceptance_passed=all(gates.values()),
        transport=diagnostics,
        fit_endpoints=len(fit),
        audit_endpoints=len(audit),
        fit_stable_endpoints=len(fit_stable),
        training=training_log,
        corpus_sha256=CORPUS_SHA,
        daily_sha256=DAILY_SHA,
        strict_seven_days=args.strict_seven_days,
        maximum_context_days=maximum_context_days,
        phenotype_or_private_test_reads=0,
        glucofm_components=0,
    )
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        dict(
            variant=args.variant,
            model=model.cpu().state_dict(),
            contract=contract,
            control=args.control,
            transport=transport_state(transported) if transported else None,
            gain=gain,
            corpus_sha256=CORPUS_SHA,
            strict_seven_days=args.strict_seven_days,
        ),
        checkpoint,
    )
    report["checkpoint_sha256"] = sha256(checkpoint)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}), flush=True)


def public_extract(args):
    from lingsante_pulse.cgm.inference.longitudinal import (
        infer_native_cadence,
        provenance_subjects,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["transport"] is None:
        raise ValueError("cannot export an unsupported canary")
    with np.load(args.public, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    n = len(payload["window_id"])
    data = dict(
        glucose=payload["glucose_mg_dl"],
        observed=payload["observed_mask"].astype(bool),
        subject=provenance_subjects(
            payload["subject_id"], payload.get("episode_index")
        ),
        starts=payload["start_time"],
        cohort=np.full(n, args.public.parent.name),
        split=np.zeros(n, dtype=np.int8),
        cadence=infer_native_cadence(
            payload["observed_mask"],
            payload.get("sensor_code"),
            payload.get("sensor_names"),
        ),
        source=payload.get("sensor_code", np.full(n, -1)),
    )
    daily_raw = _align_daily_embedding(args.daily, payload["window_id"])
    a, _ = assemble(
        data,
        daily_raw,
        control=checkpoint["control"],
        contract=checkpoint["contract"],
        strict_seven_days=checkpoint.get("strict_seven_days", False),
    )
    model = LongitudinalIdea(checkpoint["variant"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    torch.set_num_threads(2)
    out = outputs(model, tensorize(a, args.device), np.arange(n))
    transport = transport_from_state(checkpoint["transport"])
    memory = out["memory_codes"].reshape(n, 160)
    z = a["stable"] + checkpoint["gain"] * a["reliability"][
        :, None
    ] * transport.innovation(a["stable"], memory)
    z = (
        z * checkpoint["contract"]["daily_scale"]
        + checkpoint["contract"]["daily_center"]
    )
    z[a["days"] == 1] = daily_raw[a["days"] == 1]
    if not np.isfinite(z).all():
        raise ValueError("invalid public embeddings")
    keep = {
        k: v
        for k, v in payload.items()
        if k not in ("glucose_mg_dl", "observed_mask", "sensor_names")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, embedding=z.astype(np.float32), history_days=a["days"], **keep
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("train", "extract", "decide"))
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--source", type=Path)
    p.add_argument("--daily", type=Path)
    p.add_argument("--public", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control", action="store_true")
    p.add_argument("--strict-seven-days", action="store_true")
    p.add_argument("--steps", type=int, choices=(600, 6960), default=600)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--candidate-report", type=Path)
    p.add_argument("--control-report", type=Path)
    args = p.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "extract":
        public_extract(args)
    else:
        c = json.loads(args.candidate_report.read_text())
        g = json.loads(args.control_report.read_text())
        match = c["parameters"] == g["parameters"] and c["steps"] == g["steps"] == 600
        advantage = c["typed_ev"] - g["typed_ev"]
        accepted = c["acceptance_passed"] and match and advantage >= 0.02
        r = dict(
            acceptance_passed=accepted,
            equal_budget=match,
            typed_advantage=advantage,
            candidate_gates=c["gates"],
            candidate_sha256=sha256(args.candidate_report),
            control_sha256=sha256(args.control_report),
        )
        args.output.write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r), flush=True)
        raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
