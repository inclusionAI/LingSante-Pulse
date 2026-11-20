"""Train the fixed final-space structural pair on remote train/validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lingsante_pulse.cgm.models.final_space_longitudinal import (
    FinalSpaceLongitudinal,
    evidence_v2,
    final_space_loss,
    longitudinal_targets,
    supported_mean,
)
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    causal_history_indices,
    generic_multiscale_features,
    pack_histories,
    standardize_evidence,
)
from lingsante_pulse.cgm.models.parallel_longitudinal_models import bounded_histories
from lingsante_pulse.cgm.models.semantic_assimilation import supported_reliability
from lingsante_pulse.cgm.training.parallel_longitudinal import CORPUS_SHA, DAILY_SHA
from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
    _load_source,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import (
    _align_daily_embedding,
    sha256,
    mean_ev,
    effective_rank,
)
from lingsante_pulse.cgm.training.recurrent_orthogonal_canary import linear_cka


def prepare(data, raw_daily, control=False):
    fit = data["split"] == 0
    center, scale = raw_daily[fit].mean(0), np.maximum(raw_daily[fit].std(0), 1e-5)
    daily = ((raw_daily - center) / scale).astype(np.float32)
    histories = bounded_histories(
        causal_history_indices(
            data["subject"],
            data["starts"],
            data["cohort"],
            data["split"],
            data["cadence"],
            data["source"],
            max_days=7,
        ),
        data["starts"],
    )
    typed, support = evidence_v2(
        data["glucose"], data["observed"], data["cadence"], data["starts"]
    )
    raw_inputs = (
        generic_multiscale_features(data["glucose"], data["observed"])[0]
        if control
        else typed
    )
    features, feature_contract = standardize_evidence(raw_inputs, np.flatnonzero(fit))
    input_support = np.repeat(support[:, :1], 4, axis=1) if control else support
    packed = pack_histories(features, input_support, data["starts"], histories)
    target_raw = np.concatenate(
        [
            typed.reshape(len(typed), -1),
            longitudinal_targets(typed, support, histories).reshape(len(typed), -1),
        ],
        axis=1,
    )
    has_history = packed["history_mask"].any(1)
    fit_rows = np.flatnonzero(fit & has_history)
    audit_rows = np.flatnonzero((data["split"] == 1) & has_history)
    target_center = target_raw[fit_rows].mean(0)
    target_scale = np.maximum(target_raw[fit_rows].std(0), 1e-5)
    targets = ((target_raw - target_center) / target_scale).astype(np.float32)
    stable = supported_mean(daily, support[:, 0], histories)
    decoder = Ridge(alpha=1.0).fit(stable[fit_rows], targets[fit_rows])
    groups = np.unique(data["subject"], return_inverse=True)[1]
    reliability = supported_reliability(packed, support)
    contract = dict(
        daily_center=center,
        daily_scale=scale,
        features=feature_contract,
        target_center=target_center,
        target_scale=target_scale,
        decoder_weight=decoder.coef_,
        decoder_bias=decoder.intercept_,
        evidence_version=2,
    )
    return (
        dict(
            daily=daily,
            stable=stable,
            reliability=reliability,
            groups=groups,
            target=targets,
            packed=packed,
        ),
        fit_rows,
        audit_rows,
        contract,
    )


def call(model, tensors, rows, packed=None):
    p = tensors["packed"] if packed is None else packed
    return model(
        tensors["daily"][rows],
        tensors["stable"][rows],
        tensors["reliability"][rows],
        p["features"][rows],
        p["support"][rows],
        p["elapsed_days"][rows],
        p["history_mask"][rows],
    )


@torch.no_grad()
def evaluate(model, tensors, rows, packed=None):
    model.eval()
    final, losses = [], []
    for batch in np.array_split(rows, max(1, int(np.ceil(len(rows) / 256)))):
        z, prediction, _ = call(model, tensors, batch, packed)
        final.append(z.cpu().numpy())
        losses.append(
            float((prediction - tensors["target"][batch]).square().mean()) * len(batch)
        )
    return np.concatenate(final), sum(losses) / len(rows)


def readout(values, targets, fit, audit):
    decoder = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    predictions = decoder.fit(values[fit], targets[fit]).predict(values[audit])
    current = [
        mean_ev(
            targets[audit, k * 16 : (k + 1) * 16], predictions[:, k * 16 : (k + 1) * 16]
        )
        for k in range(4)
    ]
    long_term = [
        mean_ev(
            targets[audit, 64 + k * 64 : 64 + (k + 1) * 64],
            predictions[:, 64 + k * 64 : 64 + (k + 1) * 64],
        )
        for k in range(4)
    ]
    return dict(
        current=current,
        longitudinal=long_term,
        current_mean=float(np.mean(current)),
        longitudinal_mean=float(np.mean(long_term)),
    )


def train(args):
    # Claim this CUDA card before CPU-side preprocessing; idle watchdog yields.
    reservation = torch.empty(1, device=args.device)
    if sha256(args.source) != CORPUS_SHA or sha256(args.daily) != DAILY_SHA:
        raise ValueError("Frozen source/daily SHA mismatch")
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("Only train/validation allowed")
    if set(data["subject"][data["split"] == 0]) & set(
        data["subject"][data["split"] == 1]
    ):
        raise ValueError("Subject split overlap")
    a, fit, audit, contract = prepare(
        data, _align_daily_embedding(args.daily, data["window_id"]), args.control
    )
    if len(fit) < 2400 or len(audit) < 300:
        raise ValueError("Insufficient frozen support")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    torch.set_num_threads(2)
    model_class = FinalSpaceLongitudinal
    if args.physiology_guard:
        from lingsante_pulse.cgm.models.physiology_guarded_final_space import (
            PhysiologyGuardedFinalSpace,
        )

        model_class = PhysiologyGuardedFinalSpace
    model = model_class(
        args.variant, contract["decoder_weight"], contract["decoder_bias"]
    ).to(args.device)
    t = {
        key: (
            {k: torch.from_numpy(v).to(args.device) for k, v in value.items()}
            if key == "packed"
            else torch.from_numpy(value).to(args.device)
        )
        for key, value in a.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    rng = np.random.default_rng(43)
    trace = []
    for step in range(1, args.steps + 1):
        rows = rng.choice(fit, 128, replace=True)
        model.train()
        z, predictions, aux = call(model, t, rows)
        loss = final_space_loss(z, t["stable"][rows], predictions, t["target"][rows])
        if args.variant == "day_contrast":
            p = t["packed"]
            aux = model.encoder.contrast_loss(
                t["daily"][rows],
                p["features"][rows],
                p["support"][rows],
                p["elapsed_days"][rows],
                p["history_mask"][rows],
                t["groups"][rows],
            )
        loss = loss + 0.1 * aux
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            _, validation = evaluate(model, t, audit)
            row = dict(step=step, train=float(loss), validation=validation)
            trace.append(row)
            if step % 100 == 0:
                print(json.dumps(row), flush=True)
    final, _ = evaluate(model, t, np.arange(len(a["daily"])))
    _, loss = evaluate(model, t, audit)
    shuffled = {k: v.clone() for k, v in t["packed"].items()}
    for k in shuffled:
        shuffled[k][audit] = t["packed"][k][np.roll(audit, 1)]
    _, shuffled_loss = evaluate(model, t, audit, shuffled)
    baseline = readout(a["stable"], a["target"], fit, audit)
    learned = readout(final, a["target"], fit, audit)
    rank, base_rank = effective_rank(final[audit]), effective_rank(a["stable"][audit])
    cka = linear_cka(a["stable"][audit], final[audit])
    no_history = ~a["packed"]["history_mask"].any(1)
    max_days = float(a["packed"]["elapsed_days"].max() + 1)
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
        rank=rank >= 0.95 * base_rank,
        geometry=cka >= 0.9,
        history_used=shuffled_loss / max(loss, 1e-12) >= 1.01,
        no_history_identity=bool(
            np.array_equal(final[no_history], a["daily"][no_history])
        ),
        seven_day_span=max_days <= 7.0,
        subject_split_isolation=True,
    )
    protection_error = float(
        np.max(np.abs((final - a["stable"]) @ contract["decoder_weight"][:64].T))
    )
    if args.physiology_guard:
        gates["current_readout_conservation"] = protection_error <= 1e-4
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            contract=contract,
            variant=args.variant,
            control=args.control,
            steps=args.steps,
            seed=43,
            evidence_version=2,
            physiology_guard=args.physiology_guard,
            corpus_sha256=CORPUS_SHA,
            daily_sha256=DAILY_SHA,
        ),
        checkpoint,
    )
    report = dict(
        variant=args.variant,
        control=args.control,
        steps=args.steps,
        seed=43,
        fit_endpoints=len(fit),
        audit_endpoints=len(audit),
        baseline=baseline,
        learned=learned,
        stable_rank=base_rank,
        output_rank=rank,
        cka=cka,
        history_shuffle_ratio=shuffled_loss / max(loss, 1e-12),
        maximum_context_days=max_days,
        parameters=sum(p.numel() for p in model.parameters()),
        gates=gates,
        acceptance_passed=all(gates.values()),
        training=trace,
        checkpoint_sha256=sha256(checkpoint),
        corpus_sha256=CORPUS_SHA,
        daily_sha256=DAILY_SHA,
        phenotype_or_private_test_reads=0,
        glucofm_components=0,
        final_dimension=146,
        checkpoint_selection="last_step",
        physiology_guard=args.physiology_guard,
        current_readout_max_change=protection_error,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["day_contrast", "masked_scale"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--daily", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--control", action="store_true")
    p.add_argument("--physiology-guard", action="store_true")
    p.add_argument("--steps", type=int, choices=[600, 6960], default=600)
    train(p.parse_args())
