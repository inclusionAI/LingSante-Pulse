"""Fixed seed43 observation-supported attention candidate/control training."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from lingsante_pulse.cgm.models.observation_supported_attention import (
    ObservationSupportedAttention,
    pack_observations,
)
from lingsante_pulse.cgm.models.final_space_longitudinal import evidence_v2
from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
    generic_multiscale_features,
)
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


from lingsante_pulse.cgm.data.history import make_inputs


def call(model, packed, rows):
    return model(**{k: v[rows] for k, v in packed.items()})[0]


def objective(z, weight, bias, target):
    error = (z @ weight.T + bias - target).square()
    return error[:, :64].mean() + error[:, 64:].mean()


def training_readout(contract, device, joint):
    weight = torch.as_tensor(contract["decoder_weight"], device=device).clone()
    bias = torch.as_tensor(contract["decoder_bias"], device=device).clone()
    if joint:
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
    return weight, bias


@torch.no_grad()
def evaluate(model, packed, rows, weight, bias, targets):
    model.eval()
    z = torch.cat(
        [
            call(model, packed, r)
            for r in np.array_split(rows, max(1, int(np.ceil(len(rows) / 512))))
        ]
    )
    loss = objective(z, weight, bias, targets[rows])
    return z.cpu().numpy(), float(loss)


def train(args):
    if not args.device.startswith("cuda"):
        raise ValueError("Training requires remote CUDA")
    reservation = torch.empty(1, device=args.device)
    if sha256(args.source) != CORPUS_SHA or sha256(args.daily) != DAILY_SHA:
        raise ValueError("Frozen corpus/daily SHA mismatch")
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("Only split0/1 allowed")
    if set(data["subject"][data["split"] == 0]) & set(
        data["subject"][data["split"] == 1]
    ):
        raise ValueError("Subject split overlap")
    a, fit, audit, contract = prepare(
        data, _align_daily_embedding(args.daily, data["window_id"]), args.control
    )
    if len(fit) < 2400 or len(audit) < 300:
        raise ValueError("Insufficient support")
    inputs = make_inputs(data, a["daily"], contract, args.control)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    torch.set_num_threads(2)
    model = ObservationSupportedAttention().to(args.device)
    packed = {k: torch.from_numpy(v).to(args.device) for k, v in inputs.items()}
    weight, bias = training_readout(contract, args.device, args.joint_readout)
    parameters = list(model.parameters()) + (
        [weight, bias] if args.joint_readout else []
    )
    targets = torch.from_numpy(a["target"]).to(args.device)
    allrows = np.arange(len(a["daily"]))
    initial, _ = evaluate(model, packed, allrows, weight, bias, targets)
    initial_error = float(abs(initial - a["stable"]).max())
    if not np.allclose(initial, a["stable"], atol=2e-6, rtol=1e-5):
        raise ValueError("Zero-init does not reproduce stable origin")
    optimizer = torch.optim.AdamW(parameters, lr=0.0005, weight_decay=0.0001)
    rng = np.random.default_rng(43)
    batch_hash = hashlib.sha256()
    trace = []
    max_grad = 0.0
    max_readout_grad = 0.0
    for step in range(1, args.steps + 1):
        rows = rng.choice(fit, 128, replace=True)
        batch_hash.update(rows.astype("<i8").tobytes())
        model.train()
        z = call(model, packed, rows)
        loss = objective(z, weight, bias, targets[rows])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_grad = max(max_grad, float(model.score[-1].weight.grad.norm()))
        if args.joint_readout:
            max_readout_grad = max(max_readout_grad, float(weight.grad.norm()))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % 100 == 0:
            _, val = evaluate(model, packed, audit, weight, bias, targets)
            row = dict(step=step, train=float(loss), validation=val)
            trace.append(row)
            print(json.dumps(row), flush=True)
    final, _ = evaluate(model, packed, allrows, weight, bias, targets)
    _, loss = evaluate(model, packed, audit, weight, bias, targets)
    shuffled = {k: v.clone() for k, v in packed.items()}
    # Counterfactual only: keep current day; mismatch prior evidence and values.
    donors = np.roll(audit, 1)
    for k in shuffled:
        shuffled[k][audit, 1:] = packed[k][donors, 1:]
    _, shuffle_loss = evaluate(model, shuffled, audit, weight, bias, targets)
    baseline = readout(a["stable"], a["target"], fit, audit)
    learned = readout(final, a["target"], fit, audit)
    valid = inputs["mask"] & (inputs["support"][:, :, 0] > 0)
    lo = np.where(valid[:, :, None], inputs["values"], np.inf).min(1)
    hi = np.where(valid[:, :, None], inputs["values"], -np.inf).max(1)
    bound_error = float(max(0.0, np.max(lo - final), np.max(final - hi)))
    no_history = ~valid[:, 1:].any(1)
    rank, base_rank = effective_rank(final[audit]), effective_rank(a["stable"][audit])
    cka = linear_cka(a["stable"][audit], final[audit])
    ratio = shuffle_loss / max(loss, 1e-12)
    gates = dict(
        finite=bool(np.isfinite(final).all()),
        zero_init=initial_error <= 2e-5,
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
        geometry=cka >= 0.90,
        history_used=ratio >= 1.01,
        no_history_identity=bool(
            np.array_equal(final[no_history], a["daily"][no_history])
        ),
        observation_value_bounds=bound_error <= 2e-5,
        score_gradient=max_grad > 0,
        seven_day_span=bool(inputs["elapsed_days"].max() + 1 <= 7),
        subject_split_isolation=True,
    )
    checkpoint = args.output / "checkpoint.pt"
    if args.joint_readout and max_readout_grad <= 0:
        raise ValueError("Joint decoder did not receive gradients")
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            contract=contract,
            control=args.control,
            steps=args.steps,
            seed=43,
            architecture="observation_supported_attention_v1",
            joint_readout=args.joint_readout,
            training_readout=dict(
                weight=weight.detach().cpu(), bias=bias.detach().cpu()
            ),
            corpus_sha256=CORPUS_SHA,
            daily_sha256=DAILY_SHA,
        ),
        checkpoint,
    )
    report = dict(
        steps=args.steps,
        seed=43,
        control=args.control,
        baseline=baseline,
        learned=learned,
        fit_endpoints=len(fit),
        audit_endpoints=len(audit),
        initial_max_error=initial_error,
        bound_max_error=bound_error,
        max_score_gradient=max_grad,
        output_rank=rank,
        stable_rank=base_rank,
        cka=cka,
        history_shuffle_ratio=ratio,
        gates=gates,
        acceptance_passed=all(gates.values()),
        batch_sequence_sha256=batch_hash.hexdigest(),
        parameters=sum(p.numel() for p in parameters),
        inference_parameters=sum(p.numel() for p in model.parameters()),
        joint_readout=args.joint_readout,
        max_readout_gradient=max_readout_grad,
        checkpoint_sha256=sha256(checkpoint),
        corpus_sha256=CORPUS_SHA,
        daily_sha256=DAILY_SHA,
        maximum_context_days=float(inputs["elapsed_days"].max() + 1),
        training=trace,
        phenotype_or_private_test_reads=0,
        glucofm_components=0,
        final_dimension=146,
        checkpoint_selection="last_step",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "daily", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--control", action="store_true")
    p.add_argument("--joint-readout", action="store_true")
    p.add_argument("--steps", type=int, choices=[600, 6960], default=600)
    train(p.parse_args())
