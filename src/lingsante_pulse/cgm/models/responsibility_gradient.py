"""Two-objective PCGrad restricted to the CGM responsibility update layer."""

import torch


def family_losses(error):
    if error.ndim != 2 or error.shape[1] != 336:
        raise ValueError("Original336 target layout required")
    result = {}
    for i, name in enumerate(("state", "challenge", "extreme", "rhythm")):
        result[name] = (
            error[:, i * 16 : (i + 1) * 16].sum(1).mean() / 64
            + error[:, 64 + i * 64 : 64 + (i + 1) * 64].sum(1).mean() / 272
        )
    result["extreme"] = result["extreme"] + error[:, 320:].sum(1).mean() / 272
    return result


def conflict_correction(a, b):
    a = a.double()
    b = b.double()
    dot = torch.dot(a, b)
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("Nonfinite responsibility gradients")
    if dot >= 0:
        return torch.zeros_like(a), False
    # Negative dot implies nonzero norms; original gradients used symmetrically.
    return -dot / b.square().sum() * b - dot / a.square().sum() * a, True


def responsibility_backward(model, losses, scope):
    if scope not in ("readout", "all"):
        raise ValueError("Frozen scope required")
    named = list(model.named_parameters())
    params = [p for _, p in named]
    a = torch.autograd.grad(losses["state"], params, retain_graph=True)
    b = torch.autograd.grad(losses["challenge"], params, retain_graph=True)
    sum(losses.values()).backward()
    selected = [
        i
        for i, (name, _) in enumerate(named)
        if scope == "all" or name.startswith("readout.")
    ]
    if not selected:
        raise ValueError("Missing update layer")
    delta, conflict = conflict_correction(
        torch.cat([a[i].flatten() for i in selected]),
        torch.cat([b[i].flatten() for i in selected]),
    )
    if conflict:
        offset = 0
        for i in selected:
            size = params[i].numel()
            params[i].grad.add_(
                delta[offset : offset + size].reshape_as(params[i]).to(params[i].dtype)
            )
            offset += size
    return dict(conflict=conflict, correction_norm=float(delta.norm()))


def responsibility_decision(candidate, control, reference):
    from lingsante_pulse.cgm.models.local_trajectory_occupation import (
        trajectory_support_valid,
    )

    hard = (
        "finite",
        "initial_stable",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
        "shared_lift_gradient",
        "two_day_update",
        "slot_permutation",
    )
    values = lambda r: [
        r["learned"]["current_mean"],
        r["learned"]["longitudinal_mean"],
        r["event_ev"],
    ]
    c, g, ref = map(values, (candidate, control, reference))
    base = [
        candidate["baseline"]["current_mean"],
        candidate["baseline"]["longitudinal_mean"],
        candidate["stable_event_ev"],
    ]
    dominated = (
        all(y >= x for x, y in zip(c, g))
        and control["episode_validation_loss"] <= candidate["episode_validation_loss"]
        and (
            any(y > x for x, y in zip(c, g))
            or control["episode_validation_loss"] < candidate["episode_validation_loss"]
        )
    )
    gates = dict(
        structural_isolation=all(
            r["gates"].get(k) is True for r in (candidate, control) for k in hard
        ),
        support=all(
            trajectory_support_valid(r["occupation_path"])
            and r["occupation_path"]["sorted_control"] is False
            for r in (candidate, control)
        ),
        forbidden_inputs_absent=all(
            r["phenotype_or_private_test_reads"] == 0 and r["glucofm_components"] == 0
            for r in (candidate, control)
        ),
        equal_budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 5648, 146)
            for r in (candidate, control)
        ),
        projection_valid=all(
            r["responsibility_gradient"]["conflict_steps"] > 0
            and r["responsibility_gradient"]["maximum_correction_norm"] > 0
            and r["responsibility_gradient"]["reverse_passes_per_step"] == 3
            for r in (candidate, control)
        ),
        distinct_scope=candidate["responsibility_gradient"]["scope"] == "readout"
        and control["responsibility_gradient"]["scope"] == "all",
        useful_progress=any(x > max(y, z) for x, y, z in zip(c, ref, base))
        and candidate["episode_validation_loss"]
        < candidate["stable_episode_validation_loss"],
        not_dominated_by_generic=not dominated,
        frozen_reference=reference["checkpoint_sha256"]
        == "50d7bed5bbffccec8662d213f31bbfadf685b6669f1f0c4f02aeb72cbeba1e14",
        identical_episodes=len(
            {r["episode_sequence_sha256"] for r in (candidate, control, reference)}
        )
        == 1,
        identical_targets=len(
            {r["target_sha256"] for r in (candidate, control, reference)}
        )
        == 1,
    )
    return dict(
        policy="responsibility_gradient_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        candidate_minus_generic=[x - y for x, y in zip(c, g)],
        rank_policy="diagnostic_only",
    )
