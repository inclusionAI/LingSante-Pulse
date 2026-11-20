"""Balance unlabeled acquisition cohorts before subject-disjoint episode draws."""

import numpy as np


class CohortBalancedEpisodes:
    def __init__(self, subjects, cohorts, fit_rows, seed=43):
        subjects = np.asarray(subjects)
        cohorts = np.asarray(cohorts)
        fit_rows = np.asarray(fit_rows, dtype=np.int64)
        self.subjects = np.unique(subjects[fit_rows])
        self.cohorts = np.unique(cohorts[fit_rows])
        if len(self.cohorts) != 4:
            raise ValueError("Frozen four supported pretraining cohorts required")
        for s in self.subjects:
            if len(np.unique(cohorts[fit_rows[subjects[fit_rows] == s]])) != 1:
                raise ValueError("Subject spans cohorts")
        self.groups = []
        for c in self.cohorts:
            rows = fit_rows[cohorts[fit_rows] == c]
            ids = np.unique(subjects[rows])
            if len(ids) < 4:
                raise ValueError("Insufficient independent subjects in a cohort")
            self.groups.append([rows[subjects[rows] == s] for s in ids])
        self.rng = np.random.default_rng(seed)

    def next(self):
        output = [[], []]
        for groups in self.groups:
            halves = np.array_split(self.rng.permutation(len(groups)), 2)
            for side, available in enumerate(halves):
                output[side].extend(
                    self.rng.choice(groups[g])
                    for g in self.rng.choice(available, 16, replace=True)
                )
        return tuple(
            self.rng.permutation(np.asarray(rows, dtype=np.int64)) for rows in output
        )


def balanced_episode_decision(candidate, reference):
    c, r = candidate, reference
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
    long_ok = c["learned"]["longitudinal_mean"] > max(
        c["baseline"]["longitudinal_mean"], r["learned"]["longitudinal_mean"]
    )
    event_ok = c["event_ev"] > max(c["stable_event_ev"], r["event_ev"])
    gates = dict(
        structural_isolation=all(
            x.get("gates", {}).get(k) is True for x in (c, r) for k in hard
        ),
        useful_balancing_progress=c["episode_validation_loss"]
        < c["stable_episode_validation_loss"]
        and (long_ok or event_ok),
        equal_parameters=c["parameters"] == r["parameters"] == 20928,
        fixed_budget=all(
            x["seed"] == 43 and x["steps"] == 600 and x["kind"] == "candidate"
            for x in (c, r)
        ),
        identical_targets=c["target_sha256"] == r["target_sha256"],
        frozen_reference=r["checkpoint_sha256"]
        == "bf7856e1fecba37cc590454be05715f9dae6834618c743074100652d96e143fe",
        source_balanced=c.get("cohort_balanced_episodes", {}).get("gates_passed")
        is True,
    )
    return dict(
        policy="cohort_balanced_pair_progress_v1",
        gates=gates,
        acceptance_passed=all(gates.values()),
        original_proxy_acceptance_passed=c["acceptance_passed"],
        reference_reused_not_retrained=True,
        full_or_public_launched=False,
    )


def validate_balanced_exploration(report, decision, steps=6960):
    if (
        decision.get("policy") != "balanced_pair_public_transfer_exploration_v1"
        or decision.get("exploratory_advancement") is not True
        or decision.get("original_preregistered_acceptance_passed") is not False
    ):
        raise ValueError(
            "Explicit exploratory decision required; do not rewrite original failure"
        )
    if (
        decision.get("candidate_canary_sha256")
        != "02bc5c98536a0fb68820e33eb5039690ed71d073ae870dd527eaec0c7e71da2d"
    ):
        raise ValueError("Wrong frozen canary")
    if steps not in (600, 6960) or (
        report["steps"],
        report["seed"],
        report["kind"],
        report["parameters"],
        report["final_dimension"],
    ) != (steps, 43, "candidate", 20928, 146):
        raise ValueError("Wrong budget or architecture")
    if (
        steps == 600
        and report["checkpoint_sha256"] != decision["candidate_canary_sha256"]
    ):
        raise ValueError("Wrong canary checkpoint")
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
    if not all(report.get("gates", {}).get(k) is True for k in hard):
        raise ValueError("Structural/isolation failure")
    balance = report["cohort_balanced_episodes"]
    if balance["gates_passed"] is not True or balance["cohort_id_encoder_inputs"] != 0:
        raise ValueError("Cohort isolation failure")
    counts = balance["per_source_sampled_rows"]
    if set(counts) != {
        "private_source_private",
        "big_ideas",
        "shanghai_t2dm",
        "stanford",
    } or set(counts.values()) != {32 * steps}:
        raise ValueError("Incorrect source exposure")
    if (
        report["shared_pair_lift"]["control"] is not False
        or report["phenotype_or_private_test_reads"] != 0
        or report["glucofm_components"] != 0
    ):
        raise ValueError("Wrong model/data boundary")
