"""Differentiable, support-only Ridge and subject-disjoint CGM episodes."""

import hashlib
import numpy as np
import torch


def coherent_canary_decision(candidate, generic):
    c, g = candidate, generic
    hard = (
        "finite",
        "initial_stable",
        "value_bounds",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
        "joint_convex",
    )
    gates = dict(
        structural_isolation=all(
            r.get("gates", {}).get(k) is True for r in (c, g) for k in hard
        ),
        useful_progress=c["episode_validation_loss"]
        < c["stable_episode_validation_loss"]
        and (
            c["learned"]["current_mean"] > c["baseline"]["current_mean"]
            or c["learned"]["longitudinal_mean"] > c["baseline"]["longitudinal_mean"]
            or c["event_ev"] > c["stable_event_ev"]
        ),
        typed_long_noninferior=c["learned"]["longitudinal_mean"]
        >= g["learned"]["longitudinal_mean"],
        equal_parameters=c["parameters"] == g["parameters"] == 8641,
        fixed_budget=all(r["seed"] == 43 and r["steps"] == 600 for r in (c, g)),
        identical_episodes=c["episode_sequence_sha256"] == g["episode_sequence_sha256"],
        identical_targets=c["target_sha256"] == g["target_sha256"],
    )
    return dict(
        gates=gates,
        acceptance_passed=all(gates.values()),
        strict_proxy_acceptance_passed=c["acceptance_passed"],
        policy="coherent_preregistered_local_progress_v1",
        full_or_public_launched=False,
    )


def validate_coherent_export(report, decision):
    if (
        decision.get("policy") != "coherent_preregistered_local_progress_v1"
        or decision.get("acceptance_passed") is not True
    ):
        raise ValueError("Prospective coherent canary decision required")
    hard = (
        "finite",
        "initial_stable",
        "value_bounds",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
        "joint_convex",
    )
    if not all(report.get("gates", {}).get(k) is True for k in hard):
        raise ValueError("Coherent structural or isolation validity failed")
    if (
        report["steps"],
        report["seed"],
        report["kind"],
        report["final_dimension"],
        report["parameters"],
    ) != (6960, 43, "candidate", 146, 8641):
        raise ValueError("Only frozen coherent full candidate may be exported")


def validate_exploratory_export(report, advancement):
    if (
        advancement.get("original_acceptance_passed") is not False
        or advancement.get("proceed_full_for_public_test") is not True
    ):
        raise ValueError(
            "Explicit exploratory decision required; never rewrite original pass"
        )
    hard = (
        "finite",
        "initial_stable",
        "value_bounds",
        "no_history_identity",
        "outer_gradient",
        "seven_day_span",
        "subject_split_isolation",
    )
    if not all(report.get("gates", {}).get(k) is True for k in hard):
        raise ValueError("Structural or isolation validity failed")
    if (report["steps"], report["seed"], report["kind"], report["final_dimension"]) != (
        6960,
        43,
        "candidate",
        146,
    ):
        raise ValueError("Only frozen full candidate may be exported")


def ridge_predict(support, support_targets, query, stop_through_fit=False):
    """Alpha=1 with intercept; query features/targets never fit the inner solver."""
    fit_support = support.detach() if stop_through_fit else support
    x, y, q = fit_support.double(), support_targets.double(), query.double()
    center = x.mean(0, keepdim=True)
    variance = (x - center).square().mean(0, keepdim=True)
    scale = torch.where(
        variance == 0, torch.ones_like(variance), variance.clamp_min(1e-10).sqrt()
    )
    x, q = (x - center) / scale, (q - center) / scale
    target_center = y.mean(0, keepdim=True)
    gram = x @ x.T + torch.eye(len(x), dtype=x.dtype, device=x.device)
    coefficient = torch.linalg.solve(gram, y - target_center)
    return (q @ x.T @ coefficient + target_center).to(support.dtype)


def episodic_loss(
    support,
    support_targets,
    query,
    query_targets,
    stop_through_fit=False,
    target_split=64,
):
    prediction = ridge_predict(
        support, support_targets, query, stop_through_fit=stop_through_fit
    )
    error = (prediction - query_targets).square()
    if not 0 < target_split < error.shape[1]:
        raise ValueError("Target split must divide two non-empty target families")
    return error[:, :target_split].mean() + error[:, target_split:].mean()


class SubjectEpisodes:
    """IDs are used only for sampling isolation, never returned as model features."""

    def __init__(self, subjects, fit_rows, seed=43):
        subjects = np.asarray(subjects)
        fit_rows = np.asarray(fit_rows, dtype=np.int64)
        self.row_subjects = subjects
        self.subjects = np.unique(subjects[fit_rows])
        if len(self.subjects) < 4:
            raise ValueError("At least four independent fit subjects required")
        self.groups = [fit_rows[subjects[fit_rows] == s] for s in self.subjects]
        self.rng = np.random.default_rng(seed)

    def next(self):
        order = self.rng.permutation(len(self.groups))
        sides = np.array_split(order, 2)
        return tuple(
            np.asarray(
                [
                    self.rng.choice(self.groups[g])
                    for g in self.rng.choice(side, 64, replace=True)
                ],
                dtype=np.int64,
            )
            for side in sides
        )


class RandomRowRepartitionEpisodes:
    """Repartition one matched identity episode without changing row exposure."""

    def __init__(self, base, row_subjects, row_families=None, seed=9043):
        self.base = base
        self.subjects = base.subjects
        self.row_subjects = np.asarray(row_subjects)
        self.row_families = (
            np.repeat("all", len(self.row_subjects))
            if row_families is None
            else np.asarray(row_families).astype(str)
        )
        if self.row_families.shape != self.row_subjects.shape:
            raise ValueError("Row families and subjects must align")
        self.rng = np.random.default_rng(seed)
        self.base_hash = hashlib.sha256()
        self.union_hash = hashlib.sha256()
        self.repartition_hash = hashlib.sha256()
        self.episodes = 0
        self.identity_overlap_total = 0
        self.identity_overlap_episodes = 0

    def _partition_family(self, rows, target):
        unique, counts = np.unique(rows, return_counts=True)
        order = self.rng.permutation(len(unique))
        reachable = {0: ()}
        for index in order:
            count = int(counts[index])
            additions = {}
            for total, chosen in reachable.items():
                candidate = total + count
                if (
                    candidate <= target
                    and candidate not in reachable
                    and candidate not in additions
                ):
                    additions[candidate] = chosen + (int(index),)
            reachable.update(additions)
        if target not in reachable:
            return None
        selected = np.zeros(len(unique), dtype=bool)
        selected[list(reachable[target])] = True
        support = np.repeat(unique[selected], counts[selected])
        query = np.repeat(unique[~selected], counts[~selected])
        self.rng.shuffle(support)
        self.rng.shuffle(query)
        return support, query

    def next(self):
        original_support, original_query = self.base.next()
        self.base_hash.update(original_support.astype("<i8").tobytes())
        self.base_hash.update(original_query.astype("<i8").tobytes())
        original_union = np.concatenate([original_support, original_query])
        self.union_hash.update(np.sort(original_union).astype("<i8").tobytes())

        for _ in range(1000):
            support_parts = []
            query_parts = []
            feasible = True
            for family in np.unique(self.row_families[original_union]):
                family_rows = original_union[
                    self.row_families[original_union] == family
                ]
                target = int(
                    np.count_nonzero(self.row_families[original_support] == family)
                )
                partition = self._partition_family(family_rows, target)
                if partition is None:
                    feasible = False
                    break
                support_parts.append(partition[0])
                query_parts.append(partition[1])
            if not feasible:
                continue
            support = np.concatenate(support_parts)
            query = np.concatenate(query_parts)
            self.rng.shuffle(support)
            self.rng.shuffle(query)

            if set(support).intersection(query):
                continue
            identity_overlap = set(self.row_subjects[support]).intersection(
                self.row_subjects[query]
            )
            if not np.array_equal(
                np.sort(np.concatenate([support, query])), np.sort(original_union)
            ):
                raise AssertionError("Random-row repartition changed row exposure")
            for family in np.unique(self.row_families[original_union]):
                before = (
                    np.count_nonzero(self.row_families[original_support] == family),
                    np.count_nonzero(self.row_families[original_query] == family),
                )
                after = (
                    np.count_nonzero(self.row_families[support] == family),
                    np.count_nonzero(self.row_families[query] == family),
                )
                if before != after:
                    raise AssertionError("Random-row repartition changed source counts")
            self.repartition_hash.update(support.astype("<i8").tobytes())
            self.repartition_hash.update(query.astype("<i8").tobytes())
            self.episodes += 1
            self.identity_overlap_total += len(identity_overlap)
            self.identity_overlap_episodes += int(bool(identity_overlap))
            return support, query
        raise ValueError("Unable to form a row-disjoint random-row episode")

    def summary(self):
        report = {
            "policy": "matched_random_row_repartition_v1",
            "episodes": self.episodes,
            "row_overlap_rate": 0.0,
            "mean_identity_overlap": (
                self.identity_overlap_total / self.episodes if self.episodes else 0.0
            ),
            "identity_overlap_episode_rate": (
                self.identity_overlap_episodes / self.episodes if self.episodes else 0.0
            ),
            "base_episode_sequence_sha256": self.base_hash.hexdigest(),
            "episode_union_sha256": self.union_hash.hexdigest(),
            "repartitioned_episode_sequence_sha256": self.repartition_hash.hexdigest(),
        }
        if hasattr(self.base, "summary"):
            report["base_sampling"] = self.base.summary()
        return report
