"""Training-only PBQ/public sampling; never supplies source IDs as features."""

import numpy as np

from lingsante_pulse.cgm.models.episodic_evidence import SubjectEpisodes


def load_source_family(path, expected_rows):
    with np.load(path, allow_pickle=False) as archive:
        if "source_family" not in archive.files:
            raise ValueError("Explicit source_family is required for source mixing")
        families = archive["source_family"].astype(str)
    if families.shape != (expected_rows,):
        raise ValueError("source_family must align with corpus rows")
    if not set(np.unique(families)) <= {
        "pbq_private_source",
        "own_private_source",
        "public",
    }:
        raise ValueError(
            "Unknown source_family; do not infer public from cohort aliases"
        )
    return families


def load_source_sampling_stratum(path, expected_rows, key="source_replay_stratum"):
    """Load provenance-only replay strata for exposure auditing."""
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"Explicit {key} is required for replay auditing")
        strata = archive[key].astype(str)
    if strata.shape != (expected_rows,):
        raise ValueError(f"{key} must align with corpus rows")
    if not set(np.unique(strata)) <= {
        "anchor_private_source",
        "increment_private_source",
        "public",
    }:
        raise ValueError(f"Unknown {key}")
    return strata


def private_family(families):
    names = set(np.unique(families))
    private = names - {"public"}
    if len(private) != 1 or not private <= {"pbq_private_source", "own_private_source"}:
        raise ValueError("Exactly one explicit private data provenance is required")
    return next(iter(private))


class SourceMixtureEpisodes(SubjectEpisodes):
    """Stratify disjoint subject pools, then weight subject draws by source."""

    def __init__(self, subjects, fit_rows, families, pbq_probability, seed=43):
        if not np.isfinite(pbq_probability) or not 0 < pbq_probability < 1:
            raise ValueError("PBQ probability must be strictly between zero and one")
        super().__init__(subjects, fit_rows, seed)
        families = np.asarray(families).astype(str)
        self.private_name = private_family(families)
        if families.shape != np.asarray(subjects).shape:
            raise ValueError("Source and subject arrays must align")
        labels = []
        for rows in self.groups:
            source = np.unique(families[rows])
            if len(source) != 1 or source[0] not in {self.private_name, "public"}:
                raise ValueError("Each fit subject must have exactly one known source")
            labels.append(source[0])
        self.group_family = np.asarray(labels)
        self.pbq_probability = float(pbq_probability)
        self.source_groups = [
            np.flatnonzero(self.group_family == name)
            for name in (self.private_name, "public")
        ]
        if any(len(groups) < 2 for groups in self.source_groups):
            raise ValueError("Each source needs two independent fit subjects")
        self.draw_counts = np.zeros((2, 2), dtype=np.int64)
        self.seen_groups = np.zeros(len(self.groups), dtype=bool)
        self.episodes = 0

    def next(self):
        # Source-stratified partition avoids removing a rare source from one side.
        partitions = [
            np.array_split(self.rng.permutation(groups), 2)
            for groups in self.source_groups
        ]
        result = []
        for side in range(2):
            groups = np.concatenate([parts[side] for parts in partitions])
            probabilities = np.concatenate(
                [
                    np.full(len(parts[side]), mass / len(parts[side]))
                    for parts, mass in zip(
                        partitions, (self.pbq_probability, 1 - self.pbq_probability)
                    )
                ]
            )
            selected = self.rng.choice(groups, 64, replace=True, p=probabilities)
            self.seen_groups[selected] = True
            for source, name in enumerate((self.private_name, "public")):
                self.draw_counts[side, source] += np.count_nonzero(
                    self.group_family[selected] == name
                )
            result.append(
                np.asarray(
                    [self.rng.choice(self.groups[g]) for g in selected], dtype=np.int64
                )
            )
        self.episodes += 1
        return tuple(result)

    def summary(self):
        return {
            "policy": "source_stratified_subject_mixture_v1",
            "pbq_probability": self.pbq_probability,
            "episodes": self.episodes,
            "draw_counts": {
                side: dict(zip((self.private_name, "public"), map(int, counts)))
                for side, counts in zip(("support", "query"), self.draw_counts)
            },
            "fit_subject_counts": {
                name: int(np.count_nonzero(self.group_family == name))
                for name in (self.private_name, "public")
            },
            "sampled_unique_subject_counts": {
                name: int(
                    np.count_nonzero(self.seen_groups & (self.group_family == name))
                )
                for name in (self.private_name, "public")
            },
            "validation_sampler": "unchanged SubjectEpisodes(seed=43)",
        }


class WeightedStratifiedSubjectEpisodes:
    """Draw weighted rows while keeping support/query subjects disjoint.

    The supplied probabilities retain the frozen cadence/source/replay exposure.
    Subjects are repartitioned independently inside every provenance stratum so
    that neither side loses a rare source.  Identity and stratum are sampling
    metadata only and are never returned as model inputs.
    """

    def __init__(
        self, subjects, fit_rows, strata, probability, *, batch_size=64, seed=43
    ):
        subjects = np.asarray(subjects).astype(str)
        fit_rows = np.asarray(fit_rows, dtype=np.int64)
        strata = np.asarray(strata).astype(str)
        probability = np.asarray(probability, dtype=np.float64)
        if subjects.shape != strata.shape:
            raise ValueError("Subjects and strata must align")
        if probability.shape != (len(fit_rows),):
            raise ValueError("Probability must align with fit rows")
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise ValueError("Sampling probabilities must be finite and nonnegative")
        if probability.sum() <= 0:
            raise ValueError("Sampling probabilities need positive mass")
        self.subjects = subjects
        self.fit_rows = fit_rows
        self.strata = strata
        self.probability = probability / probability.sum()
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("Episode batch size must be at least two")
        self.rng = np.random.default_rng(seed)
        unique_subjects, fit_subject_index = np.unique(
            subjects[fit_rows], return_inverse=True
        )
        self.unique_subjects = unique_subjects
        self.fit_subject_index = fit_subject_index
        self.subject_stratum = np.empty(len(unique_subjects), dtype=object)
        for subject_index, subject in enumerate(unique_subjects):
            rows = fit_rows[fit_subject_index == subject_index]
            names = np.unique(strata[rows])
            if len(names) != 1:
                raise ValueError("Each subject must belong to exactly one stratum")
            self.subject_stratum[subject_index] = names[0]
        self.stratum_subjects = {
            name: np.flatnonzero(self.subject_stratum == name)
            for name in np.unique(strata[fit_rows])
        }
        if any(len(group) < 2 for group in self.stratum_subjects.values()):
            raise ValueError("Every stratum needs at least two fit subjects")
        self.row_probability = np.zeros(len(subjects), dtype=np.float64)
        self.row_probability[fit_rows] = self.probability
        self.episodes = 0
        self.draw_counts = {
            side: {name: 0 for name in self.stratum_subjects}
            for side in ("support", "query")
        }

    def next(self):
        side_assignment = np.full(len(self.unique_subjects), -1, dtype=np.int8)
        for subject_indices in self.stratum_subjects.values():
            halves = np.array_split(self.rng.permutation(subject_indices), 2)
            side_assignment[halves[0]] = 0
            side_assignment[halves[1]] = 1
        if np.any(side_assignment < 0):
            raise AssertionError("Every fit subject must receive one episode side")
        result = []
        for side in range(2):
            allowed = side_assignment[self.fit_subject_index] == side
            weights = self.probability * allowed
            if weights.sum() <= 0:
                raise ValueError("Episode side has no positive sampling mass")
            weights /= weights.sum()
            rows = self.rng.choice(
                self.fit_rows, size=self.batch_size, replace=True, p=weights
            ).astype(np.int64)
            for name in self.draw_counts[("support", "query")[side]]:
                self.draw_counts[("support", "query")[side]][name] += int(
                    np.count_nonzero(self.strata[rows] == name)
                )
            result.append(rows)
        if np.intersect1d(
            self.subjects[result[0]], self.subjects[result[1]], assume_unique=False
        ).size:
            raise AssertionError("Support/query subject isolation failed")
        self.episodes += 1
        return tuple(result)

    def summary(self):
        return {
            "policy": "weighted_stratified_subject_episodes_v1",
            "episodes": self.episodes,
            "batch_size_per_side": self.batch_size,
            "fit_subject_counts": {
                name: int(len(subjects))
                for name, subjects in self.stratum_subjects.items()
            },
            "draw_counts": self.draw_counts,
            "subject_overlap_rate": 0.0,
        }


def source_mixture_probabilities(base_probability, families, pbq_probability):
    """Change source mass while preserving conditional row weights per source."""
    if not np.isfinite(pbq_probability) or not 0 < pbq_probability < 1:
        raise ValueError("PBQ probability must be strictly between zero and one")
    families = np.asarray(families).astype(str)
    probability = np.asarray(base_probability, dtype=np.float64).copy()
    if probability.ndim != 1 or probability.shape != families.shape:
        raise ValueError("Source and probability vectors must align")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Base probabilities must be finite and nonnegative")
    private = private_family(families)
    if set(np.unique(families)) != {private, "public"}:
        raise ValueError("Both explicit PBQ and public sources are required")
    for name, mass in ((private, pbq_probability), ("public", 1 - pbq_probability)):
        selected = families == name
        total = probability[selected].sum()
        if total <= 0:
            raise ValueError("Each source needs positive base probability mass")
        probability[selected] *= mass / total
    return probability / probability.sum()


def source_replay_probabilities(
    base_probability, families, pbq_probability, strata, anchor_probability
):
    """Fix public/private mass and old/new mass while preserving within-stratum weights."""
    if not np.isfinite(anchor_probability) or not 0 < anchor_probability < 1:
        raise ValueError(
            "Anchor replay probability must be strictly between zero and one"
        )
    families = np.asarray(families).astype(str)
    strata = np.asarray(strata).astype(str)
    probability = np.asarray(base_probability, dtype=np.float64).copy()
    if (
        probability.ndim != 1
        or probability.shape != families.shape
        or strata.shape != families.shape
    ):
        raise ValueError("Source, replay stratum and probability vectors must align")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Base probabilities must be finite and nonnegative")
    private = private_family(families)
    if set(np.unique(families)) != {private, "public"}:
        raise ValueError("Both explicit private and public sources are required")
    if set(np.unique(strata)) != {
        "anchor_private_source",
        "increment_private_source",
        "public",
    }:
        raise ValueError("All three replay strata are required")
    if np.any((families == "public") != (strata == "public")):
        raise ValueError("Public source and replay stratum must agree")
    targets = (
        ("anchor_private_source", pbq_probability * anchor_probability),
        ("increment_private_source", pbq_probability * (1 - anchor_probability)),
        ("public", 1 - pbq_probability),
    )
    for name, mass in targets:
        selected = strata == name
        total = probability[selected].sum()
        if total <= 0:
            raise ValueError(f"{name} needs positive base probability mass")
        probability[selected] *= mass / total
    return probability / probability.sum()


class SourceDrawAudit:
    """Aggregate source exposure without exporting participant identifiers."""

    def __init__(self, families, strata=None):
        self.families = np.asarray(families).astype(str)
        self.counts = {private_family(self.families): 0, "public": 0}
        self.seen = np.zeros(len(self.families), dtype=bool)
        self.strata = None if strata is None else np.asarray(strata).astype(str)
        if self.strata is not None:
            if self.strata.shape != self.families.shape:
                raise ValueError("Replay strata and source arrays must align")
            if not set(np.unique(self.strata)) <= {
                "anchor_private_source",
                "increment_private_source",
                "public",
            }:
                raise ValueError("Unknown replay stratum")
            if np.any((self.families == "public") != (self.strata == "public")):
                raise ValueError("Public source and replay stratum must agree")
            self.stratum_counts = {
                name: 0
                for name in (
                    "anchor_private_source",
                    "increment_private_source",
                    "public",
                )
            }

    def update(self, rows):
        self.seen[rows] = True
        for name in self.counts:
            self.counts[name] += int(np.count_nonzero(self.families[rows] == name))
        if self.strata is not None:
            for name in self.stratum_counts:
                self.stratum_counts[name] += int(
                    np.count_nonzero(self.strata[rows] == name)
                )

    def summary(self):
        result = {
            "draw_counts": self.counts.copy(),
            "sampled_unique_windows": {
                name: int(np.count_nonzero(self.seen & (self.families == name)))
                for name in self.counts
            },
        }
        if self.strata is not None:
            result["replay_stratum_draw_counts"] = self.stratum_counts.copy()
            result["sampled_unique_replay_stratum_windows"] = {
                name: int(np.count_nonzero(self.seen & (self.strata == name)))
                for name in self.stratum_counts
            }
        return result
