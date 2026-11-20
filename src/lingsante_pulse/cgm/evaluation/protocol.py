#!/usr/bin/env python3
"""Main-table nested public14 evaluation from fixed candidate embeddings.

One representation candidate is selected jointly for all tasks in an outer
episode. No outer participant contributes any label to its inner selector.
The candidate is shared across all tasks in each outer episode.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings
import sys

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from lingsante_pulse.cgm.data.protocol import _classification_metrics

COHORTS = ("cgmacros", "shanghai_t2dm", "stanford", "hall")
CANDIDATES = tuple(
    f"{config}/replica{i}"
    for config in ("lr350_aux15", "lr400_aux175", "lr500_aux10")
    for i in range(1, 9)
)
REFERENCE = "lr400_aux175/replica5"
METRICS = ("pr_auc", "roc_auc", "macro_f1")
ROOT = Path("results/cgm/candidate_embeddings")
OUTPUT = ROOT.parent / "nested_evaluation"
DATA = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_data(root):
    data, artifacts = {}, []
    for cohort in COHORTS:
        features = []
        reference_meta = None
        for candidate in CANDIDATES:
            path = root / candidate / cohort / "embeddings.npz"
            with np.load(path, allow_pickle=False) as archive:
                x = archive["embedding"].astype(np.float64)
                meta = {k: archive[k] for k in archive.files if k != "embedding"}
            if reference_meta is None:
                reference_meta = meta
            if meta.keys() != reference_meta.keys():
                raise ValueError(f"metadata keys differ: {path}")
            for key in meta:
                left, right = meta[key], reference_meta[key]
                equal = (
                    np.array_equal(left, right, equal_nan=True)
                    if left.dtype.kind in "fc"
                    else np.array_equal(left, right)
                )
                if not equal:
                    raise ValueError(f"metadata mismatch {key}: {path}")
            if (
                x.ndim != 2
                or not np.isfinite(x).all()
                or len(x) != len(meta["subject_id"])
            ):
                raise ValueError(f"invalid features: {path}")
            features.append(x)
            artifacts.append(
                {
                    "candidate": candidate,
                    "cohort": cohort,
                    "path": str(path),
                    "sha256": sha256(path),
                }
            )
        visits = reference_meta["subject_id"].astype(str)
        # Identical linkage to build_glucofm_shanghai_t2dm_protocol.py.
        people = (
            np.array([v.split("_")[0] for v in visits])
            if cohort == "shanghai_t2dm"
            else visits
        )
        if cohort == "shanghai_t2dm" and (
            len(np.unique(visits)),
            len(np.unique(people)),
        ) != (65, 58):
            raise ValueError("unexpected Shanghai visit/person linkage")
        if "biological_subject_id" in reference_meta:
            if not np.array_equal(
                people, reference_meta["biological_subject_id"].astype(str)
            ):
                raise ValueError("biological subject linkage disagrees")
        labels = {
            k.removeprefix("label_"): v
            for k, v in reference_meta.items()
            if k.startswith("label_")
        }
        for y in labels.values():
            if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
                raise ValueError("expected contiguous nonnegative class labels")
            for visit in np.unique(visits):
                if len(np.unique(y[visits == visit])) != 1:
                    raise ValueError("inconsistent visit labels")
        data[cohort] = {
            "x": features,
            "labels": labels,
            "people": people,
            "visits": visits,
            "window_id": reference_meta["window_id"].astype(str),
        }
    if sum(len(d["labels"]) for d in data.values()) != 14:
        raise ValueError("not exactly 14 tasks")
    return data, artifacts


def common_folds(people, labels, n_splits, seed):
    """First label-valid balanced-size random group partition; no features/scores.

    All tasks share the group assignment. The rejection rule requires all classes
    on each side. This uses labels solely for stratification/coverage, not scores.
    """
    groups, inverse = np.unique(people, return_inverse=True)
    memberships = []
    for y in labels.values():
        for value in np.unique(y):
            present = np.zeros(len(groups), dtype=bool)
            present[inverse[y == value]] = True
            if present.sum() < n_splits:
                raise ValueError("too few biological people for class-covered folds")
            memberships.append(present)
    membership = np.stack(memberships, axis=1)
    rng = np.random.default_rng(seed)
    for attempt in range(100000):
        test_sets = np.array_split(rng.permutation(len(groups)), n_splits)
        if all(
            np.all(membership[test].sum(axis=0) > 0)
            and np.all(membership.sum(axis=0) - membership[test].sum(axis=0) > 0)
            for test in test_sets
        ):
            result = []
            for test in test_sets:
                mask = np.isin(inverse, test)
                result.append((np.flatnonzero(~mask), np.flatnonzero(mask)))
            return result, attempt
    raise ValueError(
        "no class-covered partition found; do not change based on outcomes"
    )


def make_plan(data, artifacts):
    episodes = [
        {"repeat": r, "fold": f, "cohorts": {}} for r in range(10) for f in range(5)
    ]
    counts = {}
    for c, cohort in enumerate(COHORTS):
        d = data[cohort]
        counts[cohort] = {
            "windows": len(d["people"]),
            "people": len(np.unique(d["people"])),
            "visits": len(np.unique(d["visits"])),
            "tasks": sorted(d["labels"]),
        }
        for repeat in range(10):
            outer, attempt = common_folds(
                d["people"], d["labels"], 5, 43 + repeat + 100 * c
            )
            for fold, (train, test) in enumerate(outer):
                inner, inner_attempt = common_folds(
                    d["people"][train],
                    {k: y[train] for k, y in d["labels"].items()},
                    3,
                    1043 + 1000 * c + 10 * repeat + fold,
                )
                episodes[repeat * 5 + fold]["cohorts"][cohort] = {
                    "train": train.tolist(),
                    "test": test.tolist(),
                    "test_people": sorted(np.unique(d["people"][test]).tolist()),
                    "outer_partition_attempt": attempt,
                    "inner_partition_attempt": inner_attempt,
                    "inner": [
                        {"train": train[a].tolist(), "validation": train[b].tolist()}
                        for a, b in inner
                    ],
                }
    return {
        "version": 1,
        "candidate_ids": list(CANDIDATES),
        "reference": REFERENCE,
        "score": "equal task mean of equal PR-AUC/ROC-AUC/Macro-F1 mean; first candidate wins exact ties",
        "classifier": "fold-local StandardScaler + L2 LogisticRegression C=1 lbfgs max_iter=1000 random_state=43",
        "outer": "10x5 common biological-person, class-covered random partitions",
        "inner": "3 folds within each outer training population, common across tasks",
        "counts": counts,
        "artifacts": artifacts,
        "episodes": episodes,
        "source_sha256": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).resolve().parents[1] / "data/protocol.py",
            )
        },
    }


def validate_plan(data, plan):
    assert plan["candidate_ids"] == list(CANDIDATES)
    for episode in plan["episodes"]:
        for cohort, split in episode["cohorts"].items():
            d = data[cohort]
            train, test = np.array(split["train"]), np.array(split["test"])
            assert sorted(np.r_[train, test].tolist()) == list(range(len(d["people"])))
            assert not set(d["people"][train]) & set(d["people"][test])
            assert split["test_people"] == sorted(np.unique(d["people"][test]).tolist())
            validation_rows = []
            for inner in split["inner"]:
                a, b = np.array(inner["train"]), np.array(inner["validation"])
                assert sorted(np.r_[a, b].tolist()) == sorted(train.tolist())
                assert not set(d["people"][a]) & set(d["people"][b])
                validation_rows.extend(b.tolist())
                for y in d["labels"].values():
                    assert np.array_equal(np.unique(y[a]), np.unique(y))
                    assert np.array_equal(np.unique(y[b]), np.unique(y))
            assert sorted(validation_rows) == sorted(train.tolist())
            for y in d["labels"].values():
                assert np.array_equal(np.unique(y[test]), np.unique(y))


def probe(x_train, y_train, x_test, y_test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=43
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    p = model.predict_proba(x_test)
    return (
        _classification_metrics(y_test, p),
        p,
        sum(issubclass(w.category, ConvergenceWarning) for w in caught),
    )


def choose_candidate(development, candidate_ids):
    """Only sees outer-training rows and locally indexed inner folds."""
    details, scores, convergence = [], [], 0
    for candidate_index, candidate in enumerate(candidate_ids):
        tasks = {}
        for cohort, d in development.items():
            x = d["x"][candidate_index]
            for task, y in sorted(d["labels"].items()):
                folds = []
                for train, val in d["inner"]:
                    metrics, _, count = probe(x[train], y[train], x[val], y[val])
                    convergence += count
                    folds.append(metrics)
                tasks[f"{cohort}/{task}"] = {
                    m: float(np.mean([f[m] for f in folds])) for m in METRICS
                }
        score = float(np.mean([v[m] for v in tasks.values() for m in METRICS]))
        details.append({"candidate": candidate, "score": score, "tasks": tasks})
        scores.append(score)
    return int(np.argmax(scores)), details, convergence


def evaluate_episode(episode):
    start = time.time()
    development = {}
    for cohort, split in episode["cohorts"].items():
        d = DATA[cohort]
        train = np.array(split["train"])
        local = {int(row): i for i, row in enumerate(train)}
        development[cohort] = {
            "x": [x[train] for x in d["x"]],
            "labels": {k: y[train] for k, y in d["labels"].items()},
            "inner": [
                (
                    np.array([local[i] for i in s["train"]]),
                    np.array([local[i] for i in s["validation"]]),
                )
                for s in split["inner"]
            ],
        }
    winner, inner_scores, warnings_count = choose_candidate(development, CANDIDATES)
    # Selection has finished. Only now fit outer probes and read outer labels.
    outer = {}
    for name, index in (
        ("nested_selected", winner),
        ("fixed_released_reference", CANDIDATES.index(REFERENCE)),
    ):
        tasks = {}
        for cohort, split in episode["cohorts"].items():
            d = DATA[cohort]
            train, test = np.array(split["train"]), np.array(split["test"])
            x = d["x"][index]
            for task, y in sorted(d["labels"].items()):
                metrics, p, count = probe(x[train], y[train], x[test], y[test])
                warnings_count += count
                tasks[f"{cohort}/{task}"] = {
                    "metrics": metrics,
                    "test_rows": test.tolist(),
                    "window_id": d["window_id"][test].tolist(),
                    "labels": y[test].tolist(),
                    "probabilities": p.tolist(),
                }
        outer[name] = tasks
    return {
        "repeat": episode["repeat"],
        "fold": episode["fold"],
        "selected_candidate": CANDIDATES[winner],
        "inner_scores": inner_scores,
        "outer": outer,
        "convergence_warnings": warnings_count,
        "elapsed_seconds": time.time() - start,
    }


def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        DATA, artifacts = load_data(args.root)
        plan = make_plan(DATA, artifacts)
        validate_plan(DATA, plan)
        plan_path = args.output / "plan.json"
        if args.mode == "preflight":
            save_new(plan_path, plan)
            print(
                json.dumps(
                    {
                        "preflight": "passed",
                        "artifacts": len(artifacts),
                        "counts": plan["counts"],
                        "plan_sha256": sha256(plan_path),
                    }
                ),
                flush=True,
            )
            return
        if plan != json.loads(plan_path.read_text()):
            raise ValueError("data/code/folds changed after preflight")
        save_new(
            args.output / "started.json",
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "plan_sha256": sha256(plan_path),
                "workers": args.workers,
                "unix_time": time.time(),
            },
        )
        start = time.time()
        results = []
        # Linux fork shares the read-only frozen feature matrices across workers.
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            jobs = {pool.submit(evaluate_episode, e): e for e in plan["episodes"]}
            for future in as_completed(jobs):
                result = future.result()
                save_new(
                    args.output
                    / f"outer_r{result['repeat']:02d}_f{result['fold']}.json",
                    result,
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            "done": len(results),
                            "total": 50,
                            "repeat": result["repeat"],
                            "fold": result["fold"],
                            "elapsed_seconds": time.time() - start,
                        }
                    ),
                    flush=True,
                )
        summary = {}
        for name in ("nested_selected", "fixed_released_reference"):
            tasks = {
                task: {
                    m: float(
                        np.mean([r["outer"][name][task]["metrics"][m] for r in results])
                    )
                    for m in METRICS
                }
                for task in results[0]["outer"][name]
            }
            summary[name] = {
                "tasks": tasks,
                "overall": {
                    m: float(np.mean([v[m] for v in tasks.values()])) for m in METRICS
                },
            }
        summary["selection_counts"] = {
            c: sum(r["selected_candidate"] == c for r in results) for c in CANDIDATES
        }
        summary["convergence_warnings"] = sum(
            r["convergence_warnings"] for r in results
        )
        summary["plan_sha256"] = sha256(plan_path)
        summary["elapsed_seconds"] = time.time() - start
        save_new(args.output / "summary.json", summary)
        print("NESTED_SELECTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
