#!/usr/bin/env python3
"""Run GlucoFM's 10x5 subject-disjoint linear probe on frozen embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lingsante_pulse.cgm.data.protocol import (
    repeated_linear_probe,
    repeated_subject_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as archive:
        features = archive["embedding"].astype(np.float64)
        subjects = archive["subject_id"].astype(str)
        labels = {
            name.removeprefix("label_"): archive[name]
            for name in archive.files
            if name.startswith("label_")
        }
    tasks = {}
    fold_fingerprints = {}
    for task, target in sorted(labels.items()):
        fold_hasher = hashlib.sha256()
        for repeat, fold, _, test in repeated_subject_folds(
            subjects, target, repeats=args.repeats, folds=args.folds, seed=args.seed
        ):
            test_subjects = sorted(np.unique(subjects[test]).tolist())
            fold_hasher.update(f"{repeat}|{fold}|{','.join(test_subjects)}\n".encode())
        fold_fingerprints[task] = fold_hasher.hexdigest()
        scores, _ = repeated_linear_probe(
            features,
            target,
            subjects,
            repeats=args.repeats,
            folds=args.folds,
            seed=args.seed,
        )
        tasks[task] = {
            **{
                metric: {
                    "mean": float(np.mean([row[metric] for row in scores])),
                    "sample_std": float(
                        np.std([row[metric] for row in scores], ddof=1)
                    ),
                }
                for metric in ("pr_auc", "roc_auc", "macro_f1")
            },
            "fold_scores": scores,
        }
    report = {
        "acceptance_passed": bool(tasks),
        "cohort": args.cohort,
        "embedding_artifact": str(args.npz),
        "implementation": "GlucoFM-like paper reimplementation; not official Google code",
        "classifier": "StandardScaler + LogisticRegression(L2, lbfgs, max_iter=1000)",
        "fold_assignment": "deterministic stratified subject folds; Google seeds unavailable",
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "fold_fingerprints": fold_fingerprints,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
