"""Synthetic tests only: no actual phenotype fits on local hardware."""

import unittest
from unittest.mock import patch

import numpy as np

import lingsante_pulse.cgm.evaluation.protocol as nested


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.people = np.repeat(np.arange(45).astype(str), 2)
        self.labels = {
            "binary": np.repeat(np.arange(45) % 2, 2),
            "three": np.repeat(np.arange(45) % 3, 2),
        }

    def test_common_folds_person_isolation_coverage_and_determinism(self):
        splits, attempt = nested.common_folds(self.people, self.labels, 5, 43)
        again, again_attempt = nested.common_folds(self.people, self.labels, 5, 43)
        self.assertEqual(attempt, again_attempt)
        seen = []
        for (train, test), (_, same) in zip(splits, again):
            self.assertFalse(set(self.people[train]) & set(self.people[test]))
            np.testing.assert_array_equal(test, same)
            seen.extend(test)
            for y in self.labels.values():
                np.testing.assert_array_equal(np.unique(y[test]), np.unique(y))
        self.assertEqual(sorted(seen), list(range(len(self.people))))

    def test_inner_rows_never_include_outer_people(self):
        outer, _ = nested.common_folds(self.people, self.labels, 5, 43)
        for train, test in outer:
            inner, _ = nested.common_folds(
                self.people[train],
                {k: y[train] for k, y in self.labels.items()},
                3,
                1043,
            )
            for a, b in inner:
                self.assertFalse(
                    set(self.people[train[a]]) & set(self.people[train[b]])
                )
                self.assertFalse(set(self.people[train[b]]) & set(self.people[test]))

    def test_impossible_class_support_fails(self):
        with self.assertRaises(ValueError):
            nested.common_folds(
                self.people, {"rare": np.repeat(np.arange(45) < 4, 2)}, 5, 43
            )

    def test_global_choice_equal_task_weight_and_tie(self):
        dev = {
            "cohort": {
                "x": [np.zeros((9, 1)), np.ones((9, 1))],
                "labels": {"a": np.arange(9) % 2, "b": np.arange(9) % 3},
                "inner": [(np.arange(6), np.arange(6, 9))],
            }
        }

        def fake_probe(x, y, xv, yv):
            score = 0.4 if x[0, 0] == 0 else 0.7
            return {m: score for m in nested.METRICS}, None, 0

        with patch.object(nested, "probe", side_effect=fake_probe):
            winner, scores, _ = nested.choose_candidate(dev, ("a", "b"))
        self.assertEqual(winner, 1)
        self.assertAlmostEqual(scores[1]["score"], 0.7)
        with patch.object(
            nested, "probe", return_value=({m: 0.5 for m in nested.METRICS}, None, 0)
        ):
            winner, _, _ = nested.choose_candidate(dev, ("a", "b"))
        self.assertEqual(winner, 0)

    def test_selector_receives_only_training_rows(self):
        train, test = np.arange(6), np.arange(6, 9)
        d = {
            "x": [np.arange(9)[:, None] for _ in nested.CANDIDATES],
            "labels": {"a": np.arange(9) % 2},
            "window_id": np.arange(9).astype(str),
        }
        episode = {
            "repeat": 0,
            "fold": 0,
            "cohorts": {
                "mock": {
                    "train": train.tolist(),
                    "test": test.tolist(),
                    "inner": [{"train": [0, 1, 2], "validation": [3, 4, 5]}],
                }
            },
        }

        def selector(dev, candidates):
            np.testing.assert_array_equal(dev["mock"]["x"][0][:, 0], train)
            np.testing.assert_array_equal(
                dev["mock"]["labels"]["a"], d["labels"]["a"][train]
            )
            return 0, [], 0

        with (
            patch.object(nested, "DATA", {"mock": d}),
            patch.object(nested, "choose_candidate", side_effect=selector),
            patch.object(
                nested,
                "probe",
                return_value=({m: 0.5 for m in nested.METRICS}, np.ones((3, 2)) / 2, 0),
            ),
        ):
            result = nested.evaluate_episode(episode)
        self.assertEqual(result["selected_candidate"], nested.CANDIDATES[0])


if __name__ == "__main__":
    unittest.main()
