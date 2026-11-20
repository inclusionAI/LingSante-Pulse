import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


from lingsante_pulse.cgm.data.protocol import (
    align_nonoverlapping_windows,
    align_strictly_causal_window,
    ppgr_endpoints,
    recover_linearly_interpolated_native_readings,
    repeated_subject_folds,
    sample_observation_fraction,
    sample_support_subjects,
)


class PPGRProtocolTest(unittest.TestCase):
    def test_strict_causal_window_excludes_onset_and_future(self):
        onset = pd.Timestamp("2024-01-02 00:00:00")
        readings = pd.DataFrame(
            {
                "time": [
                    onset - pd.Timedelta(minutes=5),
                    onset,
                    onset + pd.Timedelta(minutes=5),
                ],
                "gl": [100.0, 200.0, 300.0],
            }
        )
        glucose, mask = align_strictly_causal_window(readings, onset)
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(float(glucose[-1]), 100.0)

    def test_endpoints_use_positive_iauc_and_first_argmax(self):
        observed = np.asarray([[1.0, 3.0, 3.0, -1.0]])
        predicted = np.asarray([[1.0, 1.0, 4.0, -2.0]])
        metrics = ppgr_endpoints(observed, predicted, np.asarray([5, 10, 15, 20]))
        self.assertEqual(metrics["trajectory_mae_mg_dl"], 1.0)
        self.assertEqual(metrics["peak_rise_mae_mg_dl"], 1.0)
        self.assertEqual(metrics["peak_time_mae_minutes"], 5.0)


class WindowAlignmentTest(unittest.TestCase):
    def test_floor_bins_average_collisions_and_preserve_missing_mask(self):
        start = pd.Timestamp("2026-01-01 06:02:00")
        times = [start + pd.Timedelta(minutes=5 * i) for i in range(288)]
        frame = pd.DataFrame({"id": "a", "time": times, "gl": np.arange(288.0)})
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "id": ["a"],
                        "time": [start + pd.Timedelta(minutes=1)],
                        "gl": [2.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        windows = align_nonoverlapping_windows(
            frame, subject_col="id", timestamp_col="time", value_col="gl"
        )
        self.assertEqual(windows.glucose.shape, (1, 288))
        self.assertEqual(float(windows.glucose[0, 0]), 1.0)
        self.assertEqual(int(windows.observed_mask[0].sum()), 288)
        self.assertEqual(int(windows.circadian_start[0]), 72)

    def test_gap_over_one_hour_starts_new_episode(self):
        first = pd.date_range("2026-01-01", periods=200, freq="5min")
        second = pd.date_range(
            first[-1] + pd.Timedelta(minutes=65), periods=288, freq="5min"
        )
        frame = pd.DataFrame({"id": "a", "time": first.append(second), "gl": 100.0})
        windows = align_nonoverlapping_windows(
            frame, subject_col="id", timestamp_col="time", value_col="gl"
        )
        self.assertEqual(len(windows.glucose), 1)
        self.assertEqual(int(windows.episode_index[0]), 1)


class SubjectFoldTest(unittest.TestCase):
    def test_subjects_are_disjoint_and_each_window_is_oof_once_per_repeat(self):
        groups = np.repeat([f"s{i}" for i in range(10)], 2)
        labels = np.repeat([0, 1] * 5, 2)
        seen = {repeat: np.zeros(len(groups), dtype=int) for repeat in range(2)}
        for repeat, _, train, test in repeated_subject_folds(
            groups, labels, repeats=2, folds=5, seed=7
        ):
            self.assertFalse(set(groups[train]).intersection(groups[test]))
            seen[repeat][test] += 1
        for counts in seen.values():
            np.testing.assert_array_equal(counts, np.ones(len(groups), dtype=int))


class NativeCadenceRecoveryTest(unittest.TestCase):
    def test_recovers_exact_phase_from_linear_interpolation(self):
        times = pd.date_range("2026-01-01 00:02", periods=31, freq="1min")
        native = np.asarray([100.0, 110.0, 90.0, 95.0, 105.0, 100.0, 120.0])
        values = np.interp(np.arange(31), np.arange(0, 31, 5), native)
        recovered, audit = recover_linearly_interpolated_native_readings(
            pd.Series(times), pd.Series(values), 5
        )
        np.testing.assert_allclose(recovered["gl"], native)
        self.assertAlmostEqual(audit["reconstruction_mae_mg_dl"], 0.0)
        self.assertEqual(
            audit["phase_offset"], int((times[0].value // 60_000_000_000) % 5)
        )

    def test_datetime_microsecond_storage_keeps_the_same_phase(self):
        times = np.arange(
            np.datetime64("2026-01-01T00:02", "us"),
            np.datetime64("2026-01-01T00:33", "us"),
            np.timedelta64(1, "m"),
        )
        native = np.asarray([100.0, 110.0, 90.0, 95.0, 105.0, 100.0, 120.0])
        values = np.interp(np.arange(31), np.arange(0, 31, 5), native)
        recovered, audit = recover_linearly_interpolated_native_readings(
            pd.Series(times), pd.Series(values), 5
        )
        np.testing.assert_allclose(recovered["gl"], native)
        self.assertEqual(audit["recovered_points"], 7)


class FewShotSamplingTest(unittest.TestCase):
    def test_support_sampling_uses_exact_subject_count_and_all_subject_windows(self):
        groups = np.repeat([f"s{i}" for i in range(8)], 3)
        labels = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3)
        train = np.arange(len(groups))
        chosen = sample_support_subjects(
            train, groups, labels, per_class=2, rng=np.random.default_rng(1)
        )
        self.assertEqual(np.unique(groups[chosen]).size, 4)
        self.assertEqual(len(chosen), 12)

    def test_fraction_sampling_keeps_at_least_one_window_per_subject(self):
        groups = np.repeat(["a", "b"], [3, 7])
        chosen = sample_observation_fraction(
            np.arange(10), groups, fraction=0.01, rng=np.random.default_rng(1)
        )
        self.assertEqual(set(groups[chosen]), {"a", "b"})
        self.assertEqual(len(chosen), 2)


if __name__ == "__main__":
    unittest.main()
