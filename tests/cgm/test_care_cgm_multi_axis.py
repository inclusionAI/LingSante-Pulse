import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).parents[1] / "scripts"


from lingsante_pulse.cgm.models.dynamics import (
    CAREMultiAxisSidecar,
    DYNAMICS_TARGET_NAMES,
    STATE_TARGET_NAMES,
    physical_state_targets,
    signed_dynamics_targets,
    responsible_dynamics_coordinates,
)
from lingsante_pulse.cgm.training.dynamics import (
    checkpoint_evaluation_steps,
    corpus_audit_counts,
    deterministic_checkpoint_selection_rows,
    hierarchical_balance_probabilities,
    signed_excursion_buckets,
    variance_covariance_terms,
)


class CARECGMMultiAxisTest(unittest.TestCase):
    def test_checkpoint_evaluation_schedule_includes_first_and_last(self):
        self.assertEqual(
            checkpoint_evaluation_steps(7_972, 512),
            (1, *range(512, 7_973, 512), 7_972),
        )
        self.assertEqual(len(checkpoint_evaluation_steps(31_887, 512)), 64)
        self.assertEqual(len(checkpoint_evaluation_steps(7_972, 20)), 400)
        with self.assertRaises(ValueError):
            checkpoint_evaluation_steps(100, 0)

    def test_checkpoint_selection_keeps_public_and_caps_private_by_cadence(self):
        validation_rows = np.arange(24)
        source = np.asarray(["public"] * 4 + ["own_private_source"] * 20)
        cadence = np.asarray([5, 5, 15, 15] + [3] * 10 + [5] * 6 + [15] * 4)
        selected = deterministic_checkpoint_selection_rows(
            validation_rows,
            source,
            cadence,
            max_private_windows=10,
            seed=60,
        )
        repeated = deterministic_checkpoint_selection_rows(
            validation_rows,
            source,
            cadence,
            max_private_windows=10,
            seed=60,
        )
        np.testing.assert_array_equal(selected, repeated)
        self.assertEqual(len(selected), 14)
        self.assertTrue(set(range(4)) <= set(selected))
        private = selected[source[selected] != "public"]
        self.assertEqual(len(private), 10)
        self.assertEqual(set(np.unique(cadence[private])), {3, 5, 15})

    def test_checkpoint_selection_is_identity_when_private_count_is_below_cap(self):
        rows = np.asarray([1, 2, 3, 4])
        source = np.asarray(
            [
                "public",
                "public",
                "own_private_source",
                "own_private_source",
                "own_private_source",
            ]
        )
        cadence = np.asarray([5, 5, 3, 5, 15])
        selected = deterministic_checkpoint_selection_rows(
            rows, source, cadence, max_private_windows=8, seed=60
        )
        np.testing.assert_array_equal(selected, rows)

    def test_corpus_audit_counts_use_artifact_metadata_not_stale_template(self):
        cohort = np.asarray(
            [
                "public",
                "private_source_private",
                "private_source_private",
                "public",
                "private_source_private",
            ]
        )
        counts = corpus_audit_counts(
            cohort, {"private_source_rows": 6138, "public_rows": 2283}, len(cohort)
        )
        self.assertEqual(counts["private_source_rows"], 3)
        self.assertEqual(counts["public_rows"], 2)
        self.assertTrue(counts["derived_from_artifact"])
        self.assertFalse(counts["configuration_matches_artifact"])

    @staticmethod
    def physical_trace() -> tuple[np.ndarray, np.ndarray]:
        time = np.linspace(0.0, 4.0 * np.pi, 288)
        values = np.stack([110.0 + 18.0 * np.sin(time) + np.linspace(-8.0, 14.0, 288)])
        observed = np.zeros_like(values, dtype=bool)
        observed[:, ::3] = True
        return values.astype(np.float32), observed

    def test_targets_obey_declared_time_reversal_parity(self):
        values, observed = self.physical_trace()
        state = physical_state_targets(values, observed)
        dynamics = signed_dynamics_targets(values, observed, glucose_scale=25.0)
        reverse_state = physical_state_targets(values[:, ::-1], observed[:, ::-1])
        reverse_dynamics = signed_dynamics_targets(
            values[:, ::-1], observed[:, ::-1], glucose_scale=25.0
        )
        self.assertEqual(state.shape, (1, len(STATE_TARGET_NAMES)))
        self.assertEqual(dynamics.shape, (1, len(DYNAMICS_TARGET_NAMES)))
        np.testing.assert_allclose(state, reverse_state, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dynamics, -reverse_dynamics, rtol=1e-5, atol=1e-5)

    def test_sparse_windows_mask_only_unsupported_long_lags(self):
        values = np.zeros((1, 288), dtype=np.float32)
        observed = np.zeros_like(values, dtype=bool)
        observed[0, :30] = True
        values[0, :30] = np.linspace(90.0, 150.0, 30)
        state, state_support = physical_state_targets(
            values, observed, return_support=True
        )
        dynamics, dynamics_support = signed_dynamics_targets(
            values, observed, glucose_scale=25.0, return_support=True
        )
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.isfinite(dynamics).all())
        self.assertTrue(state_support[0, :8].all())
        self.assertTrue(dynamics_support[0, :6].all())
        self.assertFalse(dynamics_support[0, -2:].any())

    def test_interval_adaptive_targets_use_irregular_pairs_and_keep_parity(self):
        time = np.linspace(0.0, 6.0 * np.pi, 288)
        values = (115.0 + 20.0 * np.sin(time))[None, :].astype(np.float32)
        observed = np.zeros_like(values, dtype=bool)
        observed[:, ::4] = True
        _, exact_support = signed_dynamics_targets(
            values,
            observed,
            glucose_scale=25.0,
            pair_mode="exact",
            return_support=True,
        )
        state, state_support = physical_state_targets(
            values,
            observed,
            pair_mode="interval_adaptive",
            return_support=True,
        )
        dynamics, dynamics_support = signed_dynamics_targets(
            values,
            observed,
            glucose_scale=25.0,
            pair_mode="interval_adaptive",
            return_support=True,
        )
        reverse_state = physical_state_targets(
            values[:, ::-1],
            observed[:, ::-1],
            pair_mode="interval_adaptive",
        )
        reverse_dynamics = signed_dynamics_targets(
            values[:, ::-1],
            observed[:, ::-1],
            glucose_scale=25.0,
            pair_mode="interval_adaptive",
        )
        self.assertFalse(exact_support[0, :2].any())
        self.assertTrue(state_support.all())
        self.assertTrue(dynamics_support.all())
        np.testing.assert_allclose(state, reverse_state, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dynamics, -reverse_dynamics, rtol=1e-5, atol=1e-5)

    def test_hybrid_keeps_exact_short_scales_and_adapts_only_long_scales(self):
        time = np.linspace(0.0, 6.0 * np.pi, 288)
        values = (115.0 + 20.0 * np.sin(time))[None, :].astype(np.float32)
        observed = np.zeros_like(values, dtype=bool)
        observed[:, ::4] = True
        exact, exact_support = signed_dynamics_targets(
            values, observed, glucose_scale=25.0, pair_mode="exact", return_support=True
        )
        hybrid, hybrid_support = signed_dynamics_targets(
            values,
            observed,
            glucose_scale=25.0,
            pair_mode="exact_short_adaptive_long",
            return_support=True,
        )
        reverse = signed_dynamics_targets(
            values[:, ::-1],
            observed[:, ::-1],
            glucose_scale=25.0,
            pair_mode="exact_short_adaptive_long",
        )
        np.testing.assert_allclose(hybrid[:, :4], exact[:, :4])
        np.testing.assert_array_equal(hybrid_support[:, :4], exact_support[:, :4])
        self.assertTrue(hybrid_support[:, 4:].all())
        np.testing.assert_allclose(hybrid, -reverse, rtol=1e-5, atol=1e-5)

    def test_exported_codes_have_exact_parity_and_independent_gradients(self):
        torch.manual_seed(43)
        model = CAREMultiAxisSidecar()
        physical = torch.randn(4, 288)
        relative = torch.randn(4, 288)
        observed = torch.ones(4, 288, dtype=torch.bool)
        result = model(physical, relative, observed, observed)
        reverse = model(
            torch.flip(physical, dims=(1,)),
            torch.flip(relative, dims=(1,)),
            torch.flip(observed, dims=(1,)),
            torch.flip(observed, dims=(1,)),
        )
        torch.testing.assert_close(result["state"], reverse["state"])
        torch.testing.assert_close(result["dynamics"], -reverse["dynamics"])
        torch.testing.assert_close(
            result["dynamics_readout"], -reverse["dynamics_readout"]
        )
        responsible = responsible_dynamics_coordinates(
            result["dynamics"], model.dynamics_head.weight
        )
        reverse_responsible = responsible_dynamics_coordinates(
            reverse["dynamics"], model.dynamics_head.weight
        )
        torch.testing.assert_close(responsible, -reverse_responsible)
        self.assertEqual(responsible.shape, result["dynamics"].shape)
        self.assertEqual(result["state"].shape, (4, 16))
        self.assertEqual(result["dynamics"].shape, (4, 16))
        self.assertEqual(result["dynamics_readout"].shape, (4, 10))
        loss = (
            result["state_prediction"].square().mean()
            + result["dynamics_prediction"].square().mean()
        )
        loss.backward()
        state_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.state_encoder.parameters()
            if parameter.grad is not None
        )
        dynamics_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.dynamics_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(state_gradient, 0.0)
        self.assertGreater(dynamics_gradient, 0.0)

    def test_dispersion_penalty_detects_collapse_and_correlation(self):
        collapsed = torch.ones(32, 4)
        collapsed_variance, collapsed_covariance = variance_covariance_terms(collapsed)
        self.assertGreater(float(collapsed_variance), 0.9)
        self.assertEqual(float(collapsed_covariance), 0.0)

        axis = torch.linspace(-2.0, 2.0, 32)
        correlated = torch.stack([axis, axis, -axis, -axis], dim=1)
        _, correlated_covariance = variance_covariance_terms(correlated)
        self.assertGreater(float(correlated_covariance), 0.1)

    def test_hierarchical_sampler_balances_source_cadence_and_excursion(self):
        source = np.asarray(["private_source"] * 8 + ["public"] * 6)
        cadence = np.asarray([3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 15, 15, 15])
        excursion = np.asarray([-1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 1, -1, 0, 1])
        probability = hierarchical_balance_probabilities(source, cadence, excursion)
        self.assertAlmostEqual(
            float(probability[source == "private_source"].sum()), 0.5
        )
        self.assertAlmostEqual(float(probability[source == "public"].sum()), 0.5)
        for source_value in ("private_source", "public"):
            rows = source == source_value
            cadence_values = np.unique(cadence[rows])
            for cadence_value in cadence_values:
                expected = 0.5 / len(cadence_values)
                selected = rows & (cadence == cadence_value)
                self.assertAlmostEqual(float(probability[selected].sum()), expected)
                buckets = np.unique(excursion[selected])
                for bucket in buckets:
                    bucket_rows = selected & (excursion == bucket)
                    self.assertAlmostEqual(
                        float(probability[bucket_rows].sum()), expected / len(buckets)
                    )

    def test_signed_excursion_threshold_is_fit_on_train_only(self):
        dynamics = np.zeros((12, len(DYNAMICS_TARGET_NAMES)), dtype=np.float32)
        dynamics[:10, 1] = np.arange(-5.0, 5.0)
        dynamics[10:, 1] = np.asarray([-1000.0, 1000.0])
        train_rows = np.arange(10)
        buckets, threshold = signed_excursion_buckets(
            dynamics, train_rows, tail_quantile=0.9
        )
        self.assertLess(threshold, 10.0)
        self.assertEqual(int(buckets[10]), -1)
        self.assertEqual(int(buckets[11]), 1)


if __name__ == "__main__":
    unittest.main()
