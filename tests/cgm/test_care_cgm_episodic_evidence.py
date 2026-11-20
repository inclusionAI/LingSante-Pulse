import unittest
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lingsante_pulse.cgm.models.episodic_evidence import (
    RandomRowRepartitionEpisodes,
    SubjectEpisodes,
    episodic_loss,
    ridge_predict,
)


class EpisodicEvidenceTests(unittest.TestCase):
    def test_encoder_receives_outer_gradient_without_training_head(self):
        from lingsante_pulse.cgm.models.observation_supported_attention import (
            ObservationSupportedAttention,
        )
        from tests.cgm.test_care_cgm_longitudinal_statistics_tokens import example

        torch.manual_seed(43)
        inputs = [example([0, 1, -1]) for _ in range(12)]
        packed = {
            k: torch.cat([a[k] for a in inputs], 0) for k in inputs[0] if k != "events"
        }
        model = ObservationSupportedAttention()
        z, _ = model(**packed)
        loss = episodic_loss(z[:6], torch.randn(6, 336), z[6:], torch.randn(6, 336))
        loss.backward()
        self.assertTrue(
            all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
        )
        self.assertGreater(float(model.score[-1].weight.grad.norm()), 0.0)

    def test_independent_ridge_equivalence(self):
        rng = np.random.default_rng(43)
        x = rng.normal(size=(16, 23))
        q = rng.normal(size=(7, 23))
        y = rng.normal(size=(16, 8))
        x[:, 0] = 3.0
        q[:, 0] = 3.0
        actual = ridge_predict(
            torch.tensor(x), torch.tensor(y), torch.tensor(q)
        ).numpy()
        expected = (
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x, y).predict(q)
        )
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all())
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_no_query_fit_or_cross_query_dependency(self):
        torch.manual_seed(43)
        x, y, q = torch.randn(12, 9), torch.randn(12, 336), torch.randn(4, 9)
        before = ridge_predict(x, y, q)
        q[1:] *= 1000
        self.assertTrue(torch.equal(before[:1], ridge_predict(x, y, q)[:1]))
        # Query targets are absent from the prediction interface.
        self.assertTrue(torch.equal(before[:1], ridge_predict(x, y, q[:1])))

    def test_both_sides_gradient_and_constant_columns(self):
        torch.manual_seed(43)
        x = torch.randn(12, 9)
        x[:, 0] = 2.0
        x.requires_grad_()
        q = torch.randn(5, 9)
        q[:, 0] = 2.0
        q.requires_grad_()
        loss = episodic_loss(x, torch.randn(12, 336), q, torch.randn(5, 336))
        loss.backward()
        for g in (x.grad, q.grad):
            self.assertTrue(bool(torch.isfinite(g).all()))
            self.assertGreater(float(g.norm()), 0.0)

    def test_stop_through_fit_preserves_forward_and_only_query_gradient(self):
        torch.manual_seed(43)
        support = torch.randn(12, 9, requires_grad=True)
        query = torch.randn(5, 9, requires_grad=True)
        support_targets = torch.randn(12, 336)
        query_targets = torch.randn(5, 336)
        full = episodic_loss(support, support_targets, query, query_targets)
        stopped = episodic_loss(
            support,
            support_targets,
            query,
            query_targets,
            stop_through_fit=True,
        )
        self.assertTrue(torch.equal(full.detach(), stopped.detach()))
        stopped.backward()
        self.assertIsNone(support.grad)
        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.norm()), 0.0)

    def test_subject_disjoint_reproducible_no_audit(self):
        subjects = np.repeat(np.arange(10), 3)
        fit = np.arange(24)
        a, b = SubjectEpisodes(subjects, fit), SubjectEpisodes(subjects, fit)
        for _ in range(20):
            s, q = a.next()
            s2, q2 = b.next()
            np.testing.assert_array_equal(s, s2)
            np.testing.assert_array_equal(q, q2)
            self.assertEqual((len(s), len(q)), (64, 64))
            self.assertFalse(set(subjects[s]) & set(subjects[q]))
            self.assertTrue(set(s) | set(q) <= set(fit))
        with self.assertRaises(ValueError):
            SubjectEpisodes(subjects, np.arange(6))

    def test_random_row_repartition_preserves_matched_exposure(self):
        subjects = np.repeat(np.arange(20), 4)
        families = np.where(subjects < 16, "private", "public")
        fit = np.arange(len(subjects))
        reference = SubjectEpisodes(subjects, fit)
        control = RandomRowRepartitionEpisodes(
            SubjectEpisodes(subjects, fit), subjects, families
        )
        for _ in range(20):
            reference_rows = reference.next()
            support, query = control.next()
            self.assertFalse(set(support).intersection(query))
            self.assertTrue(set(subjects[support]).intersection(subjects[query]))
            np.testing.assert_array_equal(
                np.sort(np.concatenate([support, query])),
                np.sort(np.concatenate(reference_rows)),
            )
            for family in ("private", "public"):
                self.assertEqual(
                    np.count_nonzero(families[support] == family),
                    np.count_nonzero(families[reference_rows[0]] == family),
                )
                self.assertEqual(
                    np.count_nonzero(families[query] == family),
                    np.count_nonzero(families[reference_rows[1]] == family),
                )
        report = control.summary()
        self.assertEqual(report["row_overlap_rate"], 0.0)
        self.assertGreater(report["mean_identity_overlap"], 0.0)

    def test_random_row_repartition_handles_unequal_row_multiplicities(self):
        class FixedEpisode:
            def __init__(self):
                self.subjects = np.arange(8)

            def next(self):
                return (
                    np.asarray([0, 0, 0, 1, 2, 2, 3, 3]),
                    np.asarray([4, 4, 5, 5, 5, 6, 6, 7]),
                )

        subjects = np.asarray([0, 0, 1, 1, 0, 2, 2, 3])
        control = RandomRowRepartitionEpisodes(FixedEpisode(), subjects, seed=43)
        support, query = control.next()
        self.assertEqual((len(support), len(query)), (8, 8))
        self.assertFalse(set(support).intersection(query))
        self.assertTrue(set(subjects[support]).intersection(subjects[query]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([support, query])),
            np.sort(np.asarray([0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7])),
        )

    def test_random_row_repartition_allows_an_individually_unavoidable_zero_overlap(
        self,
    ):
        class FixedEpisode:
            subjects = np.arange(8)

            def next(self):
                return np.arange(4), np.arange(4, 8)

        subjects = np.arange(8)
        control = RandomRowRepartitionEpisodes(FixedEpisode(), subjects, seed=43)
        support, query = control.next()
        self.assertFalse(set(support).intersection(query))
        self.assertFalse(set(subjects[support]).intersection(subjects[query]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([support, query])), np.arange(8)
        )
        self.assertEqual(control.summary()["identity_overlap_episode_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
