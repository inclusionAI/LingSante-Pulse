import unittest
import torch
from lingsante_pulse.cgm.models.longitudinal_statistics_tokens import (
    statistics_tokens,
    LongitudinalStatisticsEncoder,
)


def example(sequence):
    x = torch.zeros(1, 7, 4, 16)
    n = len(sequence)
    x[:, :n, :, :] = torch.tensor(sequence, dtype=torch.float32)[None, :, None, None]
    mask = torch.arange(7)[None] < n
    support = mask[..., None].expand(-1, -1, 4).float()
    return dict(
        values=torch.randn(1, 7, 146),
        features=x,
        support=support,
        elapsed_days=torch.arange(7).float()[None],
        mask=mask,
        events=torch.zeros(1, 7, 2),
    )


class StatisticsTokensTests(unittest.TestCase):
    def test_same_mean_different_variation(self):
        a, b = example([0, 1, -1]), example([0, 0, 0])
        ta, _ = statistics_tokens(**{k: v for k, v in a.items() if k != "values"})
        tb, _ = statistics_tokens(**{k: v for k, v in b.items() if k != "values"})
        self.assertTrue(torch.equal(ta[:, 1, :16], tb[:, 1, :16]))
        self.assertGreater(float(ta[:, 2, 0]), float(tb[:, 2, 0]) + 0.8)
        b["values"] = a["values"].clone()
        torch.manual_seed(43)
        model = LongitudinalStatisticsEncoder().eval()
        self.assertFalse(torch.allclose(model(**a), model(**b), atol=1e-6))
        model.mean_only = True
        self.assertTrue(torch.allclose(model(**a), model(**b), atol=1e-6))

    def test_event_count_equal_persistence_differs(self):
        a, b = example([0] * 4), example([0] * 4)
        a["events"][0, :4, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        b["events"][0, :4, 0] = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ta, _ = statistics_tokens(**{k: v for k, v in a.items() if k != "values"})
        tb, _ = statistics_tokens(**{k: v for k, v in b.items() if k != "values"})
        self.assertEqual(float(ta[0, 28, 0]), float(tb[0, 28, 0]))
        self.assertGreater(float(ta[0, 28, 2]), float(tb[0, 28, 2]))
        self.assertLess(float(ta[0, 28, 3]), float(tb[0, 28, 3]))
        b["values"] = a["values"].clone()
        torch.manual_seed(43)
        model = LongitudinalStatisticsEncoder().eval()
        self.assertFalse(torch.allclose(model(**a), model(**b), atol=1e-6))
        model.mean_only = True
        self.assertTrue(torch.allclose(model(**a), model(**b), atol=1e-6))

    def test_gap_is_not_an_observed_transition(self):
        a = example([1, 0])
        a["elapsed_days"][0, 1] = 2
        a["events"][0, 0, 0] = 1
        t, m = statistics_tokens(**{k: v for k, v in a.items() if k != "values"})
        self.assertFalse(bool(m[0, 6]))
        self.assertEqual(float(t[0, 28, 3]), 0.0)

    def test_padding_identity_and_subject_independence(self):
        torch.manual_seed(43)
        model = LongitudinalStatisticsEncoder().eval()
        a = example([1, 2, 3])
        reference = model(**a)
        polluted = {k: v.clone() for k, v in a.items()}
        for key in ("values", "features", "support", "elapsed_days", "events"):
            polluted[key][:, 3:] = float("nan")
        self.assertTrue(torch.allclose(reference, model(**polluted), atol=1e-6))
        single = example([5])
        self.assertTrue(torch.equal(model(**single), single["values"][:, 0]))
        batch = {k: torch.cat([v, example([8, 9, 10])[k]], 0) for k, v in a.items()}
        self.assertTrue(torch.allclose(reference, model(**batch)[:1], atol=1e-6))

    def test_all_parameters_receive_finite_gradients(self):
        model = LongitudinalStatisticsEncoder()
        z = model(**example([0, 1, -1]))
        z.square().mean().backward()
        self.assertEqual(z.shape, (1, 146))
        self.assertTrue(
            all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
        )
        self.assertGreater(float(model.stat_projection.weight.grad.norm()), 0.0)
        control = LongitudinalStatisticsEncoder(mean_only=True)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in control.parameters()),
        )


if __name__ == "__main__":
    unittest.main()
