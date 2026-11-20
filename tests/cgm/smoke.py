"""Synthetic-only inference tests; does not train a representation or use real records."""

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from lingsante_pulse.cgm.inference.pipeline import (
    extract,
    daily_representation,
    ROOT,
    validate_input,
)


def synthetic():
    # Two independent synthetic subjects, three contiguous days each.
    t = np.arange(288, dtype=np.float32)
    glucose = np.stack(
        [
            100
            + 12 * s
            + 8 * np.sin(2 * np.pi * t / 288)
            + (35 + 3 * d) * np.exp(-(((t - 110 - 4 * d) / 17) ** 2))
            for s in range(2)
            for d in range(3)
        ]
    ).astype(np.float32)
    mask = np.ones_like(glucose, dtype=bool)
    mask[3:] = False
    mask[3:, ::3] = True
    return dict(
        glucose_mg_dl=glucose,
        observed_mask=mask,
        subject_id=np.array(["synthetic-a"] * 3 + ["synthetic-b"] * 3),
        window_id=np.array(["synthetic-" + str(i) for i in range(6)]),
        start_time=np.array(
            ["2020-01-01T00:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00"] * 2
        ),
        episode_index=np.zeros(6, dtype=np.int32),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-example", type=Path)
    a = p.parse_args()
    data = synthetic()
    if a.write_example:
        if a.write_example.exists():
            raise FileExistsError(a.write_example)
        a.write_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.write_example, **data)
    z, history = extract(data)
    assert z.shape == (6, 146) and np.isfinite(z).all()
    assert history.max() > 1
    labels = dict(data, label_demo=np.arange(6) % 2)
    other, _ = extract(labels)
    np.testing.assert_array_equal(z, other)
    # Input ordering does not define temporal order.
    order = np.array([5, 1, 3, 0, 4, 2])
    inverse = np.argsort(order)
    shuffled = {k: v[order] for k, v in data.items()}
    other, _ = extract(shuffled)
    np.testing.assert_allclose(z, other[inverse], atol=2e-5, rtol=2e-5)
    # A subject's representation is independent of other subjects in the batch.
    other, _ = extract({k: v[:3] for k, v in data.items()})
    np.testing.assert_allclose(z[:3], other, atol=2e-5, rtol=2e-5)
    single = {k: v[:1] for k, v in data.items()}
    short, h = extract(single)
    daily = daily_representation(single, ROOT / "checkpoints")
    np.testing.assert_array_equal(short, daily)
    invalid = dict(data, window_id=np.array(["duplicate"] * 6))
    try:
        validate_input(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate windows accepted")
    print(
        json.dumps(
            {
                "synthetic_windows": 6,
                "shape": list(z.shape),
                "max_history": int(history.max()),
                "finite": True,
                "label_invariance": True,
                "row_order_invariance": True,
                "cross_subject_isolation": True,
                "no_history_daily_fallback": True,
                "invalid_input_rejected": True,
            }
        )
    )


if __name__ == "__main__":
    main()
