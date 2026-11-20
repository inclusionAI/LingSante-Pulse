from pathlib import Path

import numpy as np
import pytest
import torch

from lingsante_pulse import LingSantePulse
from lingsante_pulse.waveform.preprocessing import prepare


def test_pinned_reference_features(model):
    """Golden output comes from the original pinned encoder, on synthetic data."""
    with np.load(Path(__file__).with_name("reference_features.npz")) as fixture:
        result = model.encode(fixture["signal"], channel_names="II")
        for key in ("cls", "patch_mean"):
            np.testing.assert_allclose(
                result[key].numpy(), fixture[key], rtol=1e-5, atol=1e-5
            )


def test_channels_are_independent_and_ordered(model):
    signals = np.random.default_rng(21).normal(size=(2, 2, 256)).astype(np.float32)
    together = model.encode(signals, channel_names=["I", "II"], batch_size=3)
    for channel, name in enumerate(["I", "II"]):
        separate = model.encode(signals[:, channel : channel + 1], channel_names=name)
        for key in ("cls", "patch_mean"):
            torch.testing.assert_close(
                together[key][:, channel * 256 : (channel + 1) * 256],
                separate[key],
                rtol=1e-5,
                atol=1e-5,
            )


def test_missing_and_storage_padding(model):
    signals = np.linspace(-1, 1, 130, dtype=np.float32)
    signals[32:64] = np.nan
    present = np.ones(130, bool)
    present[100:] = False
    result = model.encode(
        signals, channel_names="ecg_ii", sample_present=present, return_tokens=True
    )
    assert result["patch_tokens"].shape == (1, 5, 256)
    assert result["patch_valid"].tolist() == [
        [[True], [False], [True], [True], [False]]
    ]
    assert (result["patch_tokens"][:, [1, 4]] == 0).all()
    valid = result["patch_valid"][0, :, 0]
    torch.testing.assert_close(
        result["patch_tokens"][:, valid].mean(1), result["patch_mean"]
    )


def test_mixed_rate_dict_matches_separate_calls(model):
    rng = np.random.default_rng(7)
    ecg = rng.normal(size=(2, 256)).astype(np.float32)
    ppg = rng.normal(size=(2, 128)).astype(np.float32)
    result = model.encode({"ecg_ii": ecg, "ppg": ppg}, {"ecg_ii": 128, "ppg": 64})
    for key, values, hz, index in [("ecg_ii", ecg, 128, 0), ("ppg", ppg, 64, 1)]:
        single = model.encode(values[:, None], hz, [key])
        for readout in ("cls", "patch_mean"):
            torch.testing.assert_close(
                result[readout][:, index * 256 : (index + 1) * 256],
                single[readout],
                rtol=1e-5,
                atol=1e-5,
            )


def test_slow_channels_keep_physical_level():
    low = prepare(np.full(8, 90.0), 8, ["spo2"])[0]
    high = prepare(np.full(8, 100.0), 8, ["spo2"])[0]
    np.testing.assert_array_equal(low, -np.ones_like(low))
    np.testing.assert_array_equal(high, np.ones_like(high))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"channel_names": "invented"}, "Unknown channels"),
        ({"batch_size": 0}, "batch_size"),
        ({"channel_names": ["I", "II"]}, "one channel name"),
        ({"observed": np.zeros(256, bool)}, "No observed samples"),
    ],
)
def test_bad_metadata_fails_clearly(model, kwargs, match):
    options = {"channel_names": "ecg_ii", **kwargs}
    with pytest.raises(ValueError, match=match):
        model.encode(np.ones(256, np.float32), **options)


def test_dict_durations_are_checked_before_padding(model):
    with pytest.raises(ValueError, match="physical duration"):
        model.encode({"ecg_ii": np.ones(129), "ppg": np.ones(130)})


def test_loading_preserves_rng(model_dir):
    torch.manual_seed(113)
    before = torch.random.get_rng_state().clone()
    LingSantePulse.from_pretrained(model_dir, device="cpu", local_files_only=True)
    assert torch.equal(before, torch.random.get_rng_state())
