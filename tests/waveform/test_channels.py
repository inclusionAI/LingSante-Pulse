import numpy as np
import pytest
import torch

from lingsante_pulse.waveform.channels import resolve_metadata


@pytest.mark.parametrize("lead", ["II", "ii", "2", 2])
def test_ecg_lead_aliases_are_contextual(lead):
    assert resolve_metadata(channels=[{"modality": "ECG", "lead": lead}]) == ["ecg_ii"]


@pytest.mark.parametrize("lead", [None, "", "  "])
def test_unspecified_lead_is_generic_ecg(lead):
    assert resolve_metadata(
        channels=[{"modality": "electrocardiogram", "lead": lead}]
    ) == ["ecg"]


def test_ecg_unknown_lead_is_not_guessed():
    with pytest.raises(ValueError, match="Unknown ECG lead"):
        resolve_metadata(channels=[{"modality": "ECG", "lead": "V7_typo"}])


def test_generic_eeg_eog_do_not_imply_montage():
    for name in ["EEG", "EOG"]:
        assert resolve_metadata(channel_names=name) == [name.lower()]


def test_full_modality_name_and_case():
    for name in ["ECG", "ecg", " Electrocardiogram "]:
        assert resolve_metadata(channel_names=name) == ["ecg"]


def test_structured_channels_use_identical_cached_vectors(model):
    signal = np.sin(np.arange(128, dtype=np.float32) * 0.2)
    direct = model.encode(signal, channel_names="ecg_ii")
    structured = model.encode(signal, channels=[{"modality": "ECG", "lead": 2}])
    for key in ["cls", "patch_mean"]:
        torch.testing.assert_close(direct[key], structured[key], rtol=0, atol=0)


def test_unknown_modality_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown channels"):
        model.encode(np.arange(128, dtype=np.float32), channels=[{"modality": "IMU"}])


def test_dict_recording_labels_can_have_separate_metadata(model):
    signal = np.sin(np.arange(128, dtype=np.float32) * 0.2)
    result = model.encode(
        {"CH1": signal},
        sample_rate={"CH1": 128},
        channels=[{"modality": "ECG", "lead": "ii"}],
    )
    direct = model.encode(signal, channel_names="ecg_ii")
    torch.testing.assert_close(result["cls"], direct["cls"], rtol=0, atol=0)
