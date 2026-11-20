"""The default waveform model resolves the official Hub repository."""

from lingsante_pulse import LingSantePulse
from lingsante_pulse.waveform import hub


def test_default_model_resolves_official_repository(model_dir, monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(model_dir)

    monkeypatch.setattr(hub, "snapshot_download", download)
    model = LingSantePulse.from_pretrained(device="cpu")
    assert "ecg_ii" in model.channel_names
    assert calls == [
        {
            "repo_id": "inclusionAI/LingSante-Pulse-1.0",
            "revision": None,
            "local_files_only": False,
            "allow_patterns": list(hub.DOWNLOAD_FILES),
        }
    ]
