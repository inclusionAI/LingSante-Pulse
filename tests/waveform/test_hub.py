import json
import socket

import pytest

from lingsante_pulse import LingSantePulse
from lingsante_pulse.waveform import hub


def _mock_snapshot(directory):
    directory.mkdir(parents=True)
    for filename in hub.MODEL_FILES:
        (directory / filename).touch()
    return directory


def test_repo_id_forwards_download_options(tmp_path, monkeypatch):
    directory = _mock_snapshot(tmp_path / "snapshot")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return directory

    monkeypatch.setattr(hub, "snapshot_download", fake_download)
    revision = "a" * 40
    result = hub.resolve_model_dir(
        "example-org/lingsante-pulse", revision=revision, local_files_only=True
    )
    assert result == directory
    assert calls == [
        {
            "repo_id": "example-org/lingsante-pulse",
            "revision": revision,
            "local_files_only": True,
            "allow_patterns": list(hub.DOWNLOAD_FILES),
        }
    ]


def test_local_directory_never_downloads(tmp_path, monkeypatch):
    directory = _mock_snapshot(tmp_path / "model")
    monkeypatch.setattr(
        hub, "snapshot_download", lambda **kwargs: pytest.fail("Unexpected download")
    )
    assert hub.resolve_model_dir(directory) == directory


def test_missing_local_directory_is_not_a_repo_id(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local model directory"):
        hub.resolve_model_dir(tmp_path / "missing")


def test_incomplete_model_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        hub.resolve_model_dir(tmp_path)


def test_cached_hub_snapshot_loads_without_network(
    model_dir, model, tmp_path, monkeypatch
):
    commit = "b" * 40
    snapshot = tmp_path / "models--example-org--lingsante-pulse" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for filename in hub.MODEL_FILES:
        (snapshot / filename).symlink_to(model_dir / filename)

    def no_network(*args, **kwargs):
        raise AssertionError("Network access attempted during cached loading")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    real_snapshot_download = hub.snapshot_download
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **kwargs: real_snapshot_download(cache_dir=tmp_path, **kwargs),
    )
    loaded = LingSantePulse.from_pretrained(
        "example-org/lingsante-pulse",
        revision=commit,
        local_files_only=True,
        device="cpu",
    )
    assert loaded.channel_names == model.channel_names


def test_modified_asset_is_rejected(model_dir, tmp_path):
    for filename in hub.MODEL_FILES:
        if filename != "channels.json":
            (tmp_path / filename).symlink_to(model_dir / filename)
    (tmp_path / "channels.json").write_text(json.dumps({"ecg": "changed"}))
    with pytest.raises(ValueError, match="checksum mismatch: channels.json"):
        LingSantePulse.from_pretrained(tmp_path, device="cpu", local_files_only=True)
