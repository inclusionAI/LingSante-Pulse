"""Resolve a model directory locally or through Hugging Face Hub."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_FILES = ("config.json", "model.safetensors", "channels.npz", "channels.json")
DOWNLOAD_FILES = MODEL_FILES + ("README.md", "LICENSE")


def resolve_model_dir(source, *, revision=None, local_files_only=False):
    """Download only inference assets; never download or execute repository code."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Provide a local model directory or a Hugging Face repo ID")
    path = Path(source).expanduser()
    if path.is_file():
        raise ValueError(
            "Provide the model directory, rather than a single checkpoint file"
        )
    if path.is_dir():
        directory = path
    elif (
        isinstance(source, Path)
        or path.is_absolute()
        or str(source).startswith((".", "~"))
    ):
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    else:
        snapshot = snapshot_download(
            repo_id=str(source),
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=list(DOWNLOAD_FILES),
        )
        directory = Path(snapshot)
    missing = [
        filename for filename in MODEL_FILES if not (directory / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete LingSante-Pulse model directory {directory}: missing {missing}"
        )
    return directory
