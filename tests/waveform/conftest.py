import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from lingsante_pulse import LingSantePulse


@pytest.fixture(scope="session")
def model_dir():
    source = os.environ.get("LINGSANTE_PULSE_MODEL_DIR")
    if source is None:
        config = try_to_load_from_cache(
            "inclusionAI/LingSante-Pulse-1.0", "config.json"
        )
        if not isinstance(config, str):
            pytest.skip(
                "Waveform reference tests need cached weights or LINGSANTE_PULSE_MODEL_DIR"
            )
        source = Path(config).parent
    directory = Path(source).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Waveform model directory does not exist: {directory}")
    return directory


@pytest.fixture(scope="session")
def model(model_dir):
    torch.set_num_threads(2)
    return LingSantePulse.from_pretrained(
        model_dir, device="cpu", local_files_only=True
    )
