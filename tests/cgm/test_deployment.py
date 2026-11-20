"""Regression checks for the packaged CGM deployment interface."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
import lingsante_pulse

from lingsante_pulse.cgm import extract
from lingsante_pulse.cgm.evaluation.nested import candidate_records


def test_deployment_matches_original_synthetic_features():
    root = Path(__file__).resolve().parents[2]
    with np.load(root / "examples/cgm_synthetic.npz", allow_pickle=False) as data:
        payload = dict(data)
    z, history = extract(payload)
    with np.load(Path(__file__).with_name("reference_features.npz")) as expected:
        np.testing.assert_allclose(z, expected["embedding"], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(history, expected["history_days"])


def test_inference_does_not_import_training_or_evaluation():
    script = """
import sys
import numpy as np
from lingsante_pulse.cgm import extract
payload = dict(glucose_mg_dl=(100+np.sin(np.arange(288,dtype=np.float32)))[None,:],
               observed_mask=np.ones((1,288),bool),window_id=np.array(['example']),
               subject_id=np.array(['subject']),start_time=np.array(['2020-01-01T00:00:00']))
extract(payload)
assert not any(n.startswith(('lingsante_pulse.cgm.training',
                             'lingsante_pulse.cgm.evaluation',
                             'lingsante_pulse.cgm.diagnostics',
                             'lingsante_pulse.cgm.baselines')) for n in sys.modules)
"""
    env = {
        "PATH": os.defpath,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONPATH": str(Path(lingsante_pulse.__file__).resolve().parents[1]),
    }
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_nested_reproduction_requests_its_optional_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="separate 24-candidate"):
        candidate_records(tmp_path)
