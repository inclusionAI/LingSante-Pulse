# Validation

Run the model and protocol checks from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests cover waveform checkpoint loading, input/channel handling and reference features; CGM synthetic reference features, signed dynamics and longitudinal modules; and participant-isolated evaluation with training-only candidate selection.

CGM reference features are generated from the original deployment implementation using synthetic inputs. Inference loads the packaged weights without training or evaluation modules. File-integrity checks run when pretrained assets are loaded.

The fixed-model and optional nested evaluations use the same biological-person grouping and linear-probe protocol. Original reference aggregates are available under `lingsante_pulse/cgm/assets/benchmarks` in the installed package. Full evaluation requires the public datasets prepared as described in [DATA.md](DATA.md); optional nested selection additionally requires the 24-candidate bundle.

For numerical benchmark comparisons, use the versions in `requirements.txt`. The reference benchmark used accelerated daily extraction and CPU longitudinal extraction; floating-point outputs can vary across backends.

Waveform reference tests use downloaded weights. Set `LINGSANTE_PULSE_MODEL_DIR` to a local model directory, or run waveform inference once to populate the Hugging Face cache. CGM tests use the packaged deployment weights.
