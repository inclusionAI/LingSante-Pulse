"""Array preprocessing only; no dataset readers or data loaders."""

from fractions import Fraction
import math

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly


# Numeric channels use physical values, rather than window standardization.
FIXED_SCALES = {
    "spo2": (95.0, 5.0),
    "heart_rate": (70.0, 30.0),
    "rr_primus": (20.0, 20.0),
    "rr_solar": (20.0, 20.0),
    "bis_emg_power": (0.0, 100.0),
    "pulse_rate": (70.0, 30.0),
}


def as_batch(signals):
    """Canonical shapes: [T], [C,T], or [B,C,T] -> [B,C,T]."""
    if hasattr(signals, "detach"):
        signals = signals.detach().cpu().float().numpy()
    values = np.asarray(signals)
    if np.iscomplexobj(values) or values.dtype.kind not in "fiu":
        raise ValueError("signals must contain real numeric values")
    values = values.astype(np.float32)
    if values.ndim == 1:
        values = values[None, None]
    elif values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("signals must have nonempty shape [T], [C,T], or [B,C,T]")
    return values


def resample_channel(values, valid, hz, *, slow=False):
    """128-Hz polyphase waveform resampling or numeric zero-order hold.

    Invalid source neighborhoods remain invalid after FIR filtering. The
    output contains the nearest integer number of 128-Hz samples.
    """
    length = int(round(len(values) * 128.0 / hz))
    if length < 1:
        raise ValueError("Input duration is shorter than one 128-Hz sample")
    values = np.where(valid, values, 0.0).astype(np.float32)
    if slow:
        indices = np.minimum(
            (np.arange(length) * hz / 128).astype(int), len(values) - 1
        )
        return values[indices], valid[indices]
    if hz == 128.0:
        return values, valid.copy()
    if len(values) < 2:
        raise ValueError("Waveform resampling requires at least two source samples")
    ratio = Fraction(128.0 / hz).limit_denominator(100000)
    up, down = ratio.numerator, ratio.denominator
    if max(up, down) > 100000:
        raise ValueError(
            "sample_rate requires an impractically large resampling filter"
        )
    out = resample_poly(values, up, down, padtype="line")[:length].astype(np.float32)
    if valid.all():
        mask = np.ones(length, bool)
    else:
        radius = math.ceil(10 * max(up, down) / up)
        bad = maximum_filter1d(
            (~valid).astype(np.uint8), size=2 * radius + 1, mode="constant", cval=0
        )
        indices = np.minimum(
            (np.arange(length) * down / up).astype(int), len(valid) - 1
        )
        mask = ~bad[indices].astype(bool)
    if len(out) != length:
        raise ValueError("Resampling changed the requested physical duration")
    return np.where(mask, out, 0.0), mask


def normalize_channel(values, observed):
    """Observed-only population mean/std; float64 reductions, 1e-12 floor."""
    count = observed.sum()
    if not count:
        raise ValueError(
            "Each channel window must contain observed samples after resampling"
        )
    values = np.where(observed, values, 0.0).astype(np.float64)
    mean = values.sum() / count
    centered = np.where(observed, values - mean, 0.0)
    std = np.sqrt((centered * centered).sum() / count)
    return (centered / max(std, 1e-12)).astype(np.float32)


def _mask_batch(mask, shape, name):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype != bool:
        raise ValueError(f"{name} must be a boolean mask")
    if mask.ndim == 1:
        mask = mask[None, None]
    elif mask.ndim == 2:
        mask = mask[None]
    if mask.shape != shape:
        raise ValueError(f"{name} must have the same shape as signals")
    return mask


def prepare(signals, sample_rate, channel_names, *, observed=None, sample_present=None):
    """Resample, normalize independently, then right-pad to patch boundaries.

    ``observed`` excludes missing samples. ``sample_present`` excludes
    storage padding; missing values within physical support still occupy
    time. Optional masks use the same shape convention as signals.
    """
    values = as_batch(signals)
    batch, channels, _ = values.shape
    if len(channel_names) != channels:
        raise ValueError("Provide exactly one channel name per physical channel")
    if isinstance(sample_rate, (bool, np.bool_)):
        raise ValueError("sample_rate must be a positive finite number in Hz")
    rates = np.asarray(sample_rate, dtype=np.float64)
    if rates.ndim == 0:
        rates = np.full(channels, rates.item())
    if rates.shape != (channels,) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError(
            "sample_rate must be positive Hz, scalar or one value per channel"
        )
    obs = np.isfinite(values)
    if observed is not None:
        obs &= _mask_batch(observed, values.shape, "observed")
    pm = np.ones_like(obs)
    if sample_present is not None:
        pm = _mask_batch(sample_present, values.shape, "sample_present")
    obs &= pm
    durations = np.asarray([values.shape[-1] / rate for rate in rates])
    lengths = np.rint(durations * 128).astype(int)
    if not np.all(lengths == lengths[0]):
        raise ValueError(
            "Channels must describe the same physical duration; use a dict for different native rates"
        )
    length = int(lengths[0])
    padded = math.ceil(length / 32) * 32
    output = np.zeros((batch, channels, padded), np.float32)
    output_obs = np.zeros_like(output, bool)
    output_pm = np.zeros_like(output, bool)
    for i in range(batch):
        for c, name in enumerate(channel_names):
            slow = name in FIXED_SCALES
            x, valid = resample_channel(values[i, c], obs[i, c], rates[c], slow=slow)
            indices = np.minimum(
                (np.arange(length) * rates[c] / 128).astype(int), values.shape[-1] - 1
            )
            present = pm[i, c, indices]
            valid &= present
            if not valid.any():
                raise ValueError(
                    f"No observed samples after resampling: window {i}, channel {name}"
                )
            if slow:
                mean, scale = FIXED_SCALES[name]
                x = np.where(valid, np.clip((x - mean) / scale, -10, 10), 0.0)
            else:
                x = normalize_channel(x, valid)
            output[i, c, :length] = x
            output_obs[i, c, :length] = valid
            output_pm[i, c, :length] = present
    return output, output_obs, output_pm
