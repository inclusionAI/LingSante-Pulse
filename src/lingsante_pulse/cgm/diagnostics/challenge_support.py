"""Label-free evidence for a challenge-token successor; never trains a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def proposal_support(values, observed):
    """Reproduce legacy proposal ranking, retaining indices for support checks.

    Exact same peak index proves duplicate descriptions of a peak; it does not
    establish that distinct peak indices correspond to independent meals.
    """
    observed = np.asarray(observed, bool)
    values = np.asarray(values, np.float32)
    if values.shape != observed.shape or observed.sum() < 2:
        raise ValueError("Aligned trace with at least two observations required")
    idx = np.flatnonzero(observed)
    # Match legacy float32 interpolation and subtraction *before* Python-float
    # scoring. Otherwise near-tied proposals can be reordered by float64 math.
    filled = np.interp(np.arange(len(values)), idx, values[idx]).astype(np.float32)
    proposals = []
    for start in range(len(values) - 13):
        end = start + 12
        if not (observed[start] and observed[end]):
            continue
        # Coverage threshold is applied separately by the caller's cadence.
        rise = filled[end] - filled[start]
        if rise < 30.0:
            continue
        stop = min(len(values), end + 25)
        possible = np.flatnonzero(observed[end:stop]) + end
        if not len(possible):
            continue
        peak = int(possible[np.argmax(filled[possible])])
        after = min(len(values) - 1, peak + 24)
        amplitude = float(filled[peak] - filled[start])
        recovery = float(filled[peak] - filled[after])
        score = amplitude + max(recovery, 0.0)
        proposals.append(
            dict(
                start=start,
                peak=peak,
                after=after,
                score=float(score),
                rising_coverage=float(observed[start : end + 1].mean()),
            )
        )
    return sorted(proposals, key=lambda p: p["score"], reverse=True)


def inspect_trace(values, observed, cadence):
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("Positive native cadence required")
    observed = np.asarray(observed, bool)
    expected = min(1.0, 5.0 / cadence)
    selected = [
        p
        for p in proposal_support(values, observed)
        if p["rising_coverage"] >= 0.5 * expected
    ][:3]
    result = dict(
        windows=1,
        windows_with_proposals=int(bool(selected)),
        selected=len(selected),
        repeated_exact_peak=0,
        overlap_pairs=0,
        selected_pairs=0,
        recovery_clipped=0,
        recovery_endpoint_unobserved=0,
        recovery_long_gap=0,
    )
    result["repeated_exact_peak"] = len(selected) - len({p["peak"] for p in selected})
    for j, p in enumerate(selected):
        result["recovery_clipped"] += int(p["peak"] + 24 >= len(values))
        result["recovery_endpoint_unobserved"] += int(not observed[p["after"]])
        points = np.flatnonzero(observed[p["peak"] : p["after"] + 1]) + p["peak"]
        # Include the unsupported tail: zero observations there are not recovery.
        gaps = np.diff(np.r_[points, p["after"]]) * 5.0
        result["recovery_long_gap"] += int(
            gaps.max(initial=0.0) > max(15.0, 2.0 * cadence)
        )
        for q in selected[:j]:
            result["selected_pairs"] += 1
            result["overlap_pairs"] += int(
                max(p["start"], q["start"]) <= min(p["after"], q["after"])
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from lingsante_pulse.cgm.training.parallel_longitudinal import CORPUS_SHA
    from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
        _load_source,
    )
    from lingsante_pulse.cgm.models.longitudinal_evidence_encoder import (
        _challenge_features,
    )

    actual = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if actual != CORPUS_SHA:
        raise ValueError("Frozen corpus mismatch")
    data = _load_source(args.source)
    if set(np.unique(data["split"])) != {0, 1}:
        raise ValueError("Train/validation corpus only")
    groups = {}
    legacy_mismatch = 0
    for i, (values, mask, cadence) in enumerate(
        zip(data["glucose"], data["observed"], data["cadence"])
    ):
        row = inspect_trace(values, mask, cadence)
        # Compare reconstructed legacy amplitudes against the actual old code.
        chosen = [
            p
            for p in proposal_support(values, mask)
            if p["rising_coverage"] >= 0.5 * min(1.0, 5.0 / cadence)
        ][:3]
        old, _ = _challenge_features(values, mask, float(cadence))
        for j, p in enumerate(chosen):
            legacy_mismatch += int(
                not np.isclose(old[2 + 5 * j], values[p["peak"]] - values[p["start"]])
            )
        for key in (
            "all",
            "split_" + str(int(data["split"][i])),
            "cohort_" + str(data["cohort"][i]),
            "cadence_" + str(float(cadence)),
        ):
            target = groups.setdefault(key, {k: 0 for k in row})
            for k, v in row.items():
                target[k] += v
    report = dict(
        corpus_sha256=actual,
        groups=groups,
        legacy_amplitude_mismatch=legacy_mismatch,
        phenotype_reads=0,
        private_test_reads=0,
        model_training=False,
        interpretation="Exact-peak duplicates and recovery support only; not meal detection accuracy or disease causality.",
    )
    if legacy_mismatch:
        raise ValueError("Diagnostic no longer matches legacy extraction")
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
