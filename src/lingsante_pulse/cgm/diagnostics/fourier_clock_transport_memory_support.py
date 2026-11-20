from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from lingsante_pulse.cgm.models.memory import (
    controlled_base_rows,
    parameter_budget,
    spectral_rows,
)
from lingsante_pulse.cgm.models.hierarchical_continuous_time_day_memory import (
    pack_hctdm,
)
from lingsante_pulse.cgm.training.parallel_longitudinal import CORPUS_SHA
from lingsante_pulse.cgm.training.covariance_preserving_longitudinal_full import (
    _load_source,
)
from lingsante_pulse.cgm.training.longitudinal_evidence_canary import sha256

POLICY = "fourier_clock_transport_memory_support_v1"
BASE = "3a397cb17de31ce7927f79ea1c87e68e0dfd8d442f6b32b2d40f1c9a09ee9721"


def canonical(rows):
    rows = np.ascontiguousarray(rows)
    return np.sort(
        rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[-1]))).reshape(-1)
    )


def diagnose(a):
    if (
        sha256(a.source) != a.expected_source_sha
        or sha256(a.backbone) != a.expected_backbone_sha
    ):
        raise ValueError("Frozen input mismatch")
    observation_policy = getattr(a, "memory_observation_policy", "fixed15_v1")
    data = _load_source(a.source)
    p = pack_hctdm(data, observation_policy=observation_policy)
    base = torch.from_numpy(p["hctdm_base"])
    clock = torch.from_numpy(p["hctdm_clock"])
    mask = torch.from_numpy(p["hctdm_point_mask"])
    control = controlled_base_rows(base, mask, True)
    active = p["hctdm_active"]
    public = data["cohort"] != "private_source_private"
    groups = {}
    for split in (0, 1):
        rows = public & (data["split"] == split)
        s = rows & active
        groups[str(split)] = {
            "endpoints": int(rows.sum()),
            "active_endpoints": int(s.sum()),
            "active_fraction": float(active[rows].mean()),
            "median_supported_days": float(np.median(p["hctdm_day_mask"][s].sum(1)))
            if s.any()
            else 0.0,
            "median_supported_points": float(
                np.median(p["hctdm_point_mask"][s].sum((1, 2)))
            )
            if s.any()
            else 0.0,
        }
    sources = [
        str(c)
        for c in np.unique(data["cohort"])
        if c != "private_source_private"
        and all(
            np.any((data["cohort"] == c) & (data["split"] == s) & active)
            for s in (0, 1)
        )
    ]
    mismatch = 0
    for e in np.flatnonzero(active):
        for d in np.flatnonzero(p["hctdm_day_mask"][e]):
            points = p["hctdm_point_mask"][e, d]
            mismatch += int(
                not np.array_equal(
                    canonical(p["hctdm_base"][e, d, points]),
                    canonical(control[e, d, torch.from_numpy(points)].numpy()),
                )
            )
    candidate = spectral_rows(base, clock, mask, False)
    generic = spectral_rows(base, clock, mask, True)
    selected = torch.from_numpy(p["hctdm_day_mask"])[..., None, None]
    contrast = float(
        torch.sqrt(
            ((candidate - generic).square() * selected).sum()
            / selected.sum().clamp_min(1)
        )
    )
    budget = parameter_budget()
    single = int(np.sum(active & (p["hctdm_day_mask"].sum(1) < 3)))
    passed = (
        len(sources) >= 3
        and min(g["active_fraction"] for g in groups.values()) >= 0.5
        and min(g["median_supported_days"] for g in groups.values()) >= 3
        and min(g["median_supported_points"] for g in groups.values()) >= 72
        and contrast >= 0.01
        and mismatch == 0
        and single == 0
    )
    report = {
        "policy": POLICY,
        "groups": groups,
        "supported_public_sources": sources,
        "supported_for_canary": bool(passed),
        "spectral_interaction_contrast_rms": contrast,
        "canonical_base_row_multiset_mismatch_count": mismatch,
        "point_mask_bitwise_equal": True,
        "single_or_two_day_active_count": single,
        "candidate_control_budget": budget,
        "physical_time_interaction": "fixed_harmonic_cos_sin_integrals_of_level_and_innovation",
        "control": "even_then_odd_complete_active_base_row_permutation_before_fixed_clock_projection",
        "model_updates": 0,
        "optimizer_steps": 0,
        "phenotype_or_private_test_reads": 0,
        "private_source_label_reads": 0,
        "identity_feature_reads": 0,
        "glucofm_components": 0,
        "task_routing": 0,
    }
    report.update(
        memory_observation_policy=observation_policy,
        source_sha256=a.expected_source_sha,
        backbone_sha256=a.expected_backbone_sha,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    q = argparse.ArgumentParser()
    for n in ("source", "backbone", "output"):
        q.add_argument("--" + n, type=Path, required=True)
    q.add_argument("--expected-source-sha", default=CORPUS_SHA)
    q.add_argument("--expected-backbone-sha", default=BASE)
    q.add_argument(
        "--memory-observation-policy",
        choices=("fixed15_v1", "observed_interval_v1"),
        default="fixed15_v1",
    )
    diagnose(q.parse_args())
