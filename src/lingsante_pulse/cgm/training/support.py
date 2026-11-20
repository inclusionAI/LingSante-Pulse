"""Separate observed-memory safety from legacy density diagnostics."""

import math


def validate_memory_support(report, allow_sparse=False):
    if report.get("policy") != "fourier_clock_transport_memory_support_v1":
        raise ValueError("Unknown memory support report")
    forbidden = (
        "model_updates",
        "optimizer_steps",
        "phenotype_or_private_test_reads",
        "private_source_label_reads",
        "identity_feature_reads",
        "glucofm_components",
        "task_routing",
    )
    if any(report.get(key) != 0 for key in forbidden):
        raise ValueError("Forbidden support read/update")
    if not allow_sparse:
        if report.get("supported_for_canary") is not True:
            raise ValueError("Passing frozen support is required")
        return
    if (
        report.get("canonical_base_row_multiset_mismatch_count") != 0
        or report.get("point_mask_bitwise_equal") is not True
        or report.get("single_or_two_day_active_count") != 0
    ):
        raise ValueError("Memory observation isolation failure")
    if not math.isfinite(report.get("spectral_interaction_contrast_rms", float("nan"))):
        raise ValueError("Nonfinite memory support")
    for split, minimum in (("0", 64), ("1", 16)):
        group = report.get("groups", {}).get(split, {})
        count, total = group.get("active_endpoints", -1), group.get("endpoints", -1)
        if (
            not isinstance(count, int)
            or not isinstance(total, int)
            or not minimum <= count <= total
        ):
            raise ValueError("Insufficient real active memory endpoints")
        if not math.isclose(
            group.get("active_fraction", -1), count / total, abs_tol=1e-12
        ):
            raise ValueError("Inconsistent memory coverage")
        for key, minimum in (
            ("median_supported_days", 3),
            ("median_supported_points", 72),
        ):
            value = group.get(key, float("nan"))
            if not math.isfinite(value) or value < minimum:
                raise ValueError("Insufficient within-endpoint observations")
