"""2026-09-11 user amendment: energy-weighted rank is diagnostic, not a veto."""

POLICY = "user_20260911_rank_diagnostic_v2"
REQUIRED = (
    "finite",
    "current_mean",
    "current_branches",
    "longitudinal_gain",
    "longitudinal_branches",
    "recurrence_gain",
    "geometry",
    "history_used",
    "no_history_identity",
    "statistics_gradient",
    "seven_day_span",
    "subject_split_isolation",
)


def mechanism_passed(gates):
    missing = set(REQUIRED) - set(gates)
    if missing:
        raise ValueError("Missing mechanism evidence: " + ",".join(sorted(missing)))
    return all(gates[key] for key in REQUIRED)
