"""Fixed extra-episode control; no change to the encoder or inference."""

import numpy as np
from lingsante_pulse.cgm.models.episodic_evidence import SubjectEpisodes
from lingsante_pulse.cgm.models.distribution_phase import HARD_GATES


def extra_sampler(subjects, fit):
    sampler = SubjectEpisodes(subjects, fit, seed=43)
    for _ in range(600):
        sampler.next()
    return sampler


def compare_extra_control(control, observed):
    gates = dict(
        structural=all(
            r["gates"].get(k) is True for r in (control, observed) for k in HARD_GATES
        ),
        budget=all(
            (r["seed"], r["steps"], r["parameters"], r["final_dimension"])
            == (43, 600, 10689, 146)
            for r in (control, observed)
        ),
        isolation=all(
            r["phenotype_or_private_test_reads"] == r["glucofm_components"] == 0
            for r in (control, observed)
        ),
        main_episodes=control["episode_sequence_sha256"]
        == observed["episode_sequence_sha256"],
        targets=control["target_sha256"] == observed["target_sha256"],
        extra_gradient=control["double_main_control"]["max_extra_gradient"] > 0,
        encoder_calls=control["double_main_control"]["training_encoder_calls"] == 2400,
        control_definition=control["double_main_control"]["distribution_control"]
        is False
        and control["double_main_control"]["skipped_episodes"] == 600,
        observed_definition=observed["cross_day_prediction"]["target_kind"]
        == "observed_last_day"
        and observed["cross_day_prediction"]["distribution_control"] is False,
    )
    values = lambda r: np.array(
        [r["learned"]["current_mean"], r["learned"]["longitudinal_mean"], r["event_ev"]]
    )
    delta = values(observed) - values(control)
    gates["finite"] = bool(np.isfinite(delta).all())
    outcome = (
        "observed_physical_majority"
        if (delta > 0).sum() >= 2
        else "extra_main_majority"
        if (delta < 0).sum() >= 2
        else "mixed_or_tied"
    )
    return dict(
        gates=gates,
        valid_comparison=all(gates.values()),
        observed_minus_extra_main=delta.tolist(),
        outcome=outcome if all(gates.values()) else "invalid",
        full_authorized=False,
        rank_policy="diagnostic_only",
    )
