"""Explicit channel identity normalization without text model inference."""

from collections.abc import Mapping
from difflib import get_close_matches


ALIASES = {
    "electrocardiogram": "ecg",
    "electrocardiography": "ecg",
    "photoplethysmogram": "pleth",
    "photoplethysmography": "pleth",
    "ppg": "pleth",
    "electromyogram": "emg",
    "resp": "airflow",
    "respiration": "airflow",
    "i": "ecg_i",
    "ii": "ecg_ii",
    "iii": "ecg_iii",
    "avr": "ecg_avr",
    "avl": "ecg_avl",
    "avf": "ecg_avf",
    **{f"v{i}": f"ecg_v{i}" for i in range(1, 7)},
}
MODALITIES = {
    "electrocardiogram": "ecg",
    "electrocardiography": "ecg",
    "electroencephalogram": "eeg",
    "electrooculogram": "eog",
    "electromyogram": "emg",
    "photoplethysmogram": "ppg",
    "pleth": "ppg",
    "respiration": "airflow",
    "resp": "airflow",
}
ECG_LEADS = {
    "1": "i",
    "2": "ii",
    "3": "iii",
    "i": "i",
    "ii": "ii",
    "iii": "iii",
    "avr": "avr",
    "avl": "avl",
    "avf": "avf",
    "d3": "d3",
    "mli": "mli",
    "mlii": "mlii",
    "mliii": "mliii",
    **{f"v{i}": f"v{i}" for i in range(1, 7)},
}
GENERIC = {
    "ecg": "ecg",
    "ppg": "pleth",
    "emg": "emg",
    "airflow": "airflow",
    "spo2": "spo2",
    "heart_rate": "heart_rate",
}


def normalize_names(names):
    if isinstance(names, str):
        names = [names]
    if names is None:
        raise ValueError("Provide channel_names or channels metadata")
    result = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Channel names must be nonempty strings")
        key = name.strip().lower()
        result.append(ALIASES.get(key, key))
    return result


def resolve_metadata(channels=None, channel_names=None):
    """Return canonical identities from explicit channel metadata.

    lead=None/'' means unspecified. ECG numeric leads are interpreted only
    within structured ECG metadata. Unknown leads are never guessed.
    Generic EEG/EOG names never silently choose a montage or eye side.
    """
    if channels is not None:
        if channel_names is not None:
            raise ValueError("Use channels metadata or channel_names, not both")
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError(
                "channels must be a nonempty list of metadata dictionaries"
            )
        names = []
        for item in channels:
            if not isinstance(item, Mapping) or set(item) - {"modality", "lead"}:
                raise ValueError("Each channel accepts modality and optional lead")
            modality = item.get("modality")
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("Each channel requires a nonempty modality")
            modality = modality.strip().lower()
            modality = MODALITIES.get(modality, modality)
            lead = item.get("lead")
            if isinstance(lead, bool) or (
                lead is not None and not isinstance(lead, (str, int))
            ):
                raise ValueError("lead must be a string, integer, or None")
            lead = "" if lead is None else str(lead).strip()
            if modality == "ecg" and lead:
                canonical = ECG_LEADS.get(lead.lower())
                if canonical is None:
                    raise ValueError(
                        f"Unknown ECG lead {lead!r}; choose a supported lead"
                    )
                name = "ecg_" + canonical
            elif not lead:
                name = GENERIC.get(modality, modality)
            else:
                name = modality + "_" + lead.lower()
            names.append(name)
    else:
        names = normalize_names(channel_names)
    return names


def unknown_message(names, known):
    suggestions = {
        name: get_close_matches(name, known, n=3, cutoff=0.6) for name in names
    }
    return (
        f"Unknown channels {names}. Suggestions (not applied): {suggestions}. "
        "Choose one of model.channel_names."
    )
