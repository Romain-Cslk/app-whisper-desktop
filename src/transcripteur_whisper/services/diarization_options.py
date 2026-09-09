"""One validation contract shared by UI options, parent process and worker."""
from __future__ import annotations

import math

DEFAULT_CLUSTERING_THRESHOLD = 0.60
MAX_EXPECTED_SPEAKERS = 32


def validate_diarization_options(expected_speakers: int, clustering_threshold: float) -> tuple[int, float]:
    if type(expected_speakers) is not int or not 0 <= expected_speakers <= MAX_EXPECTED_SPEAKERS:
        raise ValueError(f"Nombre de locuteurs : Auto (0) ou entier de 1 \u00e0 {MAX_EXPECTED_SPEAKERS}.")
    if isinstance(clustering_threshold, bool) or not isinstance(clustering_threshold, (int, float)):
        raise ValueError("Le seuil de regroupement doit \u00eatre un nombre.")
    threshold = float(clustering_threshold)
    if not math.isfinite(threshold) or not 0.20 <= threshold <= 0.90:
        raise ValueError("Le seuil de regroupement doit \u00eatre compris entre 0,20 et 0,90.")
    return expected_speakers, threshold
