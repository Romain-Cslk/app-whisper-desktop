"""Serializable transcription options; credentials are deliberately separate."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.config import LANGS, MODELS_CLOUD, MODELS_LOCAL, OUTPUT_PROMPTS
from ..services.diarization_options import DEFAULT_CLUSTERING_THRESHOLD, validate_diarization_options
from .jobs import ValidationError


@dataclass(frozen=True)
class TranscriptionOptions:
    mode: str = "local"
    model: str = "base"
    language: str | None = "fr"
    output_type: str = "transcription"
    output_name: str = ""
    compute_device: str = "cpu"
    diarization_enabled: bool = False
    self_speaker_name: str = "Moi"
    expected_speakers: int = 0
    clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD

    @property
    def use_api(self) -> bool:
        return self.mode in {"api", "openai"}

    def validate(self, api_key: str = "") -> None:
        try:
            validate_diarization_options(self.expected_speakers, self.clustering_threshold)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if self.mode not in {"local", "api", "openai"}:
            raise ValidationError("Mode de transcription inconnu.")
        if self.compute_device not in {"cpu", "cuda"}:
            raise ValidationError("Calcul local : choisissez CPU ou GPU NVIDIA CUDA.")
        choices = MODELS_CLOUD if self.use_api else MODELS_LOCAL.values()
        if self.model not in choices:
            raise ValidationError("Modèle de transcription inconnu.")
        if self.language is not None and self.language not in LANGS.values():
            raise ValidationError("Langue inconnue.")
        if self.output_type not in {"transcription", *OUTPUT_PROMPTS}:
            raise ValidationError("Format de sortie inconnu.")
        name = " ".join(str(self.self_speaker_name or "").strip().split())
        if self.diarization_enabled and not name:
            raise ValidationError("Indiquez le nom à afficher lorsque vous parlez.")
        if len(name) > 80:
            raise ValidationError("Le nom personnel est limité à 80 caractères.")
        if (self.use_api or self.output_type != "transcription") and not api_key.strip():
            raise ValidationError("Une clé OpenAI est nécessaire pour ce traitement.")
